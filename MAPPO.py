#!/usr/bin/env python
# MAPPO.py
# MAPPO baseline (shared-weights PPO) + CG-HMT-compatible metrics logging.
# Robust on Windows: fixes MACAD relative-path issues and CARLA slow startup.

from __future__ import absolute_import, division, print_function

import argparse
import datetime
import math
import os
import time

import cv2
import gym
import numpy as np

# IMPORTANT: import macad_gym at module import time (matches CG-HMT behavior)
import macad_gym  # noqa: F401

import ray
from gym.spaces import Box, Discrete
from ray import tune
from ray.tune.registry import register_env

from ray.rllib.env import MultiAgentEnv
from rllib.env_wrappers import wrap_deepmind
from rllib.models import register_mnih15_shared_weights_net
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy as PPOPolicy
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor


# ------------------------------------------------------------
# Ray 0.8.6 Windows checkpoint bug workaround
# ------------------------------------------------------------
try:
    from ray.tune import trial_runner as _tr_mod

    _orig_checkpoint = _tr_mod.TrialRunner.checkpoint

    def _safe_checkpoint(self, force=False):
        try:
            return _orig_checkpoint(self, force=force)
        except FileExistsError:
            print("[WARN] Ignoring FileExistsError in TrialRunner.checkpoint() on Windows.")
            return

    _tr_mod.TrialRunner.checkpoint = _safe_checkpoint
except Exception as e:
    print("[WARN] Could not monkeypatch TrialRunner.checkpoint:", e)
# ------------------------------------------------------------


# ============================================================
# Script dir anchoring (prevents MACAD relative-path issues)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


# ============================================================
# 1) CLI
# ============================================================
parser = argparse.ArgumentParser("MAPPO baseline (shared PPO) with MACAD metrics logger")

parser.add_argument("--num-iters", type=int, default=300)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--num-gpus", type=int, default=1)

parser.add_argument("--rollout-fragment-length", type=int, default=200)
parser.add_argument("--train-batch-size", type=int, default=12000)
parser.add_argument("--envs-per-worker", type=int, default=1)

parser.add_argument("--metrics-csv", type=str, default="mappo_metrics.csv")
parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)

# CARLA boot can be slow on Windows; extend beyond MACAD's internal 30 attempts.
parser.add_argument("--reset-retries", type=int, default=6)     # additional retries
parser.add_argument("--retry-sleep", type=float, default=8.0)   # seconds between retries

parser.add_argument("--notes", default=None)

PREFERRED_GYM_ENV_ID = "HomoNcomIndePOIntrxMASS3CTWN3-v0"
FALLBACK_GYM_ENV_ID = "HomoNcomIndePOIntrxMASS3CTWN3-v0"

# Tune/RLlib env name must be different from Gym env id
RLLIB_ENV_NAME = "MACAD_MASS3_MAPPO_BASELINE"


# ============================================================
# 2) Model registration (same backbone as CG-HMT)
# ============================================================
register_mnih15_shared_weights_net()
MODEL_NAME = "mnih15_shared_weights"


# ============================================================
# 3) Resolve Gym env id on driver
# ============================================================
def _resolve_gym_env_id(preferred_id, fallback_id):
    try:
        env = gym.make(preferred_id)
        env.close()
        print("[INFO] Using MACAD env:", preferred_id)
        return preferred_id
    except gym.error.UnregisteredEnv:
        print("[WARN] Preferred env not found:", preferred_id)
        env = gym.make(fallback_id)
        env.close()
        print("[INFO] Falling back to:", fallback_id)
        return fallback_id


# ============================================================
# 4) 84x84 preprocessor (kept to match your pipeline)
# ============================================================
class ImagePreproc(Preprocessor):
    def _init_shape(self, obs_space, options):
        self.shape = (84, 84, 3)
        return self.shape

    def transform(self, observation):
        return cv2.resize(observation, (self.shape[0], self.shape[1]))

ModelCatalog.register_custom_preprocessor("sq_im_84", ImagePreproc)


# ============================================================
# 5) Robust reset wrapper (extends CARLA startup tolerance)
# ============================================================
class ResetRetryWrapper(MultiAgentEnv):
    def __init__(self, env, max_retries=6, sleep_s=8.0):
        self.env = env
        self.max_retries = int(max_retries)
        self.sleep_s = float(sleep_s)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *, seed=None, options=None):
        last_e = None
        for k in range(self.max_retries + 1):
            try:
                return self.env.reset()
            except RuntimeError as e:
                last_e = e
                msg = str(e)
                # only retry CARLA connection failures; otherwise re-raise
                if "Failed to connect to CARLA server" not in msg:
                    raise
                print(f"[WARN] CARLA not ready yet. retry {k}/{self.max_retries} in {self.sleep_s}s")
                time.sleep(self.sleep_s)
        raise last_e

    def step(self, action_dict):
        return self.env.step(action_dict)


# ============================================================
# 6) Optional spawn spacing wrapper (same as CG-HMT)
# ============================================================
class SpawnSpacingWrapper(MultiAgentEnv):
    def __init__(self, env, offset=1.0):
        self.env = env
        self.offset = offset
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        self._nudge_actors_safely()
        return obs

    def step(self, action_dict):
        return self.env.step(action_dict)

    def _nudge_actors_safely(self):
        for name in ["actors", "actor_dict", "_actors"]:
            if hasattr(self.env, name):
                actors = getattr(self.env, name)
                break
        else:
            return

        if isinstance(actors, dict):
            actor_list = [v for _, v in sorted(actors.items(), key=lambda kv: kv[0])]
        else:
            actor_list = list(actors)

        if not actor_list:
            return

        try:
            spacing = 8.0
            lead_tf = actor_list[0].get_transform()
            yaw_rad = math.radians(lead_tf.rotation.yaw)

            for i, actor in enumerate(actor_list):
                tf = actor.get_transform()
                tf.location = lead_tf.location
                tf.location.x -= i * spacing * math.cos(yaw_rad)
                tf.location.y -= i * spacing * math.sin(yaw_rad)
                tf.location.z = lead_tf.location.z
                actor.set_transform(tf)
        except Exception:
            return


# ============================================================
# 7) Metrics logger wrapper (same CSV schema as CG-HMT)
# ============================================================
class MetricsOnlyWrapper(MultiAgentEnv):
    """
    Columns match CG-HMT:
      episode,steps,time_sec,collisions_per_episode,route_completion_pct,
      infractions_per_100m,prompts_per_minute,avg_shaped_reward,cumulative_avg_shaped_reward

    For MAPPO baseline:
      prompts_per_minute = 0
      avg_shaped_reward  = avg base reward per step per agent
    """

    def __init__(self, env, metrics_csv_path, fixed_delta_seconds=0.05):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.metrics_csv_path = metrics_csv_path
        self.fixed_delta_seconds = float(fixed_delta_seconds)

        self.agent_state = {}
        self.episode_index = -1

        self._init_metrics_csv()
        self._reset_episode_counters()

        self.cumulative_sum = 0.0
        self.cumulative_count = 0

    def _init_metrics_csv(self):
        if not os.path.exists(self.metrics_csv_path):
            with open(self.metrics_csv_path, "w") as f:
                f.write(
                    "episode,steps,time_sec,collisions_per_episode,"
                    "route_completion_pct,infractions_per_100m,"
                    "prompts_per_minute,avg_shaped_reward,"
                    "cumulative_avg_shaped_reward\n"
                )

    def _reset_episode_counters(self):
        self.episode_steps = 0
        self.episode_total_reward = 0.0
        self.episode_distance_total = 0.0
        self.episode_collisions_total = 0.0
        self.episode_infractions_total = 0.0
        self.episode_prompts = 0  # always 0 in this baseline

    @staticmethod
    def _extract_xy(info_for_agent):
        if info_for_agent is None:
            return 0.0, 0.0

        if "location" in info_for_agent:
            loc = info_for_agent["location"]
            if isinstance(loc, dict):
                x = float(loc.get("x", 0.0))
                y = float(loc.get("y", 0.0))
            else:
                x = float(getattr(loc, "x", 0.0))
                y = float(getattr(loc, "y", 0.0))
        else:
            x = float(info_for_agent.get("x", 0.0))
            y = float(info_for_agent.get("y", 0.0))
        return x, y

    def _init_agent_state(self, obs_dict):
        self.agent_state = {}
        for aid in obs_dict.keys():
            self.agent_state[aid] = {
                "init_distance": None,
                "distance_last": None,
                "prev_x": None,
                "prev_y": None,
                "prev_collision_sum": 0.0,
                "prev_infractions_raw": 0.0,
            }

    def _update_per_agent_stats(self, aid, info):
        st = self.agent_state.setdefault(aid, {
            "init_distance": None,
            "distance_last": None,
            "prev_x": None,
            "prev_y": None,
            "prev_collision_sum": 0.0,
            "prev_infractions_raw": 0.0,
        })

        x, y = self._extract_xy(info)
        if st["prev_x"] is not None:
            dx = x - st["prev_x"]
            dy = y - st["prev_y"]
            step_dist = math.sqrt(dx * dx + dy * dy)
            step_dist = max(step_dist, 0.0)
            self.episode_distance_total += step_dist
        st["prev_x"], st["prev_y"] = x, y

        d_goal = None
        if info is not None:
            if "distance_to_goal" in info:
                d_goal = float(info["distance_to_goal"])
            elif "distance_to_goal_euclidean" in info:
                d_goal = float(info["distance_to_goal_euclidean"])
        if d_goal is not None:
            if st["init_distance"] is None and d_goal > 0.0:
                st["init_distance"] = d_goal
            st["distance_last"] = d_goal

        coll_sum = 0.0
        if info is not None:
            coll_sum += float(info.get("collision_vehicles", 0.0))
            coll_sum += float(info.get("collision_pedestrians", 0.0))
            coll_sum += float(info.get("collision_other", 0.0))
        delta_coll = max(coll_sum - st["prev_collision_sum"], 0.0)
        st["prev_collision_sum"] = coll_sum
        self.episode_collisions_total += delta_coll

        infr_raw = 0.0
        if info is not None:
            infr_raw += float(info.get("intersection_offroad", 0.0))
            infr_raw += float(info.get("intersection_otherlane", 0.0))
        delta_inf = max(infr_raw - st["prev_infractions_raw"], 0.0)
        st["prev_infractions_raw"] = infr_raw
        self.episode_infractions_total += delta_inf

    def _finalize_and_log_episode(self, num_agents):
        if self.episode_steps == 0:
            return

        time_sec = self.episode_steps * self.fixed_delta_seconds
        collisions_ep = self.episode_collisions_total

        if self.episode_distance_total > 1e-6:
            infractions_per_100m = (
                self.episode_infractions_total /
                max(self.episode_distance_total / 100.0, 1e-6)
            )
            infractions_per_100m = min(infractions_per_100m, 1e3)
        else:
            infractions_per_100m = 0.0

        prompts_per_minute = 0.0

        rc_vals = []
        for _, st in self.agent_state.items():
            init_d = st.get("init_distance", None)
            last_d = st.get("distance_last", None)
            if init_d is not None and init_d > 1e-3 and last_d is not None:
                rc = (init_d - last_d) / init_d * 100.0
                rc_vals.append(float(np.clip(rc, 0.0, 100.0)))
        route_completion_pct = float(sum(rc_vals) / len(rc_vals)) if rc_vals else 0.0

        avg_reward = (
            self.episode_total_reward / (self.episode_steps * num_agents)
            if num_agents > 0 else 0.0
        )

        self.cumulative_sum += avg_reward
        self.cumulative_count += 1
        cumulative_avg = self.cumulative_sum / self.cumulative_count

        with open(self.metrics_csv_path, "a") as f:
            f.write(
                f"{self.episode_index},{self.episode_steps},{time_sec:.3f},"
                f"{collisions_ep:.3f},{route_completion_pct:.6f},"
                f"{infractions_per_100m:.6f},{prompts_per_minute:.6f},"
                f"{avg_reward:.6f},{cumulative_avg:.6f}\n"
            )

        print(
            f"[METRICS] ep={self.episode_index} steps={self.episode_steps} "
            f"coll={collisions_ep:.1f} rc={route_completion_pct:.2f}% "
            f"infrac/100m={infractions_per_100m:.3f} ppm={prompts_per_minute:.2f} "
            f"avgR={avg_reward:.4f} cumAvgR={cumulative_avg:.4f}"
        )

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        self.episode_index += 1
        self._reset_episode_counters()
        self._init_agent_state(obs)
        return obs

    def step(self, action_dict):
        self.episode_steps += 1
        obs, rewards, done_dict, info_dict = self.env.step(action_dict)

        agent_ids = [aid for aid in obs.keys() if aid != "__all__"]
        for aid in agent_ids:
            info = info_dict.get(aid, {})
            self._update_per_agent_stats(aid, info)
            self.episode_total_reward += float(rewards.get(aid, 0.0))

        if done_dict.get("__all__", False):
            self._finalize_and_log_episode(num_agents=len(agent_ids))

        return obs, rewards, done_dict, info_dict


# ============================================================
# 8) Env factory (runs inside Ray worker)
# ============================================================
def make_env_with_wrappers(env_config):
    # re-anchor working directory inside worker
    worker_dir = env_config.get("script_dir", SCRIPT_DIR)
    try:
        os.chdir(worker_dir)
    except Exception:
        pass

    # ensure MACAD envs are registered in this process
    import macad_gym  # noqa: F401

    gym_env_id = env_config["gym_env_id"]
    env = gym.make(gym_env_id)

    env = ResetRetryWrapper(
        env,
        max_retries=env_config.get("reset_retries", 6),
        sleep_s=env_config.get("retry_sleep", 8.0),
    )

    env = SpawnSpacingWrapper(env, offset=1.0)

    framestack = env_config.get("framestack", 4)
    env = wrap_deepmind(env, dim=84, num_framestack=framestack)

    env = MetricsOnlyWrapper(
        env,
        metrics_csv_path=env_config.get("metrics_csv_path", "mappo_metrics.csv"),
        fixed_delta_seconds=env_config.get("fixed_delta_seconds", 0.05),
    )
    return env


# ============================================================
# 9) main
# ============================================================
if __name__ == "__main__":
    args = parser.parse_args()

    gym_env_id = _resolve_gym_env_id(PREFERRED_GYM_ENV_ID, FALLBACK_GYM_ENV_ID)

    os.environ.setdefault("TUNE_GLOBAL_CHECKPOINT_S", "0")

    ray.init(num_gpus=args.num_gpus)

    register_env(RLLIB_ENV_NAME, lambda cfg: make_env_with_wrappers(cfg))

    obs_space = Box(0.0, 255.0, shape=(84, 84, 3))
    act_space = Discrete(9)

    def gen_shared_policy():
        cfg = {
            "model": {
                "custom_model": MODEL_NAME,
                "custom_options": {"notes": {"notes": args.notes}},
                "custom_preprocessor": "sq_im_84",
                "dim": 84,
                "free_log_std": False,
                "grayscale": False,
            },
        }
        return (PPOPolicy, obs_space, act_space, cfg)

    policies = {"shared_platoon_policy": gen_shared_policy()}

    def policy_mapping_fn(agent_id, **kwargs):
        return "shared_platoon_policy"

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    local_dir = os.path.join(os.getcwd(), "ray_results", f"mappo_baseline_run_{ts}")

    analysis = tune.run(
        "PPO",
        name="MAPPO-BASELINE-SHARED-PPO-CARLA-MASS",
        stop={"training_iteration": args.num_iters},
        local_dir=local_dir,
        config={
            "env": RLLIB_ENV_NAME,
            "env_config": {
                "gym_env_id": gym_env_id,
                "framestack": 4,
                "metrics_csv_path": args.metrics_csv,
                "fixed_delta_seconds": args.fixed_delta_seconds,
                "reset_retries": args.reset_retries,
                "retry_sleep": args.retry_sleep,
                "script_dir": SCRIPT_DIR,
            },
            "log_level": "INFO",
            "framework": "torch",

            "num_gpus": args.num_gpus,
            "num_workers": args.num_workers,
            "num_envs_per_worker": args.envs_per_worker,
            "simple_optimizer": True,

            "rollout_fragment_length": args.rollout_fragment_length,
            "train_batch_size": args.train_batch_size,
            "num_sgd_iter": 10,

            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            },
        },
        checkpoint_freq=10,
        checkpoint_at_end=True,
        keep_checkpoints_num=1,
        checkpoint_score_attr="episode_reward_mean",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean")
    checkpoints = analysis.get_trial_checkpoints_paths(trial=best_trial, metric="episode_reward_mean")
    if checkpoints:
        best_checkpoint = max(checkpoints, key=lambda x: x[1])[0]
        print("Best checkpoint path:", best_checkpoint)
    else:
        print("No checkpoints found for the best trial.")
