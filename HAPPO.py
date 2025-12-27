#!/usr/bin/env python
# HAPPO.py
# ---------------------------------------------------------------------
# HAPPO baseline for MACAD-Gym (CARLA) using RLlib (Ray 0.8.6):
# - Multi-policy PPO: policy_0 ... policy_{max_agents-1}
# - Sequential per-agent PPO updates (one policy trains at a time)
# - Conservative PPO settings (clip + KL) to reduce non-stationarity
#
# IMPORTANT: This file is structured so evaluate_checkpoint.py can work
#            without modifications by importing this script and using:
#              - make_env_with_wrappers(...)
#              - get_eval_policies_and_mapping(...)
#
# Train:
#   python HAPPO.py --num-iters 100 --train-batch-size 12000 --rollout-fragment-length 200
#
# Evaluate (with your evaluate_checkpoint.py that imports --script):
#   python evaluate_checkpoint.py --script HAPPO.py --checkpoint "<...>\checkpoint-100" --seeds-file eval_seeds.txt --episodes 10 --out-prefix happo_eval
# ---------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import argparse
import datetime
import math
import os
import re
import time
from typing import Any, Dict, Tuple

import cv2
import gym
import numpy as np

import macad_gym  # noqa: F401 (registers MACAD envs)

import ray
from gym.spaces import Box, Discrete
from ray.tune.registry import register_env

# RLlib 0.8.6 APIs
from ray.rllib.env import MultiAgentEnv
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor

from rllib.env_wrappers import wrap_deepmind
from rllib.models import register_mnih15_shared_weights_net


# ============================================================
# Script dir anchoring (prevents MACAD relative-path surprises on Windows/Ray)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


# ============================================================
# 1) CLI
# ============================================================
parser = argparse.ArgumentParser("HAPPO baseline (sequential per-agent PPO) for MACAD")

parser.add_argument("--num-iters", type=int, default=300)
parser.add_argument("--num-workers", type=int, default=0)  # keep 0 for stability on Windows
parser.add_argument("--num-gpus", type=int, default=1)

parser.add_argument("--rollout-fragment-length", type=int, default=200)
parser.add_argument("--train-batch-size", type=int, default=12000)
parser.add_argument("--envs-per-worker", type=int, default=1)

parser.add_argument("--metrics-csv", type=str, default="happo_metrics.csv")
parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)

# CARLA boot can be slow on Windows; extend reset tolerance
parser.add_argument("--reset-retries", type=int, default=8)
parser.add_argument("--retry-sleep", type=float, default=8.0)

# HAPPO specifics
parser.add_argument("--max-agents", type=int, default=5)   # policies policy_0 ... policy_{max_agents-1}
parser.add_argument("--happo-steps-per-iter", type=int, default=1,
                    help="How many PPO train() calls per agent, per outer iteration (default: 1).")
parser.add_argument("--checkpoint-every", type=int, default=10)
parser.add_argument("--notes", default=None)

# Environment IDs (kept consistent with your other scripts)
PREFERRED_GYM_ENV_ID = "HomoNcomIndePOIntrxMASS3CTWN3-v0"
FALLBACK_GYM_ENV_ID = "HomoNcomIndePOIntrxMASS3CTWN3-v0"

# This is the RLlib "env name" we register into Ray's registry for training.
RLLIB_ENV_NAME = "MACAD_MASS3_HAPPO_BASELINE"

# These names are what your evaluator tries to read if present.
PREFERRED_ENV_NAME = PREFERRED_GYM_ENV_ID
FALLBACK_ENV_NAME = FALLBACK_GYM_ENV_ID


# ============================================================
# 2) Model registration (same backbone family)
# ============================================================
register_mnih15_shared_weights_net()
MODEL_NAME = "mnih15_shared_weights"


# ============================================================
# 3) Resolve Gym env id on driver (also exportable to evaluator)
# ============================================================
def _resolve_macad_env_name(preferred_id, fallback_id):
    """Helper used by evaluate_checkpoint.py if present."""
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
# 4) 84x84 preprocessor (kept to mirror your pipeline)
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
    def __init__(self, env, max_retries=8, sleep_s=8.0):
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
                if "Failed to connect to CARLA server" not in msg:
                    raise
                print(f"[WARN] CARLA not ready. retry {k}/{self.max_retries} in {self.sleep_s}s")
                time.sleep(self.sleep_s)
        raise last_e

    def step(self, action_dict):
        return self.env.step(action_dict)

    def close(self):
        try:
            return self.env.close()
        except Exception:
            return None


# ============================================================
# 6) Optional spawn spacing wrapper (same as your code)
# ============================================================
class SpawnSpacingWrapper(MultiAgentEnv):
    def __init__(self, env, offset=1.0):
        self.env = env
        self.offset = float(offset)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        self._nudge_actors_safely()
        return obs

    def step(self, action_dict):
        return self.env.step(action_dict)

    def close(self):
        try:
            return self.env.close()
        except Exception:
            return None

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
    CSV schema matches CG-HMT:
      episode,steps,time_sec,collisions_per_episode,route_completion_pct,
      infractions_per_100m,prompts_per_minute,avg_shaped_reward,cumulative_avg_shaped_reward

    For HAPPO baseline:
      prompts_per_minute = 0
      avg_shaped_reward  = avg base reward per step per agent (no shaping)
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

        # Gym wrappers often expose reward_range; keep it to avoid wrapper breakage in old gym stacks.
        self.reward_range = getattr(env, "reward_range", (-float("inf"), float("inf")))

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
        self.episode_prompts = 0  # baseline: no prompts

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
            if aid == "__all__":
                continue
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
            self.episode_distance_total += max(step_dist, 0.0)
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

    def close(self):
        try:
            return self.env.close()
        except Exception:
            return None


# ============================================================
# 8) Env factory that evaluator can call
# ============================================================
def make_env_with_wrappers(resolved_env_name: str, env_config: Dict[str, Any]):
    """
    Evaluator-friendly signature:
      make_env_with_wrappers(resolved_env_name, env_config)

    Your evaluate_checkpoint.py will pass:
      resolved_env_name = "...-v0"
      env_config contains keys like:
        - gym_env_id
        - env: {"framestack": ...}
        - reset_retries, retry_sleep
        - metrics_csv_path, fixed_delta_seconds
        - script_dir
    """
    # Re-anchor inside Ray worker or evaluator process
    try:
        os.chdir(env_config.get("script_dir", SCRIPT_DIR))
    except Exception:
        pass

    import macad_gym  # noqa: F401

    gym_env_id = env_config.get("gym_env_id", resolved_env_name)
    env = gym.make(gym_env_id)

    env = ResetRetryWrapper(
        env,
        max_retries=env_config.get("reset_retries", 8),
        sleep_s=env_config.get("retry_sleep", 8.0),
    )
    env = SpawnSpacingWrapper(env, offset=1.0)

    # Framestack config may appear in multiple places (match your other scripts + evaluator)
    framestack = 4
    if isinstance(env_config.get("env", None), dict) and "framestack" in env_config["env"]:
        framestack = int(env_config["env"]["framestack"])
    elif "framestack" in env_config:
        framestack = int(env_config["framestack"])

    env = wrap_deepmind(env, dim=84, num_framestack=framestack)

    env = MetricsOnlyWrapper(
        env,
        metrics_csv_path=env_config.get("metrics_csv_path", env_config.get("metrics_csv", "happo_metrics.csv")),
        fixed_delta_seconds=env_config.get("fixed_delta_seconds", 0.05),
    )
    return env


# ============================================================
# 9) Policy mapping helpers that evaluator can reuse
# ============================================================
def agent_to_index(agent_id, max_agents: int) -> int:
    """Robust parsing: tries trailing digits, otherwise hashes."""
    s = str(agent_id)
    m = re.findall(r"\d+", s)
    if m:
        idx = int(m[-1])
    else:
        idx = abs(hash(s)) % max_agents
    return int(idx) % int(max_agents)


def build_policies(max_agents: int, obs_space, act_space, notes=None) -> Dict[str, Any]:
    """Multi-policy spec for RLlib multiagent config."""
    policies = {}
    for i in range(int(max_agents)):
        policies[f"policy_{i}"] = (
            None,  # default PPO policy class
            obs_space,
            act_space,
            {
                "model": {
                    "custom_model": MODEL_NAME,
                    "custom_options": {"notes": {"notes": notes}},
                    "custom_preprocessor": "sq_im_84",
                    "dim": 84,
                    "free_log_std": False,
                    "grayscale": False,
                }
            },
        )
    return policies


def get_eval_policies_and_mapping(max_agents: int) -> Tuple[Dict[str, Any], Any, str]:
    """
    Hook for evaluate_checkpoint.py:
      - policies dict (multiagent["policies"])
      - policy_mapping_fn(agent_id, **kwargs)
      - default_policy_id (optional convenience; evaluator may ignore)

    This is the "multi-policy restore + policy mapping support" piece:
    it gives the evaluator everything needed to recreate the SAME multiagent
    structure as training, so restore() succeeds and actions are computed
    from the correct per-agent policy.
    """
    obs_space = Box(0.0, 255.0, shape=(84, 84, 3))
    act_space = Discrete(9)
    policies = build_policies(max_agents, obs_space, act_space, notes="eval")

    def policy_mapping_fn(agent_id, **kwargs):
        idx = agent_to_index(agent_id, max_agents)
        return f"policy_{idx}"

    return policies, policy_mapping_fn, "policy_0"


# ============================================================
# 10) main training loop
# ============================================================
if __name__ == "__main__":
    args = parser.parse_args()

    if args.num_workers != 0:
        print("[WARN] For MACAD stability on Windows, num_workers is forced to 0.")
        args.num_workers = 0

    gym_env_id = _resolve_macad_env_name(PREFERRED_GYM_ENV_ID, FALLBACK_GYM_ENV_ID)

    ray.init(num_gpus=args.num_gpus)

    # Register RLlib env used for training
    register_env(RLLIB_ENV_NAME, lambda cfg: make_env_with_wrappers(gym_env_id, cfg))

    obs_space = Box(0.0, 255.0, shape=(84, 84, 3))
    act_space = Discrete(9)

    policies = build_policies(args.max_agents, obs_space, act_space, notes=args.notes)

    def policy_mapping_fn(agent_id, **kwargs):
        idx = agent_to_index(agent_id, args.max_agents)
        return f"policy_{idx}"

    # Conservative PPO knobs (still PPO underneath)
    config = {
        "env": RLLIB_ENV_NAME,
        "env_config": {
            "gym_env_id": gym_env_id,
            "env": {"framestack": 4},
            "metrics_csv_path": args.metrics_csv,
            "fixed_delta_seconds": args.fixed_delta_seconds,
            "reset_retries": args.reset_retries,
            "retry_sleep": args.retry_sleep,
            "script_dir": SCRIPT_DIR,
        },
        "num_gpus": args.num_gpus,
        "num_workers": args.num_workers,
        "num_envs_per_worker": args.envs_per_worker,
        "simple_optimizer": True,

        "rollout_fragment_length": args.rollout_fragment_length,
        "train_batch_size": args.train_batch_size,
        "num_sgd_iter": 10,

        "lr": 5e-5,
        "clip_param": 0.10,
        "kl_target": 0.02,
        "kl_coeff": 0.2,
        "entropy_coeff": 0.0,

        "use_pytorch": True,

        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "log_level": "INFO",
    }

    trainer = PPOTrainer(config=config)

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_dir = os.path.join(os.getcwd(), "ray_results", f"happo_baseline_run_{ts}")
    os.makedirs(results_dir, exist_ok=True)

    best_mean_reward = -1e18
    best_ckpt_path = None

    train_order = [f"policy_{i}" for i in range(args.max_agents)]

    # Outer loop: one "HAPPO iteration" = sequential per-agent PPO updates
    for it in range(1, args.num_iters + 1):
        iter_rewards = []

        for pol in train_order:
            # Train one agent policy while others are frozen
            trainer.config["multiagent"]["policies_to_train"] = [pol]

            for _ in range(max(1, int(args.happo_steps_per_iter))):
                res = trainer.train()
                iter_rewards.append(float(res.get("episode_reward_mean", 0.0)))

        mean_r = float(np.mean(iter_rewards)) if iter_rewards else 0.0
        print(f"[HAPPO] outer_iter={it} mean_episode_reward_mean={mean_r:.4f}")

        if mean_r > best_mean_reward:
            best_mean_reward = mean_r
            ckpt = trainer.save(checkpoint_dir=results_dir)
            best_ckpt_path = ckpt
            print(f"[HAPPO] New best checkpoint: {best_ckpt_path} (score={best_mean_reward:.4f})")

        if args.checkpoint_every > 0 and (it % args.checkpoint_every == 0):
            ckpt = trainer.save(checkpoint_dir=results_dir)
            print(f"[HAPPO] Saved checkpoint: {ckpt}")

    final_ckpt = trainer.save(checkpoint_dir=results_dir)
    print("Final checkpoint path:", final_ckpt)
    print("Best checkpoint path:", best_ckpt_path if best_ckpt_path else final_ckpt)
    print("Best score:", best_mean_reward)

    trainer.stop()
    ray.shutdown()
