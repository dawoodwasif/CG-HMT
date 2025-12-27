#!/usr/bin/env python
# SAM.py
#
# Simplified SAM for MACAD-Gym (CARLA) with Ray/RLlib 0.8.6:
# - SAM = Potential-based reward shaping advice:
#       r'_t = r_t + alpha * (gamma * Phi(s_{t+1}) - Phi(s_t))
#   This keeps the optimal policy unchanged (policy-invariant shaping) when Phi is a potential.
#
# - Here Phi is a simple, configurable potential computed from the per-agent `info` dict
#   (distance_to_goal, forward_speed, collisions, lane/offroad intersections).
#   This is "non-expert friendly": you tune a few weights once at start of training.
#
# - Multi-agent setup:
#   Shared PPO policy across all agents ("shared_platoon_policy"), MAPPO-style.
#
# - Integration with evaluate_checkpoint.py:
#   This script exposes:
#       PREFERRED_ENV_NAME, FALLBACK_ENV_NAME, _resolve_macad_env_name,
#       MODEL_NAME, make_env_with_wrappers(resolved_env_name, env_config)
#   so your evaluate_checkpoint.py can import this file and recreate the env identically.
#
# Run training:
#   python SAM.py --num-iters 100 --train-batch-size 10000 --rollout-fragment-length 200
#
# Evaluate with your evaluate_checkpoint.py:
#   python evaluate_checkpoint.py --script SAM.py --checkpoint "<...>\checkpoint_100\checkpoint-100" --seeds-file eval_seeds.txt --episodes 10 --out-prefix sam_eval

from __future__ import absolute_import, division, print_function

import argparse
import datetime
import math
import os
import time
from typing import Any, Dict

import cv2
import gym
import numpy as np

import macad_gym  # noqa: F401  (registers MACAD envs)

import ray
from gym.spaces import Box, Discrete
from ray.tune.registry import register_env

from ray.rllib.env import MultiAgentEnv
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor

from rllib.env_wrappers import wrap_deepmind
from rllib.models import register_mnih15_shared_weights_net


# ============================================================
# Script dir anchoring (important on Windows + Ray workers)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


# ============================================================
# Env id resolution hooks expected by evaluate_checkpoint.py
# ============================================================
PREFERRED_ENV_NAME = "HomoNcomIndePOIntrxMASS3CTWN3-v0"
FALLBACK_ENV_NAME = "HomoNcomIndePOIntrxMASS3CTWN3-v0"


def _resolve_macad_env_name(preferred_id, fallback_id):
    """Pick a valid registered MACAD env id on this machine."""
    try:
        e = gym.make(preferred_id)
        e.close()
        print("[INFO] Using MACAD env:", preferred_id)
        return preferred_id
    except Exception:
        e = gym.make(fallback_id)
        e.close()
        print("[INFO] Falling back to:", fallback_id)
        return fallback_id


# ============================================================
# Model registration (same family as your baselines)
# ============================================================
register_mnih15_shared_weights_net()
MODEL_NAME = "mnih15_shared_weights"


# ============================================================
# 84x84 preprocessor (kept for compatibility with your pipeline)
# ============================================================
class ImagePreproc84(Preprocessor):
    def _init_shape(self, obs_space, options):
        self.shape = (84, 84, 3)
        return self.shape

    def transform(self, observation):
        return cv2.resize(observation, (self.shape[0], self.shape[1]))


ModelCatalog.register_custom_preprocessor("sq_im_84", ImagePreproc84)


# ============================================================
# Wrapper 1: Robust reset retry (CARLA server can be slow)
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
# Wrapper 2: Optional spawn spacing (reduces immediate collisions)
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

    def _nudge_actors_safely(self):
        # Best-effort: depends on MACAD internals, so keep this defensive.
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

    def close(self):
        try:
            return self.env.close()
        except Exception:
            return None


# ============================================================
# Wrapper 3: SAM Potential-based Shaping Advice
# ============================================================
class SAMShapingAdviceWrapper(MultiAgentEnv):
    """
    Adds SAM shaping:
      r'_t = r_t + alpha * (gamma * Phi_{t+1} - Phi_t)

    Phi is computed from agent `info` fields (simple, non-expert friendly):
      Phi = -w_dist * distance_to_goal
            +w_speed * clip(speed,0,desired_speed)/desired_speed
            -w_coll * collisions_cumulative
            -w_infr * (offroad_cum + otherlane_cum)

    Notes:
    - collisions/intersections are usually cumulative counters in MACAD.
      Using them inside Phi makes (Phi_{t+1}-Phi_t) penalize increments.
    - This wrapper also injects `init_distance` into info (first seen distance),
      which makes route completion computable in evaluate_checkpoint.py.
    """

    def __init__(
        self,
        env,
        alpha=1.0,
        gamma=0.99,
        desired_speed=8.0,
        w_dist=0.05,
        w_speed=1.0,
        w_coll=5.0,
        w_infr=2.0,
    ):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.desired_speed = float(desired_speed)
        self.w_dist = float(w_dist)
        self.w_speed = float(w_speed)
        self.w_coll = float(w_coll)
        self.w_infr = float(w_infr)

        self._phi_prev = {}          # aid -> Phi_t
        self._init_distance = {}     # aid -> initial distance_to_goal

    @staticmethod
    def _coll_sum(info: Dict[str, Any]) -> float:
        return (
            float(info.get("collision_vehicles", 0.0))
            + float(info.get("collision_pedestrians", 0.0))
            + float(info.get("collision_other", 0.0))
        )

    @staticmethod
    def _infr_sum(info: Dict[str, Any]) -> float:
        return float(info.get("intersection_offroad", 0.0)) + float(info.get("intersection_otherlane", 0.0))

    def _distance(self, info: Dict[str, Any]) -> float:
        if "distance_to_goal" in info:
            return float(info.get("distance_to_goal", 0.0))
        if "distance_to_goal_euclidean" in info:
            return float(info.get("distance_to_goal_euclidean", 0.0))
        return 0.0

    def _speed_norm(self, info: Dict[str, Any]) -> float:
        v = float(info.get("forward_speed", 0.0))
        if self.desired_speed <= 1e-6:
            return 0.0
        return float(np.clip(v, 0.0, self.desired_speed) / self.desired_speed)

    def _phi(self, info: Dict[str, Any]) -> float:
        # Potential is bounded-ish by choice of weights and clipped speed.
        d = self._distance(info)
        v = self._speed_norm(info)
        c = self._coll_sum(info)
        infr = self._infr_sum(info)

        return (
            -self.w_dist * d
            + self.w_speed * v
            - self.w_coll * c
            - self.w_infr * infr
        )

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()

        # Reset shaping state. We do not have info on reset reliably, so Phi_t starts at 0.
        self._phi_prev = {}
        self._init_distance = {}
        for aid in obs.keys():
            if aid == "__all__":
                continue
            self._phi_prev[aid] = 0.0
            self._init_distance[aid] = None
        return obs

    def step(self, action_dict):
        obs2, rewards, dones, infos = self.env.step(action_dict)

        # Apply shaping per agent.
        if rewards is None:
            rewards = {}
        if infos is None:
            infos = {}

        for aid, r in list(rewards.items()):
            if aid == "__all__":
                continue

            info = infos.get(aid, {}) or {}

            # Capture init_distance once and also expose it in info for evaluation scripts.
            d_now = None
            if "distance_to_goal" in info:
                d_now = float(info["distance_to_goal"])
            elif "distance_to_goal_euclidean" in info:
                d_now = float(info["distance_to_goal_euclidean"])

            if self._init_distance.get(aid, None) is None and d_now is not None and d_now > 0.0:
                self._init_distance[aid] = d_now

            if self._init_distance.get(aid, None) is not None and "init_distance" not in info:
                info["init_distance"] = float(self._init_distance[aid])
                infos[aid] = info

            phi_next = self._phi(info)
            phi_prev = float(self._phi_prev.get(aid, 0.0))

            shaping = self.alpha * (self.gamma * phi_next - phi_prev)

            try:
                rewards[aid] = float(r) + float(shaping)
            except Exception:
                rewards[aid] = r

            self._phi_prev[aid] = phi_next

        return obs2, rewards, dones, infos

    def close(self):
        try:
            return self.env.close()
        except Exception:
            return None


# ============================================================
# Wrapper 4: Metrics logger (same CSV schema as CG-HMT)
# ============================================================
class MetricsOnlyWrapper(MultiAgentEnv):
    """
    CSV schema matches CG-HMT:
      episode,steps,time_sec,collisions_per_episode,route_completion_pct,
      infractions_per_100m,prompts_per_minute,avg_shaped_reward,cumulative_avg_shaped_reward

    For SAM baseline:
      prompts_per_minute = 0
      avg_shaped_reward  = avg (SAM-shaped) reward per step per agent
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
        self.episode_prompts = 0

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

        # distance_to_goal
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

        # collisions (cumulative -> delta)
        coll_sum = 0.0
        if info is not None:
            coll_sum += float(info.get("collision_vehicles", 0.0))
            coll_sum += float(info.get("collision_pedestrians", 0.0))
            coll_sum += float(info.get("collision_other", 0.0))
        delta_coll = max(coll_sum - st["prev_collision_sum"], 0.0)
        st["prev_collision_sum"] = coll_sum
        self.episode_collisions_total += delta_coll

        # infractions (cumulative -> delta)
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
            info = info_dict.get(aid, {}) if info_dict else {}
            self._update_per_agent_stats(aid, info)
            if rewards is not None:
                self.episode_total_reward += float(rewards.get(aid, 0.0))

        if done_dict and done_dict.get("__all__", False):
            self._finalize_and_log_episode(num_agents=len(agent_ids))

        return obs, rewards, done_dict, info_dict

    def close(self):
        try:
            return self.env.close()
        except Exception:
            return None


# ============================================================
# Env factory required by evaluate_checkpoint.py
# ============================================================
def make_env_with_wrappers(resolved_env_name, env_config):
    """
    This is called by both:
    - RLlib (during training): via register_env
    - evaluate_checkpoint.py (during evaluation): via dynamic import
    """
    # Re-anchor working directory inside worker process
    try:
        os.chdir(env_config.get("script_dir", SCRIPT_DIR))
    except Exception:
        pass

    import macad_gym  # noqa: F401

    gym_env_id = env_config.get("gym_env_id", resolved_env_name)
    framestack = int(env_config.get("framestack", 4))

    # Build base env
    env = gym.make(gym_env_id)

    # Robustness wrappers
    env = ResetRetryWrapper(
        env,
        max_retries=int(env_config.get("reset_retries", 8)),
        sleep_s=float(env_config.get("retry_sleep", 8.0)),
    )
    env = SpawnSpacingWrapper(env, offset=float(env_config.get("spawn_offset", 1.0)))

    # Image + framestack pipeline (your existing stack)
    env = wrap_deepmind(env, dim=84, num_framestack=framestack)

    # SAM shaping advice wrapper
    env = SAMShapingAdviceWrapper(
        env,
        alpha=float(env_config.get("sam_alpha", 1.0)),
        gamma=float(env_config.get("sam_gamma", 0.99)),
        desired_speed=float(env_config.get("sam_desired_speed", 8.0)),
        w_dist=float(env_config.get("sam_w_dist", 0.05)),
        w_speed=float(env_config.get("sam_w_speed", 1.0)),
        w_coll=float(env_config.get("sam_w_coll", 5.0)),
        w_infr=float(env_config.get("sam_w_infr", 2.0)),
    )

    # Metrics logging wrapper (final object is MultiAgentEnv)
    env = MetricsOnlyWrapper(
        env,
        metrics_csv_path=str(env_config.get("metrics_csv_path", "sam_metrics.csv")),
        fixed_delta_seconds=float(env_config.get("fixed_delta_seconds", 0.05)),
    )
    return env


# ============================================================
# Training entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser("SAM baseline (potential-based shaping advice) for MACAD")

    parser.add_argument("--num-iters", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=0)  # keep 0 for stability on Windows
    parser.add_argument("--num-gpus", type=int, default=1)

    parser.add_argument("--rollout-fragment-length", type=int, default=200)
    parser.add_argument("--train-batch-size", type=int, default=10000)  # lower default to reduce OOM
    parser.add_argument("--envs-per-worker", type=int, default=1)
    parser.add_argument("--num-sgd-iter", type=int, default=10)

    parser.add_argument("--metrics-csv", type=str, default="sam_metrics.csv")
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)

    # CARLA boot can be slow on Windows; extend reset tolerance
    parser.add_argument("--reset-retries", type=int, default=8)
    parser.add_argument("--retry-sleep", type=float, default=8.0)

    # SAM shaping knobs
    parser.add_argument("--sam-alpha", type=float, default=1.0, help="Scale of shaping term.")
    parser.add_argument("--sam-gamma", type=float, default=0.99, help="Gamma used in shaping.")
    parser.add_argument("--sam-desired-speed", type=float, default=8.0, help="Speed normalization target (m/s).")

    parser.add_argument("--sam-w-dist", type=float, default=0.05, help="Weight for distance_to_goal in Phi.")
    parser.add_argument("--sam-w-speed", type=float, default=1.0, help="Weight for speed term in Phi.")
    parser.add_argument("--sam-w-coll", type=float, default=5.0, help="Weight for collision counters in Phi.")
    parser.add_argument("--sam-w-infr", type=float, default=2.0, help="Weight for lane/offroad infractions in Phi.")

    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--notes", type=str, default=None)

    args = parser.parse_args()

    if args.num_workers != 0:
        print("[WARN] For MACAD stability on Windows, num_workers is forced to 0.")
        args.num_workers = 0

    gym_env_id = _resolve_macad_env_name(PREFERRED_ENV_NAME, FALLBACK_ENV_NAME)

    ray.init(num_gpus=int(args.num_gpus))

    RLLIB_ENV_NAME = "MACAD_SAM_SHARED_PPO"

    register_env(RLLIB_ENV_NAME, lambda cfg: make_env_with_wrappers(gym_env_id, cfg))

    obs_space = Box(0.0, 255.0, shape=(84, 84, 3))
    act_space = Discrete(9)

    policies = {
        "shared_platoon_policy": (
            None,  # default PPO torch policy
            obs_space,
            act_space,
            {
                "model": {
                    "custom_model": MODEL_NAME,
                    "custom_preprocessor": "sq_im_84",
                    "dim": 84,
                    "grayscale": False,
                    "free_log_std": False,
                    # Ray 0.8.6 expects custom_options (deprecated later)
                    "custom_options": {"notes": {"notes": args.notes}},
                }
            },
        )
    }

    def policy_mapping_fn(agent_id, **kwargs):
        return "shared_platoon_policy"

    config = {
        "env": RLLIB_ENV_NAME,
        "env_config": {
            "gym_env_id": gym_env_id,
            "framestack": 4,
            "metrics_csv_path": args.metrics_csv,
            "fixed_delta_seconds": args.fixed_delta_seconds,
            "reset_retries": args.reset_retries,
            "retry_sleep": args.retry_sleep,
            "script_dir": SCRIPT_DIR,

            # SAM params
            "sam_alpha": args.sam_alpha,
            "sam_gamma": args.sam_gamma,
            "sam_desired_speed": args.sam_desired_speed,
            "sam_w_dist": args.sam_w_dist,
            "sam_w_speed": args.sam_w_speed,
            "sam_w_coll": args.sam_w_coll,
            "sam_w_infr": args.sam_w_infr,
        },
        "num_gpus": int(args.num_gpus),
        "num_workers": int(args.num_workers),
        "num_envs_per_worker": int(args.envs_per_worker),
        "simple_optimizer": True,

        "rollout_fragment_length": int(args.rollout_fragment_length),
        "train_batch_size": int(args.train_batch_size),
        "num_sgd_iter": int(args.num_sgd_iter),

        # Reasonable PPO defaults for your setup (same spirit as your baselines)
        "lr": 5e-5,
        "clip_param": 0.2,
        "entropy_coeff": 0.0,

        "use_pytorch": True,  # Ray 0.8.6 style

        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "log_level": "INFO",
    }

    trainer = PPOTrainer(config=config)

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_dir = os.path.join(os.getcwd(), "ray_results", f"sam_run_{ts}")
    os.makedirs(results_dir, exist_ok=True)

    best_mean_reward = -1e18
    best_ckpt_path = None

    for it in range(1, int(args.num_iters) + 1):
        res = trainer.train()
        mean_r = float(res.get("episode_reward_mean", 0.0))
        print(f"[SAM] iter={it} episode_reward_mean={mean_r:.4f}")

        if mean_r > best_mean_reward:
            best_mean_reward = mean_r
            best_ckpt_path = trainer.save(checkpoint_dir=results_dir)
            print(f"[SAM] New best checkpoint: {best_ckpt_path} (score={best_mean_reward:.4f})")

        if args.checkpoint_every > 0 and (it % int(args.checkpoint_every) == 0):
            ckpt = trainer.save(checkpoint_dir=results_dir)
            print(f"[SAM] Saved checkpoint: {ckpt}")

    final_ckpt = trainer.save(checkpoint_dir=results_dir)

    print("\n[SAM] Final checkpoint path:", final_ckpt)
    print("[SAM] Best checkpoint path:", best_ckpt_path if best_ckpt_path else final_ckpt)
    print("[SAM] Best score:", best_mean_reward)
    print("[SAM] Metrics CSV (per-episode):", os.path.abspath(args.metrics_csv))
    print("[SAM] Ray results dir:", os.path.abspath(results_dir))

    trainer.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
