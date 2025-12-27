#!/usr/bin/env python
# HHK.py
#
# Simplified HHK-style MARL baseline for MACAD-Gym (CARLA) using Ray/RLlib 0.8.6.
#
# What it does (HHK-inspired, but kept simple + robust for your pipeline):
# 1) Human suboptimal knowledge is represented as fuzzy rules over info signals:
#       forward_speed, collisions, infractions, distance_to_goal, etc.
# 2) A graph-based group controller aggregates neighbor signals using (x,y) positions
#    and produces a macro guidance signal (target speed factor) shared across agents.
# 3) Agents can "decide" how much to use knowledge via a trust gate (risk -> lower trust).
# 4) Knowledge is injected safely via potential-based shaping:
#       r' = r + w * (gamma * phi(s') - phi(s))
#    and optionally (OFF by default) via a small action nudge.
#
# Integration with your evaluate_checkpoint.py:
# - Exposes: PREFERRED_ENV_NAME, FALLBACK_ENV_NAME, _resolve_macad_env_name,
#            MODEL_NAME, make_env_with_wrappers(resolved_env_name, env_config)
# - Uses a single shared policy id: "shared_platoon_policy"
#
# Train:
#   python HHK.py --num-iters 100 --train-batch-size 10000 --rollout-fragment-length 200
#
# Evaluate (with your evaluate_checkpoint.py):
#   python evaluate_checkpoint.py --script HHK.py --checkpoint "<.../checkpoint-XX>" --seeds-file eval_seeds.txt --episodes 10 --out-prefix hhk_eval

from __future__ import absolute_import, division, print_function

import argparse
import datetime
import math
import os
import time
from typing import Dict, Any, List

import gym
import numpy as np

import macad_gym  # noqa: F401  (registers MACAD envs)

import ray
from gym.spaces import Box, Discrete
from ray.tune.registry import register_env

# RLlib 0.8.6
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor

from rllib.env_wrappers import wrap_deepmind
from rllib.models import register_mnih15_shared_weights_net


# ============================================================
# Script dir anchoring (Windows + Ray workers)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


# ============================================================
# 0) Env selection (match your other scripts)
# ============================================================
PREFERRED_ENV_NAME = "HomoNcomIndePOIntrxMASS3CTWN3-v0"
FALLBACK_ENV_NAME = "HomoNcomIndePOIntrxMASS3CTWN3-v0"

RLLIB_ENV_NAME = "MACAD_MASS3_HHK_BASELINE"


def _resolve_macad_env_name(preferred_id, fallback_id):
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
# 1) Model registration (same backbone family)
# ============================================================
register_mnih15_shared_weights_net()
MODEL_NAME = "mnih15_shared_weights"


# ============================================================
# 2) 84x84 preprocessor (Ray 0.8.6 style)
# ============================================================
class ImagePreproc84(Preprocessor):
    def _init_shape(self, obs_space, options):
        self.shape = (84, 84, 3)
        return self.shape

    def transform(self, observation):
        # observation is already processed by wrap_deepmind in most setups
        # but we keep this to match your pipeline.
        try:
            import cv2
            return cv2.resize(observation, (84, 84))
        except Exception:
            arr = np.asarray(observation)
            if arr.ndim < 2:
                return arr
            sy = max(1, int(arr.shape[0] / 84))
            sx = max(1, int(arr.shape[1] / 84))
            return arr[::sy, ::sx][:84, :84]


ModelCatalog.register_custom_preprocessor("sq_im_84", ImagePreproc84)


# ============================================================
# 3) Robust reset wrapper (CARLA startup tolerance)
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
# 4) Optional spawn spacing wrapper (kept consistent)
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
        # Best-effort: if actor handles exist, offset them a bit to reduce spawn collisions.
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
# 5) Metrics logger wrapper (same CSV schema as your baselines)
# ============================================================
class MetricsOnlyWrapper(MultiAgentEnv):
    """
    CSV schema aligned with your CG-HMT/HAPPO baseline:
      episode,steps,time_sec,collisions_per_episode,route_completion_pct,
      infractions_per_100m,prompts_per_minute,avg_shaped_reward,cumulative_avg_shaped_reward

    Here:
      prompts_per_minute = 0
      avg_shaped_reward  = avg base env reward per step per agent
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
# 6) HHK knowledge wrapper (graph + fuzzy + shaping)
# ============================================================
class HHKKnowledgeWrapper(MultiAgentEnv):
    """
    Outer wrapper that injects HHK-style guidance as potential-based shaping
    and optionally a light action nudge.

    Important design choice for comparability:
    - It wraps MetricsOnlyWrapper as the inner env.
    - MetricsOnlyWrapper logs *base* env rewards (before shaping),
      because shaping is applied after the inner env step returns.
    - RLlib receives shaped rewards, so training benefits from guidance.
    """

    def __init__(
        self,
        env,
        gamma=0.99,
        shaping_weight=0.25,
        neighbor_k=2,
        graph_sigma=20.0,
        # suboptimal knowledge knobs
        advice_noise_prob=0.15,
        # optional action nudge (OFF by default)
        use_action_nudge=False,
        nudge_prob=0.05,
        nudge_action_id=3,
    ):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.gamma = float(gamma)
        self.shaping_weight = float(shaping_weight)

        self.neighbor_k = int(neighbor_k)
        self.graph_sigma = float(graph_sigma)

        self.advice_noise_prob = float(advice_noise_prob)

        self.use_action_nudge = bool(use_action_nudge)
        self.nudge_prob = float(nudge_prob)
        self.nudge_action_id = int(nudge_action_id)

        self._phi_prev = {}          # per-agent last potential
        self._init_dist = {}         # per-agent init distance (for RC + progress)
        self._trust = {}             # per-agent trust gate in [0,1]
        self._last_info = {}         # per-agent last info for gating

    @staticmethod
    def _safe_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        self._phi_prev = {}
        self._init_dist = {}
        self._trust = {}
        self._last_info = {}
        for aid in obs.keys():
            if aid == "__all__":
                continue
            self._phi_prev[aid] = 0.0
            self._trust[aid] = 1.0
        return obs

    def step(self, action_dict):
        # Optional action nudge (kept OFF by default)
        if self.use_action_nudge and action_dict:
            action_dict = dict(action_dict)
            for aid in list(action_dict.keys()):
                if aid == "__all__":
                    continue
                t = float(self._trust.get(aid, 1.0))
                p = self.nudge_prob * t
                if np.random.rand() < p:
                    action_dict[aid] = self.nudge_action_id

        obs, rewards, dones, infos = self.env.step(action_dict)

        # Apply shaping after inner env step (so metrics already logged base rewards)
        shaped_rewards = {}
        agent_ids = [aid for aid in (infos or {}).keys() if aid != "__all__"]
        group_ctrl = self._compute_group_controller(infos, agent_ids)

        for aid in agent_ids:
            base_r = self._safe_float(rewards.get(aid, 0.0), 0.0)
            info = infos.get(aid, {}) or {}

            # Track init distance for progress and for eval RC extraction
            d_goal = None
            if "distance_to_goal" in info:
                d_goal = self._safe_float(info.get("distance_to_goal", None), None)
            elif "distance_to_goal_euclidean" in info:
                d_goal = self._safe_float(info.get("distance_to_goal_euclidean", None), None)

            if d_goal is not None and aid not in self._init_dist and d_goal > 0.0:
                self._init_dist[aid] = float(d_goal)

            # Make init_distance visible to your evaluate_checkpoint.py (fixes RC=NA)
            if aid in self._init_dist:
                info["init_distance"] = float(self._init_dist[aid])

            # Update trust gate from risk deltas (collisions/infractions)
            self._trust[aid] = self._update_trust(aid, info)

            # Potential function (fuzzy + group + progress)
            phi_now = self._potential(aid, info, group_ctrl)

            # Suboptimal knowledge: occasionally perturb phi (noise in advice quality)
            if np.random.rand() < self.advice_noise_prob:
                phi_now = phi_now + np.random.uniform(-0.2, 0.2)

            phi_prev = float(self._phi_prev.get(aid, 0.0))
            self._phi_prev[aid] = float(phi_now)

            shaping = self.shaping_weight * (self.gamma * phi_now - phi_prev)

            shaped_r = base_r + float(self._trust.get(aid, 1.0)) * shaping
            shaped_rewards[aid] = shaped_r

            # Put debug-friendly fields into info
            info["base_reward"] = base_r
            info["hhk_phi"] = float(phi_now)
            info["hhk_shaping"] = float(shaping)
            info["hhk_trust"] = float(self._trust.get(aid, 1.0))
            infos[aid] = info

        # keep __all__ if present
        if "__all__" in (rewards or {}):
            shaped_rewards["__all__"] = rewards["__all__"]

        return obs, shaped_rewards, dones, infos

    def _compute_group_controller(self, infos: Dict[str, Dict[str, Any]], agent_ids: List[str]) -> Dict[str, Any]:
        # Graph-based neighbor aggregation using x,y and a Gaussian kernel.
        # Output: per-agent neighbor risk + a shared target_speed_factor.
        pos = {}
        risk = {}
        for aid in agent_ids:
            info = infos.get(aid, {}) or {}
            x = self._safe_float(info.get("x", 0.0), 0.0)
            y = self._safe_float(info.get("y", 0.0), 0.0)
            pos[aid] = (x, y)

            coll = (
                self._safe_float(info.get("collision_vehicles", 0.0), 0.0)
                + self._safe_float(info.get("collision_pedestrians", 0.0), 0.0)
                + self._safe_float(info.get("collision_other", 0.0), 0.0)
            )
            infr = (
                self._safe_float(info.get("intersection_offroad", 0.0), 0.0)
                + self._safe_float(info.get("intersection_otherlane", 0.0), 0.0)
            )
            # risk in [0, +inf)
            risk[aid] = float(coll + infr)

        # Neighbor risk aggregation
        neighbor_risk = {}
        aids = list(agent_ids)
        for i, a in enumerate(aids):
            xa, ya = pos[a]
            dists = []
            for j, b in enumerate(aids):
                if a == b:
                    continue
                xb, yb = pos[b]
                dx = xa - xb
                dy = ya - yb
                dist = math.sqrt(dx * dx + dy * dy)
                dists.append((dist, b))
            dists.sort(key=lambda t: t[0])
            nbrs = dists[: max(0, self.neighbor_k)]
            if not nbrs:
                neighbor_risk[a] = float(risk[a])
                continue
            wsum = 0.0
            rsum = 0.0
            for dist, b in nbrs:
                w = math.exp(-(dist * dist) / max(self.graph_sigma * self.graph_sigma, 1e-6))
                wsum += w
                rsum += w * float(risk[b])
            neighbor_risk[a] = float(rsum / max(wsum, 1e-6))

        # Macro guidance: target speed factor decreases if group risk is high
        group_risk_mean = float(np.mean([neighbor_risk[a] for a in aids])) if aids else 0.0
        # Factor in [0.4, 1.0]
        target_speed_factor = float(np.clip(1.0 - 0.15 * group_risk_mean, 0.4, 1.0))

        return {
            "neighbor_risk": neighbor_risk,
            "target_speed_factor": target_speed_factor,
            "group_risk_mean": group_risk_mean,
        }

    def _update_trust(self, aid: str, info: Dict[str, Any]) -> float:
        # Trust drops if collisions/infractions appear, rises slowly otherwise.
        prev = self._last_info.get(aid, {}) or {}

        def csum(x):
            return (
                self._safe_float(x.get("collision_vehicles", 0.0), 0.0)
                + self._safe_float(x.get("collision_pedestrians", 0.0), 0.0)
                + self._safe_float(x.get("collision_other", 0.0), 0.0)
            )

        def isum(x):
            return (
                self._safe_float(x.get("intersection_offroad", 0.0), 0.0)
                + self._safe_float(x.get("intersection_otherlane", 0.0), 0.0)
            )

        dc = max(csum(info) - csum(prev), 0.0)
        di = max(isum(info) - isum(prev), 0.0)

        t = float(self._trust.get(aid, 1.0))

        # drop sharply on new risk events
        t = t * math.exp(-0.8 * dc - 0.4 * di)

        # slow recovery
        t = t + 0.01 * (1.0 - t)

        t = float(np.clip(t, 0.0, 1.0))
        self._last_info[aid] = dict(info)
        return t

    def _fuzzy_membership(self, x: float, a: float, b: float, c: float) -> float:
        # Triangular membership function
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return (x - a) / max(b - a, 1e-6)
        return (c - x) / max(c - b, 1e-6)

    def _potential(self, aid: str, info: Dict[str, Any], group_ctrl: Dict[str, Any]) -> float:
        """
        Potential phi(s) in roughly [-1, +1], using fuzzy rules + group macro guidance.

        Signals:
          speed, progress (normalized), lane-safety, collision-free, neighbor-risk.
        """
        v = self._safe_float(info.get("forward_speed", 0.0), 0.0)
        coll = (
            self._safe_float(info.get("collision_vehicles", 0.0), 0.0)
            + self._safe_float(info.get("collision_pedestrians", 0.0), 0.0)
            + self._safe_float(info.get("collision_other", 0.0), 0.0)
        )
        infr = (
            self._safe_float(info.get("intersection_offroad", 0.0), 0.0)
            + self._safe_float(info.get("intersection_otherlane", 0.0), 0.0)
        )

        d_goal = None
        if "distance_to_goal" in info:
            d_goal = self._safe_float(info.get("distance_to_goal", None), None)
        elif "distance_to_goal_euclidean" in info:
            d_goal = self._safe_float(info.get("distance_to_goal_euclidean", None), None)

        init_d = float(self._init_dist.get(aid, d_goal if (d_goal is not None) else 1.0))
        if init_d < 1e-6:
            init_d = 1.0
        if d_goal is None:
            d_goal = init_d

        progress = float(np.clip((init_d - float(d_goal)) / init_d, 0.0, 1.0))

        # Macro guidance influences "target speed" (suboptimal but structured)
        target_speed_factor = float(group_ctrl.get("target_speed_factor", 1.0))
        # Base target speed (m/s). Scaled down when group risk rises.
        v_target = 9.0 * target_speed_factor

        # Fuzzy memberships for speed relative to target
        speed_low = self._fuzzy_membership(v, 0.0, 0.4 * v_target, 0.8 * v_target)
        speed_ok = self._fuzzy_membership(v, 0.6 * v_target, 1.0 * v_target, 1.4 * v_target)
        speed_high = self._fuzzy_membership(v, 1.2 * v_target, 1.6 * v_target, 2.0 * v_target)

        # Safety signals
        lane_safe = 1.0 if infr <= 0.0 else float(np.exp(-0.5 * infr))
        collision_free = float(np.exp(-1.0 * coll))

        # Neighbor risk penalty
        nbr_risk = float(group_ctrl.get("neighbor_risk", {}).get(aid, 0.0))
        nbr_ok = float(np.exp(-0.4 * nbr_risk))

        # Rule-like combination (interpretable weights)
        # Encourage: progress + ok speed + safe lane + collision free + neighbor ok
        # Discourage: too high speed especially when neighbor risk is high
        phi = (
            0.45 * progress
            + 0.20 * speed_ok
            + 0.15 * lane_safe
            + 0.15 * collision_free
            + 0.10 * nbr_ok
            - 0.15 * speed_high * (1.0 - nbr_ok)
            - 0.05 * (1.0 - speed_low) * (1.0 - progress)  # mild penalty if stuck and not accelerating
        )

        # Keep bounded
        return float(np.clip(phi, -1.0, 1.0))

    def close(self):
        try:
            return self.env.close()
        except Exception:
            return None


# ============================================================
# 7) Env factory (evaluate_checkpoint imports and calls this)
# ============================================================
def make_env_with_wrappers(resolved_env_name, env_config):
    """
    Must return an RLlib MultiAgentEnv (NOT a gym.Wrapper that hides the type),
    so your evaluate_checkpoint.py multi-agent checks pass.
    """
    # Anchor working dir inside Ray worker
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

    framestack = int(env_config.get("framestack", 4))
    env = wrap_deepmind(env, dim=84, num_framestack=framestack)

    # Inner metrics wrapper logs base rewards (no shaping yet)
    env = MetricsOnlyWrapper(
        env,
        metrics_csv_path=env_config.get("metrics_csv_path", "hhk_metrics.csv"),
        fixed_delta_seconds=float(env_config.get("fixed_delta_seconds", 0.05)),
    )

    # Outer HHK wrapper applies shaping to rewards that RLlib trains on
    env = HHKKnowledgeWrapper(
        env,
        gamma=float(env_config.get("hhk_gamma", 0.99)),
        shaping_weight=float(env_config.get("hhk_shaping_weight", 0.25)),
        neighbor_k=int(env_config.get("hhk_neighbor_k", 2)),
        graph_sigma=float(env_config.get("hhk_graph_sigma", 20.0)),
        advice_noise_prob=float(env_config.get("hhk_advice_noise_prob", 0.15)),
        use_action_nudge=bool(env_config.get("hhk_use_action_nudge", False)),
        nudge_prob=float(env_config.get("hhk_nudge_prob", 0.05)),
        nudge_action_id=int(env_config.get("hhk_nudge_action_id", 3)),
    )
    return env


# ============================================================
# 8) Training entry (PPO with shared policy)
# ============================================================
def build_trainer_config(args, gym_env_id):
    obs_space = Box(0.0, 255.0, shape=(84, 84, 3))
    act_space = Discrete(9)

    # Shared policy for all agents (keeps eval compatibility simple)
    def gen_shared_policy():
        cfg = {
            "model": {
                "custom_model": MODEL_NAME,
                "custom_preprocessor": "sq_im_84",
                "dim": 84,
                "grayscale": False,
                "free_log_std": False,
                "custom_options": {"notes": {"hhk": True, "notes": args.notes}},
            },
            "env_config": {"env": {"framestack": int(args.framestack)}},
        }
        return (None, obs_space, act_space, cfg)

    policies = {"shared_platoon_policy": gen_shared_policy()}

    def policy_mapping_fn(agent_id, **kwargs):
        return "shared_platoon_policy"

    cfg = {
        "env": RLLIB_ENV_NAME,
        "env_config": {
            "gym_env_id": gym_env_id,
            "framestack": int(args.framestack),
            "metrics_csv_path": args.metrics_csv,
            "fixed_delta_seconds": float(args.fixed_delta_seconds),
            "reset_retries": int(args.reset_retries),
            "retry_sleep": float(args.retry_sleep),
            "script_dir": SCRIPT_DIR,
            # HHK knobs
            "hhk_gamma": float(args.hhk_gamma),
            "hhk_shaping_weight": float(args.hhk_shaping_weight),
            "hhk_neighbor_k": int(args.hhk_neighbor_k),
            "hhk_graph_sigma": float(args.hhk_graph_sigma),
            "hhk_advice_noise_prob": float(args.hhk_advice_noise_prob),
            "hhk_use_action_nudge": bool(args.hhk_use_action_nudge),
            "hhk_nudge_prob": float(args.hhk_nudge_prob),
            "hhk_nudge_action_id": int(args.hhk_nudge_action_id),
        },
        "num_gpus": int(args.num_gpus),
        "num_workers": 0,  # Windows stability
        "num_envs_per_worker": int(args.envs_per_worker),
        "simple_optimizer": True,

        "rollout_fragment_length": int(args.rollout_fragment_length),
        "train_batch_size": int(args.train_batch_size),
        "num_sgd_iter": int(args.num_sgd_iter),

        # Conservative PPO settings (helps stability with multi-agent non-stationarity)
        "lr": float(args.lr),
        "clip_param": float(args.clip_param),
        "kl_target": float(args.kl_target),
        "kl_coeff": float(args.kl_coeff),
        "entropy_coeff": float(args.entropy_coeff),

        "use_pytorch": True,

        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "log_level": "INFO",
    }
    return cfg


def main():
    parser = argparse.ArgumentParser("HHK baseline (fuzzy + graph-guidance + PPO) for MACAD")

    parser.add_argument("--num-iters", type=int, default=300)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--envs-per-worker", type=int, default=1)

    parser.add_argument("--framestack", type=int, default=4)
    parser.add_argument("--rollout-fragment-length", type=int, default=200)
    parser.add_argument("--train-batch-size", type=int, default=10000)
    parser.add_argument("--num-sgd-iter", type=int, default=10)

    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--metrics-csv", type=str, default="hhk_metrics.csv")
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)

    parser.add_argument("--reset-retries", type=int, default=8)
    parser.add_argument("--retry-sleep", type=float, default=8.0)

    # PPO stability knobs
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--clip-param", type=float, default=0.10)
    parser.add_argument("--kl-target", type=float, default=0.02)
    parser.add_argument("--kl-coeff", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.0)

    # HHK knobs (human knowledge as interpretable numbers)
    parser.add_argument("--hhk-gamma", type=float, default=0.99)
    parser.add_argument("--hhk-shaping-weight", type=float, default=0.25)
    parser.add_argument("--hhk-neighbor-k", type=int, default=2)
    parser.add_argument("--hhk-graph-sigma", type=float, default=20.0)
    parser.add_argument("--hhk-advice-noise-prob", type=float, default=0.15)

    # Optional action nudge
    parser.add_argument("--hhk-use-action-nudge", action="store_true")
    parser.add_argument("--hhk-nudge-prob", type=float, default=0.05)
    parser.add_argument("--hhk-nudge-action-id", type=int, default=3)

    parser.add_argument("--notes", default=None)

    args = parser.parse_args()

    gym_env_id = _resolve_macad_env_name(PREFERRED_ENV_NAME, FALLBACK_ENV_NAME)

    ray.init(num_gpus=int(args.num_gpus))

    # Register env for RLlib
    register_env(RLLIB_ENV_NAME, lambda cfg: make_env_with_wrappers(gym_env_id, cfg))

    config = build_trainer_config(args, gym_env_id)
    trainer = PPOTrainer(config=config)

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_dir = os.path.join(os.getcwd(), "ray_results", f"hhk_baseline_run_{ts}")
    os.makedirs(results_dir, exist_ok=True)

    best_mean_reward = -1e18
    best_ckpt_path = None

    for it in range(1, int(args.num_iters) + 1):
        res = trainer.train()
        mean_r = float(res.get("episode_reward_mean", 0.0))
        print(f"[HHK] iter={it} episode_reward_mean={mean_r:.6f}")

        # Track best checkpoint by mean reward
        if mean_r > best_mean_reward:
            best_mean_reward = mean_r
            ckpt = trainer.save(checkpoint_dir=results_dir)
            best_ckpt_path = ckpt
            print(f"[HHK] New best checkpoint: {best_ckpt_path} (score={best_mean_reward:.6f})")

        if args.checkpoint_every > 0 and (it % int(args.checkpoint_every) == 0):
            ckpt = trainer.save(checkpoint_dir=results_dir)
            print(f"[HHK] Saved checkpoint: {ckpt}")

    final_ckpt = trainer.save(checkpoint_dir=results_dir)
    print("Final checkpoint path:", final_ckpt)
    print("Best checkpoint path:", best_ckpt_path if best_ckpt_path else final_ckpt)
    print("Best score:", best_mean_reward)

    trainer.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
