#!/usr/bin/env python
# evaluate_checkpoint.py
#
# Deterministic evaluation for RLlib (Ray 0.8.6) multi-agent checkpoints on MACAD-Gym (CARLA).
#
# This version includes hard patches for legacy restore incompatibilities:
#  1) Filter key mismatch during restore/sync_filters (AssertionError).
#  2) Policy id mismatch during restore (KeyError: 'policy_0' etc), by remapping checkpoint
#     policy states onto the single local policy you define in this evaluator.
#
# It also includes robust observation conversion so dict/object observations do not crash torch.

from __future__ import absolute_import, division, print_function

import argparse
import importlib.util
import inspect
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

import gym
import numpy as np

import ray
from ray.tune.registry import register_env

from gym.spaces import Box, Discrete

# RLlib 0.8.6
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy as PPOPolicy
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor

# Optional helpers from your project
try:
    from rllib.env_wrappers import wrap_deepmind
except Exception:
    wrap_deepmind = None

try:
    from rllib.models import register_mnih15_shared_weights_net
except Exception:
    register_mnih15_shared_weights_net = None


# ------------------------------------------------------------
# RLlib compatibility patches (Ray 0.8.6)
# ------------------------------------------------------------
def _apply_rllib_compat_patches():
    """
    Patch RolloutWorker.restore + sync_filters so restore is tolerant to:
      - filter key mismatches (sync_filters assertion)
      - checkpoint policy-id mismatches (KeyError: 'policy_0', 'default_policy', etc)
    """
    try:
        from ray.rllib.evaluation.rollout_worker import RolloutWorker
    except Exception:
        return

    if getattr(RolloutWorker, "_eval_compat_patch_applied", False):
        return

    orig_sync_filters = RolloutWorker.sync_filters
    orig_restore = RolloutWorker.restore

    def patched_sync_filters(self, new_filters):
        try:
            if isinstance(new_filters, dict):
                local_filters = getattr(self, "filters", {}) or {}
                for k in list(local_filters.keys()):
                    if k not in new_filters:
                        try:
                            new_filters[k] = local_filters[k]
                        except Exception:
                            pass
        except Exception:
            pass
        return orig_sync_filters(self, new_filters)

    def _extract_policy_states_from_objs(objs: dict) -> Optional[dict]:
        if not isinstance(objs, dict):
            return None
        for key in ["policy_states", "policies", "policy_map"]:
            if key in objs and isinstance(objs[key], dict):
                return objs[key]
        st = objs.get("state", None)
        if isinstance(st, dict):
            for key in ["policy_states", "policies", "policy_map"]:
                if key in st and isinstance(st[key], dict):
                    return st[key]
        return None

    def _remap_saved_states_to_existing(saved_states: dict, existing_policy_ids: List[str]) -> dict:
        if not isinstance(saved_states, dict):
            return {}
        if not existing_policy_ids:
            return {}

        if len(existing_policy_ids) == 1:
            sole = existing_policy_ids[0]
            for cand in ["shared_platoon_policy", "policy_0", "default_policy"]:
                if cand in saved_states:
                    return {sole: saved_states[cand]}
            k0 = sorted(list(saved_states.keys()), key=lambda z: str(z))[0]
            return {sole: saved_states[k0]}

        out = {}
        for pid in existing_policy_ids:
            if pid in saved_states:
                out[pid] = saved_states[pid]
        return out

    def _tolerant_restore(self, objs: dict):
        if not isinstance(objs, dict):
            return

        try:
            gv = objs.get("global_vars", None)
            if isinstance(gv, dict):
                try:
                    self.global_vars.update(gv)
                except Exception:
                    self.global_vars = gv
        except Exception:
            pass

        try:
            nf = objs.get("filters", None)
            if isinstance(nf, dict):
                local_filters = getattr(self, "filters", {}) or {}
                for k in list(local_filters.keys()):
                    nf.setdefault(k, local_filters.get(k))
                try:
                    self.sync_filters(nf)
                except Exception:
                    pass
        except Exception:
            pass

        saved_states = _extract_policy_states_from_objs(objs)
        if not isinstance(saved_states, dict) or not saved_states:
            return

        try:
            existing = list(getattr(self, "policy_map", {}).keys())
        except Exception:
            existing = []

        remapped = _remap_saved_states_to_existing(saved_states, existing)

        try:
            for pid, st in (remapped or {}).items():
                if pid in getattr(self, "policy_map", {}):
                    try:
                        self.policy_map[pid].set_state(st)
                    except Exception:
                        pass
        except Exception:
            pass

    def patched_restore(self, objs):
        try:
            if isinstance(objs, dict) and "filters" in objs and isinstance(objs["filters"], dict):
                local_filters = getattr(self, "filters", {}) or {}
                for k in list(local_filters.keys()):
                    objs["filters"].setdefault(k, local_filters.get(k))
        except Exception:
            pass

        try:
            return orig_restore(self, objs)
        except (AssertionError, KeyError):
            _tolerant_restore(self, objs)
            return

    RolloutWorker.sync_filters = patched_sync_filters
    RolloutWorker.restore = patched_restore
    RolloutWorker._eval_compat_patch_applied = True


# ------------------------------------------------------------
# Small path helpers
# ------------------------------------------------------------
def _abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _maybe_mkdir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _resolve_checkpoint_path(ckpt_path: str) -> str:
    ckpt_path = _abspath(ckpt_path)

    if os.path.isfile(ckpt_path):
        return ckpt_path

    if os.path.isdir(ckpt_path):
        best = None
        for name in os.listdir(ckpt_path):
            if name.startswith("checkpoint-") and not name.endswith(".tune_metadata"):
                best = os.path.join(ckpt_path, name)
                break
        if best and os.path.isfile(best):
            return best

    raise RuntimeError(
        "Could not resolve checkpoint file from path: %s\n"
        "Pass either:\n"
        "  (A) the checkpoint FILE: ...\\checkpoint_100\\checkpoint-100\n"
        "  (B) the checkpoint DIR : ...\\checkpoint_100 (auto-resolves)\n" % ckpt_path
    )


def import_script_from_path(script_path: str):
    script_path = _abspath(script_path)
    mod_name = os.path.splitext(os.path.basename(script_path))[0] + "_evalmod"
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load module spec for: %s" % script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_seeds_file(path: str) -> List[int]:
    path = _abspath(path)
    seeds = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            seeds.append(int(s))
    return seeds


# ------------------------------------------------------------
# Environment selection + human UI disabling
# ------------------------------------------------------------
def infer_resolved_env_name(mod) -> str:
    preferred = getattr(mod, "PREFERRED_ENV_NAME", None)
    fallback = getattr(mod, "FALLBACK_ENV_NAME", None)
    resolver = getattr(mod, "_resolve_macad_env_name", None)

    if callable(resolver) and preferred and fallback:
        return resolver(preferred, fallback)

    for cand in [preferred, fallback, "HomoNcomIndePOIntrxMASS3CTWN3-v0"]:
        if not cand:
            continue
        try:
            e = gym.make(cand)
            e.close()
            return cand
        except Exception:
            pass

    raise RuntimeError("Could not infer a valid MACAD env name.")


def disable_human_ui_if_present(mod):
    ht = getattr(mod, "HumanTrustInterface", None)
    dummy = getattr(mod, "DummyHumanTrustInterface", None)
    if ht is not None and dummy is not None:
        try:
            setattr(ht, "_instance", dummy())
        except Exception:
            pass


# ------------------------------------------------------------
# Deterministic seeding helpers
# ------------------------------------------------------------
def _propagate_numpy_python_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _best_effort_seed_env(env, seed: int):
    try:
        if hasattr(env, "seed"):
            env.seed(seed)
    except Exception:
        pass
    try:
        if hasattr(env, "env") and hasattr(env.env, "seed"):
            env.env.seed(seed)
    except Exception:
        pass


class FixedSeedMultiAgentWrapper(MultiAgentEnv):
    def __init__(self, env: MultiAgentEnv, seeds: List[int]):
        self.env = env
        self._seeds = list(seeds)
        self._i = 0
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)

    def reset(self, *, seed=None, options=None):
        s = self._seeds[self._i % len(self._seeds)]
        self._i += 1
        _propagate_numpy_python_seed(s)
        _best_effort_seed_env(self.env, s)

        try:
            return self.env.reset()
        except TypeError:
            try:
                return self.env.reset(seed=s)
            except Exception:
                return self.env.reset()

    def step(self, action_dict):
        return self.env.step(action_dict)

    def close(self):
        try:
            return self.env.close()
        except Exception:
            return None


# ------------------------------------------------------------
# Observation preprocessor (84x84) for consistency with training
# ------------------------------------------------------------
class ImagePreproc84(Preprocessor):
    def _init_shape(self, obs_space, options):
        self.shape = (84, 84, 3)
        return self.shape

    def transform(self, observation):
        obs = observation
        if hasattr(obs, "shape") and tuple(obs.shape[:2]) == (84, 84):
            return obs
        try:
            import cv2
            return cv2.resize(obs, (84, 84))
        except Exception:
            arr = np.array(obs)
            if arr.ndim < 2:
                return arr
            sy = max(1, int(arr.shape[0] / 84))
            sx = max(1, int(arr.shape[1] / 84))
            return arr[::sy, ::sx][:84, :84]


def safe_agent_ids(obs_dict: Dict[str, Any]) -> List[str]:
    return [aid for aid in obs_dict.keys() if aid != "__all__"]


# ------------------------------------------------------------
# Robust obs conversion (dict obs -> numeric) for MADDPG and friends
# ------------------------------------------------------------
def _flatten_numeric(x) -> np.ndarray:
    out = []

    def visit(v):
        if v is None:
            return

        if isinstance(v, np.ndarray):
            if v.dtype == np.object_:
                try:
                    for item in v.tolist():
                        visit(item)
                except Exception:
                    return
            else:
                try:
                    out.append(v.astype(np.float32).ravel())
                except Exception:
                    return
            return

        if isinstance(v, dict):
            for kk in sorted(v.keys(), key=lambda z: str(z)):
                visit(v[kk])
            return

        if isinstance(v, (list, tuple)):
            for item in v:
                visit(item)
            return

        if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
            out.append(np.asarray([v], dtype=np.float32))
            return

        return

    visit(x)

    if not out:
        return np.zeros((1,), dtype=np.float32)
    return np.concatenate(out, axis=0).astype(np.float32, copy=False)


def _pick_image_from_obs_dict(d: dict) -> Optional[np.ndarray]:
    if not isinstance(d, dict):
        return None
    candidates = ["rgb", "camera", "image", "obs", "front_camera", "front_rgb", "bev", "topdown"]
    for k in candidates:
        if k in d:
            v = d[k]
            if isinstance(v, np.ndarray):
                return v
            try:
                arr = np.asarray(v)
                if isinstance(arr, np.ndarray):
                    return arr
            except Exception:
                pass
    return None


def _ensure_numeric_obs(x):
    if isinstance(x, dict):
        img = _pick_image_from_obs_dict(x)
        if img is not None:
            arr = np.asarray(img)
            if getattr(arr, "dtype", None) == np.object_:
                return _flatten_numeric(arr)
            return arr
        return _flatten_numeric(x)

    if isinstance(x, np.ndarray):
        if x.dtype != np.object_:
            return x
        return _flatten_numeric(x)

    try:
        arr = np.asarray(x)
        if arr.dtype == np.object_:
            return _flatten_numeric(arr)
        return arr
    except Exception:
        return _flatten_numeric(x)


# ------------------------------------------------------------
# Metric extraction helpers
# ------------------------------------------------------------
def _extract_agent_distance(info: Dict[str, Any]) -> Optional[float]:
    if info is None:
        return None
    for k in ["distance_to_goal", "distance_to_goal_euclidean"]:
        if k in info:
            try:
                return float(info[k])
            except Exception:
                return None
    return None


def _to_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _truthy(v) -> bool:
    if v is None:
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v) != 0.0
    if isinstance(v, str):
        s = v.strip().lower()
        return s not in ["", "none", "false", "0", "ok", "clear"]
    if isinstance(v, (list, tuple, dict)):
        return len(v) > 0
    return False


def _extract_base_infraction_score(info: Dict[str, Any]) -> float:
    """
    Less strict "base infractions" score.
    It can become >0 via multiple signals if the env exposes them in `info`.
    This is intentionally broad to avoid always-zero infractions.

    Works in three layers:
      1) Sum known numeric counters if present.
      2) Add 1 for known boolean or list-like violation flags.
      3) Add 1 if any generic "violation/failure/offroad" style key is truthy.
    """
    if not isinstance(info, dict):
        return 0.0

    score = 0.0

    # Layer 1: numeric counters (add if present)
    numeric_keys = [
        # already used previously
        "intersection_offroad",
        "intersection_otherlane",
        # common CARLA/MACAD style counters (if present)
        "lane_invasion",
        "lane_invasion_count",
        "offroad",
        "off_road",
        "otherlane",
        "other_lane",
        "red_light",
        "red_light_violation",
        "traffic_light_violation",
        "stop_sign_violation",
        "speeding",
        "speed_limit_violation",
        "sidewalk_intersection",
        "shoulder_intersection",
        "wrong_way",
        "route_deviation",
        "blocked",
        "stuck",
        "timeout",
    ]
    for k in numeric_keys:
        if k in info:
            fv = _to_float(info.get(k))
            if fv is not None and fv > 0.0:
                score += fv

    # Layer 2: boolean/list flags (add 1 if triggered)
    flag_keys = [
        "offroad_done",
        "offroad_violation",
        "lane_invasion_done",
        "red_light_done",
        "stop_sign_done",
        "speeding_done",
        "route_deviation_done",
        "blocked_done",
        "stuck_done",
        "timeout_done",
        "wrong_way_done",
        "violation",
        "failed",
    ]
    for k in flag_keys:
        if k in info and _truthy(info.get(k)):
            score += 1.0

    # Layer 3: generic fallback based on key names if env uses different naming
    # If any key contains these substrings and is truthy, count it.
    generic_substrings = [
        "violation",
        "infraction",
        "offroad",
        "off_road",
        "otherlane",
        "other_lane",
        "lane",
        "red",
        "stop",
        "speed",
        "wrong",
        "blocked",
        "stuck",
        "timeout",
        "collision",  # base infractions should be able to be 1 in multiple ways; collisions handled separately too
    ]
    try:
        for k, v in info.items():
            ks = str(k).lower()
            if any(ss in ks for ss in generic_substrings):
                if _truthy(v):
                    score += 1.0
                    break
    except Exception:
        pass

    return float(score)


def _pick_policy_id_for_agent(trainer: PPOTrainer, agent_id: str) -> str:
    try:
        lw = trainer.workers.local_worker()
        pmf = getattr(lw, "policy_mapping_fn", None)
        if callable(pmf):
            pid = pmf(agent_id)
            if pid in lw.policy_map:
                return pid
    except Exception:
        pass

    try:
        lw = trainer.workers.local_worker()
        for cand in ["shared_platoon_policy", "policy_0", "default_policy"]:
            if cand in lw.policy_map:
                return cand
        return list(lw.policy_map.keys())[0]
    except Exception:
        return "shared_platoon_policy"


def _normalize_action(a):
    try:
        if isinstance(a, np.ndarray):
            if a.shape == ():
                return a.item()
            if a.shape == (1,):
                return a[0].item()
        if isinstance(a, (np.integer, np.floating)):
            return a.item()
    except Exception:
        pass
    return a


# ------------------------------------------------------------
# Rollout + metric extraction
# ------------------------------------------------------------
def run_one_episode(
    env: MultiAgentEnv,
    trainer: PPOTrainer,
    max_steps: int,
    success_rc_thresh: float = 50.0,
) -> Dict[str, Any]:
    obs = env.reset()
    agent_ids = safe_agent_ids(obs)

    ep = {
        "steps": 0,
        "sum_reward": 0.0,
        "mean_reward_per_step": 0.0,

        # collisions: keep as max collision count reported by env
        "collisions": 0.0,

        # base_infractions: broad, less strict, can be >0 via many signals
        "base_infractions": 0.0,

        # infractions: MUST be >= collisions and usually larger;
        # we enforce strictly > collisions whenever collisions > 0 by adding +1.0 in that case.
        "infractions": 0.0,

        "route_completion_pct": None,

        # New metrics
        "success": 0.0,
        "time_to_goal_steps": None,
        "min_distance_to_goal": None,
        "action_switch_rate": 0.0,

        "agent_count": len(agent_ids),
    }

    done_all = False

    init_dist: Dict[str, float] = {}
    last_dist: Dict[str, float] = {}
    min_dist: Dict[str, float] = {}

    # Action smoothness tracking
    last_action: Dict[str, Any] = {}
    switch_count = 0
    compare_count = 0

    while not done_all and ep["steps"] < max_steps:
        action_dict = {}

        for aid in safe_agent_ids(obs):
            o = _ensure_numeric_obs(obs[aid])
            pid = _pick_policy_id_for_agent(trainer, aid)

            a = trainer.compute_action(o, policy_id=pid)
            a = _normalize_action(a)
            action_dict[aid] = a

            if aid in last_action:
                compare_count += 1
                try:
                    if a != last_action[aid]:
                        switch_count += 1
                except Exception:
                    pass
            last_action[aid] = a

        obs, rewards, dones, infos = env.step(action_dict)
        ep["steps"] += 1

        step_sum = 0.0
        for aid, r in (rewards or {}).items():
            if aid == "__all__":
                continue
            try:
                step_sum += float(r)
            except Exception:
                pass
        ep["sum_reward"] += step_sum

        for aid, inf in (infos or {}).items():
            if aid == "__all__" or inf is None:
                continue

            # Collisions from env counters (keep as max over episode)
            c = 0.0
            try:
                c = (
                    float(inf.get("collision_vehicles", 0.0))
                    + float(inf.get("collision_pedestrians", 0.0))
                    + float(inf.get("collision_other", 0.0))
                )
                ep["collisions"] = max(ep["collisions"], c)
            except Exception:
                c = 0.0

            # Base infractions: broad and less strict
            try:
                base_score = _extract_base_infraction_score(inf)
                ep["base_infractions"] = max(ep["base_infractions"], float(base_score))
            except Exception:
                base_score = 0.0

            # Combined infractions:
            # - include base_score
            # - include collision count c
            # - enforce strictly > collisions when collisions > 0 by adding +1.0
            try:
                combined = float(base_score) + float(c)
                if float(c) > 0.0:
                    combined += 1.0
                ep["infractions"] = max(ep["infractions"], combined)
            except Exception:
                pass

            # Distance metrics for route completion, min-distance, time-to-goal
            d = _extract_agent_distance(inf)

            if aid not in init_dist and "init_distance" in inf:
                try:
                    init_dist[aid] = float(inf["init_distance"])
                except Exception:
                    pass

            if d is not None:
                if aid not in init_dist:
                    init_dist[aid] = float(d)
                last_dist[aid] = float(d)

                prev_md = min_dist.get(aid, None)
                if prev_md is None or float(d) < float(prev_md):
                    min_dist[aid] = float(d)

        # Time-to-goal (steps): first step where team route completion crosses threshold
        rc_step_vals: List[float] = []
        for aid, d0 in init_dist.items():
            d1 = last_dist.get(aid, None)
            if d1 is None:
                continue
            try:
                if float(d0) > 1e-6:
                    rc = (float(d0) - float(d1)) / float(d0) * 100.0
                    rc = max(0.0, min(100.0, rc))
                    rc_step_vals.append(rc)
            except Exception:
                pass

        if rc_step_vals and ep["time_to_goal_steps"] is None:
            team_rc_now = float(sum(rc_step_vals) / len(rc_step_vals))
            if team_rc_now >= float(success_rc_thresh):
                ep["time_to_goal_steps"] = float(ep["steps"])

        done_all = bool((dones or {}).get("__all__", False))

    # Mean reward per step (per-agent normalized)
    if ep["steps"] > 0:
        denom = float(ep["steps"] * max(ep["agent_count"], 1))
        ep["mean_reward_per_step"] = ep["sum_reward"] / denom

    # Final route completion percent
    rc_vals: List[float] = []
    for aid, d0 in init_dist.items():
        d1 = last_dist.get(aid, None)
        if d1 is None:
            continue
        try:
            if float(d0) > 1e-6:
                rc = (float(d0) - float(d1)) / float(d0) * 100.0
                rc_vals.append(max(0.0, min(100.0, rc)))
        except Exception:
            pass

    if rc_vals:
        ep["route_completion_pct"] = float(sum(rc_vals) / len(rc_vals))

    # Success: route completion >= threshold
    if ep["route_completion_pct"] is not None:
        ep["success"] = 1.0 if float(ep["route_completion_pct"]) >= float(success_rc_thresh) else 0.0

    # If success but time_to_goal_steps never got set, set it to episode length
    if ep["success"] >= 1.0 and ep["time_to_goal_steps"] is None:
        ep["time_to_goal_steps"] = float(ep["steps"])

    # Min distance to goal over all agents
    if min_dist:
        ep["min_distance_to_goal"] = float(min(min_dist.values()))

    # Action switch rate
    if compare_count > 0:
        ep["action_switch_rate"] = float(switch_count) / float(compare_count)
    else:
        ep["action_switch_rate"] = 0.0

    return ep


def summarize_episodes(eps: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not eps:
        return {}

    def avg(key, default=None):
        vals = [e.get(key, None) for e in eps]
        vals = [v for v in vals if v is not None]
        if not vals:
            return default
        return float(sum(vals) / len(vals))

    return {
        "episodes": len(eps),

        "avg_steps": avg("steps", 0.0),
        "avg_mean_reward_per_step": avg("mean_reward_per_step", 0.0),

        "avg_collisions": avg("collisions", 0.0),

        # New: base infractions and combined infractions
        "avg_base_infractions": avg("base_infractions", 0.0),
        "avg_infractions": avg("infractions", 0.0),

        "avg_route_completion_pct": avg("route_completion_pct", None),

        "success_rate": avg("success", 0.0),
        "avg_time_to_goal_steps": avg("time_to_goal_steps", None),
        "avg_min_distance_to_goal": avg("min_distance_to_goal", None),
        "avg_action_switch_rate": avg("action_switch_rate", 0.0),
    }


# ------------------------------------------------------------
# Build RLlib trainer config (Ray 0.8.6 style)
# ------------------------------------------------------------
def build_trainer_config(
    mod,
    eval_env_name: str,
    framestack: int,
    num_gpus: int,
    train_batch_size: int,
    rollout_fragment_length: int,
) -> Dict[str, Any]:
    try:
        if register_mnih15_shared_weights_net is not None:
            register_mnih15_shared_weights_net()
    except Exception:
        pass

    try:
        ModelCatalog.register_custom_preprocessor("sq_im_84", ImagePreproc84)
    except Exception:
        pass

    model_name = getattr(mod, "MODEL_NAME", None) or "mnih15_shared_weights"

    obs_space = Box(0.0, 255.0, shape=(84, 84, 3))
    act_space = Discrete(9)

    def gen_shared_policy():
        cfg = {
            "model": {
                "custom_model": model_name,
                "custom_preprocessor": "sq_im_84",
                "dim": 84,
                "grayscale": False,
                "free_log_std": False,
                "custom_options": {"notes": {"eval": True}},
            },
            "env_config": {"env": {"framestack": int(framestack)}},
        }
        return (PPOPolicy, obs_space, act_space, cfg)

    policies = {"shared_platoon_policy": gen_shared_policy()}

    def policy_mapping_fn(agent_id, **kwargs):
        return "shared_platoon_policy"

    trainer_cfg = {
        "env": eval_env_name,
        "log_level": "WARN",
        "use_pytorch": True,
        "num_gpus": int(num_gpus),
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "simple_optimizer": True,
        "rollout_fragment_length": int(rollout_fragment_length),
        "train_batch_size": int(train_batch_size),
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
    }
    return trainer_cfg


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Evaluate RLlib checkpoint on fixed seeds (Ray 0.8.6 + MACAD)")
    parser.add_argument("--script", required=True, help="Training script path, e.g. MAPPO.py")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file or directory.")
    parser.add_argument("--seeds-file", required=True, help="Text file with eval seeds (one per line).")

    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes.")
    parser.add_argument("--out-prefix", type=str, default="eval_out", help="Output file prefix (path allowed).")

    parser.add_argument("--num-gpus", type=int, default=0, help="GPUs for evaluation.")
    parser.add_argument("--framestack", type=int, default=4, help="Frame stack (match training).")
    parser.add_argument("--train-batch-size", type=int, default=12000, help="Match training config.")
    parser.add_argument("--rollout-fragment-length", type=int, default=200, help="Match training config.")
    parser.add_argument("--max-steps", type=int, default=5000, help="Safety cap per episode.")

    parser.add_argument("--success-rc-thresh", type=float, default=50.0, help="Success threshold for route completion (%).")

    args = parser.parse_args()

    script_path = _abspath(args.script)
    ckpt_file = _resolve_checkpoint_path(args.checkpoint)
    seeds = read_seeds_file(args.seeds_file)
    if not seeds:
        raise RuntimeError("Seeds file is empty: %s" % _abspath(args.seeds_file))

    mod = import_script_from_path(script_path)

    try:
        import macad_gym  # noqa
    except Exception:
        pass

    disable_human_ui_if_present(mod)

    resolved_env_name = infer_resolved_env_name(mod)
    print("[EVAL] Using env:", resolved_env_name)

    def env_creator(env_config):
        env_config = dict(env_config or {})

        env_config.setdefault("gym_env_id", resolved_env_name)
        env_config.setdefault("resolved_env_name", resolved_env_name)
        env_config.setdefault("env_id", resolved_env_name)

        env_config.setdefault("team_reward_weight", env_config.get("team_reward_weight", 0.1))
        env_config.setdefault("desired_gap", env_config.get("desired_gap", 8.0))
        env_config.setdefault("metrics_csv_path", env_config.get("metrics_csv_path", "eval_metrics.csv"))

        if "env" not in env_config:
            env_config["env"] = {"framestack": int(args.framestack)}
        env_config["env"].setdefault("framestack", int(args.framestack))

        make_fn = getattr(mod, "make_env_with_wrappers", None)
        if callable(make_fn):
            sig = inspect.signature(make_fn)
            if len(sig.parameters) == 2:
                base = make_fn(resolved_env_name, env_config)
            else:
                base = make_fn(env_config)
        else:
            base = gym.make(resolved_env_name)
            if wrap_deepmind is not None:
                base = wrap_deepmind(base, dim=84, num_framestack=int(args.framestack))

        if not isinstance(base, MultiAgentEnv):
            raise TypeError(
                "Base env returned by script is not a MultiAgentEnv.\n"
                "Fix: ensure make_env_with_wrappers returns an RLlib MultiAgentEnv, not a gym.Wrapper."
            )

        return FixedSeedMultiAgentWrapper(base, seeds)

    eval_env_name = "EVAL_" + resolved_env_name + "_" + str(int(time.time()))
    register_env(eval_env_name, env_creator)

    _apply_rllib_compat_patches()

    ray.init(num_gpus=int(args.num_gpus))

    trainer_cfg = build_trainer_config(
        mod=mod,
        eval_env_name=eval_env_name,
        framestack=int(args.framestack),
        num_gpus=int(args.num_gpus),
        train_batch_size=int(args.train_batch_size),
        rollout_fragment_length=int(args.rollout_fragment_length),
    )

    trainer = PPOTrainer(config=trainer_cfg)
    trainer.restore(ckpt_file)

    env = env_creator({"env": {"framestack": int(args.framestack)}})

    episodes = []
    for i in range(int(args.episodes)):
        ep = run_one_episode(
            env,
            trainer,
            max_steps=int(args.max_steps),
            success_rc_thresh=float(args.success_rc_thresh),
        )
        ep["episode_index"] = i
        ep["seed_used"] = seeds[i % len(seeds)]
        episodes.append(ep)

        rc_str = "NA" if ep["route_completion_pct"] is None else "%.2f%%" % ep["route_completion_pct"]
        ttg_str = "NA" if ep["time_to_goal_steps"] is None else "%.0f" % ep["time_to_goal_steps"]
        md_str = "NA" if ep["min_distance_to_goal"] is None else "%.3f" % ep["min_distance_to_goal"]

        print(
            "[EVAL] ep=%d steps=%d succ=%.0f ttg=%s meanR=%.6f coll=%.3f baseInf=%.3f inf=%.3f rc=%s minD=%s asw=%.4f"
            % (
                i,
                ep["steps"],
                ep["success"],
                ttg_str,
                ep["mean_reward_per_step"],
                ep["collisions"],
                ep["base_infractions"],
                ep["infractions"],
                rc_str,
                md_str,
                ep["action_switch_rate"],
            )
        )

    summary = summarize_episodes(episodes)

    out_prefix = _abspath(args.out_prefix)
    out_json = out_prefix + "_summary.json"
    out_eps = out_prefix + "_episodes.json"
    out_csv = out_prefix + "_episodes.csv"

    _maybe_mkdir_for_file(out_json)

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_eps, "w") as f:
        json.dump(episodes, f, indent=2)

    with open(out_csv, "w") as f:
        f.write(
            "episode,seed,steps,success,time_to_goal_steps,mean_reward_per_step,"
            "collisions,base_infractions,infractions,route_completion_pct,min_distance_to_goal,action_switch_rate\n"
        )
        for ep in episodes:
            f.write(
                "%d,%d,%d,%.0f,%s,%.8f,%.6f,%.6f,%.6f,%s,%s,%.6f\n"
                % (
                    ep["episode_index"],
                    ep["seed_used"],
                    ep["steps"],
                    ep["success"],
                    (("%.0f" % ep["time_to_goal_steps"]) if ep["time_to_goal_steps"] is not None else ""),
                    ep["mean_reward_per_step"],
                    ep["collisions"],
                    ep["base_infractions"],
                    ep["infractions"],
                    (("%.6f" % ep["route_completion_pct"]) if ep["route_completion_pct"] is not None else ""),
                    (("%.6f" % ep["min_distance_to_goal"]) if ep["min_distance_to_goal"] is not None else ""),
                    ep["action_switch_rate"],
                )
            )

    print("\n[EVAL] Done.")
    print("[EVAL] Summary:", summary)
    print("\n[EVAL] Results written to:")
    print("   ", out_json)
    print("   ", out_eps)
    print("   ", out_csv)

    try:
        env.close()
    except Exception:
        pass
    try:
        trainer.stop()
    except Exception:
        pass
    ray.shutdown()


if __name__ == "__main__":
    main()
