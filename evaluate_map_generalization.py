#!/usr/bin/env python
# evaluate_map_generalization.py
#
# Map Generalization Evaluation Script
# Tests a trained model (trained on Town03) on different CARLA towns
# to measure map generalization performance.
#
# Usage:
#   python evaluate_map_generalization.py \
#       --script CG-HMT-PPO.py \
#       --checkpoint <path_to_checkpoint> \
#       --seeds-file eval_seeds.txt \
#       --episodes-per-town 20 \
#       --out-prefix map_gen_results

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Any

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

# Import custom scenarios for different towns
try:
    import scenarios.custom_stop_sign_3c_town01  # noqa
    import scenarios.custom_stop_sign_3c_town02  # noqa
    # Town03 uses the standard MACAD env or custom 5c version
    try:
        import scenarios.custom_stop_sign_5c_town03  # noqa
    except ImportError:
        pass
    import scenarios.custom_stop_sign_3c_town04  # noqa
    import scenarios.custom_stop_sign_3c_town05  # noqa
    import scenarios.custom_stop_sign_3c_town10  # noqa
except ImportError as e:
    print(f"[WARN] Could not import some custom scenarios: {e}")

# Import helpers from evaluate_checkpoint.py
try:
    from evaluate_checkpoint import (
        _abspath,
        _maybe_mkdir_for_file,
        _resolve_checkpoint_path,
        import_script_from_path,
        read_seeds_file,
        infer_resolved_env_name,
        disable_human_ui_if_present,
        _apply_rllib_compat_patches,
        build_trainer_config,
        FixedSeedMultiAgentWrapper,
        run_one_episode,
        summarize_episodes,
    )
except ImportError:
    # If evaluate_checkpoint is not importable, define minimal versions
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
            for name in os.listdir(ckpt_path):
                if name.startswith("checkpoint-") and not name.endswith(".tune_metadata"):
                    return os.path.join(ckpt_path, name)
        raise RuntimeError(f"Could not resolve checkpoint: {ckpt_path}")

    def import_script_from_path(script_path: str):
        import importlib.util
        script_path = _abspath(script_path)
        mod_name = os.path.splitext(os.path.basename(script_path))[0] + "_evalmod"
        spec = importlib.util.spec_from_file_location(mod_name, script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module: {script_path}")
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
                if s and not s.startswith("#"):
                    seeds.append(int(s))
        return seeds

    def infer_resolved_env_name(mod) -> str:
        preferred = getattr(mod, "PREFERRED_ENV_NAME", None)
        fallback = getattr(mod, "FALLBACK_ENV_NAME", None)
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

    def _apply_rllib_compat_patches():
        pass  # Minimal version

    def build_trainer_config(mod, eval_env_name, framestack, num_gpus, train_batch_size, rollout_fragment_length):
        from rllib.models import register_mnih15_shared_weights_net
        try:
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

        return {
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

    class FixedSeedMultiAgentWrapper(MultiAgentEnv):
        def __init__(self, env: MultiAgentEnv, seeds: List[int]):
            self.env = env
            self._seeds = list(seeds)
            self._i = 0
            self.observation_space = getattr(env, "observation_space", None)
            self.action_space = getattr(env, "action_space", None)

        def reset(self, *, seed=None, options=None):
            import random
            s = self._seeds[self._i % len(self._seeds)]
            self._i += 1
            random.seed(s)
            np.random.seed(s)
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

    def run_one_episode(env, trainer, max_steps):
        obs = env.reset()
        agent_ids = [aid for aid in obs.keys() if aid != "__all__"]

        ep = {
            "steps": 0,
            "sum_reward": 0.0,
            "mean_reward_per_step": 0.0,
            "collisions": 0.0,
            "infractions": 0.0,
            "route_completion_pct": None,
            "agent_count": len(agent_ids),
        }

        done_all = False
        init_dist = {}
        last_dist = {}

        while not done_all and ep["steps"] < max_steps:
            action_dict = {}
            for aid in agent_ids:
                o = obs[aid]
                if isinstance(o, dict):
                    # Extract image if dict
                    o = o.get("rgb", o.get("camera", o.get("image", None)))
                if o is None:
                    continue
                o = np.asarray(o)
                pid = "shared_platoon_policy"
                a = trainer.compute_action(o, policy_id=pid)
                action_dict[aid] = a if not isinstance(a, np.ndarray) or a.shape == () else a.item()

            obs, rewards, dones, infos = env.step(action_dict)
            ep["steps"] += 1

            step_sum = 0.0
            for aid, r in (rewards or {}).items():
                if aid == "__all__":
                    continue
                step_sum += float(r)
            ep["sum_reward"] += step_sum

            for aid, inf in (infos or {}).items():
                if aid == "__all__" or inf is None:
                    continue
                try:
                    c = float(inf.get("collision_vehicles", 0.0)) + \
                        float(inf.get("collision_pedestrians", 0.0)) + \
                        float(inf.get("collision_other", 0.0))
                    ep["collisions"] = max(ep["collisions"], c)
                except Exception:
                    pass

                try:
                    infr = float(inf.get("intersection_offroad", 0.0)) + \
                           float(inf.get("intersection_otherlane", 0.0))
                    ep["infractions"] = max(ep["infractions"], infr)
                except Exception:
                    pass

                d = None
                for k in ["distance_to_goal", "distance_to_goal_euclidean"]:
                    if k in inf:
                        try:
                            d = float(inf[k])
                            break
                        except Exception:
                            pass

                if d is not None:
                    if aid not in init_dist:
                        init_dist[aid] = d
                    last_dist[aid] = d

            done_all = bool((dones or {}).get("__all__", False))

        if ep["steps"] > 0:
            ep["mean_reward_per_step"] = ep["sum_reward"] / float(ep["steps"] * max(ep["agent_count"], 1))

        rc_vals = []
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
            "avg_infractions": avg("infractions", 0.0),
            "avg_route_completion_pct": avg("route_completion_pct", None),
        }


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


# Town environment IDs mapping
TOWN_ENV_IDS = {
    "Town01": "HomoNcomIndePOIntrxMASS3CTWN1-v0",
    "Town02": "HomoNcomIndePOIntrxMASS3CTWN2-v0",
    "Town03": "HomoNcomIndePOIntrxMASS3CTWN3-v0",  # Training town
    "Town04": "HomoNcomIndePOIntrxMASS3CTWN4-v0",
    "Town05": "HomoNcomIndePOIntrxMASS3CTWN5-v0",
    "Town10": "HomoNcomIndePOIntrxMASS3CTWN10-v0",
}


def evaluate_on_town(
    trainer: PPOTrainer,
    town_name: str,
    env_id: str,
    seeds: List[int],
    episodes_per_town: int,
    max_steps: int,
    framestack: int,
) -> Dict[str, Any]:
    """Evaluate trainer on a specific town."""
    print(f"\n[EVAL] Evaluating on {town_name} ({env_id})...")

    try:
        import macad_gym  # noqa
    except Exception:
        pass

    try:
        # Try to make the environment
        test_env = gym.make(env_id)
        test_env.close()
    except Exception as e:
        print(f"[WARN] Could not create environment {env_id}: {e}")
        return {
            "town": town_name,
            "env_id": env_id,
            "status": "failed",
            "error": str(e),
            "episodes": [],
            "summary": {},
        }

    def env_creator(env_config):
        env_config = dict(env_config or {})
        env_config.setdefault("gym_env_id", env_id)
        env_config.setdefault("env", {"framestack": int(framestack)})

        # Try to use the script's make_env_with_wrappers if available
        try:
            from rllib.env_wrappers import wrap_deepmind
            base = gym.make(env_id)
            if wrap_deepmind is not None:
                base = wrap_deepmind(base, dim=84, num_framestack=int(framestack))
            if not isinstance(base, MultiAgentEnv):
                raise TypeError("Base env is not MultiAgentEnv")
            return FixedSeedMultiAgentWrapper(base, seeds)
        except Exception as e:
            print(f"[WARN] Error creating env for {town_name}: {e}")
            base = gym.make(env_id)
            return FixedSeedMultiAgentWrapper(base, seeds)

    eval_env_name = f"EVAL_{town_name}_{int(time.time())}"
    register_env(eval_env_name, env_creator)

    try:
        env = env_creator({"env": {"framestack": int(framestack)}})
    except Exception as e:
        print(f"[ERROR] Failed to create environment for {town_name}: {e}")
        return {
            "town": town_name,
            "env_id": env_id,
            "status": "failed",
            "error": str(e),
            "episodes": [],
            "summary": {},
        }

    episodes = []
    for i in range(episodes_per_town):
        try:
            ep = run_one_episode(env, trainer, max_steps)
            ep["episode_index"] = i
            ep["seed_used"] = seeds[i % len(seeds)]
            episodes.append(ep)

            rc_str = "NA" if ep["route_completion_pct"] is None else f"{ep['route_completion_pct']:.2f}%"
            print(
                f"[EVAL] {town_name} ep={i} steps={ep['steps']} meanR={ep['mean_reward_per_step']:.6f} "
                f"coll={ep['collisions']:.3f} infr={ep['infractions']:.3f} rc={rc_str}"
            )
        except Exception as e:
            print(f"[ERROR] Episode {i} failed on {town_name}: {e}")
            episodes.append({
                "episode_index": i,
                "status": "failed",
                "error": str(e),
            })

    summary = summarize_episodes([e for e in episodes if "status" not in e])

    try:
        env.close()
    except Exception:
        pass

    return {
        "town": town_name,
        "env_id": env_id,
        "status": "success",
        "episodes": episodes,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        "Map Generalization Evaluation - Test trained model on different CARLA towns"
    )
    parser.add_argument("--script", required=True, help="Training script path, e.g. CG-HMT-PPO.py")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file or directory")
    parser.add_argument("--seeds-file", required=True, help="Text file with eval seeds (one per line)")

    parser.add_argument("--episodes-per-town", type=int, default=20, help="Episodes per town")
    parser.add_argument("--out-prefix", type=str, default="map_gen_results", help="Output file prefix")
    parser.add_argument("--towns", nargs="+", default=None, help="Specific towns to test (default: all)")

    parser.add_argument("--num-gpus", type=int, default=0, help="GPUs for evaluation")
    parser.add_argument("--framestack", type=int, default=4, help="Frame stack (match training)")
    parser.add_argument("--train-batch-size", type=int, default=12000, help="Match training config")
    parser.add_argument("--rollout-fragment-length", type=int, default=200, help="Match training config")
    parser.add_argument("--max-steps", type=int, default=5000, help="Safety cap per episode")

    args = parser.parse_args()

    script_path = _abspath(args.script)
    ckpt_file = _resolve_checkpoint_path(args.checkpoint)
    seeds = read_seeds_file(args.seeds_file)
    if not seeds:
        raise RuntimeError(f"Seeds file is empty: {_abspath(args.seeds_file)}")

    mod = import_script_from_path(script_path)

    try:
        import macad_gym  # noqa
    except Exception:
        pass

    disable_human_ui_if_present(mod)

    # Determine which towns to test
    towns_to_test = args.towns if args.towns else list(TOWN_ENV_IDS.keys())
    print(f"[EVAL] Testing on towns: {towns_to_test}")

    # Apply compatibility patches
    _apply_rllib_compat_patches()

    ray.init(num_gpus=int(args.num_gpus))

    # Build trainer config (use training town env for config, but we'll test on others)
    training_env_id = infer_resolved_env_name(mod)
    print(f"[EVAL] Training was on: {training_env_id}")

    trainer_cfg = build_trainer_config(
        mod=mod,
        eval_env_name=training_env_id,  # Just for config
        framestack=int(args.framestack),
        num_gpus=int(args.num_gpus),
        train_batch_size=int(args.train_batch_size),
        rollout_fragment_length=int(args.rollout_fragment_length),
    )

    trainer = PPOTrainer(config=trainer_cfg)
    trainer.restore(ckpt_file)
    print(f"[EVAL] Loaded checkpoint: {ckpt_file}")

    # Evaluate on each town
    results = {}
    for town_name in towns_to_test:
        env_id = TOWN_ENV_IDS.get(town_name)
        if not env_id:
            print(f"[WARN] Unknown town: {town_name}, skipping")
            continue

        result = evaluate_on_town(
            trainer=trainer,
            town_name=town_name,
            env_id=env_id,
            seeds=seeds,
            episodes_per_town=int(args.episodes_per_town),
            max_steps=int(args.max_steps),
            framestack=int(args.framestack),
        )
        results[town_name] = result

    # Save results
    out_prefix = _abspath(args.out_prefix)
    out_json = out_prefix + "_all_towns.json"
    out_csv = out_prefix + "_summary.csv"

    _maybe_mkdir_for_file(out_json)

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Write summary CSV
    with open(out_csv, "w") as f:
        f.write("town,status,episodes,avg_steps,avg_reward,avg_collisions,avg_infractions,avg_route_completion_pct\n")
        for town_name, result in results.items():
            if result["status"] == "success" and result["summary"]:
                s = result["summary"]
                f.write(
                    f"{town_name},{result['status']},{s.get('episodes', 0)},"
                    f"{s.get('avg_steps', 0):.2f},{s.get('avg_mean_reward_per_step', 0):.8f},"
                    f"{s.get('avg_collisions', 0):.6f},{s.get('avg_infractions', 0):.6f},"
                    f"{s.get('avg_route_completion_pct', 0) or 0:.6f}\n"
                )
            else:
                f.write(f"{town_name},{result.get('status', 'unknown')},0,0,0,0,0,0\n")

    print("\n[EVAL] Map Generalization Evaluation Complete!")
    print(f"[EVAL] Results written to:")
    print(f"   {out_json}")
    print(f"   {out_csv}")

    # Print summary table
    print("\n[EVAL] Summary by Town:")
    print(f"{'Town':<10} {'Status':<10} {'Episodes':<10} {'Avg RC %':<12} {'Avg Collisions':<15}")
    print("-" * 70)
    for town_name, result in results.items():
        if result["status"] == "success" and result["summary"]:
            s = result["summary"]
            rc = s.get("avg_route_completion_pct", 0) or 0
            coll = s.get("avg_collisions", 0)
            eps = s.get("episodes", 0)
            print(f"{town_name:<10} {result['status']:<10} {eps:<10} {rc:>10.2f}% {coll:>13.3f}")
        else:
            print(f"{town_name:<10} {result.get('status', 'unknown'):<10} {'0':<10} {'N/A':<12} {'N/A':<15}")

    try:
        trainer.stop()
    except Exception:
        pass
    ray.shutdown()


if __name__ == "__main__":
    main()

