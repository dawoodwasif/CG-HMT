#!/usr/bin/env python
# MADDPG.py (MADDPG-style centralized critic + PPO, Ray 0.8.6 compatible)

from __future__ import absolute_import, division, print_function

import argparse
import datetime
import math
import os
import re
import time
from typing import Dict, Any

import gym
import numpy as np

import macad_gym  # noqa: F401

import ray
from gym.spaces import Box, Discrete, Dict as SpaceDict
from ray.tune.registry import register_env

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

import torch
import torch.nn as nn


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

PREFERRED_ENV_NAME = "HomoNcomIndePOIntrxMASS3CTWN3-v0"
FALLBACK_ENV_NAME = "HomoNcomIndePOIntrxMASS3CTWN3-v0"

RLLIB_ENV_NAME = "MACAD_MASS3_MADDPG_STYLE_PPO"

EVAL_POLICY_MODE = "multi"
DEFAULT_MAX_AGENTS = 5
STATE_DIM_PER_AGENT = 5
DEFAULT_STATE_DIM = DEFAULT_MAX_AGENTS * STATE_DIM_PER_AGENT

EVAL_OBS_SPACE = SpaceDict({
    "image": Box(0.0, 255.0, shape=(84, 84, 3), dtype=np.float32),
    "state": Box(-1e9, 1e9, shape=(DEFAULT_STATE_DIM,), dtype=np.float32),
})
EVAL_ACT_SPACE = Discrete(9)


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


class ResetRetryWrapper(MultiAgentEnv):
    def __init__(self, env, max_retries=8, sleep_s=8.0):
        self.env = env
        self.max_retries = int(max_retries)
        self.sleep_s = float(sleep_s)
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)

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
                print("[WARN] CARLA not ready. retry %d/%d in %.1fs" % (k, self.max_retries, self.sleep_s))
                time.sleep(self.sleep_s)
        raise last_e

    def step(self, action_dict):
        return self.env.step(action_dict)

    def close(self):
        try:
            return self.env.close()
        except Exception:
            return None


class SpawnSpacingWrapper(MultiAgentEnv):
    def __init__(self, env, offset=1.0):
        self.env = env
        self.offset = float(offset)
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)

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


def agent_to_index(agent_id, max_agents):
    s = str(agent_id)
    m = re.findall(r"\d+", s)
    if m:
        idx = int(m[-1])
        if idx > 0:
            idx -= 1
    else:
        idx = abs(hash(s))
    return int(idx) % int(max_agents)


class GlobalStateObsWrapper(MultiAgentEnv):
    def __init__(self, env, max_agents=5):
        self.env = env
        self.max_agents = int(max_agents)

        self.image_space = Box(0.0, 255.0, shape=(84, 84, 3), dtype=np.float32)
        self.state_space = Box(-1e9, 1e9, shape=(self.max_agents * STATE_DIM_PER_AGENT,), dtype=np.float32)

        self.observation_space = SpaceDict({"image": self.image_space, "state": self.state_space})
        self.action_space = getattr(env, "action_space", None)

        self._last_global_state = np.zeros((self.max_agents * STATE_DIM_PER_AGENT,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        self._last_global_state = np.zeros((self.max_agents * STATE_DIM_PER_AGENT,), dtype=np.float32)
        return self._wrap_obs(obs)

    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)
        self._last_global_state = self._build_global_state(infos)
        return self._wrap_obs(obs), rewards, dones, infos

    def close(self):
        try:
            return self.env.close()
        except Exception:
            return None

    def _wrap_obs(self, obs_dict):
        out = {}
        for aid, img in obs_dict.items():
            if aid == "__all__":
                continue
            out[aid] = {
                "image": np.asarray(img, dtype=np.float32),
                "state": self._last_global_state.copy(),
            }
        return out

    def _build_global_state(self, infos):
        g = np.zeros((self.max_agents * STATE_DIM_PER_AGENT,), dtype=np.float32)
        if not infos:
            return g

        for aid, inf in infos.items():
            if aid == "__all__" or inf is None:
                continue

            idx = agent_to_index(aid, self.max_agents)
            base = idx * STATE_DIM_PER_AGENT

            x = float(inf.get("x", 0.0))
            y = float(inf.get("y", 0.0))
            yaw = float(inf.get("yaw", 0.0))
            spd = float(inf.get("forward_speed", 0.0))
            dgoal = inf.get("distance_to_goal", None)
            if dgoal is None:
                dgoal = inf.get("distance_to_goal_euclidean", 0.0)
            dgoal = float(dgoal)

            g[base + 0] = x
            g[base + 1] = y
            g[base + 2] = yaw
            g[base + 3] = spd
            g[base + 4] = dgoal

        return g


class MADDPGStyleCentralCritic(TorchModelV2, nn.Module):
    """
    RLlib 0.8.6 note:
    - Items in config["model"]["custom_model_config"] are passed as **kwargs
      into the model constructor.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # RLlib 0.8.6 passes custom_model_config as kwargs (ex: state_dim=...)
        state_dim = kwargs.get("state_dim", None)
        if state_dim is None:
            state_dim = model_config.get("custom_model_config", {}).get("state_dim", DEFAULT_STATE_DIM)
        state_dim = int(state_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 84, 84)
            out = self.cnn(dummy)
            conv_out = int(np.prod(out.shape[1:]))

        self.actor_head = nn.Sequential(
            nn.Linear(conv_out, 256), nn.ReLU(),
            nn.Linear(256, num_outputs),
        )

        self.v_head = nn.Sequential(
            nn.Linear(conv_out + state_dim, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._last_img_emb = None
        self._last_state = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        # Dict obs: {"image": Tensor, "state": Tensor}
        img = obs["image"]
        st = obs["state"]

        # (B,84,84,3) -> (B,3,84,84)
        if img.dim() == 4 and img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2).contiguous()

        x = self.cnn(img.float())
        x = x.view(x.size(0), -1)

        self._last_img_emb = x
        self._last_state = st.float()

        logits = self.actor_head(x)
        return logits, state

    def value_function(self):
        if self._last_img_emb is None or self._last_state is None:
            return torch.zeros(1)
        z = torch.cat([self._last_img_emb, self._last_state], dim=1)
        v = self.v_head(z)
        return v.squeeze(1)


ModelCatalog.register_custom_model("maddpg_style_cc", MADDPGStyleCentralCritic)


def make_env_with_wrappers(resolved_env_name, env_config):
    try:
        os.chdir(env_config.get("script_dir", SCRIPT_DIR))
    except Exception:
        pass

    from rllib.env_wrappers import wrap_deepmind

    gym_env_id = env_config.get("gym_env_id", resolved_env_name)
    env = gym.make(gym_env_id)

    env = ResetRetryWrapper(env,
                            max_retries=env_config.get("reset_retries", 8),
                            sleep_s=env_config.get("retry_sleep", 8.0))
    env = SpawnSpacingWrapper(env, offset=1.0)

    framestack = int(env_config.get("framestack", 4))
    env = wrap_deepmind(env, dim=84, num_framestack=framestack)

    max_agents = int(env_config.get("max_agents", DEFAULT_MAX_AGENTS))
    env = GlobalStateObsWrapper(env, max_agents=max_agents)

    return env


def main():
    parser = argparse.ArgumentParser("MADDPG-style (central critic + separate actors) using PPO (Ray 0.8.6)")

    parser.add_argument("--num-iters", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)

    parser.add_argument("--framestack", type=int, default=4)
    parser.add_argument("--rollout-fragment-length", type=int, default=200)
    parser.add_argument("--train-batch-size", type=int, default=10000)

    parser.add_argument("--reset-retries", type=int, default=8)
    parser.add_argument("--retry-sleep", type=float, default=8.0)

    parser.add_argument("--max-agents", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=10)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num-sgd-iter", type=int, default=10)
    parser.add_argument("--clip-param", type=float, default=0.2)

    args = parser.parse_args()

    if args.num_workers != 0:
        print("[WARN] For MACAD stability on Windows, num_workers forced to 0.")
        args.num_workers = 0

    gym_env_id = _resolve_macad_env_name(PREFERRED_ENV_NAME, FALLBACK_ENV_NAME)

    ray.init(num_gpus=int(args.num_gpus))

    register_env(RLLIB_ENV_NAME, lambda cfg: make_env_with_wrappers(gym_env_id, cfg))

    policies = {}
    for i in range(int(args.max_agents)):
        policies["policy_%d" % i] = (
            None,
            EVAL_OBS_SPACE,
            EVAL_ACT_SPACE,
            {
                "model": {
                    "custom_model": "maddpg_style_cc",
                    "custom_model_config": {
                        "state_dim": int(args.max_agents) * STATE_DIM_PER_AGENT
                    },
                }
            },
        )

    def policy_mapping_fn(agent_id, **kwargs):
        idx = agent_to_index(agent_id, int(args.max_agents))
        return "policy_%d" % idx

    config = {
        "env": RLLIB_ENV_NAME,
        "env_config": {
            "gym_env_id": gym_env_id,
            "framestack": int(args.framestack),
            "reset_retries": int(args.reset_retries),
            "retry_sleep": float(args.retry_sleep),
            "script_dir": SCRIPT_DIR,
            "max_agents": int(args.max_agents),
        },
        "num_gpus": int(args.num_gpus),
        "num_workers": int(args.num_workers),
        "num_envs_per_worker": 1,
        "simple_optimizer": True,

        "rollout_fragment_length": int(args.rollout_fragment_length),
        "train_batch_size": int(args.train_batch_size),
        "num_sgd_iter": int(args.num_sgd_iter),
        "lr": float(args.lr),
        "clip_param": float(args.clip_param),

        "use_pytorch": True,

        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "log_level": "INFO",
    }

    trainer = PPOTrainer(config=config)

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_dir = os.path.join(os.getcwd(), "ray_results", "maddpg_style_run_%s" % ts)
    os.makedirs(results_dir, exist_ok=True)

    best = -1e18
    best_ckpt = None

    for it in range(1, int(args.num_iters) + 1):
        res = trainer.train()
        mean_r = float(res.get("episode_reward_mean", 0.0))
        print("[MADDPG-STYLE] iter=%d episode_reward_mean=%.6f" % (it, mean_r))

        if mean_r > best:
            best = mean_r
            best_ckpt = trainer.save(checkpoint_dir=results_dir)
            print("[MADDPG-STYLE] New best checkpoint: %s (score=%.6f)" % (best_ckpt, best))

        if args.checkpoint_every > 0 and (it % int(args.checkpoint_every) == 0):
            ckpt = trainer.save(checkpoint_dir=results_dir)
            print("[MADDPG-STYLE] Saved checkpoint:", ckpt)

    final_ckpt = trainer.save(checkpoint_dir=results_dir)
    print("Final checkpoint path:", final_ckpt)
    print("Best checkpoint path:", best_ckpt if best_ckpt else final_ckpt)
    print("Best score:", best)

    trainer.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
