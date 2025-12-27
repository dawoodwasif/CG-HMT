#!/usr/bin/env python
# CG-HMT  (with --attack support)
from __future__ import absolute_import, division, print_function

import argparse
import datetime
import math
import os
import sys
import random

import cv2
import gym
import macad_gym  # registers MACAD envs
import numpy as np
import tkinter as tk

try:
    import scenarios.custom_stop_sign_5c_town03   # noqa
    _HAS_CUSTOM_5C = True
except ImportError:
    _HAS_CUSTOM_5C = False
    custom_stop_sign_5c_town03 = None

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

# --- Ray 0.8.6 Windows checkpoint bug workaround ---
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
# ---------------------------------------------------


# ============================================================
# 1) CLI
# ============================================================
parser = argparse.ArgumentParser("Multi-agent CARLA (IMM+Trust+Human Tk UI)")

parser.add_argument("--num-iters", type=int, default=300,
                    help="Number of PPO training iterations (outer loop).")

parser.add_argument("--num-workers", type=int, default=0,
                    help="Use 0 workers so Tk UI runs only on local worker.")

parser.add_argument("--num-gpus", type=int, default=1)

parser.add_argument("--rollout-fragment-length", type=int, default=200,
                    help="rollout_fragment_length for PPO (steps per worker per sample).")

parser.add_argument("--train-batch-size", type=int, default=12000,
                    help="train_batch_size (number of timesteps per PPO update).")

parser.add_argument("--envs-per-worker", type=int, default=1)

parser.add_argument("--team-reward-weight", type=float, default=0.1,
                    help="Weight for trust/uncertainty bonus in shaped reward (currently down-weighted).")

parser.add_argument("--desired-gap", type=float, default=8.0,
                    help="Kept for compatibility; not used directly in wrapper.")

parser.add_argument("--notes", default=None)

parser.add_argument("--metrics-csv", type=str, default="cghmt_metrics.csv",
                    help="Path to CSV file where per-episode metrics will be logged.")

parser.add_argument(
    "--attack",
    type=str,
    default="none",
    choices=["none", "v2x", "weather", "byzantine"],
    help="Apply a simple, severe test-time attack: "
         "'none' (default), 'v2x' (comm blackouts via stale observations), "
         "'weather' (heavy blur + darken), 'byzantine' (flip one agent's actions)."
)

# Use your MASS3 env for stability; MASS5 works if spawn config is correct.
PREFERRED_ENV_NAME = "HomoNcomIndePOIntrxMASS3CTWN3-v0"
FALLBACK_ENV_NAME = "HomoNcomIndePOIntrxMASS3CTWN3-v0"

# ============================================================
# 2) Model registration
# ============================================================
register_mnih15_shared_weights_net()
MODEL_NAME = "mnih15_shared_weights"

# ============================================================
# 3) Resolve env on the DRIVER
# ============================================================
def _resolve_macad_env_name(preferred_name: str, fallback_name: str) -> str:
    try:
        env = gym.make(preferred_name)
        env.close()
        print(f"[INFO] Using MACAD env: {preferred_name}")
        return preferred_name
    except gym.error.UnregisteredEnv:
        print(f"[WARN] Preferred env '{preferred_name}' not found.")
        try:
            env = gym.make(fallback_name)
            env.close()
            print(f"[INFO] Falling back to: {fallback_name}")
            return fallback_name
        except gym.error.UnregisteredEnv:
            available = [e.id for e in gym.envs.registry.all()]
            raise RuntimeError(
                f"No suitable env found. Registered IDs: {available}"
            )

# ============================================================
# 4) Simple 84x84 preprocessor
# ============================================================
class ImagePreproc(Preprocessor):
    """
    Resize to 84x84 and keep 3 channels.
    Works with wrap_deepmind(dim=84) output.
    """
    def _init_shape(self, obs_space, options):
        self.shape = (84, 84, 3)
        return self.shape

    def transform(self, observation):
        resized = cv2.resize(observation, (self.shape[0], self.shape[1]))
        return resized

ModelCatalog.register_custom_preprocessor("sq_im_84", ImagePreproc)

# ============================================================
# 5) Optional spawn spacing wrapper
# ============================================================
class SpawnSpacingWrapper(MultiAgentEnv):
    """
    Light-touch nudge after reset to reduce accidental overlaps
    if CARLA perturbs spawns.
    """
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
        # Try to get underlying actor container
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
            spacing = 8.0  # meters between bumpers
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
# 6) Tkinter Human Interface (no help button, safe shutdown)
# ============================================================
class HumanTrustInterface:
    """
    Tk-based dashboard for human trust input, in a separate window.
    """
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            try:
                cls._instance = HumanTrustInterface()
            except tk.TclError as e:
                print(f"[WARN] Could not initialize Tk UI: {e}. Disabling human UI.")
                cls._instance = DummyHumanTrustInterface()
        return cls._instance

    def __init__(self):
        self.alive = True

        self.root = tk.Tk()
        self.root.title("Human-Team Trust Dashboard")

        try:
            self.root.attributes("-topmost", True)
        except tk.TclError:
            pass

        # Header instructions
        self.instructions = tk.Label(
            self.root,
            text=(
                "Quick controls:\n"
                "Keyboard: 1–5 = select agent | q = Unreliable | w = Monitor | e = Reliable\n"
                "Mouse: click Unreliable / Monitor / Reliable buttons for each agent"
            ),
            font=("Arial", 11),
            justify="left"
        )
        self.instructions.pack(fill="x", padx=8, pady=(4, 2))

        self.global_label = tk.Label(
            self.root,
            text="Global human trust: 0.50 | Selected: -",
            font=("Arial", 11, "bold")
        )
        self.global_label.pack(fill="x", padx=8, pady=(2, 4))

        # Table header
        header = tk.Frame(self.root)
        header.pack(fill="x", padx=8)

        tk.Label(header, text="#", width=3, font=("Arial", 10, "bold")).grid(row=0, column=0, padx=2)
        tk.Label(header, text="Agent", width=8, font=("Arial", 10, "bold")).grid(row=0, column=1, padx=2)
        tk.Label(header, text="Auto", width=8, font=("Arial", 10, "bold")).grid(row=0, column=2, padx=2)
        tk.Label(header, text="Human", width=8, font=("Arial", 10, "bold")).grid(row=0, column=3, padx=2)
        tk.Label(header, text="Combined", width=10, font=("Arial", 10, "bold")).grid(row=0, column=4, padx=2)
        tk.Label(header, text="U", width=6, font=("Arial", 10, "bold")).grid(row=0, column=5, padx=2)
        tk.Label(header, text="Reward", width=10, font=("Arial", 10, "bold")).grid(row=0, column=6, padx=2)
        tk.Label(header, text="Status", width=14, font=("Arial", 10, "bold")).grid(row=0, column=7, padx=2)
        tk.Label(header, text="Feedback", width=24, font=("Arial", 10, "bold")).grid(
            row=0, column=8, columnspan=3, padx=2
        )

        self.table_frame = tk.Frame(self.root)
        self.table_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self.agent_widgets = {}
        self.selected_agent = None
        self.pending_nudges = {}
        self.last_global_trust = 0.5

        self.root.bind("<Key>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        print("[INFO] Human UI closed. Disabling further UI updates.")
        self.alive = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _set_nudge(self, aid, value):
        self.selected_agent = aid
        self.pending_nudges[aid] = value
        self._update_global_label()

    def _update_global_label(self):
        sel = self.selected_agent if self.selected_agent is not None else "-"
        self.global_label.config(
            text=f"Global human trust: {self.last_global_trust:.2f} | Selected: {sel}"
        )

    def on_key(self, event):
        if not self.alive:
            return
        c = (event.char or "").lower()
        if not c:
            return

        if c in "12345":
            idx = int(c) - 1
            agent_ids = list(self.agent_widgets.keys())
            if 0 <= idx < len(agent_ids):
                self.selected_agent = agent_ids[idx]
                self._update_global_label()
            return

        if self.selected_agent is not None and self.selected_agent in self.agent_widgets:
            if c == "q":
                self.pending_nudges[self.selected_agent] = -0.2
            elif c == "w":
                self.pending_nudges[self.selected_agent] = 0.0
            elif c == "e":
                self.pending_nudges[self.selected_agent] = 0.2
            self._update_global_label()

    def _ensure_agent_row(self, aid):
        if aid in self.agent_widgets:
            return

        row_idx = len(self.agent_widgets) + 1
        row_frame = tk.Frame(self.table_frame, bd=1, relief="solid", padx=2, pady=2)
        row_frame.pack(fill="x", pady=2)

        idx_lbl = tk.Label(row_frame, text=str(row_idx), width=3)
        idx_lbl.grid(row=0, column=0, padx=2)

        name_lbl = tk.Label(row_frame, text=str(aid), width=8)
        name_lbl.grid(row=0, column=1, padx=2)

        auto_lbl = tk.Label(row_frame, text="0.00", width=8)
        auto_lbl.grid(row=0, column=2, padx=2)

        human_lbl = tk.Label(row_frame, text="0.50", width=8)
        human_lbl.grid(row=0, column=3, padx=2)

        comb_lbl = tk.Label(row_frame, text="0.25", width=10)
        comb_lbl.grid(row=0, column=4, padx=2)

        unc_lbl = tk.Label(row_frame, text="1.00", width=6)
        unc_lbl.grid(row=0, column=5, padx=2)

        rew_lbl = tk.Label(row_frame, text="0.000", width=10)
        rew_lbl.grid(row=0, column=6, padx=2)

        status_lbl = tk.Label(row_frame, text="nominal", width=14)
        status_lbl.grid(row=0, column=7, padx=2)

        btn_unrel = tk.Button(
            row_frame,
            text="Unreliable",
            width=8,
            command=lambda a=aid: self._set_nudge(a, -0.2),
        )
        btn_unrel.grid(row=0, column=8, padx=2)

        btn_mon = tk.Button(
            row_frame,
            text="Monitor",
            width=8,
            command=lambda a=aid: self._set_nudge(a, 0.0),
        )
        btn_mon.grid(row=0, column=9, padx=2)

        btn_rel = tk.Button(
            row_frame,
            text="Reliable",
            width=8,
            command=lambda a=aid: self._set_nudge(a, 0.2),
        )
        btn_rel.grid(row=0, column=10, padx=2)

        self.agent_widgets[aid] = {
            "frame": row_frame,
            "idx": idx_lbl,
            "name": name_lbl,
            "auto": auto_lbl,
            "human": human_lbl,
            "comb": comb_lbl,
            "unc": unc_lbl,
            "rew": rew_lbl,
            "status": status_lbl,
            "btn_unrel": btn_unrel,
            "btn_mon": btn_mon,
            "btn_rel": btn_rel,
        }

    def update(self, agent_states):
        if not self.alive:
            return

        for st in agent_states:
            self._ensure_agent_row(st["id"])

        if agent_states:
            comb_vals = [float(s["trust_combined"]) for s in agent_states]
            self.last_global_trust = float(sum(comb_vals) / len(comb_vals))
        else:
            self.last_global_trust = 0.5

        for st in agent_states:
            aid = st["id"]
            w = self.agent_widgets[aid]

            ta = float(st["trust_auto"])
            th = float(st["trust_human"])
            tc = float(st["trust_combined"])
            U = float(st["uncertainty"])
            r = float(st["last_reward"])
            divergent = bool(st["divergent"])

            w["auto"].config(text=f"{ta:.2f}")
            w["human"].config(text=f"{th:.2f}")
            w["comb"].config(text=f"{tc:.2f}")
            w["unc"].config(text=f"{U:.2f}")
            w["rew"].config(text=f"{r:.3f}")

            if divergent:
                status_text = "DIVERGENT"
                frame_bg = "#552222"
                label_fg = "white"
            else:
                status_text = "nominal"
                frame_bg = "#222222"
                label_fg = "white"

            w["status"].config(text=status_text)
            w["frame"].config(bg=frame_bg)
            for key in ["idx", "name", "auto", "human", "comb", "unc", "rew", "status"]:
                w[key].config(bg=frame_bg, fg=label_fg)

        self._update_global_label()

        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError as e:
            print(f"[WARN] Tk UI error ({e}). Disabling further UI updates.")
            self.alive = False

    def pop_nudges(self):
        nudges = dict(self.pending_nudges)
        self.pending_nudges.clear()
        return nudges


class DummyHumanTrustInterface:
    """Fallback when Tk init fails: no-op interface."""
    def __init__(self):
        self.last_global_trust = 0.5

    def update(self, agent_states):
        return

    def pop_nudges(self):
        return {}

# ============================================================
# 7) IMM + Trust + Human wrapper with metric logging
# ============================================================
class IMMTrustHumanWrapper(MultiAgentEnv):
    """
    Implements:
      - Individual Mental Model per agent: mu, phi, U.
      - Automatic trust from performance + agreement + calibration.
      - Human trust via Tk-based UI nudges.
      - Combined trust per agent (auto+human).
      - Global human trust.
      - Reward shaping + per-episode metrics CSV.
    """

    def __init__(self,
                 env,
                 metrics_csv_path="cghmt_metrics.csv",
                 num_actions=9,
                 mu_dim=3,
                 mu_ema=0.2,
                 trust_lr=0.1,
                 lambda_P=0.4,
                 lambda_A=0.3,
                 lambda_C=0.3,
                 lambda_unc=0.2,
                 trust_reward_weight=0.1,
                 consistency_weight=0.01,
                 prog_bonus=2.0,
                 coll_penalty=2.0,
                 infr_penalty=1.0,
                 fixed_delta_seconds=0.05):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.num_actions = num_actions
        self.mu_dim = mu_dim
        self.mu_ema = mu_ema

        self.trust_lr = trust_lr
        self.lambda_P = lambda_P
        self.lambda_A = lambda_A
        self.lambda_C = lambda_C
        self.lambda_unc = lambda_unc

        self.trust_reward_weight = trust_reward_weight
        self.consistency_weight = consistency_weight

        self.prog_bonus = prog_bonus
        self.coll_penalty = coll_penalty
        self.infr_penalty = infr_penalty

        self.fixed_delta_seconds = fixed_delta_seconds

        self.metrics_csv_path = metrics_csv_path
        self._init_metrics_csv()

        self.imm_state = {}
        self.global_human_trust = 0.5

        self.episode_index = -1
        self._reset_episode_counters()

        self.cumulative_shaped_sum = 0.0
        self.cumulative_episode_count = 0

    # ---------- metrics CSV ----------
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
        self.episode_total_shaped_reward = 0.0
        self.episode_distance_total = 0.0
        self.episode_collisions_total = 0.0
        self.episode_infractions_total = 0.0
        self.episode_prompts = 0

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

        if time_sec > 0:
            prompts_per_minute = self.episode_prompts / (time_sec / 60.0)
        else:
            prompts_per_minute = 0.0

        rc_vals = []
        for aid, st in self.imm_state.items():
            init_d = st.get("init_distance", None)
            last_d = st.get("distance_last", None)
            if init_d is not None and init_d > 1e-3 and last_d is not None:
                rc = (init_d - last_d) / init_d * 100.0
                rc = float(np.clip(rc, 0.0, 100.0))
                rc_vals.append(rc)
        if rc_vals:
            route_completion_pct = float(sum(rc_vals) / len(rc_vals))
        else:
            route_completion_pct = 0.0

        if num_agents > 0:
            avg_shaped_reward = self.episode_total_shaped_reward / (
                self.episode_steps * num_agents
            )
        else:
            avg_shaped_reward = 0.0

        self.cumulative_shaped_sum += avg_shaped_reward
        self.cumulative_episode_count += 1
        cumulative_avg_shaped_reward = (
            self.cumulative_shaped_sum / self.cumulative_episode_count
            if self.cumulative_episode_count > 0
            else 0.0
        )

        with open(self.metrics_csv_path, "a") as f:
            f.write(
                f"{self.episode_index},{self.episode_steps},{time_sec:.3f},"
                f"{collisions_ep:.3f},{route_completion_pct:.6f},"
                f"{infractions_per_100m:.6f},{prompts_per_minute:.6f},"
                f"{avg_shaped_reward:.6f},{cumulative_avg_shaped_reward:.6f}\n"
            )

        print(
            f"[METRICS] ep={self.episode_index} steps={self.episode_steps} "
            f"coll={collisions_ep:.1f} rc={route_completion_pct:.2f}% "
            f"infrac/100m={infractions_per_100m:.3f} ppm={prompts_per_minute:.2f} "
            f"avgR={avg_shaped_reward:.4f} cumAvgR={cumulative_avg_shaped_reward:.4f}"
        )

    # ---------- IMM helpers ----------
    @staticmethod
    def _extract_xy_speed(info_for_agent):
        if info_for_agent is None:
            return 0.0, 0.0, 0.0

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

        speed = info_for_agent.get("speed", 0.0)
        if isinstance(speed, dict):
            speed = speed.get("value", 0.0)
        speed = float(speed)

        return x, y, speed

    @staticmethod
    def _js_divergence(p, q, eps=1e-8):
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        p /= p.sum()
        q /= q.sum()
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * (np.log(p) - np.log(m)))
        kl_qm = np.sum(q * (np.log(q) - np.log(m)))
        return 0.5 * (kl_pm + kl_qm)

    def _init_imm_state(self, obs_dict):
        self.imm_state = {}
        for aid in obs_dict.keys():
            self.imm_state[aid] = {
                "mu": np.zeros(self.mu_dim, dtype=np.float32),
                "phi": np.ones(self.num_actions, dtype=np.float32) / float(self.num_actions),
                "alpha0": float(self.num_actions),
                "U": 1.0,
                "action_hist": np.zeros(self.num_actions, dtype=np.float32),
                "trust_auto": 0.5,
                "trust_human": 0.5,
                "trust_combined": 0.5,
                "last_reward": 0.0,
                # metrics
                "init_distance": None,
                "distance_last": None,
                "prev_distance_last": None,
                "distance_travelled": 0.0,
                "prev_x": None,
                "prev_y": None,
                "prev_collision_sum": 0.0,
                "collisions": 0.0,
                "prev_infractions_raw": 0.0,
                "infractions": 0.0,
                "divergent_prev": False,
            }
        self.global_human_trust = 0.5

    def _update_imm_for_agent(self, aid, action, info):
        st = self.imm_state.setdefault(aid, {
            "mu": np.zeros(self.mu_dim, dtype=np.float32),
            "phi": np.ones(self.num_actions, dtype=np.float32) / float(self.num_actions),
            "alpha0": float(self.num_actions),
            "U": 1.0,
            "action_hist": np.zeros(self.num_actions, dtype=np.float32),
            "trust_auto": 0.5,
            "trust_human": 0.5,
            "trust_combined": 0.5,
            "last_reward": 0.0,
            "init_distance": None,
            "distance_last": None,
            "prev_distance_last": None,
            "distance_travelled": 0.0,
            "prev_x": None,
            "prev_y": None,
            "prev_collision_sum": 0.0,
            "collisions": 0.0,
            "prev_infractions_raw": 0.0,
            "infractions": 0.0,
            "divergent_prev": False,
        })

        x, y, speed = self._extract_xy_speed(info)

        if st["prev_x"] is not None:
            dx = x - st["prev_x"]
            dy = y - st["prev_y"]
            step_dist = math.sqrt(dx * dx + dy * dy)
            step_dist = max(step_dist, 0.0)
            st["distance_travelled"] += step_dist
            self.episode_distance_total += step_dist
        st["prev_x"], st["prev_y"] = x, y

        new_feat = np.array([x, y, speed], dtype=np.float32)
        st["mu"] = (1.0 - self.mu_ema) * st["mu"] + self.mu_ema * new_feat

        if action is not None and isinstance(action, int) and 0 <= action < self.num_actions:
            st["action_hist"] *= 0.95
            st["action_hist"][action] += 1.0

        e = np.maximum(st["action_hist"], 0.0)
        alpha = e + 1.0
        alpha0 = float(alpha.sum()) + 1e-6
        phi = alpha / alpha0

        st["phi"] = phi.astype(np.float32)
        st["alpha0"] = alpha0
        K = float(self.num_actions)
        st["U"] = float(np.clip(K / alpha0, 0.0, 1.0))

        # distance to goal
        d_goal = None
        if info is not None:
            if "distance_to_goal" in info:
                d_goal = float(info["distance_to_goal"])
            elif "distance_to_goal_euclidean" in info:
                d_goal = float(info["distance_to_goal_euclidean"])

        if d_goal is not None:
            if st["init_distance"] is None and d_goal > 0.0:
                st["init_distance"] = d_goal
            st["prev_distance_last"] = st["distance_last"]
            st["distance_last"] = d_goal

        # collisions
        coll_sum = 0.0
        if info is not None:
            coll_sum += float(info.get("collision_vehicles", 0.0))
            coll_sum += float(info.get("collision_pedestrians", 0.0))
            coll_sum += float(info.get("collision_other", 0.0))
        delta_coll = max(coll_sum - st["prev_collision_sum"], 0.0)
        st["prev_collision_sum"] = coll_sum
        st["collisions"] += delta_coll
        self.episode_collisions_total += delta_coll
        st["delta_collisions"] = delta_coll

        # infractions (offroad + otherlane)
        infr_raw = 0.0
        if info is not None:
            infr_raw += float(info.get("intersection_offroad", 0.0))
            infr_raw += float(info.get("intersection_otherlane", 0.0))
        delta_inf = max(infr_raw - st["prev_infractions_raw"], 0.0)
        st["prev_infractions_raw"] = infr_raw
        st["infractions"] += delta_inf
        self.episode_infractions_total += delta_inf
        st["delta_infractions"] = delta_inf

    def _update_auto_trust(self, reward_dict, mu_bar, phi_bar):
        for aid, st in self.imm_state.items():
            r = float(reward_dict.get(aid, 0.0))
            P = 0.5 * (1.0 + np.tanh(r))

            mu = st["mu"]
            phi = st["phi"]
            mu_gap = np.linalg.norm(mu - mu_bar)
            js = self._js_divergence(phi, phi_bar)
            A = np.exp(-mu_gap) * (1.0 / (1.0 + js))

            U = float(st["U"])
            C = 1.0 - U

            hat_w = (
                self.lambda_P * P +
                self.lambda_A * A +
                self.lambda_C * C
            )
            hat_w = hat_w - self.lambda_unc * U

            w_old = float(st["trust_auto"])
            w_new = (1.0 - self.trust_lr) * w_old + self.trust_lr * hat_w
            st["trust_auto"] = float(np.clip(w_new, 0.0, 1.0))

    # ---------- MultiAgentEnv API ----------
    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        self.episode_index += 1
        self._reset_episode_counters()
        self._init_imm_state(obs)
        return obs

    def step(self, action_dict):
        self.episode_steps += 1

        obs, base_rewards, done_dict, info_dict = self.env.step(action_dict)
        agent_ids = [aid for aid in obs.keys() if aid != "__all__"]
        if not agent_ids:
            return obs, base_rewards, done_dict, info_dict

        # 1) IMM update per agent
        for aid in agent_ids:
            act = action_dict.get(aid, None)
            info = info_dict.get(aid, {})
            self._update_imm_for_agent(aid, act, info)

        # 2) Global shared belief/intent
        mus = np.stack([self.imm_state[aid]["mu"] for aid in agent_ids], axis=0)
        phi_stack = np.stack([self.imm_state[aid]["phi"] for aid in agent_ids], axis=0)
        mu_bar = mus.mean(axis=0)
        phi_bar = phi_stack.mean(axis=0)
        phi_bar /= phi_bar.sum() + 1e-8

        # 3) Automatic trust update based on base rewards
        self._update_auto_trust(base_rewards, mu_bar, phi_bar)

        # 4) Human nudges via Tk UI
        ui = HumanTrustInterface.get()
        agent_states_for_ui = []
        for aid in agent_ids:
            st = self.imm_state[aid]
            base_r = float(base_rewards.get(aid, 0.0))
            st["last_reward"] = base_r

            mu_gap = float(np.linalg.norm(st["mu"] - mu_bar))
            divergent = (mu_gap > 5.0) or (st["U"] > 0.6)

            if divergent and not st["divergent_prev"]:
                self.episode_prompts += 1
            st["divergent_prev"] = divergent

            agent_states_for_ui.append({
                "id": aid,
                "trust_auto": st["trust_auto"],
                "trust_human": st["trust_human"],
                "trust_combined": st["trust_combined"],
                "uncertainty": st["U"],
                "last_reward": st["last_reward"],
                "divergent": divergent,
            })

        ui.update(agent_states_for_ui)
        human_nudges = ui.pop_nudges()

        for aid, delta in human_nudges.items():
            if aid in self.imm_state:
                st = self.imm_state[aid]
                new_h = float(np.clip(st["trust_human"] + delta, 0.0, 1.0))
                st["trust_human"] = new_h

        # 5) Combine auto + human trust and compute global human trust
        combined_vals = []
        for aid in agent_ids:
            st = self.imm_state[aid]
            combined = 0.5 * (st["trust_auto"] + st["trust_human"])
            st["trust_combined"] = float(np.clip(combined, 0.0, 1.0))
            combined_vals.append(st["trust_combined"])
        if combined_vals:
            self.global_human_trust = float(sum(combined_vals) / len(combined_vals))
        else:
            self.global_human_trust = 0.5

        # 6) Reward shaping: step-wise progress + collisions + infractions
        new_rewards = {}
        for aid in agent_ids:
            st = self.imm_state[aid]
            base_r = float(base_rewards.get(aid, 0.0))

            # step-wise progress toward goal
            delta_prog = 0.0
            init_d = st.get("init_distance", None)
            last_d = st.get("distance_last", None)
            prev_d = st.get("prev_distance_last", None)

            if init_d is not None and last_d is not None and prev_d is not None:
                raw_step_prog = prev_d - last_d  # >0 if moved closer
                raw_step_prog = max(raw_step_prog, 0.0)
                delta_prog = raw_step_prog / max(init_d, 1e-6)
            else:
                delta_prog = 0.0

            # collisions / infractions
            delta_coll = float(st.get("delta_collisions", 0.0))
            delta_inf = float(st.get("delta_infractions", 0.0))

            shaped = (
                base_r +
                self.prog_bonus * delta_prog -
                self.coll_penalty * delta_coll -
                self.infr_penalty * delta_inf
            )

            new_rewards[aid] = shaped
            self.episode_total_shaped_reward += shaped

            info = info_dict.get(aid, {})
            info["imm_mu"] = st["mu"].tolist()
            info["imm_phi"] = st["phi"].tolist()
            info["imm_U"] = st["U"]
            info["trust_auto"] = st["trust_auto"]
            info["trust_human"] = st["trust_human"]
            info["trust_combined"] = st["trust_combined"]
            info["global_human_trust"] = self.global_human_trust
            info_dict[aid] = info

        # 7) If episode ended, log metrics
        if done_dict.get("__all__", False):
            self._finalize_and_log_episode(num_agents=len(agent_ids))

        return obs, new_rewards, done_dict, info_dict

# ============================================================
# 7.5) Attack wrapper (easy-to-integrate severe stressors)
# ============================================================
class AttackWrapper(MultiAgentEnv):
    """
    Applies simple test-time attacks:
      - 'v2x'      : observation staleness (per-agent blackout with prob p).
      - 'weather'  : heavy blur + brightness drop on camera observations.
      - 'byzantine': flip one agent's actions frequently.
      - 'none'     : pass-through.
    """
    def __init__(self, env, mode="none"):
        self.env = env
        self.mode = mode
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Buffers for attacks
        self.prev_obs = {}
        self.agent_ids_last = []
        self.byz_agents = set()
        # Fixed severe settings
        self.v2x_drop_prob = 0.4         # 40% of the time, use stale frame
        self.byz_flip_prob = 0.5         # 50% steps flip action
        self.rand = random.Random(1337)

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset(seed=seed, options=options)
        # Store initial prev_obs for v2x
        self.prev_obs = {aid: np.copy(o) for aid, o in obs.items() if aid != "__all__"}
        self.agent_ids_last = [aid for aid in obs.keys() if aid != "__all__"]
        self.byz_agents = set()
        return self._maybe_weather(obs)

    def step(self, action_dict):
        # Byzantine: pick one agent, flip its actions often
        if self.mode == "byzantine":
            if not self.byz_agents:
                # deterministically pick first seen agent id for reproducibility
                if self.agent_ids_last:
                    self.byz_agents = {sorted(self.agent_ids_last)[0]}
            # Apply flips
            for aid in list(action_dict.keys()):
                if aid in self.byz_agents and self.rand.random() < self.byz_flip_prob:
                    a = action_dict.get(aid, None)
                    if isinstance(a, int):
                        # pick a different action uniformly
                        new_a = a
                        while new_a == a:
                            new_a = self.rand.randrange(0, 9)
                        action_dict[aid] = new_a

        obs, rewards, dones, infos = self.env.step(action_dict)
        self.agent_ids_last = [aid for aid in obs.keys() if aid != "__all__"]

        # V2X: with probability p, replace current obs with previous obs (per agent)
        if self.mode == "v2x":
            for aid in self.agent_ids_last:
                if aid in self.prev_obs and self.rand.random() < self.v2x_drop_prob:
                    obs[aid] = np.copy(self.prev_obs[aid])
                else:
                    self.prev_obs[aid] = np.copy(obs[aid])

        # Weather: blur + darken every frame
        obs = self._maybe_weather(obs)

        return obs, rewards, dones, infos

    def _maybe_weather(self, obs_dict):
        if self.mode != "weather":
            return obs_dict
        out = {}
        for k, v in obs_dict.items():
            if k == "__all__":
                out[k] = v
                continue
            img = v
            # Expect HxWxC uint8; if not, try to convert safely
            if isinstance(img, np.ndarray) and img.ndim == 3 and img.dtype == np.uint8:
                # heavy blur
                blurred = cv2.GaussianBlur(img, (7, 7), 3)
                # darken and reduce contrast
                dark = cv2.convertScaleAbs(blurred, alpha=0.7, beta=-30)
                out[k] = dark
            else:
                out[k] = v
        return out

# ============================================================
# 8) Env factory for Ray workers
# ============================================================
def make_env_with_wrappers(resolved_env_name, env_config):
    import macad_gym  # noqa

    env = gym.make(resolved_env_name)
    env = SpawnSpacingWrapper(env, offset=1.0)

    num_framestack = env_config.get("env", {}).get("framestack", 4)
    env = wrap_deepmind(env, dim=84, num_framestack=num_framestack)

    env = IMMTrustHumanWrapper(
        env,
        metrics_csv_path=env_config.get("metrics_csv_path", "cghmt_metrics.csv"),
        num_actions=9,
        trust_reward_weight=env_config.get("team_reward_weight", 0.1),
        consistency_weight=0.01,
        prog_bonus=2.0,
        coll_penalty=2.0,
        infr_penalty=1.0,
        fixed_delta_seconds=0.05,
    )

    # Attack wrapper last, so it sees policy actions and tampers or corrupts observations
    attack_mode = env_config.get("attack", "none")
    env = AttackWrapper(env, mode=attack_mode)
    return env

# ============================================================
# 9) main
# ============================================================
if __name__ == "__main__":
    args = parser.parse_args()

    if args.num_workers != 0:
        print("[WARN] For the Tk human interface, num_workers must be 0. Overriding to 0.")
        args.num_workers = 0

    RESOLVED_ENV_NAME = _resolve_macad_env_name(PREFERRED_ENV_NAME, FALLBACK_ENV_NAME)

    env_actor_configs = {"env": {"framestack": 4}}

    # disable Tune global checkpointing to avoid Windows rename race
    os.environ.setdefault("TUNE_GLOBAL_CHECKPOINT_S", "0")

    ray.init(num_gpus=args.num_gpus)

    register_env(
        RESOLVED_ENV_NAME,
        lambda config: make_env_with_wrappers(
            RESOLVED_ENV_NAME,
            {
                **config,
                "team_reward_weight": args.team_reward_weight,
                "desired_gap": args.desired_gap,
                "env": env_actor_configs["env"],
                "metrics_csv_path": args.metrics_csv,
                "attack": args.attack,  # <--- pass attack setting into env
            },
        ),
    )

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
            "env_config": {
                **env_actor_configs,
                "team_reward_weight": args.team_reward_weight,
                "desired_gap": args.desired_gap,
            },
        }
        return (PPOPolicy, obs_space, act_space, cfg)

    policies = {"shared_platoon_policy": gen_shared_policy()}

    def policy_mapping_fn(agent_id, **kwargs):
        return "shared_platoon_policy"

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    local_dir = os.path.join(
        "C:/Users/dawoo/ray_results",
        f"platoon_mass5_imm_trust_human_run_{ts}",
    )

    analysis = tune.run(
        "PPO",
        name="MA-PPO-IMMTrust-Human-CARLA-MASS5",
        stop={"training_iteration": args.num_iters},
        local_dir=local_dir,
        config={
            "env": RESOLVED_ENV_NAME,
            "log_level": "INFO",
            "use_pytorch": True,
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

    # Ray 0.8.6 compatible way to get the best checkpoint
    best_trial = analysis.get_best_trial("episode_reward_mean")
    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=best_trial,
        metric="episode_reward_mean",
    )
    if checkpoints:
        # each item is (checkpoint_path, metric_value)
        best_checkpoint = max(checkpoints, key=lambda x: x[1])[0]
        print("Best checkpoint path:", best_checkpoint)
    else:
        print("No checkpoints found for the best trial.")
