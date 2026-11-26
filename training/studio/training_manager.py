"""Thread-safe RL Training Manager connecting DoomEnv, ESRHA, PPOTrainer, and Dashboard."""

from __future__ import annotations

import copy
import io
import json
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from doom_env import make_env
from doom_env import spaces as S
from training.algorithms.ppo import PPOConfig, PPOTrainer
from training.models.doom_model import ESRHAAgentModel, obs_to_tensors
from training.models.policy import STRATEGIES
from training.rewards.reward import DoomRewardShaper
from training.rollout import RolloutBuffer
from training.studio.feedback_engine import (
    PROBLEM_CATEGORIES,
    FeedbackShapingModifiers,
    apply_modifiers_to_shaper,
    check_candidate_improvement,
    diagnose_feedback,
    evaluate_policy_episodes,
)
from training.studio.map_catalog import ROOT, discover_maps

EXPERIMENTS_DIR = ROOT / "training" / "runs" / "studio_experiments"
_ENEMY_TYPE_IDS = {i for i, t in enumerate(S.ENTITY_TYPES) if t.startswith("enemy")}


@dataclass
class TrainStudioConfig:
    experiment_name: str = "combat_studio_exp"
    map_id: str = "vizdoom_corridor"
    num_episodes: int = 200
    max_ticks: int = 1400
    lr: float = 2.5e-5
    batch_seqs: int = 8
    ppo_epochs: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.001
    checkpoint_interval: int = 10
    seed: int = 42
    device: str = "cpu"
    step_delay_ms: int = 115  # ~8.7 steps/s (1x real-time DOOM speed)
    deterministic_eval: bool = True  # True = clean deterministic playback without corrupting trained weights


def _encode_jpeg(rgb: np.ndarray) -> bytes:
    try:
        from PIL import Image, ImageEnhance

        buf = io.BytesIO()
        img = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8))
        # Upscale 160x120 engine buffer to 640x480 with crisp-smooth resampling for clean Studio playback
        img = img.resize((640, 480), Image.Resampling.BICUBIC)
        img = ImageEnhance.Sharpness(img).enhance(1.35)
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception:
        import struct, zlib

        arr = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w, _ = arr.shape
        raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b"")


class TrainingManager:
    """Manages real ViZDoom + ESRHA + PPO training lifecycle and live telemetry."""

    def __init__(self):
        self._lock = threading.RLock()
        self._frame_cond = threading.Condition()
        self.maps = discover_maps()
        self._map_by_id = {m["id"]: m for m in self.maps}

        self.config = TrainStudioConfig()
        self.status = "Not Started"
        self.env_status = "Closed"
        self.error_msg = ""

        self._env = None
        self._model: ESRHAAgentModel | None = None
        self._train_model: ESRHAAgentModel | None = None
        self._trainer: PPOTrainer | None = None
        self._shaper = DoomRewardShaper()
        self._hidden: torch.Tensor | None = None
        self._prev_action = np.zeros((1, len(S.BUTTONS)), dtype=np.float32)

        # Dual-policy, rolling history, & continuous online feedback learning state
        self._gameplay_history: deque[dict] = deque(maxlen=256)
        self._feedback_modifiers = FeedbackShapingModifiers()
        self._feedback_lock = threading.Lock()
        self._feedback_busy = False
        self._pending_promotion: dict | None = None
        self._prev_stable_snapshot: dict | None = None
        self.policy_version_num = 0
        self.policy_version = "v13.0"
        self.latest_policy_version = "v13.0"
        self.learning_status: dict = {"status": "Idle", "active_report_id": None}
        self.problem_history: list[dict] = []

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._reset_ep_requested = False

        # Live frame
        self._latest_frame: bytes = b""
        self._last_frame_time: float = 0.0

        # Episode & cumulative counters
        self.episode_num = 0
        self.total_steps = 0
        self.update_idx = 0
        self.ep_kills = 0
        self.total_kills = 0
        self.ep_step = 0
        self.best_reward = -1e9
        self.current_checkpoint = "uninitialized"

        # HUD & behavior state
        self.hud = {
            "health": 100.0, "armor": 0.0, "kills": 0, "episode": 0,
            "ammo_bullets": 50, "ammo_shells": 0, "ammo_rockets": 0, "ammo_cells": 0,
            "selected_weapon": 2, "selected_weapon_ammo": 50,
            "survival_tics": 0, "episode_ticks": 1400, "damage_taken": 0.0,
            "cells_visited": 0, "ep_reward": 0.0,
        }
        self.behavior = {
            "visible_enemies": 0, "enemy_detected": False, "strategy": "-",
            "action": [0] * 9, "action_probs": [0.0] * 9,
            "aimed_shots": 0, "wasted_shots": 0, "ammo_used": 0,
            "attack_pct": 0.0, "explore_pct": 0.0,
            "move_count": 0, "turn_count": 0, "strafe_count": 0,
            "low_health_warning": False, "medikit_detected": False,
            "medikit_target": "-", "hp_recovery": "-",
            "weapon_name": "Pistol", "low_ammo_warning": False,
            "reload_supported": False, "weapon_switches": 0,
            "target_info": "-", "target_aligned": False,
            "firing_justified": False, "shots_without_target": 0, "shots_after_kill": 0,
            "behavior_mode": "Exploration", "explore_dir": "Forward",
            "repeated_turn": False, "exit_reached": False,
        }
        self.memory_viz = {
            "slots": [],
            "sectors": [],
            "target": None,
            "alive_count": 0,
            "killed_count": 0,
        }
        self.history = {
            "ep_rewards": [], "avg_rewards": [], "ep_kills": [],
            "ep_survival": [], "ep_cells": [], "policy_losses": [], "value_losses": [],
            "fwd_probs": [], "turn_probs": [], "atk_probs": [],
        }
        self.logs: deque[dict] = deque(maxlen=150)
        self._ep_atk_steps = 0
        self._ep_exp_steps = 0
        self._ep_move = 0
        self._ep_turn = 0
        self._ep_strafe = 0
        self._prev_hp = 100.0
        self._hp_before_med = None
        self._hp_after_med = None
        self._ep_wpn_switches = 0
        self._ep_shots_no_tgt = 0
        self._ep_shots_after_kill = 0
        self._steps_since_kill = 999
        self._prev_items = 0
        self._prev_enemy_seen = False

        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self._log("info", "DOOM AI Training Studio initialized.")
        default_weights = ROOT / "submission" / "weights" / "esrha_best.pt"
        if default_weights.is_file():
            self.load_checkpoint(str(default_weights), resume_optimizer=False)

    def _log(self, level: str, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.logs.append({"time": ts, "level": level, "msg": msg})

    def _exp_dir(self, name: str | None = None) -> Path:
        return EXPERIMENTS_DIR / (name or self.config.experiment_name)

    def _init_model_if_needed(self, force_fresh: bool = False) -> None:
        if self._model is None or force_fresh:
            self._model = ESRHAAgentModel(hidden_dim=256, use_visual=False, use_depth=False).to(self.config.device)
            self._train_model = ESRHAAgentModel(hidden_dim=256, use_visual=False, use_depth=False).to(self.config.device)
            self._train_model.load_state_dict(copy.deepcopy(self._model.state_dict()))
            cfg = PPOConfig(
                lr=self.config.lr, gamma=self.config.gamma, gae_lambda=self.config.gae_lambda,
                ent_coef=self.config.ent_coef, ppo_epochs=self.config.ppo_epochs,
                batch_seqs=self.config.batch_seqs, seq_len=32, target_kl=0.015,
            )
            self._trainer = PPOTrainer(self._train_model, cfg, total_updates=max(10, self.config.num_episodes), device=self.config.device)
            self._hidden = self._model.init_hidden(1, self.config.device)
            self._prev_action = np.zeros((1, len(S.BUTTONS)), dtype=np.float32)
            self.current_checkpoint = "fresh_esrha_256"

    def _open_env(self, map_id: str) -> dict:
        spec = self._map_by_id.get(map_id)
        if not spec or not spec["available"]:
            raise FileNotFoundError(f"Map '{map_id}' not found on disk.")
        if self._env is not None:
            self._env.close()
            self._env = None
        ticks = int(self.config.max_ticks or spec["episode_ticks"])
        self._env = make_env(spec["path"], "MAP01", episode_ticks=ticks, render_buffers=True)
        obs, _ = self._env.reset(seed=self.config.seed + self.episode_num)
        self._shaper.reset(self._env.privileged_state(), obs=obs)
        self.env_status = f"Loaded ({spec['filename']})"
        self._log("info", f"Environment initialized & map loaded: {spec['name']} ({spec['filename']})")
        return obs

    def _reset_episode_state(self, obs: dict) -> None:
        self.ep_kills = 0
        self.ep_step = 0
        self._ep_atk_steps = 0
        self._ep_exp_steps = 0
        self._ep_move = 0
        self._ep_turn = 0
        self._ep_strafe = 0
        self._prev_hp = float(obs["vitals"][0])
        self._hp_before_med = None
        self._hp_after_med = None
        self._ep_wpn_switches = 0
        self._ep_shots_no_tgt = 0
        self._ep_shots_after_kill = 0
        self._steps_since_kill = 999
        self._prev_items = 0
        self._prev_enemy_seen = False
        if self._model is not None:
            self._hidden = self._model.init_hidden(1, self.config.device)
        self._prev_action[:] = 0.0
        self._update_hud_and_frame(obs, np.zeros(9, np.int8), np.zeros(9, np.float32), "-", force_frame=True)

    def _update_hud_and_frame(
        self, obs: dict, action: np.ndarray, probs: np.ndarray, strategy: str, force_frame: bool = False
    ) -> None:
        priv = self._env.privileged_state() if self._env is not None else {}
        v = obs["vitals"]
        ve = obs["visible_entities"]
        mask = ve["mask"].astype(bool)
        vis_enemies = sum(1 for i in np.flatnonzero(mask) if int(ve["entity_type"][i]) in _ENEMY_TYPE_IDS)
        vis_meds = [i for i in np.flatnonzero(mask) if int(ve["entity_type"][i]) == 5]
        enemy_idxs = [i for i in np.flatnonzero(mask) if int(ve["entity_type"][i]) in _ENEMY_TYPE_IDS]
        wpn_names = {1: "Fist/Melee", 2: "Pistol", 3: "Shotgun", 4: "Chaingun", 5: "Rocket", 6: "Plasma", 7: "BFG"}
        cur_wpn = int(v[2])
        cur_ammo = float(v[3])
        med_str = "-"
        if vis_meds:
            bm = min(vis_meds, key=lambda i: float(ve["features"][i, 0]))
            md, ms = float(ve["features"][bm, 0]) * 1024.0, math.degrees(math.asin(max(-1.0, min(1.0, float(ve["features"][bm, 1])))))
            med_str = f"d={md:.0f}u ang={ms:+.0f}°"
        tgt_str, tgt_aligned = "-", False
        if enemy_idxs:
            be = max(enemy_idxs, key=lambda i: float(ve["features"][i, 2]))
            td = float(ve["features"][be, 0]) * 1024.0
            ts = math.degrees(math.asin(max(-1.0, min(1.0, float(ve["features"][be, 1])))))
            tc = float(ve["features"][be, 2])
            tgt_aligned = tc >= 0.982
            tgt_str = f"d={td:.0f}u ang={ts:+.0f}° cos={tc:.3f}"
        hp_rec_str = f"{self._hp_before_med:.0f}% -> {self._hp_after_med:.0f}%" if self._hp_before_med is not None else "-"

        self.hud.update({
            "health": float(v[0]), "armor": float(v[1]),
            "selected_weapon": int(v[2]), "selected_weapon_ammo": float(v[3]),
            "ammo_bullets": float(v[4]), "ammo_shells": float(v[5]),
            "ammo_rockets": float(v[6]), "ammo_cells": float(v[7]),
            "kills": self.ep_kills, "episode": self.episode_num,
            "survival_tics": int(priv.get("tic", 0)),
            "episode_ticks": int(priv.get("episode_ticks", self.config.max_ticks)),
            "damage_taken": float(priv.get("damage_taken", 0.0)),
            "cells_visited": int(self._shaper._stats.cells_visited),
            "ep_reward": round(float(self._shaper._stats.reward), 3),
        })
        steps_safe = max(1, self.ep_step)
        self.behavior.update({
            "visible_enemies": vis_enemies,
            "enemy_detected": vis_enemies > 0,
            "strategy": strategy,
            "action": [int(x) for x in action],
            "action_probs": [round(float(p), 3) for p in probs],
            "aimed_shots": int(self._shaper._stats.aimed_shots),
            "wasted_shots": int(self._shaper._stats.wasted_shots),
            "ammo_used": int(self._shaper._stats.ammo_used),
            "attack_pct": round(100.0 * self._ep_atk_steps / steps_safe, 1),
            "explore_pct": round(100.0 * self._ep_exp_steps / steps_safe, 1),
            "move_count": self._ep_move,
            "turn_count": self._ep_turn,
            "strafe_count": self._ep_strafe,
            "distance_travelled": round(float(self._shaper._stats.distance_travelled), 1),
            "low_health_warning": float(v[0]) <= 45.0,
            "medikit_detected": len(vis_meds) > 0,
            "medikit_target": med_str,
            "hp_recovery": hp_rec_str,
            "weapon_name": wpn_names.get(cur_wpn, f"Slot {cur_wpn}"),
            "low_ammo_warning": (cur_wpn <= 1) or (0.0 <= cur_ammo <= 5.0),
            "reload_supported": False,
            "weapon_switches": self._ep_wpn_switches,
            "target_info": tgt_str,
            "target_aligned": tgt_aligned,
            "firing_justified": bool(enemy_idxs and tgt_aligned and not ((cur_wpn <= 1) or (0.0 <= cur_ammo <= 0.5))),
            "shots_without_target": self._ep_shots_no_tgt,
            "shots_after_kill": self._ep_shots_after_kill,
            "behavior_mode": (
                "Recovery" if float(v[0]) <= 45.0 and len(vis_meds) > 0
                else ("Combat" if vis_enemies > 0
                      else ("Re-engagement" if self.memory_viz.get("alive_count", 0) > 0 else "Exploration"))
            ),
            "explore_dir": (
                "Forward Clear" if float(obs["rays"]["distance"][0]) > 0.22
                else ("Turn Left (Open)" if float(np.mean(obs["rays"]["distance"][1:9])) >= float(np.mean(obs["rays"]["distance"][-8:])) else "Turn Right (Open)")
            ),
            "repeated_turn": bool(self._shaper._no_enemy_turn_steps > 2),
            "exit_reached": bool(priv.get("exit_reached", False)),
        })
        if self.ep_step % 4 == 0:
            self.history["fwd_probs"].append(round(float(probs[0]), 3))
            self.history["turn_probs"].append(round(float(min(1.0, probs[4] + probs[5])), 3))
            self.history["atk_probs"].append(round(float(probs[6]), 3))
            for k in ("fwd_probs", "turn_probs", "atk_probs"):
                if len(self.history[k]) > 120:
                    self.history[k] = self.history[k][-120:]

        now = time.perf_counter()
        if force_frame or (now - self._last_frame_time) >= 0.03:  # cap encoding at ~33 Hz max
            self._last_frame_time = now
            jpg = _encode_jpeg(obs["rgb"])
            with self._frame_cond:
                self._latest_frame = jpg
                self._frame_cond.notify_all()

    def _update_memory_viz(self, em_state: torch.Tensor | None, mem_summary: torch.Tensor | None) -> None:
        if em_state is None or mem_summary is None:
            return
        from training.models.enemy_memory import MAX_MEM_SLOTS, NUM_SECTORS, SLOT_DIM
        s_np = em_state.squeeze(0).detach().cpu().numpy()
        sum_np = mem_summary.squeeze(0).detach().cpu().numpy()
        slots_arr = s_np[: MAX_MEM_SLOTS * SLOT_DIM].reshape(MAX_MEM_SLOTS, SLOT_DIM)
        slots_out = []
        alive_c, killed_c = 0, 0
        for row in slots_arr:
            st = int(round(float(row[2])))
            if st == 0:
                continue
            if st == 1:
                alive_c += 1
            elif st == 2:
                killed_c += 1
            slots_out.append({
                "x": round(float(row[0]), 3),
                "y": round(float(row[1]), 3),
                "status": "alive" if st == 1 else "killed",
                "conf": round(float(row[3]), 2),
                "etype": int(round(float(row[4]))),
                "age": int(round(float(row[5]))),
                "visible": float(row[5]) < 0.5 and st == 1,
            })
        sectors_out = []
        for k in range(NUM_SECTORS):
            sectors_out.append({
                "idx": k,
                "angle_deg": k * (360 // NUM_SECTORS),
                "living": round(float(sum_np[12 + k]), 2),
                "cleared": round(float(sum_np[12 + NUM_SECTORS + k]), 2),
            })
        target = None
        if float(sum_np[5]) > 0.5:
            d, sn, cs = float(sum_np[0]), float(sum_np[1]), float(sum_np[2])
            target = {"x": round(d * cs, 3), "y": round(d * sn, 3), "dist": round(d, 3), "sin": round(sn, 2)}
        self.memory_viz = {
            "slots": slots_out,
            "sectors": sectors_out,
            "target": target,
            "alive_count": alive_c,
            "killed_count": killed_c,
        }

    # ---------------- Controls ----------------
    def start(self, cfg_updates: dict | None = None) -> dict:
        with self._lock:
            if self.status in ("Running", "Initializing"):
                return {"ok": False, "error": "Training is already running."}
            if cfg_updates:
                for k, v in cfg_updates.items():
                    if hasattr(self.config, k):
                        typ = type(getattr(self.config, k))
                        setattr(self.config, k, typ(v))
            self.status = "Initializing"
            self.error_msg = ""
            self._stop_event.clear()
            self._pause_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            return {"ok": True}

    def pause(self) -> dict:
        with self._lock:
            if self.status != "Running":
                return {"ok": False, "error": f"Cannot pause from status '{self.status}'."}
            self._pause_event.set()
            self.status = "Paused"
            self._log("warn", "Training paused.")
            return {"ok": True}

    def resume(self) -> dict:
        with self._lock:
            if self.status != "Paused":
                return {"ok": False, "error": f"Cannot resume from status '{self.status}'."}
            self._pause_event.clear()
            self.status = "Running"
            self._log("info", "Training resumed.")
            return {"ok": True}

    def reset_episode(self) -> dict:
        with self._lock:
            if self._env is None:
                return {"ok": False, "error": "No active environment to reset."}
            self._reset_ep_requested = True
            self._log("info", "Episode reset requested (weights & optimizer preserved).")
            return {"ok": True}

    def stop(self) -> dict:
        with self._lock:
            if self.status not in ("Running", "Paused", "Initializing"):
                return {"ok": False, "error": "Training is not active."}
            self._stop_event.set()
            self._pause_event.clear()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return {"ok": True}

    def select_map(self, map_id: str) -> dict:
        with self._lock:
            if self.status == "Running":
                return {"ok": False, "error": "Pause or stop training before switching maps."}
            if map_id not in self._map_by_id:
                return {"ok": False, "error": f"Unknown map '{map_id}'."}
            self.config.map_id = map_id
            spec = self._map_by_id[map_id]
            self.config.max_ticks = spec["episode_ticks"]
            if self._env is not None:
                try:
                    obs = self._open_env(map_id)
                    self._reset_episode_state(obs)
                except Exception as e:
                    self.error_msg = str(e)
                    self._log("error", f"Map switch failed: {e}")
                    return {"ok": False, "error": str(e)}
            else:
                self._log("info", f"Selected map: {spec['name']} ({spec['filename']})")
            return {"ok": True}

    def restart(self, experiment_name: str | None = None) -> dict:
        self.stop()
        with self._lock:
            if experiment_name:
                self.config.experiment_name = experiment_name.strip() or f"exp_{int(time.time())}"
            self._model = None
            self._trainer = None
            self.episode_num = 0
            self.total_steps = 0
            self.update_idx = 0
            self.ep_kills = 0
            self.total_kills = 0
            self.best_reward = -1e9
            for k in self.history:
                self.history[k].clear()
            self.status = "Not Started"
            self._log("info", f"Fresh experiment prepared: {self.config.experiment_name}")
            return {"ok": True}

    def save_checkpoint(self, tag: str | None = None) -> dict:
        with self._lock:
            if self._model is None or self._trainer is None:
                return {"ok": False, "error": "No initialized model to save."}
            exp_dir = self._exp_dir()
            exp_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{tag}.pt" if tag else f"ckpt_ep{self.episode_num}_step{self.total_steps}.pt"
            path = exp_dir / fname
            self._trainer.save_checkpoint(
                path, self.update_idx, self.total_steps,
                extra={
                    "episode_num": self.episode_num, "total_kills": self.total_kills,
                    "best_reward": self.best_reward, "config": asdict(self.config),
                    "history": self.history,
                },
            )
            self.current_checkpoint = fname
            self._log("info", f"Saved checkpoint: {fname}")
            return {"ok": True, "path": str(path)}

    def load_checkpoint(self, path_str: str, resume_optimizer: bool = True) -> dict:
        with self._lock:
            if self.status == "Running":
                return {"ok": False, "error": "Pause or stop training before loading a checkpoint."}
            p = Path(path_str)
            if not p.is_file():
                return {"ok": False, "error": f"Checkpoint not found: {path_str}"}
            ckpt = torch.load(p, map_location=self.config.device, weights_only=False)
            mcfg = ckpt.get("model_config", {})
            self._model = ESRHAAgentModel(
                hidden_dim=int(mcfg.get("hidden_dim", 256)),
                use_visual=bool(mcfg.get("use_visual", False)),
                use_depth=bool(mcfg.get("use_depth", False)),
            ).to(self.config.device)
            self._train_model = ESRHAAgentModel(
                hidden_dim=int(mcfg.get("hidden_dim", 256)),
                use_visual=bool(mcfg.get("use_visual", False)),
                use_depth=bool(mcfg.get("use_depth", False)),
            ).to(self.config.device)
            cfg = PPOConfig(lr=self.config.lr, gamma=self.config.gamma, gae_lambda=self.config.gae_lambda, ent_coef=self.config.ent_coef, ppo_epochs=self.config.ppo_epochs, target_kl=0.015)
            self._trainer = PPOTrainer(self._train_model, cfg, total_updates=max(10, self.config.num_episodes), device=self.config.device)
            self._model.load_state_dict(ckpt["model_state_dict"])
            self._train_model.load_state_dict(copy.deepcopy(ckpt["model_state_dict"]))
            if resume_optimizer and "optimizer_state_dict" in ckpt:
                self._trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    self._trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            extra = ckpt.get("extra", {})
            self.update_idx = int(ckpt.get("update_idx", 0))
            self.total_steps = int(ckpt.get("total_steps", 0))
            self.episode_num = int(extra.get("episode_num", 0))
            self.total_kills = int(extra.get("total_kills", 0))
            self.best_reward = float(extra.get("best_reward", -1e9))
            if "history" in extra:
                for k, v in extra["history"].items():
                    if k in self.history:
                        self.history[k] = list(v)
            self._hidden = self._model.init_hidden(1, self.config.device)
            self.current_checkpoint = p.name
            self._log("info", f"Loaded checkpoint {p.name} (resume_opt={resume_optimizer}, steps={self.total_steps})")
            return {"ok": True}

    def list_checkpoints(self) -> list[dict]:
        out = []
        for base in (self._exp_dir(), ROOT / "submission" / "weights", ROOT / "training" / "runs" / "combat_v1"):
            if not base.is_dir():
                continue
            for p in sorted(base.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
                st = p.stat()
                is_resumable = "submission" not in p.parts
                out.append({
                    "name": p.name, "path": str(p),
                    "kind": "Resumable" if is_resumable else "Inference",
                    "size_mb": round(st.st_size / 1024**2, 2),
                    "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                })
        return out

    def list_experiments(self) -> list[str]:
        if not EXPERIMENTS_DIR.is_dir():
            return [self.config.experiment_name]
        names = sorted([d.name for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()])
        if self.config.experiment_name not in names:
            names.append(self.config.experiment_name)
        return names

    # ---------------- Training Loop ----------------
    def _run_loop(self) -> None:
        try:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
            self._init_model_if_needed(force_fresh=False)
            obs = self._open_env(self.config.map_id)
            self.episode_num += 1
            self._reset_episode_state(obs)
            self.status = "Running"
            self._log("info", f"Training started (exp={self.config.experiment_name}, ep={self.episode_num}).")

            rollout_steps = 128
            buf = RolloutBuffer(rollout_steps, 1, self._model.state_dim)
            target_episodes = self.episode_num + self.config.num_episodes - 1

            while not self._stop_event.is_set() and self.episode_num <= target_episodes:
                if self._reset_ep_requested:
                    with self._lock:
                        prev_st = self.status
                        self._reset_ep_requested = False
                        self.status = "Resetting"
                        obs, _ = self._env.reset(seed=self.config.seed + self.episode_num)
                        self._shaper.reset(self._env.privileged_state(), obs=obs)
                        self._reset_episode_state(obs)
                        self.status = prev_st if prev_st == "Paused" else "Running"
                        self._log("info", f"Episode {self.episode_num} reset; weights & total kills ({self.total_kills}) preserved.")

                if self._pause_event.is_set():
                    time.sleep(0.05)
                    continue

                self._model.eval()
                obs_t = obs_to_tensors(obs, device=self.config.device)
                pa_t = torch.as_tensor(self._prev_action, dtype=torch.float32, device=self.config.device)
                with torch.no_grad():
                    out = self._model.forward_step(obs_t, hidden=self._hidden, prev_action=pa_t)
                    if self.config.deterministic_eval:
                        action_t = (out["tactical_dist"].base_dist.probs > 0.5).to(torch.float32)
                    else:
                        action_t = out["tactical_dist"].sample()
                    log_prob_t = out["tactical_dist"].log_prob(action_t)
                    value_t = out["value"]
                    probs_np = out["tactical_dist"].base_dist.probs.squeeze(0).cpu().numpy()
                    strat = STRATEGIES[int(out["strategy_probs"].argmax(dim=-1).item())]
                    self._update_memory_viz(out.get("enemy_mem_state"), out.get("mem_summary"))

                action_np = action_t.cpu().numpy().astype(np.float32)
                action_int = action_np.squeeze(0).astype(np.int8)
                hid_in = self._hidden.squeeze(0).cpu().numpy()

                next_obs, _, term, trunc, _ = self._env.step(action_int)
                priv = self._env.privileged_state()
                prev_kills = self._shaper._stats.kills
                prev_items = self._shaper._stats.items
                r, ep_stats = self._shaper.step(priv, action=action_int, obs=next_obs)

                # Reliable delta kill tracking (never double counts)
                d_kills = max(0, int(priv["kills"]) - prev_kills)
                if d_kills > 0:
                    self.ep_kills += d_kills
                    self.total_kills += d_kills
                    self._steps_since_kill = 0
                    self._log("info", f"Enemy killed! (ep_kills={self.ep_kills}, total_kills={self.total_kills})")
                else:
                    self._steps_since_kill += 1
                if action_int[6]:
                    if not self.behavior["enemy_detected"]:
                        self._ep_shots_no_tgt += 1
                        if self._steps_since_kill <= 5:
                            self._ep_shots_after_kill += 1

                if priv["health"] < self._prev_hp:
                    self._log("warn", f"Player damaged: HP {self._prev_hp:.0f} -> {priv['health']:.0f}")
                elif priv["health"] > self._prev_hp:
                    self._hp_before_med = self._prev_hp
                    self._hp_after_med = float(priv["health"])
                    self._log("info", f"Medikit collected: HP {self._prev_hp:.0f} -> {priv['health']:.0f}")
                self._prev_hp = float(priv["health"])
                if action_int[8]:
                    self._ep_wpn_switches += 1

                if int(priv["items"]) > prev_items:
                    self._log("info", "Player collected an item.")

                # Behavior counters
                self.ep_step += 1
                self.total_steps += 1
                if action_int[6]:
                    self._ep_atk_steps += 1
                if strat == "exploration":
                    self._ep_exp_steps += 1
                if action_int[0] or action_int[1]:
                    self._ep_move += 1
                if action_int[4] or action_int[5]:
                    self._ep_turn += 1
                if action_int[2] or action_int[3]:
                    self._ep_strafe += 1

                done = bool(term or trunc)
                if not self.config.deterministic_eval:
                    buf.add(
                        [obs], self._prev_action, action_np,
                        log_prob_t.cpu().numpy(), value_t.cpu().numpy(),
                        np.array([r], np.float32), np.array([float(done)], np.float32), hid_in,
                    )

                self._gameplay_history.append({
                    "step": self.total_steps,
                    "episode": self.episode_num,
                    "map_id": self.config.map_id,
                    "health": float(priv["health"]),
                    "armor": float(obs["vitals"][1]),
                    "weapon": int(obs["vitals"][2]),
                    "ammo": float(obs["vitals"][3]),
                    "kills": int(priv["kills"]),
                    "visible_enemies": int(self.behavior["visible_enemies"]),
                    "action": [int(x) for x in action_int],
                    "action_probs": [round(float(p), 3) for p in probs_np],
                    "reward": round(float(r), 4),
                    "disp": round(float(math.hypot(obs["vitals"][14], obs["vitals"][15])), 2),
                    "front_ray": round(float(obs["rays"]["distance"][0]), 3),
                    "cells_visited": int(self._shaper._stats.cells_visited),
                })

                self._hidden = out["next_hidden"]
                self._prev_action = action_np
                obs = next_obs
                self._update_hud_and_frame(obs, action_int, probs_np, strat)

                if self.behavior["enemy_detected"] and not self._prev_enemy_seen:
                    self._log("info", f"Enemy detected ({self.behavior['visible_enemies']} visible).")
                self._prev_enemy_seen = self.behavior["enemy_detected"]

                if self.config.step_delay_ms > 0:
                    time.sleep(self.config.step_delay_ms / 1000.0)

                if buf.ptr >= rollout_steps:
                    if not self.config.deterministic_eval and self._train_model is not None:
                        self._train_model.load_state_dict(copy.deepcopy(self._model.state_dict()))
                        with torch.no_grad():
                            lo = self._train_model.forward_step(
                                obs_to_tensors(obs, device=self.config.device),
                                hidden=self._hidden,
                                prev_action=torch.as_tensor(self._prev_action, dtype=torch.float32, device=self.config.device),
                            )
                        t_stats = self._trainer.update(buf, lo["value"].cpu().numpy(), np.array([float(done)], np.float32))
                        self._model.load_state_dict(copy.deepcopy(self._train_model.state_dict()))
                        self.update_idx += 1
                        self.history["policy_losses"].append(round(t_stats["policy_loss"], 5))
                        self.history["value_losses"].append(round(t_stats["value_loss"], 5))
                    buf = RolloutBuffer(rollout_steps, 1, self._model.state_dim)

                if done:
                    st = ep_stats or self._shaper._stats
                    self.history["ep_rewards"].append(round(st.reward, 3))
                    win = self.history["ep_rewards"][-10:]
                    self.history["avg_rewards"].append(round(float(np.mean(win)), 3))
                    self.history["ep_kills"].append(st.kills)
                    self.history["ep_survival"].append(st.length)
                    self.history["ep_cells"].append(st.cells_visited)
                    if st.reward > self.best_reward:
                        self.best_reward = st.reward
                        self.save_checkpoint("best")
                    if st.dead:
                        self._log("warn", f"Player died on episode {self.episode_num} (rew={st.reward:+.2f}, kills={st.kills}).")
                    else:
                        self._log("info", f"Episode {self.episode_num} completed (rew={st.reward:+.2f}, kills={st.kills}, exit={st.exit_reached}).")

                    if self.episode_num % max(1, self.config.checkpoint_interval) == 0:
                        self.save_checkpoint("latest")

                    self.episode_num += 1
                    if self._pending_promotion is not None:
                        promo = self._pending_promotion
                        self._pending_promotion = None
                        self._model.load_state_dict(copy.deepcopy(promo["state_dict"]))
                        self.policy_version = promo["new_version"]
                        self.current_checkpoint = promo.get("ckpt_name", self.current_checkpoint)
                        for r_item in self.problem_history:
                            if r_item["id"] == promo["report"]["id"]:
                                r_item["status"] = "Improvement verified"
                                r_item["applied"] = True
                                self.learning_status = dict(r_item)
                                break
                        self._log("info", f"Applied updated policy {self.policy_version} at episode boundary.")
                    if self.episode_num <= target_episodes and not self._stop_event.is_set():
                        obs, _ = self._env.reset(seed=self.config.seed + self.episode_num)
                        self._shaper.reset(self._env.privileged_state(), obs=obs)
                        self._reset_episode_state(obs)

            if self._stop_event.is_set():
                self.save_checkpoint("latest")
                self.status = "Stopped"
                self._log("info", "Training stopped safely and checkpoint saved.")
            else:
                self.save_checkpoint("latest")
                self.status = "Completed"
                self._log("info", "Training completed.")
        except Exception as e:
            self.status = "Failed"
            self.error_msg = str(e)
            self._log("error", f"Training failed: {e}")
        finally:
            if self._env is not None:
                self._env.close()
                self._env = None
                self.env_status = "Closed"

    # ---------------- Human Feedback & Online Learning ----------------
    def submit_feedback(self, categories: list[str], description: str = "") -> dict:
        if not categories and not description.strip():
            return {"ok": False, "error": "Select at least one problem category or enter a description."}
        with self._feedback_lock:
            if self._feedback_busy:
                return {"ok": False, "error": "An online learning update is already in progress."}
            self._init_model_if_needed(force_fresh=False)
            self._feedback_busy = True

        rep_id = len(self.problem_history) + 1
        ctx = {
            "map_id": self.config.map_id,
            "episode": self.episode_num,
            "step": self.total_steps,
            "health": self.hud["health"],
            "armor": self.hud["armor"],
            "weapon_name": self.behavior.get("weapon_name", "Pistol"),
            "ammo": self.hud["selected_weapon_ammo"],
            "kills": self.hud["kills"],
            "checkpoint": self.current_checkpoint,
            "policy_version": self.policy_version,
        }
        hist_copy = list(self._gameplay_history)
        report = {
            "id": rep_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "categories": list(categories),
            "description": description.strip(),
            "map_id": self.config.map_id,
            "episode": self.episode_num,
            "step": self.total_steps,
            "status": "Feedback received",
            "origin": "-",
            "detected_cause": "Analyzing gameplay context...",
            "learning_action": "Pending",
            "prev_version": self.policy_version,
            "new_version": self.policy_version,
            "before_metrics": None,
            "after_metrics": None,
            "verification": "Pending",
            "applied": False,
        }
        self.problem_history.insert(0, report)
        self.learning_status = dict(report)
        threading.Thread(
            target=self._feedback_worker, args=(rep_id, categories, description, hist_copy, ctx), daemon=True
        ).start()
        return {"ok": True, "report_id": rep_id}

    def _feedback_worker(self, rep_id: int, categories: list[str], description: str, hist_copy: list[dict], ctx: dict) -> None:
        report = next((r for r in self.problem_history if r["id"] == rep_id), None)
        if report is None:
            with self._feedback_lock:
                self._feedback_busy = False
            return

        def _set_status(st: str, **kw):
            report["status"] = st
            for k, v in kw.items():
                report[k] = v
            self.learning_status = dict(report)

        try:
            _set_status("Capturing gameplay context")
            spec = self._map_by_id.get(ctx["map_id"]) or next((m for m in self.maps if m["available"]), None)
            if spec is None or not spec["available"]:
                _set_status("Correction unsuccessful", verification="Map not available on disk.")
                return

            _set_status("Analyzing behaviour")
            diag, new_mod = diagnose_feedback(categories, description, hist_copy, ctx, self._feedback_modifiers)
            self._feedback_modifiers = new_mod
            _set_status(
                "Problem identified",
                origin=diag["origin"],
                detected_cause=diag["detected_cause"],
                learning_action=diag["learning_action"],
            )

            # Save stable snapshot for rollback before modifying _train_model
            self._prev_stable_snapshot = {
                "model_state_dict": copy.deepcopy(self._model.state_dict()),
                "train_state_dict": copy.deepcopy(self._train_model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(self._trainer.optimizer.state_dict()),
                "scheduler_state_dict": copy.deepcopy(self._trainer.scheduler.state_dict()),
                "policy_version": self.policy_version,
                "policy_version_num": self.policy_version_num,
                "modifiers": copy.deepcopy(self._feedback_modifiers),
            }

            _set_status("Generating learning signal")
            before_m = evaluate_policy_episodes(
                self._model, spec["path"], ticks=min(700, spec["episode_ticks"]), n_episodes=2, seed_base=700, device=self.config.device
            )
            _set_status("Online learning update in progress", before_metrics=before_m)

            # Run 2 valid online PPO rollout + optimization updates on _train_model (with retention mix!)
            self._train_model.load_state_dict(copy.deepcopy(self._model.state_dict()))
            retention_map = ROOT / "training" / "maps" / "grandmaster" / "center_lr_sym.wad"
            env_paths = [spec["path"], str(retention_map) if retention_map.is_file() else spec["path"]]
            n_envs, r_steps = len(env_paths), 256
            envs = [make_env(p, "MAP01", episode_ticks=min(700, spec["episode_ticks"]), render_buffers=False) for p in env_paths]
            shapers = [DoomRewardShaper() for _ in envs]
            obs_l = []
            for idx_e, e in enumerate(envs):
                o, _ = e.reset(seed=800 + rep_id * 10 + idx_e)
                shapers[idx_e].reset(e.privileged_state(), obs=o)
                obs_l.append(o)
            hid = self._train_model.init_hidden(n_envs, self.config.device)
            pa = np.zeros((n_envs, len(S.BUTTONS)), dtype=np.float32)

            try:
                for upd_i in range(2):
                    self._train_model.eval()
                    obuf = RolloutBuffer(r_steps, n_envs, self._train_model.state_dim)
                    for _ in range(r_steps):
                        from training.rollout import _batch_obs
                        ot = _batch_obs(obs_l, self.config.device)
                        pat = torch.as_tensor(pa, dtype=torch.float32, device=self.config.device)
                        with torch.no_grad():
                            out = self._train_model.forward_step(ot, hidden=hid, prev_action=pat)
                            at = out["tactical_dist"].sample()
                            lpt = out["tactical_dist"].log_prob(at)
                            vt = out["value"]
                        anp = at.cpu().numpy().astype(np.float32)
                        aint = anp.astype(np.int8)
                        hin = hid.squeeze(0).cpu().numpy()
                        rews = np.zeros(n_envs, dtype=np.float32)
                        dones = np.zeros(n_envs, dtype=np.float32)
                        next_ol = []
                        nhid = out["next_hidden"]
                        npa = anp.copy()
                        ms_np = out["mem_summary"].cpu().numpy()
                        for i_e, e in enumerate(envs):
                            no, _, term, trunc, _ = e.step(aint[i_e])
                            priv = e.privileged_state()
                            raw_r, _ = shapers[i_e].step(priv, action=aint[i_e], obs=no, mem_summary=ms_np[i_e])
                            rews[i_e] = apply_modifiers_to_shaper(shapers[i_e], raw_r, aint[i_e], no, self._feedback_modifiers)
                            dn = bool(term or trunc)
                            dones[i_e] = float(dn)
                            if dn:
                                no, _ = e.reset(seed=900 + rep_id * 10 + i_e)
                                shapers[i_e].reset(e.privileged_state(), obs=no)
                                nhid[:, i_e, :] = 0.0
                                npa[i_e] = 0.0
                            next_ol.append(no)
                        obuf.add(obs_l, pa, anp, lpt.cpu().numpy(), vt.cpu().numpy(), rews, dones, hin)
                        obs_l = next_ol
                        hid = nhid
                        pa = npa
                    with torch.no_grad():
                        from training.rollout import _batch_obs
                        lo = self._train_model.forward_step(
                            _batch_obs(obs_l, self.config.device), hidden=hid,
                            prev_action=torch.as_tensor(pa, dtype=torch.float32, device=self.config.device),
                        )
                    self._trainer.update(obuf, lo["value"].cpu().numpy(), np.zeros(n_envs, dtype=np.float32))
                    self.update_idx += 1
                    self.total_steps += r_steps * n_envs
            finally:
                for e in envs:
                    e.close()

            _set_status("Candidate policy ready")
            _set_status("Evaluating updated policy")
            after_m = evaluate_policy_episodes(
                self._train_model, spec["path"], ticks=min(700, spec["episode_ticks"]), n_episodes=2, seed_base=700, device=self.config.device
            )
            passed, ver_msg = check_candidate_improvement(categories, before_m, after_m)
            report["after_metrics"] = after_m

            if passed:
                self.policy_version_num += 1
                new_ver = f"v13.{self.policy_version_num}"
                self.latest_policy_version = new_ver
                cand_name = f"candidate_{new_ver}.pt"
                cand_dir = self._exp_dir()
                cand_dir.mkdir(parents=True, exist_ok=True)
                self._trainer.save_checkpoint(cand_dir / cand_name, self.update_idx, self.total_steps, extra={"policy_version": new_ver})
                report["new_version"] = new_ver
                report["verification"] = ver_msg
                report["applied"] = True
                if self.status == "Running":
                    _set_status("Applying updated policy")
                    self._pending_promotion = {
                        "state_dict": copy.deepcopy(self._train_model.state_dict()),
                        "new_version": new_ver,
                        "ckpt_name": cand_name,
                        "report": dict(report),
                    }
                else:
                    self._model.load_state_dict(copy.deepcopy(self._train_model.state_dict()))
                    self.policy_version = new_ver
                    self.current_checkpoint = cand_name
                    _set_status("Improvement verified")
                self._log("info", f"Online feedback {new_ver} verified: {ver_msg}")
            else:
                # Rollback _train_model & optimizer
                snap = self._prev_stable_snapshot
                self._train_model.load_state_dict(copy.deepcopy(snap["train_state_dict"]))
                self._trainer.optimizer.load_state_dict(copy.deepcopy(snap["optimizer_state_dict"]))
                self._trainer.scheduler.load_state_dict(copy.deepcopy(snap["scheduler_state_dict"]))
                _set_status("Correction unsuccessful", verification=f"Rolled back to {self.policy_version}: {ver_msg}", applied=False)
                self._log("warn", f"Candidate rejected & rolled back: {ver_msg}")
        except Exception as e:
            _set_status("Correction unsuccessful", verification=f"Error during online update: {e}", applied=False)
            self._log("error", f"Feedback learning error: {e}")
        finally:
            with self._feedback_lock:
                self._feedback_busy = False

    def rollback_policy(self) -> dict:
        if self._prev_stable_snapshot is None:
            return {"ok": False, "error": "No previous stable policy snapshot available to roll back."}
        snap = self._prev_stable_snapshot
        with self._lock:
            self._pending_promotion = None
            self._model.load_state_dict(copy.deepcopy(snap["model_state_dict"]))
            self._train_model.load_state_dict(copy.deepcopy(snap["train_state_dict"]))
            self._trainer.optimizer.load_state_dict(copy.deepcopy(snap["optimizer_state_dict"]))
            self._trainer.scheduler.load_state_dict(copy.deepcopy(snap["scheduler_state_dict"]))
            self.policy_version = snap["policy_version"]
            self.policy_version_num = snap["policy_version_num"]
            self._feedback_modifiers = copy.deepcopy(snap["modifiers"])
            self.learning_status = {"status": f"Rolled back to {self.policy_version}", "active_report_id": None}
            self._log("warn", f"Manually rolled back active policy to {self.policy_version}.")
        return {"ok": True, "policy_version": self.policy_version}

    def snapshot(self) -> dict:
        with self._lock:
            n_eps = len(self.history["ep_rewards"])
            return {
                "status": self.status,
                "env_status": self.env_status,
                "error_msg": self.error_msg,
                "config": asdict(self.config),
                "maps": self.maps,
                "selected_map": self._map_by_id.get(self.config.map_id),
                "current_checkpoint": self.current_checkpoint,
                "policy_version": self.policy_version,
                "latest_policy_version": self.latest_policy_version,
                "can_rollback": self._prev_stable_snapshot is not None,
                "problem_categories": PROBLEM_CATEGORIES,
                "learning_status": self.learning_status,
                "problem_history": self.problem_history[:25],
                "episode_num": self.episode_num,
                "total_steps": self.total_steps,
                "total_kills": self.total_kills,
                "best_reward": round(self.best_reward, 3) if n_eps > 0 else 0.0,
                "avg_reward": self.history["avg_rewards"][-1] if n_eps > 0 else 0.0,
                "avg_kills": round(float(np.mean(self.history["ep_kills"])), 2) if n_eps > 0 else 0.0,
                "avg_survival": round(float(np.mean(self.history["ep_survival"])), 1) if n_eps > 0 else 0.0,
                "completed_episodes": n_eps,
                "hud": self.hud,
                "behavior": self.behavior,
                "memory_viz": self.memory_viz,
                "history": self.history,
                "checkpoints": self.list_checkpoints(),
                "experiments": self.list_experiments(),
                "logs": list(self.logs)[-40:],
            }
