"""Thread-safe RL Training Manager connecting DoomEnv, ESRHA, PPOTrainer, and Dashboard."""

from __future__ import annotations

import io
import json
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
from training.studio.map_catalog import ROOT, discover_maps

EXPERIMENTS_DIR = ROOT / "training" / "runs" / "studio_experiments"
_ENEMY_TYPE_IDS = {i for i, t in enumerate(S.ENTITY_TYPES) if t.startswith("enemy")}


@dataclass
class TrainStudioConfig:
    experiment_name: str = "combat_studio_exp"
    map_id: str = "vizdoom_corridor"
    num_episodes: int = 200
    max_ticks: int = 1400
    lr: float = 3e-4
    batch_seqs: int = 8
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    checkpoint_interval: int = 10
    seed: int = 42
    device: str = "cpu"
    step_delay_ms: int = 80  # ~12.5 steps/s (smooth ~1.4x real-time DOOM speed; 0 = max speed)
    deterministic_eval: bool = False  # True = clean deterministic playback without stochastic jitter


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
        self._trainer: PPOTrainer | None = None
        self._shaper = DoomRewardShaper()
        self._hidden: torch.Tensor | None = None
        self._prev_action = np.zeros((1, len(S.BUTTONS)), dtype=np.float32)

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
            cfg = PPOConfig(
                lr=self.config.lr, gamma=self.config.gamma, gae_lambda=self.config.gae_lambda,
                ent_coef=self.config.ent_coef, ppo_epochs=self.config.ppo_epochs,
                batch_seqs=self.config.batch_seqs, seq_len=32,
            )
            self._trainer = PPOTrainer(self._model, cfg, total_updates=max(10, self.config.num_episodes), device=self.config.device)
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
            cfg = PPOConfig(lr=self.config.lr, gamma=self.config.gamma, gae_lambda=self.config.gae_lambda, ent_coef=self.config.ent_coef)
            self._trainer = PPOTrainer(self._model, cfg, total_updates=max(10, self.config.num_episodes), device=self.config.device)
            self._model.load_state_dict(ckpt["model_state_dict"])
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
                    self._log("info", f"Enemy killed! (ep_kills={self.ep_kills}, total_kills={self.total_kills})")

                if priv["health"] < self._prev_hp:
                    self._log("warn", f"Player damaged: HP {self._prev_hp:.0f} -> {priv['health']:.0f}")
                self._prev_hp = float(priv["health"])

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
                buf.add(
                    [obs], self._prev_action, action_np,
                    log_prob_t.cpu().numpy(), value_t.cpu().numpy(),
                    np.array([r], np.float32), np.array([float(done)], np.float32), hid_in,
                )

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
                    with torch.no_grad():
                        lo = self._model.forward_step(
                            obs_to_tensors(obs, device=self.config.device),
                            hidden=self._hidden,
                            prev_action=torch.as_tensor(self._prev_action, dtype=torch.float32, device=self.config.device),
                        )
                    t_stats = self._trainer.update(buf, lo["value"].cpu().numpy(), np.array([float(done)], np.float32))
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
