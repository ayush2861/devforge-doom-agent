# DevForge DOOM AI Agent — ESRHA

Submission for **Yugaantar 2026 · DevForge — Track 1: "Can It Play DOOM?"**

**ESRHA (Ego-Spatial Recurrent Hierarchical Agent)** is a lightweight (`1.46M`–`1.60M` parameters, `5.75 MB` weights) PyTorch reinforcement learning agent trained from scratch with Recurrent PPO, Intrinsic Curiosity (ICM), Differentiable Episodic Slot Memory, and a 6-stage Performance-Based Curriculum.

See [PROJECT_BRIEF.pdf](PROJECT_BRIEF.pdf) for the two-page technical brief.

---

## Architecture Overview (`training/models/`)

1. **Spatial Encoder (`spatial_encoder.py`)**: Circular 1D CNN (`padding_mode="circular"`) over 64 ego-centric 360° rays (`distance`, `1 - distance`, `hit_type` embedding), strided 2D CNN over `(120, 160)` depth, and normalized 17-d vitals (one-hot weapon, log-scaled ammo, keys, ego velocity) + `prev_action`.
2. **Entity Attention Encoder (`entity_encoder.py`)**: Per-entity MLP + `Embedding(12, 8)` + prepended `[CLS]` token + 4-head masked self-attention (`key_padding_mask` ignoring padded slots, zero-NaN safe for 0 visible entities) + masked max pool.
3. **Modular Visual Encoder (`visual_encoder.py`)**: Optional strided RGB CNN (`enabled=True/False`).
4. **Recurrent Spatial-Episodic Memory (`memory.py`)**: `GRU(384, 384)` augmented with an 8-slot × 32-dim differentiable gated episodic memory bank (`state_dim = 640`).
5. **Hierarchical Policy & Critic (`policy.py`)**: 5-mode Strategic Head (`exploration`, `combat`, `item_collection`, `door_progression`, `exit_seeking`) conditioning a Tactical `MultiBinary(9)` head (`Independent(Bernoulli, 1)`) and scalar Value Network.

---

## Setup & Validation

```bash
uv sync
uv run python validate.py
# Or in Docker (Linux GPU):
python validate.py --docker
```

## Training & Evaluation

```bash
# 1. Generate procedural long-horizon progression WADs
PYTHONPATH=. uv run python training/maps/progression_wads.py

# 2. Train with custom Recurrent PPO + ICM (optional --web live dashboard on :8765)
PYTHONPATH=. uv run python training/train.py \
  --wad maps/two_rooms.wad --total-steps 204800 --use-icm --web

# 3. Train with 6-stage performance-based curriculum
PYTHONPATH=. uv run python training/train_curriculum.py --total-steps 500000

# 4. Evaluate zero-shot generalization on unseen test maps
PYTHONPATH=. uv run python training/eval_generalization.py \
  --checkpoint training/runs/curriculum_v1/best.pt --split test
```

## Inference (`submission/agent.py`)

`DoomAgent` loads `submission/weights/esrha_best.pt` (`5.75 MB`) once in `__init__`, resets GRU + episodic slot memory in `reset()`, and returns `np.int8` `(9,)` binary actions in `act()` at ~`4–6 ms/step` on 1 CPU thread.
