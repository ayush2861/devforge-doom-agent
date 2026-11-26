# Training Pipeline (`training/`)

Custom PyTorch RL pipeline for **ESRHA (Ego-Spatial Recurrent Hierarchical Agent)**:

- `models/`: `SpatialEncoder`, `EntityEncoder`, `VisualEncoder`, `RecurrentMemory` (GRU + Episodic Slot Bank), `HierarchicalPolicy`, `ValueNetwork`, `ESRHAAgentModel`.
- `algorithms/ppo.py`: Clipped PPO with truncated BPTT (`seq_len=32`), GAE, cosine LR schedule, and joint ICM optimization.
- `curiosity.py`: Prediction-error `IntrinsicCuriosityModule` with Welford `RunningMeanStd` normalization.
- `rewards/reward.py`: Training-only `DoomRewardShaper` using `env.privileged_state()` (exploration grid coverage, kills, items, keys, doors opened, stuck/repetition penalties).
- `maps/progression_wads.py` & `maps/registry.py`: Deterministic ASCII-to-UDMF WAD compiler and 48-map disjoint `train`/`val`/`test` registry across Stages 1–6.
- `curriculum.py` & `train_curriculum.py`: Performance-gated 6-stage curriculum trainer.
- `eval_generalization.py`: Zero-shot evaluation across `train`, `val`, and unseen `test` splits.
- `diagnostics/` & `web_viewer.py` / `viewer.py`: Sensory visualizers and live MJPEG+JSON training dashboard (`--web`).
