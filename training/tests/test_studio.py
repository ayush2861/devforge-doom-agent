"""End-to-end integration tests for DOOM AI Training Studio."""

from __future__ import annotations

import time

import torch

from training.studio.training_manager import TrainingManager


def test_studio_full_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr("training.studio.training_manager.EXPERIMENTS_DIR", tmp_path)
    mgr = TrainingManager()
    assert len(mgr.maps) >= 15
    assert mgr.status == "Not Started"

    # 1. Select map & start training
    assert mgr.select_map("basic_00")["ok"]
    assert mgr.start({"experiment_name": "test_exp", "num_episodes": 10, "max_ticks": 120})["ok"]

    # 2. Prevent duplicate training loop
    dup = mgr.start()
    assert not dup["ok"]

    # 3. Wait for real steps & frames
    deadline = time.time() + 15.0
    while mgr.total_steps < 15 and time.time() < deadline:
        time.sleep(0.1)
    assert mgr.status == "Running"
    assert mgr.total_steps >= 15
    assert len(mgr._latest_frame) > 100

    # 4. Block map switch while Running
    assert not mgr.select_map("two_rooms")["ok"]

    # 5. Pause & Resume
    assert mgr.pause()["ok"]
    assert mgr.status == "Paused"
    time.sleep(0.05)  # allow any in-flight step to finish
    steps_paused = mgr.total_steps
    time.sleep(0.2)
    assert mgr.total_steps == steps_paused

    # Switch map while paused
    assert mgr.select_map("two_rooms")["ok"]
    assert mgr.resume()["ok"]
    assert mgr.status == "Running"

    # 6. Reset episode (while paused) preserves weights & total_steps
    assert mgr.pause()["ok"]
    time.sleep(0.05)
    w_before = next(mgr._model.parameters()).detach().clone()
    assert mgr.reset_episode()["ok"]
    time.sleep(0.2)
    w_after = next(mgr._model.parameters()).detach().clone()
    torch.testing.assert_close(w_before, w_after)

    # 7. Stop & save checkpoint, then resume from checkpoint
    assert mgr.stop()["ok"]
    assert mgr.status == "Stopped"
    ckpts = mgr.list_checkpoints()
    assert any(c["name"] == "latest.pt" for c in ckpts)
    latest = [c for c in ckpts if c["name"] == "latest.pt"][0]["path"]
    assert mgr.load_checkpoint(latest, resume_optimizer=True)["ok"]

    # 8. Restart fresh
    assert mgr.restart("fresh_exp")["ok"]
    assert mgr.status == "Not Started"
    assert mgr.total_steps == 0
    assert mgr.config.experiment_name == "fresh_exp"
