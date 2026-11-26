"""Unit tests and CPU inference benchmark for the ESRHA neural network."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from doom_env import make_env
from doom_env import spaces as S
from tests.fixtures import fixture_wad
from training.models import (
    ESRHAAgentModel,
    EntityEncoder,
    HierarchicalPolicy,
    RecurrentMemory,
    SpatialEncoder,
    ValueNetwork,
    VisualEncoder,
    obs_to_tensors,
)


def _fake_obs(n_entities: int = 3, batch: int | None = None) -> dict:
    rng = np.random.default_rng(0)
    prefix = () if batch is None else (batch,)
    mask = np.zeros((*prefix, S.MAX_ENTITIES), dtype=np.int8)
    if batch is None:
        mask[:n_entities] = 1
    else:
        mask[:, :n_entities] = 1
    return {
        "rgb": rng.integers(0, 256, size=(*prefix, S.SCREEN_HEIGHT, S.SCREEN_WIDTH, 3), dtype=np.uint8),
        "depth": rng.integers(0, 256, size=(*prefix, S.SCREEN_HEIGHT, S.SCREEN_WIDTH), dtype=np.uint8),
        "vitals": rng.uniform(0, 100, size=(*prefix, len(S.VITALS_FIELDS))).astype(np.float32),
        "rays": {
            "distance": rng.uniform(0, 1, size=(*prefix, S.NUM_RAYS)).astype(np.float32),
            "hit_type": rng.integers(0, len(S.RAY_HIT_TYPES), size=(*prefix, S.NUM_RAYS), dtype=np.int64),
        },
        "visible_entities": {
            "features": rng.normal(0, 1, size=(*prefix, S.MAX_ENTITIES, len(S.ENTITY_FEATURES))).astype(np.float32),
            "entity_type": rng.integers(1, len(S.ENTITY_TYPES), size=(*prefix, S.MAX_ENTITIES), dtype=np.int64),
            "mask": mask,
        },
    }


def test_spatial_encoder_shapes_and_depth_toggle():
    b = 4
    obs_t = obs_to_tensors(_fake_obs(batch=b))
    enc_with_depth = SpatialEncoder(out_dim=192, use_depth=True)
    out = enc_with_depth(obs_t["ray_distance"], obs_t["ray_hit_type"], obs_t["vitals"], obs_t["depth"])
    assert out.shape == (b, 192)
    assert torch.isfinite(out).all()

    enc_no_depth = SpatialEncoder(out_dim=192, use_depth=False)
    out_nd = enc_no_depth(obs_t["ray_distance"], obs_t["ray_hit_type"], obs_t["vitals"])
    assert out_nd.shape == (b, 192)
    assert torch.isfinite(out_nd).all()


def test_entity_encoder_zero_entities_and_padding_invariance():
    enc = EntityEncoder(out_dim=128)
    enc.eval()

    # 1. Zero visible entities (mask all 0) must not produce NaNs
    obs_zero = obs_to_tensors(_fake_obs(n_entities=0, batch=2))
    out_zero = enc(obs_zero["ent_features"], obs_zero["ent_type"], obs_zero["ent_mask"])
    assert out_zero.shape == (2, 128)
    assert torch.isfinite(out_zero).all()

    # 2. Padded slots must be ignored completely
    obs_a = obs_to_tensors(_fake_obs(n_entities=2, batch=1))
    obs_b = {k: v.clone() for k, v in obs_a.items()}
    obs_b["ent_features"][:, 2:, :] = 999.0
    obs_b["ent_type"][:, 2:] = 5
    out_a = enc(obs_a["ent_features"], obs_a["ent_type"], obs_a["ent_mask"])
    out_b = enc(obs_b["ent_features"], obs_b["ent_type"], obs_b["ent_mask"])
    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)


def test_visual_encoder_enabled_and_disabled():
    rgb = torch.randint(0, 256, (3, 120, 160, 3), dtype=torch.uint8)
    vis_on = VisualEncoder(out_dim=128, enabled=True)
    out_on = vis_on(rgb)
    assert out_on.shape == (3, 128)
    assert torch.isfinite(out_on).all()

    vis_off = VisualEncoder(out_dim=128, enabled=False)
    out_off = vis_off(rgb)
    assert out_off.shape == (3, 128)
    assert (out_off == 0).all()
    assert sum(p.numel() for p in vis_off.parameters()) == 0


def test_memory_step_and_sequence_reset():
    mem = RecurrentMemory(input_dim=64, hidden_dim=128)
    x = torch.randn(2, 64)
    out1, h1 = mem.forward_step(x)
    assert out1.shape == (2, 128) and h1.shape == (1, 2, mem.state_dim)

    # Sequence with done at t=2 should reset hidden state for t=3
    x_seq = torch.randn(2, 5, 64)
    dones = torch.zeros(2, 5)
    dones[:, 2] = 1.0
    out_seq, h_last = mem.forward_sequence(x_seq, dones=dones)
    assert out_seq.shape == (2, 5, 128)
    assert h_last.shape == (1, 2, mem.state_dim)
    assert torch.isfinite(out_seq).all()


def test_hierarchical_policy_and_value():
    pol = HierarchicalPolicy(hidden_dim=128)
    val = ValueNetwork(hidden_dim=128)
    mem_out = torch.randn(4, 128)
    t_dist, s_dist, logits, s_probs = pol(mem_out)
    assert logits.shape == (4, 9)
    assert s_probs.shape == (4, 5)
    torch.testing.assert_close(s_probs.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=1e-5)

    act = t_dist.sample()
    assert act.shape == (4, 9)
    assert ((act == 0) | (act == 1)).all()
    assert t_dist.log_prob(act).shape == (4,)

    v = val(mem_out, s_probs)
    assert v.shape == (4,)
    assert torch.isfinite(v).all()


def test_full_esrha_step_sequence_and_real_env():
    model = ESRHAAgentModel(use_visual=True, use_depth=True)
    n_params = sum(p.numel() for p in model.parameters())
    assert 100_000 < n_params < 5_000_000  # lightweight

    # Batched sequence BPTT forward
    b, t = 2, 6
    obs_seq = obs_to_tensors(_fake_obs(n_entities=4, batch=b * t))
    obs_seq = {k: v.reshape(b, t, *v.shape[1:]) for k, v in obs_seq.items()}
    dones = torch.zeros(b, t)
    dones[0, 3] = 1.0
    out_seq = model.forward_sequence(obs_seq, dones=dones)
    assert out_seq["action_logits"].shape == (b, t, 9)
    assert out_seq["strategy_probs"].shape == (b, t, 5)
    assert out_seq["values"].shape == (b, t)

    # Real DoomEnv single-agent inference
    env = make_env(fixture_wad("#####\n#P.z#\n#####"), episode_ticks=100)
    obs, _ = env.reset(seed=0)
    model.reset_state()
    for _ in range(5):
        action = model.act(obs)
        assert action.shape == (9,) and action.dtype == np.int8
        assert env.action_space.contains(action)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    env.close()


if __name__ == "__main__":
    # Parameter count & single-thread CPU inference benchmark
    torch.set_num_threads(1)
    for use_vis in (True, False):
        m = ESRHAAgentModel(use_visual=use_vis, use_depth=True)
        m.eval()
        total = sum(p.numel() for p in m.parameters())
        print(f"ESRHA (use_visual={use_vis}): {total:,} params ({total * 4 / 1024**2:.2f} MB fp32)")
        for name, mod in [
            ("spatial_encoder", m.spatial_encoder),
            ("entity_encoder", m.entity_encoder),
            ("visual_encoder", m.visual_encoder),
            ("pre_memory", m.pre_memory),
            ("memory", m.memory),
            ("policy", m.policy),
            ("value_net", m.value_net),
        ]:
            print(f"  {name:16s}: {sum(p.numel() for p in mod.parameters()):,}")

        obs = _fake_obs(n_entities=8)
        m.reset_state()
        for _ in range(10):  # warm-up
            m.act(obs)
        lats = []
        for _ in range(200):
            t0 = time.perf_counter()
            m.act(obs)
            lats.append((time.perf_counter() - t0) * 1000.0)
        print(
            f"  CPU 1-thread latency (200 steps): "
            f"p50={np.percentile(lats, 50):.2f} ms, p99={np.percentile(lats, 99):.2f} ms, max={max(lats):.2f} ms\n"
        )
