"""Watch a trained ESRHA checkpoint play DOOM live in a pygame window."""

from __future__ import annotations

import argparse

import torch

from doom_env import make_env
from training.algorithms.ppo import PPOTrainer
from training.models.doom_model import ESRHAAgentModel, obs_to_tensors
from training.models.policy import STRATEGIES
from training.rewards.reward import DoomRewardShaper
from training.viewer import DoomLiveViewer


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch a trained ESRHA checkpoint play live")
    ap.add_argument("--checkpoint", default="training/runs/ppo_smoke/latest.pt")
    ap.add_argument("--wad", default="maps/two_rooms.wad")
    ap.add_argument("--map", default="MAP01")
    ap.add_argument("--episode-ticks", type=int, default=2100)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--fps", type=int, default=15, help="Display FPS (15 ~= 1.7x real-time)")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--web", action="store_true", help="Stream live gameplay to http://127.0.0.1:8765")
    ap.add_argument("--web-port", type=int, default=8765)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    mcfg = ckpt.get("model_config", {})
    model = ESRHAAgentModel(
        hidden_dim=mcfg.get("hidden_dim", 384),
        use_visual=mcfg.get("use_visual", False),
        use_depth=mcfg.get("use_depth", True),
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = make_env(args.wad, args.map, episode_ticks=args.episode_ticks, render_buffers=True)
    shaper = DoomRewardShaper()
    if args.web:
        from training.web_viewer import WebLiveViewer

        viewer = WebLiveViewer(port=args.web_port)
        print(f"Live gameplay stream: {viewer.url}", flush=True)
    else:
        viewer = DoomLiveViewer(title=f"ESRHA Watch — {args.wad}:{args.map}", fps=args.fps)

    try:
        for ep in range(1, args.episodes + 1):
            obs, _ = env.reset(seed=ep)
            shaper.reset(env.privileged_state())
            model.reset_state(device=args.device)
            ep_rew = 0.0
            step = 0
            while True:
                # Query strategic mode for HUD + act
                obs_t = obs_to_tensors(obs, device=args.device)
                with torch.no_grad():
                    out = model.forward_step(obs_t, hidden=model._hidden, prev_action=model._prev_action)
                strat = STRATEGIES[int(out["strategy_probs"].argmax(dim=-1).item())]
                action = model.act(obs, deterministic=args.deterministic)

                if not viewer.render(obs, step=step, episode=ep, ep_reward=ep_rew, strategy=strat, action=action):
                    return
                if args.web:
                    import time

                    time.sleep(1.0 / max(1, args.fps))

                obs, _, term, trunc, info = env.step(action)
                r, stats = shaper.step(env.privileged_state())
                ep_rew += r
                step += 1
                if term or trunc:
                    print(
                        f"Episode {ep}: steps={step}, reward={ep_rew:+.2f}, "
                        f"exit={info['exit_reached']}, kills={stats.kills if stats else 0}"
                    )
                    break
    finally:
        viewer.close()
        env.close()


if __name__ == "__main__":
    main()
