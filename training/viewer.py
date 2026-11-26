"""Live pygame viewer for watching ESRHA play DOOM (RGB + 360° Ray Radar + HUD)."""

from __future__ import annotations

import numpy as np

from doom_env import spaces as S
from training.diagnostics.visualize_rays import render_rays_polar
from training.diagnostics.visualize_vitals import decode_vitals


class DoomLiveViewer:
    """Real-time pygame window displaying upscaled RGB, 64-ray radar, and agent telemetry."""

    def __init__(self, title: str = "DevForge DOOM — ESRHA Live", fps: int = 35):
        import pygame

        pygame.init()
        pygame.display.set_caption(title)
        self.pygame = pygame
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((840, 480))
        self.font = pygame.font.SysFont("monospace", 14)
        self.font_bold = pygame.font.SysFont("monospace", 15, bold=True)
        self.closed = False

    def render(
        self,
        obs: dict,
        *,
        step: int = 0,
        episode: int = 1,
        ep_reward: float = 0.0,
        strategy: str = "-",
        action: np.ndarray | None = None,
        throttle: bool = True,
    ) -> bool:
        """Render one frame. Returns False if the user closed the window."""
        if self.closed:
            return False

        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.close()
                return False

        self.screen.fill((18, 20, 24))

        # 1. Upscaled RGB (160x120 -> 640x480)
        rgb = np.ascontiguousarray(obs["rgb"])  # (H, W, 3)
        surf_rgb = pg.surfarray.make_surface(rgb.swapaxes(0, 1))
        surf_scaled = pg.transform.scale(surf_rgb, (640, 480))
        self.screen.blit(surf_scaled, (0, 0))

        # 2. Live 360° Ray Radar (200x200)
        radar = render_rays_polar(obs["rays"], size=200)
        surf_radar = pg.surfarray.make_surface(radar.swapaxes(0, 1))
        self.screen.blit(surf_radar, (640, 0))

        # 3. Telemetry & Strategy HUD
        v = decode_vitals(obs["vitals"])
        n_ent = int(obs["visible_entities"]["mask"].sum())
        act_str = "".join(
            name[0] if action is not None and action[i] else "."
            for i, name in enumerate(S.BUTTONS)
        )
        lines = [
            (f"Episode: {episode}  Step: {step}", (220, 225, 235), True),
            (f"Ep Rew : {ep_reward:+.2f}", (100, 240, 140), True),
            (f"Mode   : {strategy}", (90, 200, 255), True),
            (f"HP/Arm : {v['health']:.0f} / {v['armor']:.0f}", (240, 210, 100), False),
            (f"Ammo   : {v['selected_weapon_ammo']:.0f} (W{int(v['selected_weapon'])})", (200, 200, 210), False),
            (f"VisEnt : {n_ent} / {S.MAX_ENTITIES}", (255, 140, 120), False),
            (f"Vel F/R: {v['vel_forward']:+.1f} / {v['vel_right']:+.1f}", (180, 185, 195), False),
            (f"Action : {act_str}", (160, 220, 180), False),
            ("ESC / Close to stop", (120, 125, 135), False),
        ]
        y = 212
        for text, color, bold in lines:
            fnt = self.font_bold if bold else self.font
            self.screen.blit(fnt.render(text, True, color), (650, y))
            y += 26

        pg.display.flip()
        if throttle and self.fps > 0:
            self.clock.tick(self.fps)
        return True

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.pygame.quit()
