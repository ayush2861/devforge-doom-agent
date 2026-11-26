"""Zero-dependency HTTP live training viewer (MJPEG stream + live metrics JSON)."""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from doom_env import spaces as S
from training.diagnostics.visualize_depth import write_png
from training.diagnostics.visualize_rays import render_rays_polar
from training.diagnostics.visualize_vitals import decode_vitals

_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>DevForge DOOM — Live ESRHA Training</title>
<style>
  body { background:#111318; color:#e6e9ef; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; margin:12px; }
  h1 { font-size:15px; margin:0 0 10px; color:#7dd3fc; }
  .grid { display:flex; gap:12px; flex-wrap:nowrap; align-items:stretch; }
  .card { background:#1a1d24; border:1px solid #2b303b; border-radius:8px; padding:10px; box-sizing:border-box; }
  .video-card { flex:1 1 auto; min-width:0; display:flex; align-items:center; justify-content:center; }
  .side-col { flex:0 0 270px; display:flex; flex-direction:column; gap:10px; }
  .radar-card { display:flex; justify-content:center; padding:8px; }
  img { display:block; border-radius:6px; image-rendering:pixelated; max-width:100%; height:auto; }
  .stat { display:flex; justify-content:space-between; gap:8px; padding:2px 0; border-bottom:1px solid #242832; font-size:11px; line-height:1.25; }
  .k { color:#94a3b8; } .v { color:#f8fafc; font-weight:600; text-align:right; }
</style>
</head>
<body>
  <h1>DevForge DOOM — ESRHA Live Training Stream</h1>
  <div class="grid">
    <div class="card video-card">
      <img id="stream" src="/stream.mjpg" width="640" height="480" alt="Live DOOM stream"/>
    </div>
    <div class="side-col">
      <div class="card radar-card">
        <img id="radar" src="/radar.mjpg" width="180" height="180" alt="360 Ray Radar"/>
      </div>
      <div class="card" style="flex:1 1 auto; overflow-y:auto;">
        <div style="font-weight:700;font-size:12px;margin-bottom:6px;color:#86efac">Live Telemetry &amp; Training</div>
        <div id="stats">Waiting for first frame...</div>
      </div>
    </div>
  </div>
<script>
async function poll() {
  try {
    const r = await fetch('/metrics.json');
    const d = await r.json();
    const rows = Object.entries(d).map(([k,v]) =>
      `<div class="stat"><span class="k">${k}</span><span class="v">${typeof v==='number'? (Number.isInteger(v)? v : v.toFixed(2)) : v}</span></div>`
    ).join('');
    document.getElementById('stats').innerHTML = rows || 'Running...';
  } catch (e) {}
}
setInterval(poll, 500);
poll();
</script>
</body>
</html>
"""


def _encode_jpeg_or_png(rgb: np.ndarray) -> tuple[bytes, str]:
    try:
        import pygame

        if not pygame.get_init():
            pygame.init()
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(rgb).swapaxes(0, 1))
        buf = io.BytesIO()
        pygame.image.save(surf, buf, "JPEG")
        return buf.getvalue(), "image/jpeg"
    except Exception:
        # Fallback to pure-Python PNG via write_png-compatible encoding
        import struct
        import zlib

        arr = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w, _ = arr.shape
        raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""), "image/png"


class WebLiveViewer:
    """Serves a live MJPEG stream and JSON metrics at http://localhost:<port>/."""

    def __init__(self, port: int = 8765, host: str = "127.0.0.1"):
        self.port = port
        self.host = host
        self.closed = False
        self._lock = threading.Lock()
        self._frame: bytes = b""
        self._radar: bytes = b""
        self._mime: str = "image/jpeg"
        self._metrics: dict = {"status": "starting"}
        self._cond = threading.Condition(self._lock)

        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = _HTML.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/metrics.json":
                    with parent._lock:
                        body = json.dumps(parent._metrics).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path in ("/stream.mjpg", "/radar.mjpg"):
                    is_radar = self.path == "/radar.mjpg"
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while not parent.closed:
                            with parent._cond:
                                parent._cond.wait(timeout=1.0)
                                frame = parent._radar if is_radar else parent._frame
                                mime = parent._mime
                            if not frame:
                                continue
                            self.wfile.write(
                                b"--frame\r\nContent-Type: " + mime.encode() + b"\r\nContent-Length: "
                                + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
                            )
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                else:
                    self.send_error(404)

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def update_training_metrics(self, record: dict) -> None:
        with self._lock:
            self._metrics["status"] = "training"
            self._metrics.update(record)

    def render(
        self,
        obs: dict,
        *,
        step: int = 0,
        episode: int = 1,
        ep_reward: float = 0.0,
        strategy: str = "-",
        action: np.ndarray | None = None,
        throttle: bool = False,
    ) -> bool:
        if self.closed:
            return False

        rgb_big = np.repeat(np.repeat(obs["rgb"], 4, axis=0), 4, axis=1)  # (480, 640, 3)
        radar = render_rays_polar(obs["rays"], size=180)

        frame, mime = _encode_jpeg_or_png(rgb_big)
        radar_bytes, _ = _encode_jpeg_or_png(radar)
        v = decode_vitals(obs["vitals"])
        act_str = "".join(name[0] if action is not None and action[i] else "." for i, name in enumerate(S.BUTTONS))

        with self._cond:
            self._frame = frame
            self._radar = radar_bytes
            self._mime = mime
            self._metrics.update(
                {
                    "episode": episode,
                    "step": step,
                    "live_ep_reward": round(float(ep_reward), 3),
                    "strategy": strategy,
                    "health": float(v["health"]),
                    "ammo": float(v["selected_weapon_ammo"]),
                    "visible_entities": int(obs["visible_entities"]["mask"].sum()),
                    "action": act_str,
                }
            )
            self._cond.notify_all()
        return True

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            with self._cond:
                self._cond.notify_all()
            self._server.shutdown()
            self._server.server_close()
