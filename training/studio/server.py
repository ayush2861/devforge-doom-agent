"""HTTP server for DOOM AI Training Studio."""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from training.studio.training_manager import TrainingManager

STUDIO_DIR = Path(__file__).resolve().parent


def run_studio(host: str = "127.0.0.1", port: int = 8765) -> None:
    mgr = TrainingManager()
    html_bytes = (STUDIO_DIR / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)
            elif parsed.path == "/api/state":
                self._json(200, mgr.snapshot())
            elif parsed.path == "/stream.mjpg":
                qs = parse_qs(parsed.query)
                fps = max(1, min(30, int(qs.get("fps", ["15"])[0])))
                interval = 1.0 / fps
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with mgr._frame_cond:
                            mgr._frame_cond.wait(timeout=0.5)
                            frame = mgr._latest_frame
                        if frame:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
                            )
                            self.wfile.flush()
                        time.sleep(interval)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/api/feedback":
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    data = json.loads(raw.decode())
                except json.JSONDecodeError:
                    self._json(400, {"ok": False, "error": "Invalid JSON"})
                    return
                cats = data.get("categories") or []
                desc = str(data.get("description", ""))
                self._json(200, mgr.submit_feedback(cats, desc))
                return
            if self.path == "/api/rollback":
                self._json(200, mgr.rollback_policy())
                return
            if self.path != "/api/control":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                data = json.loads(raw.decode())
            except json.JSONDecodeError:
                self._json(400, {"ok": False, "error": "Invalid JSON"})
                return

            act = data.pop("action", "")
            if act == "start":
                self._json(200, mgr.start(data))
            elif act == "pause":
                self._json(200, mgr.pause())
            elif act == "resume":
                self._json(200, mgr.resume())
            elif act == "reset_episode":
                self._json(200, mgr.reset_episode())
            elif act == "stop":
                self._json(200, mgr.stop())
            elif act == "restart":
                self._json(200, mgr.restart(data.get("experiment_name")))
            elif act == "select_map":
                self._json(200, mgr.select_map(str(data.get("map_id", ""))))
            elif act == "save_checkpoint":
                self._json(200, mgr.save_checkpoint(data.get("tag")))
            elif act == "load_checkpoint":
                self._json(200, mgr.load_checkpoint(str(data.get("path", "")), bool(data.get("resume_optimizer", True))))
            elif act == "set_speed":
                mgr.config.step_delay_ms = max(0, int(data.get("step_delay_ms", 80)))
                self._json(200, {"ok": True})
            elif act == "set_mode":
                mgr.config.deterministic_eval = bool(data.get("deterministic_eval", False))
                self._json(200, {"ok": True})
            elif act == "clear_logs":
                mgr.logs.clear()
                self._json(200, {"ok": True})
            else:
                self._json(400, {"ok": False, "error": f"Unknown action: {act}"})

    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"DOOM AI Training Studio running at http://{host}:{port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        mgr.stop()
        srv.server_close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run DOOM AI Training Studio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    run_studio(host=args.host, port=args.port)
