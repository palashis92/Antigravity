"""Lightweight MJPEG Camera Feed Server for LUMI.

Provides a browser-accessible live view of what LUMI's camera sees,
with optional face detection bounding box overlays.

Usage:
    server = CameraFeedServer(camera, face_service, port=5555)
    server.start()  # Runs in a daemon thread
    # Access at http://<pi-ip>:5555/
"""

from __future__ import annotations

import io
import socket
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import TYPE_CHECKING, Any, Optional

from ..core.logger import get_logger

if TYPE_CHECKING:
    from ..vision.camera import CameraInterface
    from ..vision.face import FaceRecognitionService

logger = get_logger("web.camera_server")

# Shared state for the latest annotated frame
_latest_jpeg: bytes = b""
_frame_lock = threading.Lock()


def _get_local_ip() -> str:
    """Get the local IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"


_HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LUMI Camera Feed</title>
    <style>
        body {{ background: #111; color: #0ff; font-family: monospace; text-align: center; margin: 0; padding: 20px; }}
        h1 {{ color: #0ff; font-size: 1.5em; }}
        img {{ max-width: 100%; border: 2px solid #0ff; border-radius: 8px; }}
        .info {{ color: #888; font-size: 0.9em; margin-top: 10px; }}
    </style>
</head>
<body>
    <h1>🤖 LUMI Live Camera Feed</h1>
    <img src="/camera" alt="LUMI Camera">
    <div class="info">Stream from LUMI's eyes — Refresh page if stream stops</div>
</body>
</html>
"""


class _MJPEGHandler(BaseHTTPRequestHandler):
    """HTTP handler for MJPEG streaming and the HTML viewer page."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP logs to keep console clean."""
        pass

    def do_GET(self) -> None:
        if self.path == "/":
            self._serve_html()
        elif self.path == "/camera":
            self._serve_mjpeg()
        else:
            self.send_error(404)

    def _serve_html(self) -> None:
        content = _HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_mjpeg(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                with _frame_lock:
                    jpeg = _latest_jpeg
                if jpeg:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                time.sleep(0.1)  # ~10 FPS to browser
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # Client disconnected


def _draw_text_pil(frame, text: str, position: tuple, color: tuple, font_size: int = 22):
    """Draw Unicode text (including Bengali) on an OpenCV frame using PIL."""
    try:
        import cv2
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        # Convert BGR OpenCV frame to RGB PIL Image
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Try to find a Unicode-capable font
        font = None
        font_paths = [
            "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
            "/usr/share/fonts/truetype/lohit-bengali/Lohit-Bengali.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for fp in font_paths:
            try:
                import os
                if os.path.exists(fp):
                    font = ImageFont.truetype(fp, font_size)
                    break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()

        # PIL uses RGB color, OpenCV provides BGR
        rgb_color = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=rgb_color)

        # Convert back to BGR numpy array
        frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except (ImportError, Exception):
        # PIL not available or failed, fallback to ASCII-safe cv2.putText
        import cv2
        safe_text = text.encode("ascii", "replace").decode("ascii")
        cv2.putText(frame, safe_text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


class CameraFeedServer:
    """Runs an MJPEG web server in a background thread."""

    def __init__(
        self,
        camera: CameraInterface,
        face_service: Optional[FaceRecognitionService] = None,
        port: int = 5555,
    ) -> None:
        self.camera = camera
        self.face_service = face_service
        self.port = port
        self._running = False
        self._server: Optional[HTTPServer] = None

    def start(self) -> None:
        """Start the camera feed server in background threads."""
        self._running = True

        # Thread 1: Capture frames and update shared JPEG buffer
        capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="CameraFeed_Capture"
        )
        capture_thread.start()

        # Thread 2: HTTP server
        server_thread = threading.Thread(
            target=self._run_server, daemon=True, name="CameraFeed_HTTP"
        )
        server_thread.start()

        local_ip = _get_local_ip()
        logger.info(f"📷 Camera Feed Server started at http://{local_ip}:{self.port}/")

    def stop(self) -> None:
        self._running = False
        if self._server:
            self._server.shutdown()

    def _capture_loop(self) -> None:
        """Continuously capture frames, annotate with face boxes, encode as JPEG."""
        global _latest_jpeg
        while self._running:
            if not self.camera.is_available():
                time.sleep(1.0)
                continue
            try:
                frame = self.camera.get_frame()
                if frame is None or not hasattr(frame, "shape"):
                    time.sleep(0.1)
                    continue

                import cv2

                # Optionally overlay face detection boxes
                if self.face_service:
                    try:
                        faces = self.face_service.detect_and_recognize(frame)
                        for face in faces:
                            x, y, w, h = face.bounding_box
                            if face.is_known and face.person:
                                # Green box + name for known persons
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                label = face.person.name
                                _draw_text_pil(frame, label, (x, y - 30), (0, 255, 0))
                            else:
                                # Red box for unknown persons
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                _draw_text_pil(frame, "Unknown", (x, y - 30), (0, 0, 255))
                    except Exception:
                        pass  # Don't let annotation errors kill the feed

                _, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                with _frame_lock:
                    _latest_jpeg = buffer.tobytes()

            except Exception as e:
                logger.debug(f"Camera feed capture error: {e}")

            time.sleep(0.1)  # ~10 FPS capture rate

    def _run_server(self) -> None:
        """Run the HTTP server."""
        try:
            self._server = HTTPServer(("0.0.0.0", self.port), _MJPEGHandler)
            self._server.serve_forever()
        except Exception as e:
            logger.error(f"Camera feed server error: {e}")
