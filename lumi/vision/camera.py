"""Camera Interface and Swappable Backends (Phone IP WebCam, USB Webcam, Pi Camera Module)."""

from __future__ import annotations

import time
from typing import Any, Optional

from ..core.logger import get_logger
from ..hardware.base import CameraBackendBase
from ..hardware.mocks import MockCameraBackend

logger = get_logger("vision.camera")


import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "-8"

class PhoneCameraBackend(CameraBackendBase):
    """Captures network MJPEG or RTSP stream from mobile phone camera (IP Webcam / DroidCam)."""

    def __init__(self, stream_url: str = "http://192.168.1.100:8080/video") -> None:
        self.stream_url = stream_url
        self._cap: Optional[object] = None
        self._running = False

    def start(self) -> bool:
        try:
            import cv2  # type: ignore
            if hasattr(cv2, "setLogLevel"):
                try:
                    cv2.setLogLevel(0)
                except Exception:
                    pass
            self._cap = cv2.VideoCapture(self.stream_url)
            self._running = self._cap.isOpened()
            if self._running:
                logger.info(f"Phone Camera Stream connected at '{self.stream_url}'.")
                return True
        except Exception as e:
            logger.warning(f"Failed to connect to phone camera stream ({e}). Fallback active.")
        self._running = False
        return False

    def stop(self) -> None:
        self._running = False
        if self._cap is not None:
            try:
                self._cap.release()  # type: ignore
            except Exception:
                pass
        logger.info("Phone camera backend stopped.")

    def is_available(self) -> bool:
        return self._running

    def get_frame(self) -> Optional[Any]:
        if not self._running or self._cap is None:
            return None
        try:
            ret, frame = self._cap.read()  # type: ignore
            if ret:
                return frame
        except Exception as e:
            logger.error(f"Error reading phone stream frame: {e}")
        return None


class USBWebcamBackend(CameraBackendBase):
    """Captures from standard USB V4L2 video devices."""

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self._cap: Optional[object] = None
        self._running = False

    def start(self) -> bool:
        try:
            import cv2  # type: ignore
            self._cap = cv2.VideoCapture(self.device_index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._running = self._cap.isOpened()
            if self._running:
                logger.info(f"USB Webcam initialized on device index {self.device_index}.")
                return True
        except Exception as e:
            logger.warning(f"Could not open USB Webcam ({e}).")
        self._running = False
        return False

    def stop(self) -> None:
        self._running = False
        if self._cap is not None:
            try:
                self._cap.release()  # type: ignore
            except Exception:
                pass

    def is_available(self) -> bool:
        return self._running

    def get_frame(self) -> Optional[Any]:
        if not self._running or self._cap is None:
            return None
        try:
            ret, frame = self._cap.read()  # type: ignore
            return frame if ret else None
        except Exception:
            return None


class PiCameraBackend(CameraBackendBase):
    """Captures from Raspberry Pi Camera Module v2/v3 via Picamera2."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._picam: Optional[object] = None
        self._running = False

    def start(self) -> bool:
        try:
            from picamera2 import Picamera2  # type: ignore
            self._picam = Picamera2()
            config = self._picam.create_preview_configuration(main={"size": (self.width, self.height)})
            self._picam.configure(config)
            self._picam.start()
            self._running = True
            logger.info("Picamera2 backend initialized for Pi 5.")
            return True
        except Exception as e:
            logger.warning(f"Picamera2 initialization failed ({e}).")
        self._running = False
        return False

    def stop(self) -> None:
        self._running = False
        if self._picam is not None:
            try:
                self._picam.stop()  # type: ignore
            except Exception:
                pass

    def is_available(self) -> bool:
        return self._running

    def get_frame(self) -> Optional[Any]:
        if not self._running or self._picam is None:
            return None
        try:
            frame = self._picam.capture_array()  # type: ignore
            # PiCamera2 returns RGB array, but OpenCV expects BGR. Convert it.
            import cv2
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception:
            return None


class CameraInterface:
    """Unified camera abstraction manager for LUMI."""

    def __init__(self, backend: Optional[CameraBackendBase] = None) -> None:
        self.backend: CameraBackendBase = backend or MockCameraBackend()

    def set_backend(self, backend: CameraBackendBase) -> None:
        if self.backend.is_available():
            self.backend.stop()
        self.backend = backend

    def start(self) -> bool:
        return self.backend.start()

    def stop(self) -> None:
        self.backend.stop()

    def is_available(self) -> bool:
        return self.backend.is_available()

    def get_frame(self) -> Optional[Any]:
        return self.backend.get_frame()

    def capture_frame(self) -> Optional[Any]:
        """Convenience alias for get_frame()."""
        return self.backend.get_frame()
