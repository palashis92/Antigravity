"""Hardware Manager for monitoring hardware health."""

import os
from typing import Dict, Any

from ..core.logger import get_logger

logger = get_logger("hardware.manager")

class HardwareManager:
    """Checks which peripherals are connected at startup."""

    def __init__(self) -> None:
        self.is_linux = os.name == 'posix'

    def detect_all(self) -> Dict[str, Any]:
        """Runs all detection checks, returns a status dict."""
        status = {
            "i2c": self.check_i2c_devices(),
            "spi": self.check_spi_devices(),
            "camera": self.check_camera(),
            "audio": self.check_audio()
        }
        logger.info(f"Hardware Manager detection results: {status}")
        return status

    def check_i2c_devices(self) -> bool:
        """Scan I2C bus for PCA9685 (address 0x40)."""
        if not self.is_linux:
            return False
        try:
            import smbus2 # type: ignore
            bus = smbus2.SMBus(1)
            bus.read_byte(0x40)
            return True
        except (ImportError, Exception):
            return False

    def check_spi_devices(self) -> bool:
        """Check if SPI buses are available."""
        if not self.is_linux:
            return False
        return os.path.exists("/dev/spidev0.0") and os.path.exists("/dev/spidev0.1")

    def check_camera(self) -> bool:
        """Verify camera is accessible."""
        if not self.is_linux:
            return False
        return os.path.exists("/dev/video0")

    def check_audio(self) -> bool:
        """Verify ALSA audio devices exist."""
        if not self.is_linux:
            return False
        return os.path.exists("/dev/snd")

    def get_status_report(self) -> str:
        """Human-readable status of all peripherals."""
        status = self.detect_all()
        report = []
        report.append(f"I2C Bus (PCA9685): {'OK' if status['i2c'] else 'Not Found'}")
        report.append(f"SPI Displays: {'OK' if status['spi'] else 'Not Found'}")
        report.append(f"Camera (/dev/video0): {'OK' if status['camera'] else 'Not Found'}")
        report.append(f"Audio System: {'OK' if status['audio'] else 'Not Found'}")
        return "\n".join(report)
