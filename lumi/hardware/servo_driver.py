"""PCA9685 I2C Hardware Servo Driver Adapter for Raspberry Pi 5."""

from __future__ import annotations

from typing import Optional

from ..core.logger import get_logger
from .base import ServoDriverBase
from .mocks import MockServoDriver

logger = get_logger("hardware.pca9685")


class PCA9685ServoDriver(ServoDriverBase):
    """Hardware I2C driver for PCA9685 16-channel PWM controller."""

    def __init__(
        self,
        i2c_bus: int = 1,
        i2c_address: int = 0x40,
        pwm_frequency_hz: int = 50,
    ) -> None:
        self.i2c_bus = i2c_bus
        self.i2c_address = i2c_address
        self.pwm_frequency_hz = pwm_frequency_hz
        self._pca: Optional[object] = None
        self._is_hardware = False

    def initialize(self) -> bool:
        """Attempt to bind to the hardware I2C bus via Adafruit CircuitPython or SMBus."""
        try:
            import board  # type: ignore
            import busio  # type: ignore
            from adafruit_pca9685 import PCA9685  # type: ignore

            i2c = busio.I2C(board.SCL, board.SDA)
            self._pca = PCA9685(i2c, address=self.i2c_address)
            self._pca.frequency = self.pwm_frequency_hz  # type: ignore
            self._is_hardware = True
            logger.info(
                f"PCA9685 Hardware Servo Driver initialized at I2C address 0x{self.i2c_address:02X} ({self.pwm_frequency_hz}Hz)."
            )
            return True
        except (ImportError, Exception) as e:
            logger.warning(
                f"Could not initialize physical PCA9685 I2C hardware ({e}). Operating in software simulation mode."
            )
            self._pca = None
            self._is_hardware = False
            return True

    def set_pwm_us(self, channel: int, pulse_us: int) -> None:
        if not (0 <= channel <= 15):
            return

        if self._is_hardware and self._pca is not None:
            try:
                # PCA9685 has 4096 counts per period.
                # Period for 50Hz = 20,000 us (20ms).
                period_us = 1_000_000.0 / self.pwm_frequency_hz
                duty_cycle = int((pulse_us / period_us) * 65535.0)
                duty_cycle = max(0, min(65535, duty_cycle))
                self._pca.channels[channel].duty_cycle = duty_cycle  # type: ignore
            except Exception as e:
                logger.error(f"Failed to write PWM to PCA9685 channel {channel}: {e}")
        else:
            logger.debug(f"[MOCK PCA9685] Ch {channel} -> {pulse_us} us")

    def set_angle(self, channel: int, angle_deg: float) -> None:
        # Standard 500us (-90 deg) to 2500us (+90 deg) mapping
        norm = (angle_deg + 90.0) / 180.0
        pulse = 500 + norm * 2000
        self.set_pwm_us(channel, int(pulse))

    def release_channel(self, channel: int) -> None:
        if self._is_hardware and self._pca is not None:
            try:
                self._pca.channels[channel].duty_cycle = 0  # type: ignore
            except Exception:
                pass

    def shutdown(self) -> None:
        if self._is_hardware and self._pca is not None:
            try:
                self._pca.deinit()  # type: ignore
            except Exception:
                pass
        logger.info("PCA9685 driver shutdown complete.")
