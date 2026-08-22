"""Dual & Single GC9A01 1.28" Round IPS LCD (240x240 SPI) Hardware Display Driver.

Supports Raspberry Pi 5 (RP1 I/O controller via lgpio / pinctrl / gpiod) as well as Pi 4/3.
Each eye display gets its own dedicated DC/RST GPIO controller to prevent signal conflicts.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Optional

# Ensure system dist-packages is accessible inside virtualenvs on Pi
for p in ["/usr/lib/python3/dist-packages", "/usr/local/lib/python3/dist-packages"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from ..core.logger import get_logger
from .base import DisplayBackendBase

logger = get_logger("hardware.gc9a01")


class GPIOController:
    """Universal GPIO pin controller with native Pi 5 pinctrl / lgpio / gpiod support."""

    def __init__(self, dc_pin: int = 24, rst_pin: int = 25, label: str = "") -> None:
        self.dc_pin = dc_pin
        self.rst_pin = rst_pin
        self.label = label
        self.backend: Optional[str] = None
        self._lgpio_handle: Optional[Any] = None
        self._gpiod_request: Optional[Any] = None
        self._has_pinctrl = shutil.which("pinctrl") is not None
        self._init_gpio()

    def _init_gpio(self) -> None:
        # 1. Try lgpio (best for Pi 5)
        try:
            import lgpio  # type: ignore

            for chip_num in [4, 0, 1, 2, 3]:
                h = None
                try:
                    h = lgpio.gpiochip_open(chip_num)
                    lgpio.gpio_claim_output(h, self.dc_pin)
                    lgpio.gpio_claim_output(h, self.rst_pin)
                    self._lgpio_handle = h
                    self.backend = "lgpio"
                    logger.info(
                        f"Display GPIO{' ' + self.label if self.label else ''} initialized via lgpio "
                        f"(gpiochip {chip_num}, DC={self.dc_pin}, RST={self.rst_pin})."
                    )
                    return
                except Exception:
                    if h is not None:
                        try:
                            lgpio.gpiochip_close(h)
                        except Exception:
                            pass
        except Exception:
            pass

        # 2. Try Pi 5 native pinctrl tool
        if self._has_pinctrl:
            try:
                subprocess.run(["pinctrl", "set", str(self.dc_pin), "op"], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["pinctrl", "set", str(self.rst_pin), "op"], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.backend = "pinctrl"
                logger.info(
                    f"Display GPIO{' ' + self.label if self.label else ''} initialized via pinctrl "
                    f"(DC={self.dc_pin}, RST={self.rst_pin})."
                )
                return
            except Exception:
                pass

        # 3. Try gpiod v2
        try:
            import gpiod  # type: ignore

            if hasattr(gpiod, "request_lines"):
                for chip_path in ["/dev/gpiochip4", "/dev/gpiochip0", "/dev/gpiochip1"]:
                    try:
                        req = gpiod.request_lines(
                            chip_path,
                            consumer=f"lumi-display-{self.label or 'default'}",
                            config={
                                self.dc_pin: gpiod.LineSettings(direction=gpiod.line.Direction.OUTPUT),
                                self.rst_pin: gpiod.LineSettings(direction=gpiod.line.Direction.OUTPUT),
                            },
                        )
                        self._gpiod_request = req
                        self.backend = "gpiod_v2"
                        logger.info(
                            f"Display GPIO{' ' + self.label if self.label else ''} initialized via gpiod "
                            f"({chip_path}, DC={self.dc_pin}, RST={self.rst_pin})."
                        )
                        return
                    except Exception:
                        pass
        except Exception:
            pass

        # 4. Try RPi.GPIO
        try:
            import RPi.GPIO as GPIO  # type: ignore

            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.dc_pin, GPIO.OUT)
            GPIO.setup(self.rst_pin, GPIO.OUT)
            self.backend = "rpi_gpio"
            logger.info(
                f"Display GPIO{' ' + self.label if self.label else ''} initialized via RPi.GPIO "
                f"(DC={self.dc_pin}, RST={self.rst_pin})."
            )
            return
        except Exception:
            pass

        logger.warning(
            f"No hardware GPIO library detected for{' ' + self.label if self.label else ''} "
            f"(DC={self.dc_pin}, RST={self.rst_pin}). Software rendering active."
        )

    def set_pin(self, pin: int, high: bool) -> None:
        val = 1 if high else 0
        if self.backend == "lgpio" and self._lgpio_handle is not None:
            import lgpio

            lgpio.gpio_write(self._lgpio_handle, pin, val)
        elif self.backend == "pinctrl":
            arg = "dh" if high else "dl"
            subprocess.run(
                ["pinctrl", "set", str(pin), arg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif self.backend == "gpiod_v2" and self._gpiod_request is not None:
            import gpiod

            self._gpiod_request.set_value(
                pin, gpiod.line.Value.ACTIVE if high else gpiod.line.Value.INACTIVE
            )
        elif self.backend == "rpi_gpio":
            import RPi.GPIO as GPIO

            GPIO.output(pin, GPIO.HIGH if high else GPIO.LOW)

    def cleanup(self) -> None:
        if self.backend == "lgpio" and self._lgpio_handle is not None:
            try:
                import lgpio

                lgpio.gpiochip_close(self._lgpio_handle)
            except Exception:
                pass


class GC9A01DisplayDriver(DisplayBackendBase):
    """Hardware driver for dual GC9A01 240x240 round SPI displays with independent GPIO control.

    Each eye display gets its own DC and RST pin controller so SPI commands
    to the left eye don't corrupt the right eye's data/command signaling.
    """

    def __init__(
        self,
        spi_speed_hz: int = 40_000_000,
        width: int = 240,
        height: int = 240,
        left_dc_pin: int = 24,
        left_rst_pin: int = 25,
        right_dc_pin: int = 23,
        right_rst_pin: int = 22,
    ) -> None:
        self.spi_speed_hz = spi_speed_hz
        self.width = width
        self.height = height
        self.left_dc_pin = left_dc_pin
        self.left_rst_pin = left_rst_pin
        self.right_dc_pin = right_dc_pin
        self.right_rst_pin = right_rst_pin
        self._is_hardware = False
        self._spi_left: Optional[Any] = None
        self._spi_right: Optional[Any] = None
        self._gpio_left: Optional[GPIOController] = None
        self._gpio_right: Optional[GPIOController] = None

    def initialize(self) -> bool:
        """Attempt to open hardware SPI buses and initialize GC9A01 display(s)."""
        try:
            import spidev  # type: ignore

            # Initialize GPIO controllers — one per display with independent DC/RST pins
            self._gpio_left = GPIOController(
                dc_pin=self.left_dc_pin, rst_pin=self.left_rst_pin, label="left-eye"
            )
            self._gpio_right = GPIOController(
                dc_pin=self.right_dc_pin, rst_pin=self.right_rst_pin, label="right-eye"
            )

            # Hardware Reset each display independently
            self._reset_display(self._gpio_left)
            self._reset_display(self._gpio_right)

            # Display 0 (Left Eye): SPI Bus 0, Device 0 (CE0)
            try:
                self._spi_left = spidev.SpiDev()
                self._spi_left.open(0, 0)
                self._spi_left.max_speed_hz = self.spi_speed_hz
                self._spi_left.mode = 0
                self._send_init_sequence(self._spi_left, self._gpio_left)
                logger.info(f"GC9A01 Left Eye (CE0) initialized @ {self.spi_speed_hz / 1e6:.1f}MHz.")
            except Exception as e:
                logger.warning(f"Could not open SPI 0.0 (left eye): {e}")
                self._spi_left = None

            # Display 1 (Right Eye): SPI Bus 0, Device 1 (CE1)
            try:
                self._spi_right = spidev.SpiDev()
                self._spi_right.open(0, 1)
                self._spi_right.max_speed_hz = self.spi_speed_hz
                self._spi_right.mode = 0
                self._send_init_sequence(self._spi_right, self._gpio_right)
                logger.info(f"GC9A01 Right Eye (CE1) initialized @ {self.spi_speed_hz / 1e6:.1f}MHz.")
            except Exception as e:
                logger.warning(f"Could not open SPI 0.1 (right eye): {e}")
                self._spi_right = None

            if self._spi_left is not None or self._spi_right is not None:
                self._is_hardware = True
                self.clear()
                return True
            else:
                self._is_hardware = False
                return True

        except (ImportError, Exception) as e:
            logger.warning(f"Physical GC9A01 SPI hardware not detected ({e}).")
            self._is_hardware = False
            return True

    def _reset_display(self, gpio: GPIOController) -> None:
        """Hardware reset pulse: RST LOW -> wait 100ms -> RST HIGH -> wait 100ms."""
        if gpio.backend is None:
            return
        gpio.set_pin(gpio.rst_pin, False)
        time.sleep(0.1)
        gpio.set_pin(gpio.rst_pin, True)
        time.sleep(0.1)

    def _send_command(
        self, spi_dev: Any, gpio: GPIOController, cmd: int, data: Optional[list[int]] = None
    ) -> None:
        """Send a command byte (DC LOW) optionally followed by data bytes (DC HIGH)."""
        if not spi_dev or gpio.backend is None:
            return
        # DC LOW for Command
        gpio.set_pin(gpio.dc_pin, False)
        spi_dev.writebytes([cmd])
        # DC HIGH for Data parameters
        if data:
            gpio.set_pin(gpio.dc_pin, True)
            spi_dev.writebytes(data)

    def _send_init_sequence(self, spi_dev: Any, gpio: GPIOController) -> None:
        """Full Waveshare GC9A01 240x240 Round LCD initialization register sequence."""
        if not spi_dev:
            return
        try:
            self._send_command(spi_dev, gpio, 0xEF)
            self._send_command(spi_dev, gpio, 0xEB, [0x14])
            self._send_command(spi_dev, gpio, 0xFE)
            self._send_command(spi_dev, gpio, 0xEF)
            self._send_command(spi_dev, gpio, 0x84, [0x40])
            self._send_command(spi_dev, gpio, 0x85, [0xFF])
            self._send_command(spi_dev, gpio, 0x86, [0xFF])
            self._send_command(spi_dev, gpio, 0x87, [0xFF])
            self._send_command(spi_dev, gpio, 0x88, [0x0A])
            self._send_command(spi_dev, gpio, 0x89, [0x21])
            self._send_command(spi_dev, gpio, 0x8A, [0x00])
            self._send_command(spi_dev, gpio, 0x8B, [0x80])
            self._send_command(spi_dev, gpio, 0x8C, [0x01])
            self._send_command(spi_dev, gpio, 0x8D, [0x01])
            self._send_command(spi_dev, gpio, 0x8E, [0xFF])
            self._send_command(spi_dev, gpio, 0x8F, [0xFF])
            self._send_command(spi_dev, gpio, 0xB6, [0x00, 0x00])
            self._send_command(spi_dev, gpio, 0x36, [0x08])   # MADCTL
            self._send_command(spi_dev, gpio, 0x3A, [0x05])   # COLMOD: RGB565
            self._send_command(spi_dev, gpio, 0x90, [0x08, 0x08, 0x08, 0x08])
            self._send_command(spi_dev, gpio, 0xBD, [0x06])
            self._send_command(spi_dev, gpio, 0xBC, [0x00])
            self._send_command(spi_dev, gpio, 0xFF, [0x60, 0x01, 0x04])
            self._send_command(spi_dev, gpio, 0xC3, [0x13])
            self._send_command(spi_dev, gpio, 0xC4, [0x13])
            self._send_command(spi_dev, gpio, 0xC9, [0x22])
            self._send_command(spi_dev, gpio, 0xBE, [0x11])
            self._send_command(spi_dev, gpio, 0xE1, [0x10, 0x0E])
            self._send_command(spi_dev, gpio, 0xDF, [0x21, 0x0C, 0x02])
            self._send_command(spi_dev, gpio, 0xF0, [0x45, 0x09, 0x08, 0x08, 0x26, 0x2A])
            self._send_command(spi_dev, gpio, 0xF1, [0x43, 0x70, 0x72, 0x36, 0x37, 0x6F])
            self._send_command(spi_dev, gpio, 0xF2, [0x45, 0x09, 0x08, 0x08, 0x26, 0x2A])
            self._send_command(spi_dev, gpio, 0xF3, [0x43, 0x70, 0x72, 0x36, 0x37, 0x6F])
            self._send_command(spi_dev, gpio, 0xED, [0x1B, 0x0B])
            self._send_command(spi_dev, gpio, 0xAE, [0x77])
            self._send_command(spi_dev, gpio, 0xCD, [0x63])
            self._send_command(spi_dev, gpio, 0x70, [0x07, 0x07, 0x04, 0x0E, 0x0F, 0x09, 0x07, 0x08, 0x03])
            self._send_command(spi_dev, gpio, 0xE8, [0x34])
            self._send_command(spi_dev, gpio, 0x62, [0x18, 0x0D, 0x71, 0xED, 0x70, 0x70, 0x18, 0x0F, 0x71, 0xEF, 0x70, 0x70])
            self._send_command(spi_dev, gpio, 0x63, [0x18, 0x11, 0x71, 0xF1, 0x70, 0x70, 0x18, 0x13, 0x71, 0xF3, 0x70, 0x70])
            self._send_command(spi_dev, gpio, 0x64, [0x28, 0x29, 0xF1, 0x01, 0xF1, 0x00, 0x07])
            self._send_command(spi_dev, gpio, 0x66, [0x3C, 0x00, 0xCD, 0x67, 0x45, 0x45, 0x10, 0x00, 0x00, 0x00])
            self._send_command(spi_dev, gpio, 0x67, [0x00, 0x3C, 0x00, 0x00, 0x00, 0x01, 0x54, 0x10, 0x32, 0x98])
            self._send_command(spi_dev, gpio, 0x74, [0x10, 0x85, 0x80, 0x00, 0x00, 0x4E, 0x00])
            self._send_command(spi_dev, gpio, 0x98, [0x3E, 0x07])
            self._send_command(spi_dev, gpio, 0x35)          # TEON
            self._send_command(spi_dev, gpio, 0x21)          # Display Inversion ON
            self._send_command(spi_dev, gpio, 0x11)          # Sleep Out
            time.sleep(0.120)
            self._send_command(spi_dev, gpio, 0x29)          # Display ON
            time.sleep(0.020)

            # Set address window: Column 0-239, Row 0-239
            self._send_command(spi_dev, gpio, 0x2A, [0x00, 0x00, 0x00, 0xEF])
            self._send_command(spi_dev, gpio, 0x2B, [0x00, 0x00, 0x00, 0xEF])
        except Exception as e:
            logger.debug(f"GC9A01 init sequence note: {e}")

    def _image_to_rgb565_bytearray(self, image: Any) -> bytearray:
        """Convert a PIL Image to raw 16-bit big-endian RGB565 bytearray."""
        if hasattr(image, "convert"):
            if image.mode != "RGB":
                image = image.convert("RGB")
            if image.size != (self.width, self.height):
                image = image.resize((self.width, self.height))
            pixels = image.getdata()
        else:
            return bytearray(self.width * self.height * 2)

        buffer = bytearray(self.width * self.height * 2)
        idx = 0
        for r, g, b in pixels:
            # RGB565: r(5 bits), g(6 bits), b(5 bits)
            val = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
            buffer[idx] = (val >> 8) & 0xFF
            buffer[idx + 1] = val & 0xFF
            idx += 2
        return buffer

    def _write_frame_to_device(self, spi_dev: Any, gpio: GPIOController, raw_bytes: bytearray) -> None:
        """Stream RGB565 frame to display via RAMWR (0x2C)."""
        if not spi_dev or gpio.backend is None:
            return
        # RAMWR command (DC LOW)
        self._send_command(spi_dev, gpio, 0x2C)
        # Pixel data stream (DC HIGH)
        gpio.set_pin(gpio.dc_pin, True)

        chunk_size = 4096
        use_writebytes2 = hasattr(spi_dev, "writebytes2")
        for i in range(0, len(raw_bytes), chunk_size):
            chunk = raw_bytes[i : i + chunk_size]
            if use_writebytes2:
                spi_dev.writebytes2(chunk)
            else:
                spi_dev.writebytes(list(chunk))

    def draw_eyes(self, left_image: Any, right_image: Any) -> None:
        """Transfer rendered eye image frames to the hardware displays."""
        if not self._is_hardware:
            return

        try:
            # Left eye -> SPI CE0 with left GPIO controller
            if self._spi_left is not None and self._gpio_left is not None:
                left_bytes = self._image_to_rgb565_bytearray(left_image)
                self._write_frame_to_device(self._spi_left, self._gpio_left, left_bytes)

            # Right eye -> SPI CE1 with right GPIO controller
            if self._spi_right is not None and self._gpio_right is not None and right_image is not None:
                right_bytes = self._image_to_rgb565_bytearray(right_image)
                self._write_frame_to_device(self._spi_right, self._gpio_right, right_bytes)

        except Exception as e:
            logger.error(f"SPI frame write error: {e}")

    def set_brightness(self, level_percent: int) -> None:
        """Backlight PWM control (placeholder for BL pin PWM)."""
        pass

    def clear(self) -> None:
        """Clear both displays to black."""
        if not self._is_hardware:
            return
        black_buf = bytearray(self.width * self.height * 2)
        if self._spi_left and self._gpio_left:
            self._write_frame_to_device(self._spi_left, self._gpio_left, black_buf)
        if self._spi_right and self._gpio_right:
            self._write_frame_to_device(self._spi_right, self._gpio_right, black_buf)

    def shutdown(self) -> None:
        """Close SPI buses and release GPIO handles."""
        if self._is_hardware:
            try:
                self.clear()
            except Exception:
                pass
            try:
                if self._spi_left:
                    self._spi_left.close()
                if self._spi_right:
                    self._spi_right.close()
                if self._gpio_left:
                    self._gpio_left.cleanup()
                if self._gpio_right:
                    self._gpio_right.cleanup()
            except Exception:
                pass
        logger.info("GC9A01 display driver shut down cleanly.")
