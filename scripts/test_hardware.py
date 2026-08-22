#!/usr/bin/env python3
"""LUMI Hardware Diagnostics and Calibration Tool.

Run this script directly on Raspberry Pi 5 to individually test:
1. GC9A01 Display (Clear screen & render procedural animated eyes)
2. MAX98357A I2S Audio (Play clean Bengali speech without white noise)
3. PCA9685 Servos (Smooth calibration sweep)
4. Phone IP Webcam (Test video & audio network streaming)
"""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lumi.audio.speaker import I2SSpeakerBackend
from lumi.core.logger import setup_logger
from lumi.eyes.expressions import EXPRESSIONS
from lumi.eyes.renderer import EyeRenderer
from lumi.hardware.display_driver import GC9A01DisplayDriver
from lumi.hardware.servo_driver import PCA9685ServoDriver
from lumi.speech.tts import BanglaTTS

setup_logger(name="diag", log_level="INFO")


def test_display():
    print("\n" + "=" * 50)
    print("TEST 1: GC9A01 SPI Round LCD Display")
    print("=" * 50)
    driver = GC9A01DisplayDriver(spi_speed_hz=40_000_000, width=240, height=240)
    success = driver.initialize()
    if not success or not driver._is_hardware:
        print("❌ Display hardware initialization failed.")
        print("   Make sure SPI is enabled: sudo raspi-config nonint do_spi 0")
        print("   Check wiring: DC=Pin18(G24), RST=Pin22(G25), CS=Pin24(CE0), SCLK=Pin23, MOSI=Pin19")
        return

    print("✅ Display driver initialized. Rendering test eyes...")
    renderer = EyeRenderer(display_backend=driver, single_display_both_eyes=True)
    renderer.start()

    expressions = ["happy", "curious", "thinking", "sleepy", "neutral"]
    for expr in expressions:
        print(f"   Setting expression: '{expr}'")
        renderer.set_expression(expr)
        time.sleep(1.5)

    renderer.stop()
    print("✅ Display test complete.")


def test_speaker():
    print("\n" + "=" * 50)
    print("TEST 2: MAX98357A I2S Amplifier Audio")
    print("=" * 50)
    tts = BanglaTTS()
    test_text = "আসসালামু আলাইকুম পলাশ ভাই! আমি লুমি, আপনার এআই রোবট। আমার স্পিকার এখন পরিষ্কারভাবে কাজ করছে।"
    print("   Synthesizing Bengali speech...")
    audio_path = tts.synthesize(test_text)
    if not audio_path:
        print("❌ TTS synthesis failed.")
        return

    print(f"   Audio saved at: {audio_path}")
    print("   Playing through MAX98357A I2S DAC...")
    speaker = I2SSpeakerBackend(volume=85)
    speaker.play_audio_file(audio_path, block=True)
    print("✅ Speaker test complete.")


def test_servos():
    print("\n" + "=" * 50)
    print("TEST 3: PCA9685 Servo Controller (I2C 0x40)")
    print("=" * 50)
    driver = PCA9685ServoDriver(i2c_bus=1, i2c_address=0x40)
    if not driver.initialize():
        print("❌ PCA9685 not detected on I2C bus 1 address 0x40.")
        return

    print("✅ PCA9685 online. Testing servo channels 0-3 (Head & Arms)...")
    # Home position (90 deg)
    for ch in range(4):
        driver.set_angle(ch, 90.0)
    time.sleep(0.5)

    # Gentle head sweep (ch 0)
    print("   Testing Head Pan (Channel 0)...")
    driver.set_angle(0, 70.0)
    time.sleep(0.4)
    driver.set_angle(0, 110.0)
    time.sleep(0.4)
    driver.set_angle(0, 90.0)
    time.sleep(0.3)

    # Arm wave (ch 2 & 3)
    print("   Testing Left/Right Arms (Channels 2 & 3)...")
    driver.set_angle(2, 120.0)
    driver.set_angle(3, 60.0)
    time.sleep(0.5)
    driver.set_angle(2, 90.0)
    driver.set_angle(3, 90.0)
    time.sleep(0.3)

    driver.shutdown()
    print("✅ Servos test complete.")


def main():
    print("LUMI Hardware Diagnostic Suite")
    print("1. Test Display (Eyes on GC9A01)")
    print("2. Test Speaker (MAX98357A I2S audio)")
    print("3. Test Servos (PCA9685)")
    print("4. Run ALL Tests")

    choice = input("Enter choice (1/2/3/4) [default: 4]: ").strip() or "4"
    if choice == "1":
        test_display()
    elif choice == "2":
        test_speaker()
    elif choice == "3":
        test_servos()
    else:
        test_display()
        test_speaker()
        test_servos()


if __name__ == "__main__":
    main()
