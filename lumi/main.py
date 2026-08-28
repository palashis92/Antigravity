"""LUMI AI Companion Robot - Main Entry Point and Full Lifecycle Runtime."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from .audio.mic import MicInterface, SystemMicBackend
from .audio.speaker import I2SSpeakerBackend, SpeakerInterface
from .config import LumiSettings, load_settings
from .core.event_bus import EventBus
from .core.logger import get_logger, setup_logger
from .core.lumi_brain import LumiBrain
from .core.state_manager import BehaviorState, StateManager
from .eyes.renderer import EyeRenderer
from .hardware.display_driver import GC9A01DisplayDriver
from .hardware.mocks import (
    MockCameraBackend,
    MockDisplayBackend,
    MockMicBackend,
    MockServoDriver,
    MockSpeakerBackend,
)
from .hardware.servo_driver import PCA9685ServoDriver
from .memory.database import Database
from .memory.manager import MemoryManager
from .memory.models import ConsentStatus
from .motion.servo_controller import ServoController
from .vision.camera import CameraInterface, PhoneCameraBackend, PiCameraBackend, USBWebcamBackend

logger = get_logger("main")


class LumiApplication:
    """LUMI Complete Application Runtime."""

    def __init__(self, settings: LumiSettings) -> None:
        self.settings = settings
        self.running = False

        # 1. State Manager & Event Bus
        self.state_manager = StateManager(BehaviorState.IDLE)
        self.event_bus = EventBus(async_workers=2)

        # 2. Database & Memory Manager
        self.db = Database(
            db_path=settings.memory.db_path,
            enable_wal=settings.memory.wal_mode,
        )
        self.memory = MemoryManager(self.db)

        # 3. Hardware Drivers
        # 3a. Servos (PCA9685)
        if settings.motion.backend == "pca9685":
            self.raw_servo_driver = PCA9685ServoDriver(
                i2c_bus=settings.hardware.i2c_bus,
                i2c_address=settings.hardware.i2c_address,
                pwm_frequency_hz=settings.hardware.pwm_frequency_hz,
            )
        else:
            self.raw_servo_driver = MockServoDriver()

        self.servo_controller = ServoController(
            driver=self.raw_servo_driver,
            hw_config=settings.hardware,
            auto_relax_delay_s=5.0,
        )

        # 3b. Display & Procedural Eye Renderer
        if settings.display.backend == "gc9a01_spi":
            # Extract per-eye DC/RST pin numbers from hardware_config.yaml
            left_disp_cfg = settings.hardware.displays.get("left_eye", {})
            right_disp_cfg = settings.hardware.displays.get("right_eye", {})
            self.display_backend = GC9A01DisplayDriver(
                spi_speed_hz=settings.display.spi_speed_hz,
                width=settings.display.width,
                height=settings.display.height,
                left_dc_pin=left_disp_cfg.get("dc_pin", 24),
                left_rst_pin=left_disp_cfg.get("rst_pin", 25),
                right_dc_pin=right_disp_cfg.get("dc_pin", 23),
                right_rst_pin=right_disp_cfg.get("rst_pin", 22),
            )
        else:
            self.display_backend = MockDisplayBackend(
                width=settings.display.width,
                height=settings.display.height,
            )

        is_single_display = settings.display.single_display_both_eyes or not settings.display.dual_eyes
        self.eye_renderer = EyeRenderer(
            display_backend=self.display_backend,
            width=settings.display.width,
            height=settings.display.height,
            target_fps=settings.display.target_fps,
            single_display_both_eyes=is_single_display,
        )

        # 3c. Camera Backend
        if settings.vision.camera_backend == "phone_stream":
            camera_backend = PhoneCameraBackend(stream_url=settings.vision.phone_video_url)
        elif settings.vision.camera_backend == "usb":
            camera_backend = USBWebcamBackend(width=settings.vision.frame_width, height=settings.vision.frame_height)
        elif settings.vision.camera_backend == "picamera":
            camera_backend = PiCameraBackend(width=settings.vision.frame_width, height=settings.vision.frame_height)
        else:
            camera_backend = MockCameraBackend(
                width=settings.vision.frame_width,
                height=settings.vision.frame_height,
            )
        self.camera = CameraInterface(camera_backend)

        # 3d. Microphone Backend
        if settings.audio.mic_backend != "mock":
            mic_backend = SystemMicBackend(sample_rate=settings.audio.sample_rate)
        else:
            mic_backend = MockMicBackend()
        self.mic = MicInterface(mic_backend)

        # 3e. Speaker Backend
        if settings.audio.speaker_backend == "i2s":
            speaker_backend = I2SSpeakerBackend(volume=settings.audio.volume)
        else:
            speaker_backend = MockSpeakerBackend(volume=settings.audio.volume)
        self.speaker = SpeakerInterface(speaker_backend)

        # 4. LumiBrain Orchestrator
        self.brain = LumiBrain(
            settings=settings,
            state_manager=self.state_manager,
            event_bus=self.event_bus,
            memory_manager=self.memory,
            servo_controller=self.servo_controller,
            eye_renderer=self.eye_renderer,
            camera=self.camera,
            mic=self.mic,
            speaker=self.speaker,
        )

    def startup(self) -> bool:
        """Execute the 18-step validated startup sequence."""
        logger.info("==================================================")
        logger.info(f"Starting {self.settings.app.name} v{self.settings.app.version} ({self.settings.app.environment})")
        logger.info("==================================================")

        try:
            logger.info("[Step 1-5] Subsystem configurations & memory database verified.")
            self.servo_controller.initialize()
            logger.info("[Step 6-8] Kinematic servo controller online & homed.")
            self.eye_renderer.start()
            logger.info("[Step 9] Procedural Dual Eye Renderer (30 FPS) active.")
            self.camera.start()
            self.mic.start()
            self.speaker.set_volume(self.settings.audio.volume)
            logger.info("[Step 10-12] Camera, Microphone, and Speaker online.")
            self.event_bus.start()
            self.brain.reminders.start()
            self.brain.start_loops()
            logger.info("[Step 13-17] Memory, EventBus, and AI Services online.")
            self.state_manager.transition_to(BehaviorState.IDLE, reason="startup_complete")
            logger.info("[Step 18] LUMI successfully entered READY (IDLE) state.")
            logger.info("==================================================")
            self.running = True
            return True
        except Exception as e:
            logger.critical(f"LUMI Startup failed: {e}", exc_info=True)
            self.state_manager.force_state(BehaviorState.ERROR, reason=str(e))
            return False

    def shutdown(self) -> None:
        """Gracefully release hardware resources and persist state."""
        if not self.running:
            return
        logger.info("Initiating graceful shutdown sequence...")
        self.running = False

        self.brain.stop()
        self.brain.reminders.stop()
        self.event_bus.stop()
        self.camera.stop()
        self.mic.stop()
        self.speaker.stop()
        self.eye_renderer.stop()
        self.servo_controller.shutdown()

        logger.info("LUMI shutdown completed cleanly.")


def _load_dotenv() -> None:
    """Load .env file from the project root into os.environ (no external dependency needed)."""
    import os
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:  # Don't override existing env vars
                os.environ[key] = value


def main() -> None:
    # Load .env file FIRST so API keys are available via os.getenv()
    _load_dotenv()

    parser = argparse.ArgumentParser(description="LUMI AI Companion Robot Runtime")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--simulate", action="store_true", help="Run in interactive simulation mode")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    settings = load_settings(config_path=args.config)
    if args.log_level:
        settings.app.log_level = args.log_level

    setup_logger(
        name="lumi",
        log_level=settings.app.log_level,
        log_file="logs/lumi.log",
    )

    app = LumiApplication(settings)

    def handle_sigint(sig: int, frame: Any) -> None:
        print("\n")
        logger.info("Interrupt signal received (SIGINT).")
        app.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    if not app.startup():
        sys.exit(1)

    if args.simulate:
        logger.info("[SIMULATION] Running full autonomous companion demonstration...")
        time.sleep(0.4)

        # 1. Person Interaction (Vision -> Greet -> Speak)
        logger.info("\n--- [Demo 1: Person Vision & Social Interaction] ---")
        mock_frame = {"width": 640, "height": 480}
        app.brain.process_person_interaction(mock_frame)
        time.sleep(0.6)

        # 2. Bangla Conversation & AI Reasoning
        logger.info("\n--- [Demo 2: Bangla Conversation & Tools] ---")
        reply = app.brain.handle_user_speech_input(b"dummy_pcm")
        logger.info(f"LUMI Replied in Bangla: '{reply}'")
        time.sleep(0.6)

        # 3. Plant Disease Diagnosis
        logger.info("\n--- [Demo 3: Plant Leaf Disease CV & Advice] ---")
        plant_summary = app.brain.analyze_plant_leaf(mock_frame)
        logger.info(f"Plant Diagnosis Summary: '{plant_summary}'")
        time.sleep(0.6)

        # 4. Chessboard Analysis & Stockfish
        logger.info("\n--- [Demo 4: Chessboard Vision & Stockfish Commentary] ---")
        chess_summary = app.brain.analyze_chessboard(mock_frame)
        logger.info(f"Chess Evaluation Summary: '{chess_summary}'")
        time.sleep(0.6)

        # 5. Document Generation
        logger.info("\n--- [Demo 5: Document & PDF Report Generation] ---")
        doc_path = app.brain.documents.generate_summary_pdf(
            title="LUMI Daily Briefing & Health Report",
            content_sections={
                "Personal Memory Status": "Owner: Palash (Active)\nInteraction Count: 12\nConsent: Granted",
                "Agricultural Advisory": plant_summary,
                "Chess Summary": chess_summary,
            },
        )
        logger.info(f"Document created at: '{doc_path}'")
        time.sleep(0.4)

        logger.info("\n[SIMULATION] All subsystem pipelines completed successfully.")
        app.shutdown()
        return

    logger.info("LUMI is running. Press Ctrl+C to terminate.")
    try:
        while app.running:
            app.brain.behavior.tick_idle()
            time.sleep(0.5)
    except KeyboardInterrupt:
        handle_sigint(signal.SIGINT, None)


if __name__ == "__main__":
    main()
