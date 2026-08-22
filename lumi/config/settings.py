"""Type-safe configuration engine for LUMI."""

from __future__ import annotations

import os
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class AppConfig:
    name: str = "LUMI"
    version: str = "0.1.0"
    environment: str = "development"  # "development" | "production"
    log_level: str = "INFO"
    data_dir: str = "data"
    primary_language: str = "bn"
    fallback_language: str = "en"
    owner_name: str = "Palash"


@dataclass
class MemoryConfig:
    db_path: str = "data/lumi.db"
    wal_mode: bool = True
    backup_interval_hours: int = 24
    auto_consent_prompt: bool = True
    max_conversation_history: int = 100


@dataclass
class AudioConfig:
    mic_backend: str = "mock"
    speaker_backend: str = "mock"
    phone_audio_url: str = "http://192.168.1.100:8080/audio.wav"
    sample_rate: int = 16000
    channels: int = 1
    volume: int = 85
    stt_provider: str = "cloud"
    tts_provider: str = "cloud"


@dataclass
class DisplayConfig:
    backend: str = "mock"
    target_fps: int = 30
    width: int = 240
    height: int = 240
    dual_eyes: bool = True
    single_display_both_eyes: bool = False
    spi_speed_hz: int = 40000000


@dataclass
class MotionConfig:
    backend: str = "mock"
    smoothing_factor: float = 0.15
    head_enabled: bool = True
    arms_enabled: bool = True


@dataclass
class VisionConfig:
    camera_backend: str = "mock"
    phone_video_url: str = "http://192.168.1.100:8080/video"
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 15
    face_recognition_enabled: bool = True
    object_detection_enabled: bool = True


@dataclass
class HardwareConfig:
    servo_driver_model: str = "PCA9685"
    i2c_bus: int = 1
    i2c_address: int = 0x40
    pwm_frequency_hz: int = 50
    channels: Dict[str, Any] = field(default_factory=dict)
    displays: Dict[str, Any] = field(default_factory=dict)
    audio_i2s: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LumiSettings:
    app: AppConfig = field(default_factory=AppConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    def ensure_directories(self) -> None:
        """Create configured data directories if they do not exist."""
        data_path = Path(self.app.data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        db_path = Path(self.memory.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        logs_path = Path("logs")
        logs_path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8")
    if _HAS_YAML:
        try:
            loaded = yaml.safe_load(content)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
    # Fallback to json if formatted or key-value
    try:
        return json.loads(content)
    except Exception:
        return {}


def load_settings(
    config_path: Optional[str | Path] = None,
    hardware_config_path: Optional[str | Path] = None,
) -> LumiSettings:
    """Load settings from YAML/JSON files, merged with environment variables."""
    root_dir = Path(__file__).resolve().parent.parent

    # Default paths
    if config_path is None:
        config_path = root_dir / "config" / "default_config.yaml"
    else:
        config_path = Path(config_path)

    if hardware_config_path is None:
        hardware_config_path = root_dir / "config" / "hardware_config.yaml"
    else:
        hardware_config_path = Path(hardware_config_path)

    raw_main = _load_yaml_or_json(config_path)
    raw_hw = _load_yaml_or_json(hardware_config_path)

    app_data = raw_main.get("app", {})
    mem_data = raw_main.get("memory", {})
    audio_data = raw_main.get("audio", {})
    disp_data = raw_main.get("display", {})
    motion_data = raw_main.get("motion", {})
    vision_data = raw_main.get("vision", {})

    # Environment variable overrides
    env_mode = os.getenv("LUMI_ENV", app_data.get("environment", "development"))
    env_log = os.getenv("LUMI_LOG_LEVEL", app_data.get("log_level", "INFO"))
    env_data_dir = os.getenv("LUMI_DATA_DIR", app_data.get("data_dir", "data"))
    env_db_path = os.getenv("LUMI_DB_PATH", mem_data.get("db_path", f"{env_data_dir}/lumi.db"))

    app_cfg = AppConfig(
        name=app_data.get("name", "LUMI"),
        version=app_data.get("version", "0.1.0"),
        environment=env_mode,
        log_level=env_log,
        data_dir=env_data_dir,
        primary_language=app_data.get("primary_language", "bn"),
        fallback_language=app_data.get("fallback_language", "en"),
        owner_name=app_data.get("owner_name", "Palash"),
    )

    mem_cfg = MemoryConfig(
        db_path=env_db_path,
        wal_mode=mem_data.get("wal_mode", True),
        backup_interval_hours=mem_data.get("backup_interval_hours", 24),
        auto_consent_prompt=mem_data.get("auto_consent_prompt", True),
        max_conversation_history=mem_data.get("max_conversation_history", 100),
    )

    audio_cfg = AudioConfig(
        mic_backend=audio_data.get("mic_backend", "mock"),
        speaker_backend=audio_data.get("speaker_backend", "mock"),
        phone_audio_url=audio_data.get("phone_audio_url", "http://192.168.1.100:8080/audio.wav"),
        sample_rate=audio_data.get("sample_rate", 16000),
        channels=audio_data.get("channels", 1),
        volume=audio_data.get("volume", 85),
        stt_provider=audio_data.get("stt_provider", "cloud"),
        tts_provider=audio_data.get("tts_provider", "cloud"),
    )

    disp_cfg = DisplayConfig(
        backend=disp_data.get("backend", "mock"),
        target_fps=disp_data.get("target_fps", 30),
        width=disp_data.get("width", 240),
        height=disp_data.get("height", 240),
        dual_eyes=disp_data.get("dual_eyes", True),
        single_display_both_eyes=disp_data.get("single_display_both_eyes", False),
        spi_speed_hz=disp_data.get("spi_speed_hz", 40000000),
    )

    motion_cfg = MotionConfig(
        backend=motion_data.get("backend", "mock"),
        smoothing_factor=motion_data.get("smoothing_factor", 0.15),
        head_enabled=motion_data.get("head_enabled", True),
        arms_enabled=motion_data.get("arms_enabled", True),
    )

    vision_cfg = VisionConfig(
        camera_backend=vision_data.get("camera_backend", "mock"),
        phone_video_url=vision_data.get("phone_video_url", "http://192.168.1.100:8080/video"),
        frame_width=vision_data.get("frame_width", 640),
        frame_height=vision_data.get("frame_height", 480),
        fps=vision_data.get("fps", 15),
        face_recognition_enabled=vision_data.get("face_recognition_enabled", True),
        object_detection_enabled=vision_data.get("object_detection_enabled", True),
    )

    hw_driver = raw_hw.get("servo_driver", {})
    hw_cfg = HardwareConfig(
        servo_driver_model=hw_driver.get("model", "PCA9685"),
        i2c_bus=hw_driver.get("i2c_bus", 1),
        i2c_address=hw_driver.get("i2c_address", 0x40),
        pwm_frequency_hz=hw_driver.get("pwm_frequency_hz", 50),
        channels=raw_hw.get("channels", {}),
        displays=raw_hw.get("displays", {}),
        audio_i2s=raw_hw.get("audio_i2s", {}),
    )

    settings = LumiSettings(
        app=app_cfg,
        memory=mem_cfg,
        audio=audio_cfg,
        display=disp_cfg,
        motion=motion_cfg,
        vision=vision_cfg,
        hardware=hw_cfg,
    )
    settings.ensure_directories()
    return settings
