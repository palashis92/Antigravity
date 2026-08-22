# LUMI — AI Companion Robot

LUMI is an expressive, socially interactive physical AI companion robot powered by Raspberry Pi 5.

## Confirmed Hardware Setup
- **Compute**: Raspberry Pi 5 (4GB RAM)
- **Eyes**: Dual GC9A01 1.28" Round IPS LCDs (240×240, SPI interface)
- **Audio Output**: MAX98357A I2S Mono Class-D Amplifier
- **Vision (Phased)**: Phone Camera (Dev) $\rightarrow$ USB Webcam $\rightarrow$ Raspberry Pi Camera Module v2/v3
- **Microphone (Phased)**: Phone Mic (Dev) $\rightarrow$ ReSpeaker 2-Mic Pi HAT

---

## Phase 1 Deliverables (Foundation)
- **Configuration Engine (`lumi/config/`)**: Type-safe settings with YAML and `.env` support.
- **Structured Logging (`lumi/core/logger.py`)**: Console and rotating file logging.
- **State Machine (`lumi/core/state_manager.py`)**: 12 behavior states with transition guards.
- **Event Bus (`lumi/core/event_bus.py`)**: Thread-safe publish/subscribe asynchronous event router.
- **Database & Memory (`lumi/memory/`)**: SQLite WAL database with explicit privacy consent enforcement.
- **Hardware Abstraction Layer (`lumi/hardware/`)**: Abstract HAL base classes and mock drivers.
- **Application Orchestrator (`lumi/main.py`)**: 18-step validated boot sequence and lifecycle manager.

---

## Running Locally

### 1. Run Foundation Simulation
```powershell
python -m lumi.main --simulate
```

### 2. Run Automated Unit Tests
```powershell
python -m pytest tests/ -v
```
