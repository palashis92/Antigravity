# Directive: Simulate Robot Motion

## Purpose
Provide a visual, interactive simulation environment for LUMI's 6-channel servo motion system. This allows developers and users to verify kinematics, gestures, coordinated multi-joint movements, angle constraints, and power-cut recovery calibration without requiring physical hardware connected.

## Architecture
- **Layer 1 (Directive)**: This file defines the simulation capabilities, controls, and testing procedures.
- **Layer 2 (Orchestration)**: Antigravity/Developer commands or test automations triggering motion scenarios.
- **Layer 3 (Execution)**: `execution/simulate_movement.py` (and root wrapper `simulate_movement.py`).

## Capabilities
1. **Interactive Dual-Projection Visualization**:
   - Front View: Head tilt, waist rotation compass, 2-DOF Left and Right arms with elevation (X) and reach (Y).
   - Side View: Pitch profile for head tilt and arm front/back reach.
2. **Real Kinematics Stack**:
   - Directly executes `lumi.motion.servo_controller.ServoController` with 50Hz cubic easing interpolation.
   - Tests real `HeadController`, `ArmController`, and `GestureManager` choreographies.
3. **Interactive Gesture Suite**:
   - Greet, Wave, Happy, Celebrate, Dance, Thinking, Curious, Excited, Sleep, Bored, Home.
4. **Manual Slider Control**:
   - Direct angle control for all 6 ground-truth calibrated channels.
5. **Power-Cut & Recovery Testing**:
   - Simulate sudden power loss mid-motion.
   - Simulate reboot homing (`startup_self_check_and_home`) to visually verify soft calibration recovery.
6. **Web Mode Fallback**:
   - Run with `--web [port]` to access browser-based interactive simulation on headless machines (e.g. Raspberry Pi via SSH).

## How to Run
```bash
# Graphical Desktop GUI (Tkinter)
python simulate_movement.py

# Web Browser Dashboard (Headless / Remote Pi)
python simulate_movement.py --web 8080
```
