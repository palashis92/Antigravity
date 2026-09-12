"""LUMI Robot Motion Visual Simulator.

Provides both:
1. A rich desktop GUI (Tkinter) with Front View, Side Profile, Gauges, and Gesture triggers.
2. An interactive Web Dashboard (HTML5 Canvas) for headless/remote use (e.g. Raspberry Pi).

Usage:
    python simulate_movement.py          # Launches desktop GUI
    python simulate_movement.py --web    # Launches web browser dashboard (port 8080)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lumi.hardware.mocks import MockServoDriver
from lumi.motion.arms import ArmController
from lumi.motion.gestures import GestureManager
from lumi.motion.head import HeadController
from lumi.motion.servo_controller import ServoController


class SimulationBackend:
    """Core kinematic backend wrapping the actual LUMI motion stack with telemetry hooks."""

    def __init__(self, force_mock: bool = False) -> None:
        self.is_real_hardware = False
        if not force_mock:
            try:
                from lumi.hardware.servo_driver import PCA9685ServoDriver
                hw = PCA9685ServoDriver()
                if hw.initialize() and hw._is_hardware:
                    self.driver = hw
                    self.is_real_hardware = True
                else:
                    self.driver = MockServoDriver()
            except Exception:
                self.driver = MockServoDriver()
        else:
            self.driver = MockServoDriver()

        self.controller = ServoController(self.driver, auto_relax_delay_s=10.0)
        self.head = HeadController(self.controller)
        self.arms = ArmController(self.controller)
        self.gestures = GestureManager(self.controller, self.head, self.arms)

        # Real-time state telemetry: channel -> current angle
        self.angles: Dict[int, float] = {
            0: 0.0,  # Head tilt
            1: 0.0,  # Right arm Y
            2: 0.0,  # Left arm Y
            3: 0.0,  # Right arm X
            4: 0.0,  # Left arm X
            5: 0.0,  # Body / waist
        }
        self.current_expression: str = "happy"
        self.active_gesture: str = "IDLE"
        self.logs: list[str] = []
        self._lock = threading.RLock()

        # Intercept driver set_angle to capture 50Hz kinematic frames
        orig_set_angle = getattr(self.driver, "set_angle", None)
        if callable(orig_set_angle):
            def _intercept_set_angle(channel: int, angle: float) -> None:
                orig_set_angle(channel, angle)
                with self._lock:
                    self.angles[channel] = round(float(angle), 2)
            self.driver.set_angle = _intercept_set_angle

        # Start controller background loop
        self.controller.initialize()
        mode_str = "REAL HARDWARE (PCA9685)" if self.is_real_hardware else "VIRTUAL SIMULATION"
        self.log(f"Simulation backend online in [{mode_str}] mode.")

    def run_full_demo(self, print_fn=print) -> None:
        """Execute a comprehensive, step-by-step motion demonstration."""
        mode_str = "REAL PCA9685 HARDWARE" if self.is_real_hardware else "VIRTUAL SIMULATION"
        print_fn("\n=======================================================")
        print_fn(f"🤖 LUMI ROBOT FULL MOTION DEMO")
        print_fn(f"   Mode: {mode_str}")
        print_fn("=======================================================\n")

        # Step 1: Head Tilt
        print_fn("[1/7] Head Tilt Test (Ch 0)...")
        print_fn("      -> Looking UP (-15°)...")
        self.head.look_up(15.0, duration_s=0.35)
        time.sleep(0.4)
        print_fn("      -> Looking DOWN (+15°)...")
        self.head.look_down(15.0, duration_s=0.35)
        time.sleep(0.4)
        print_fn("      -> Centering Head (0°)...")
        self.head.look_center(duration_s=0.3)
        time.sleep(0.3)

        # Step 2: Body Waist
        print_fn("[2/7] Body Waist Rotation (Ch 5)...")
        print_fn("      -> Turning RIGHT (-45°)...")
        self.head.look_right(45.0, duration_s=0.4)
        time.sleep(0.4)
        print_fn("      -> Turning LEFT (+45°)...")
        self.head.look_left(45.0, duration_s=0.4)
        time.sleep(0.4)
        print_fn("      -> Returning Waist to Center (0°)...")
        self.head.look_center(duration_s=0.3)
        time.sleep(0.3)

        # Step 3: Right Arm
        print_fn("[3/7] Right Arm Test (Ch 1 Reach, Ch 3 Lift)...")
        print_fn("      -> Raising Right Arm (-25° X)...")
        self.arms.raise_right(duration_s=0.3)
        time.sleep(0.3)
        print_fn("      -> Waving Right Arm...")
        self.arms.wave_right(count=2)
        time.sleep(0.3)

        # Step 4: Left Arm
        print_fn("[4/7] Left Arm Test (Ch 2 Reach, Ch 4 Lift)...")
        print_fn("      -> Raising Left Arm (+25° X)...")
        self.arms.raise_left(duration_s=0.3)
        time.sleep(0.3)
        print_fn("      -> Waving Left Arm...")
        self.arms.wave_left(count=2)
        time.sleep(0.3)

        # Step 5: Expressive Choreographies
        print_fn("[5/7] Expressive Gestures...")
        print_fn("      -> Greet Gesture...")
        self.gestures.greet()
        time.sleep(0.3)
        print_fn("      -> Happy Gesture...")
        self.gestures.happy()
        time.sleep(0.3)
        print_fn("      -> Dance Gesture...")
        self.gestures.dance()
        time.sleep(0.3)

        # Step 6: Coordinated Simultaneous Motion
        print_fn("[6/7] Simultaneous 6-Channel Parallel Motion...")
        print_fn("      -> Both Arms Raised + Head Tilted...")
        self.controller.move_multiple(
            {
                "head_tilt": -10.0,
                "head_pan": 0.0,
                "left_arm_x": 20.0,
                "left_arm_y": 25.0,
                "right_arm_x": -20.0,
                "right_arm_y": -25.0,
            },
            duration_s=0.35,
        )
        time.sleep(0.5)

        # Step 7: Homing
        print_fn("[7/7] Returning all channels to safe Home (0.0°)...")
        self.home_all(duration_s=0.35)
        time.sleep(0.4)
        print_fn("\n✅ FULL MOTION DEMO COMPLETED SUCCESSFULLY!\n")

    def get_angles(self) -> Dict[int, float]:
        """Fetch canonical physical angles directly from the kinematic controller."""
        with self.controller._lock:
            return {
                0: round(float(self.controller.current_angles.get("head_tilt", 0.0)), 1),
                1: round(float(self.controller.current_angles.get("right_arm_y", 0.0)), 1),
                2: round(float(self.controller.current_angles.get("left_arm_y", 0.0)), 1),
                3: round(float(self.controller.current_angles.get("right_arm_x", 0.0)), 1),
                4: round(float(self.controller.current_angles.get("left_arm_x", 0.0)), 1),
                5: round(float(self.controller.current_angles.get("head_pan", 0.0)), 1),
            }

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        with self._lock:
            self.logs.append(entry)
            if len(self.logs) > 50:
                self.logs.pop(0)

    def trigger_gesture(self, name: str) -> None:
        gesture_map = {
            "greet": (self.gestures.greet, "happy"),
            "wave": (self.gestures.wave, "happy"),
            "happy": (self.gestures.happy, "excited"),
            "celebrate": (self.gestures.celebrate, "excited"),
            "dance": (self.gestures.dance, "happy"),
            "thinking": (self.gestures.thinking, "thinking"),
            "curious": (self.gestures.curious, "curious"),
            "excited": (self.gestures.excited, "excited"),
            "sleep": (self.gestures.sleep, "sleep"),
            "bored": (self.gestures.bored, "bored"),
            "home": (self.home_all, "happy"),
        }
        if name in gesture_map:
            fn, expr = gesture_map[name]
            self.active_gesture = name.upper()
            self.current_expression = expr
            self.log(f"Triggering gesture: {name.upper()}")

            def _run():
                try:
                    fn()
                finally:
                    self.active_gesture = "IDLE"
                    self.current_expression = "happy"
                    self.log(f"Gesture {name.upper()} completed.")

            threading.Thread(target=_run, daemon=True).start()

    def home_all(self, duration_s: float = 0.3) -> None:
        self.controller.move_multiple(
            {
                "head_tilt": 0.0,
                "head_pan": 0.0,
                "left_arm_x": 0.0,
                "left_arm_y": 0.0,
                "right_arm_x": 0.0,
                "right_arm_y": 0.0,
            },
            duration_s=duration_s,
        )

    def set_single_joint(self, channel: int, angle: float, duration_s: float = 0.3) -> None:
        ch_to_name = {
            0: "head_tilt",
            1: "right_arm_y",
            2: "left_arm_y",
            3: "right_arm_x",
            4: "left_arm_x",
            5: "head_pan",
        }
        name = ch_to_name.get(channel)
        if name:
            self.log(f"Manual move: {name} (Ch {channel}) -> {angle:.1f}° ({duration_s}s)")
            self.controller.move_joint(name, angle, duration_s=duration_s)

    def simulate_power_cut(self) -> None:
        self.log("⚠️ SIMULATING POWER CUT! Servos frozen at current angles.")
        # Stop background interpolation thread immediately
        self.controller.shutdown()
        self.active_gesture = "POWER_CUT"
        self.current_expression = "sleep"

    def simulate_reboot_recovery(self) -> None:
        self.log("🔄 SIMULATING POWER REBOOT: Running startup self-check homing...")
        self.controller.initialize()
        self.active_gesture = "HOMING"
        self.current_expression = "happy"
        self.log("Homing complete. Servos safely calibrated to 0.0°.")


# =============================================================================
# Desktop Tkinter GUI Simulator
# =============================================================================

class DesktopSimulatorGUI:
    """Modern dark-themed Tkinter graphical simulator."""

    def __init__(self, backend: SimulationBackend) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.backend = backend
        self.root = tk.Tk()
        self.root.title("LUMI Robot Kinematic Motion Simulator")
        self.root.geometry("1120x760")
        self.root.minsize(980, 680)
        self.root.configure(bg="#0f172a")  # Dark slate

        self._setup_styles()
        self._build_layout()
        self._schedule_render_loop()

    def _setup_styles(self) -> None:
        from tkinter import ttk
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#0f172a")
        style.configure("Card.TFrame", background="#1e293b", relief="flat")
        style.configure("TLabel", background="#0f172a", foreground="#e2e8f0", font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#0f172a", foreground="#38bdf8", font=("Segoe UI", 12, "bold"))
        style.configure("SubHeader.TLabel", background="#1e293b", foreground="#94a3b8", font=("Segoe UI", 9))
        style.configure("Value.TLabel", background="#1e293b", foreground="#38bdf8", font=("Consolas", 10, "bold"))
        style.configure("TButton", font=("Segoe UI", 9, "bold"), background="#334155", foreground="#f8fafc")
        style.map("TButton", background=[("active", "#38bdf8"), ("pressed", "#0284c7")])

    def _build_layout(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        # Main Header
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=16, pady=(12, 8))
        title = tk.Label(
            header,
            text="🤖 LUMI ROBOT DUAL-PROJECTION MOTION SIMULATOR",
            font=("Segoe UI", 14, "bold"),
            fg="#38bdf8",
            bg="#0f172a",
        )
        title.pack(side="left")
        self.status_badge = tk.Label(
            header,
            text="STATUS: READY",
            font=("Consolas", 10, "bold"),
            fg="#22c55e",
            bg="#1e293b",
            padx=10,
            pady=3,
        )
        self.status_badge.pack(side="right")

        # Container
        main_box = ttk.Frame(self.root)
        main_box.pack(fill="both", expand=True, padx=16, pady=8)

        # Left Column: Visual Canvas (Front View + Side Profile)
        left_col = ttk.Frame(main_box)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        canvas_frame = tk.Frame(left_col, bg="#020617", bd=2, relief="groove")
        canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#020617", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Bottom of Left Column: Real-time Telemetry Gauges
        gauges_card = tk.Frame(left_col, bg="#1e293b", bd=1, relief="ridge", pady=6, padx=8)
        gauges_card.pack(fill="x", pady=(8, 0))

        gauges_title = tk.Label(
            gauges_card, text="REAL-TIME SERVO GAUGES (GROUND TRUTH)", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#1e293b"
        )
        gauges_title.pack(anchor="w", pady=(0, 4))

        self.gauge_labels: Dict[int, tk.Label] = {}
        grid_frame = tk.Frame(gauges_card, bg="#1e293b")
        grid_frame.pack(fill="x")

        labels_spec = [
            (0, "CH 0 Head Tilt", "[-15° .. +15°]"),
            (5, "CH 5 Body Waist", "[-90° .. +90°]"),
            (1, "CH 1 R-Arm Pitch (Y)", "[-60° .. +25°]"),
            (3, "CH 3 R-Arm Lift (X)", "[-25° .. +5°]"),
            (2, "CH 2 L-Arm Pitch (Y)", "[-25° .. +60°]"),
            (4, "CH 4 L-Arm Lift (X)", "[-5° .. +25°]"),
        ]
        for idx, (ch, name, limits) in enumerate(labels_spec):
            r = idx // 2
            c = (idx % 2) * 2
            lbl_title = tk.Label(grid_frame, text=f"{name}:", font=("Segoe UI", 9), fg="#cbd5e1", bg="#1e293b")
            lbl_title.grid(row=r, column=c, sticky="w", padx=4, pady=2)
            lbl_val = tk.Label(
                grid_frame, text=f" 0.0° {limits}", font=("Consolas", 9, "bold"), fg="#38bdf8", bg="#1e293b"
            )
            lbl_val.grid(row=r, column=c + 1, sticky="w", padx=(0, 16), pady=2)
            self.gauge_labels[ch] = lbl_val

        # Right Column: Controls
        right_col = ttk.Frame(main_box, width=380)
        right_col.pack(side="right", fill="both", padx=(10, 0))
        right_col.pack_propagate(False)

        # 1. Gestures Card
        gestures_card = tk.Frame(right_col, bg="#1e293b", bd=1, relief="ridge", padx=10, pady=8)
        gestures_card.pack(fill="x", pady=(0, 8))

        g_title = tk.Label(gestures_card, text="🎭 EXPRESSIVE GESTURES", font=("Segoe UI", 10, "bold"), fg="#38bdf8", bg="#1e293b")
        g_title.pack(anchor="w", pady=(0, 6))

        btn_grid = tk.Frame(gestures_card, bg="#1e293b")
        btn_grid.pack(fill="x")

        gestures = [
            ("👋 Greet", "greet"),
            ("🙋 Wave", "wave"),
            ("😄 Happy", "happy"),
            ("🎉 Celebrate", "celebrate"),
            ("🕺 Dance", "dance"),
            ("🤔 Thinking", "thinking"),
            ("🧐 Curious", "curious"),
            ("⚡ Excited", "excited"),
            ("😴 Sleep", "sleep"),
            ("😐 Bored", "bored"),
            ("🏠 Soft Home", "home"),
        ]
        for idx, (label, g_name) in enumerate(gestures):
            r = idx // 3
            c = idx % 3
            btn = tk.Button(
                btn_grid,
                text=label,
                font=("Segoe UI", 9, "bold"),
                bg="#334155",
                fg="#f8fafc",
                activebackground="#38bdf8",
                activeforeground="#0f172a",
                bd=0,
                padx=6,
                pady=4,
                command=lambda name=g_name: self.backend.trigger_gesture(name),
            )
            btn.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")
        for c in range(3):
            btn_grid.columnconfigure(c, weight=1)

        # 2. Manual Sliders Card
        sliders_card = tk.Frame(right_col, bg="#1e293b", bd=1, relief="ridge", padx=10, pady=8)
        sliders_card.pack(fill="x", pady=(0, 8))

        s_title = tk.Label(sliders_card, text="🎛️ MANUAL JOINT SLIDERS", font=("Segoe UI", 10, "bold"), fg="#38bdf8", bg="#1e293b")
        s_title.pack(anchor="w", pady=(0, 6))

        slider_specs = [
            (0, "Head Tilt (-15° up, +15° down)", -15.0, 15.0),
            (5, "Body Waist (-90° right, +90° left)", -90.0, 90.0),
            (1, "Right Arm Y (-60° front, +25° back)", -60.0, 25.0),
            (3, "Right Arm X (-25° up, +5° down)", -25.0, 5.0),
            (2, "Left Arm Y (-25° back, +60° front)", -25.0, 60.0),
            (4, "Left Arm X (-5° down, +25° up)", -5.0, 25.0),
        ]
        self.sliders: Dict[int, tk.Scale] = {}
        for ch, label, min_val, max_val in slider_specs:
            row = tk.Frame(sliders_card, bg="#1e293b")
            row.pack(fill="x", pady=1)
            lbl = tk.Label(row, text=label, font=("Segoe UI", 8), fg="#94a3b8", bg="#1e293b")
            lbl.pack(anchor="w")

            scale = tk.Scale(
                row,
                from_=min_val,
                to=max_val,
                orient="horizontal",
                resolution=1.0,
                bg="#1e293b",
                fg="#38bdf8",
                highlightthickness=0,
                troughcolor="#0f172a",
                activebackground="#38bdf8",
                command=lambda val, c=ch: self._on_slider_change(c, float(val)),
            )
            scale.set(0.0)
            scale.pack(fill="x")
            self.sliders[ch] = scale

        # 3. Power Cut & Homing Recovery Test Card
        power_card = tk.Frame(right_col, bg="#1e293b", bd=1, relief="ridge", padx=10, pady=8)
        power_card.pack(fill="x", pady=(0, 8))

        p_title = tk.Label(power_card, text="⚡ POWER-CUT & RECOVERY TEST", font=("Segoe UI", 10, "bold"), fg="#f59e0b", bg="#1e293b")
        p_title.pack(anchor="w", pady=(0, 4))

        p_desc = tk.Label(
            power_card,
            text="Test behavior when power is cut mid-move and rebooted with homing.",
            font=("Segoe UI", 8),
            fg="#94a3b8",
            bg="#1e293b",
        )
        p_desc.pack(anchor="w", pady=(0, 4))

        p_btns = tk.Frame(power_card, bg="#1e293b")
        p_btns.pack(fill="x")

        cut_btn = tk.Button(
            p_btns,
            text="⚡ Cut Power",
            font=("Segoe UI", 8, "bold"),
            bg="#ef4444",
            fg="white",
            activebackground="#dc2626",
            bd=0,
            padx=8,
            pady=4,
            command=self.backend.simulate_power_cut,
        )
        cut_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        home_btn = tk.Button(
            p_btns,
            text="🔄 Reboot & Soft Home",
            font=("Segoe UI", 8, "bold"),
            bg="#10b981",
            fg="white",
            activebackground="#059669",
            bd=0,
            padx=8,
            pady=4,
            command=self.backend.simulate_reboot_recovery,
        )
        home_btn.pack(side="right", expand=True, fill="x", padx=(4, 0))

        # 4. Event Log Console Card
        log_card = tk.Frame(right_col, bg="#1e293b", bd=1, relief="ridge", padx=10, pady=6)
        log_card.pack(fill="both", expand=True)

        l_title = tk.Label(log_card, text="📋 LIVE EVENT LOG", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#1e293b")
        l_title.pack(anchor="w")

        self.log_text = tk.Text(
            log_card,
            height=6,
            bg="#020617",
            fg="#a7f3d0",
            font=("Consolas", 8),
            bd=0,
            highlightthickness=0,
            state="disabled",
        )
        self.log_text.pack(fill="both", expand=True, pady=(4, 0))

    def _on_slider_change(self, channel: int, value: float) -> None:
        # Avoid circular slider updates while gesture is running
        if self.backend.active_gesture == "IDLE":
            self.backend.set_single_joint(channel, value, duration_s=0.25)

    def _schedule_render_loop(self) -> None:
        """Render loop running at ~40 FPS."""
        self._render_frame()
        self.root.after(25, self._schedule_render_loop)

    def _render_frame(self) -> None:
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 100 or h < 100:
            return

        self.canvas.delete("all")

        # Background grid
        self._draw_grid(w, h)

        # Split canvas into two viewports: Front View (65% width) and Side Profile (35% width)
        front_w = int(w * 0.65)
        side_w = w - front_w

        # Draw divider
        self.canvas.create_line(front_w, 20, front_w, h - 20, fill="#1e293b", width=2, dash=(4, 4))
        self.canvas.create_text(front_w // 2, 25, text="FRONT VIEW (WITH DUAL ARMS & SCREEN)", font=("Segoe UI", 10, "bold"), fill="#64748b")
        self.canvas.create_text(front_w + side_w // 2, 25, text="SIDE PROFILE (PITCH & REACH)", font=("Segoe UI", 10, "bold"), fill="#64748b")

        # Current Angles
        angles = self.backend.get_angles()
        with self.backend._lock:
            expr = self.backend.current_expression
            gesture = self.backend.active_gesture
            logs = list(self.backend.logs)

        # Render Front View
        self._draw_front_view(front_w // 2, int(h * 0.52), angles, expr)

        # Render Side Profile View
        self._draw_side_view(front_w + side_w // 2, int(h * 0.52), angles)

        # Update telemetry labels
        for ch, lbl in self.gauge_labels.items():
            ang = angles.get(ch, 0.0)
            limits = {
                0: "[-15° .. +15°]",
                5: "[-90° .. +90°]",
                1: "[-60° .. +25°]",
                3: "[-25° .. +5°]",
                2: "[-25° .. +60°]",
                4: "[-5° .. +25°]",
            }.get(ch, "")
            lbl.config(text=f"{ang:+5.1f}° {limits}")

        # Update status badge
        if gesture != "IDLE":
            self.status_badge.config(text=f"GESTURE: {gesture}", fg="#f59e0b")
        else:
            self.status_badge.config(text="STATUS: READY", fg="#22c55e")

        # Update log console
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        for line in logs[-6:]:
            self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _draw_grid(self, w: int, h: int) -> None:
        for x in range(0, w, 40):
            self.canvas.create_line(x, 0, x, h, fill="#0b1120", width=1)
        for y in range(0, h, 40):
            self.canvas.create_line(0, y, w, y, fill="#0b1120", width=1)

    def _draw_front_view(self, cx: int, cy: int, angles: Dict[int, float], expr: str) -> None:
        head_tilt = angles.get(0, 0.0)  # -15° up, +15° down
        waist_pan = angles.get(5, 0.0)  # -90° right, +90° left
        r_arm_x = angles.get(3, 0.0)    # -25° up, +5° down
        r_arm_y = angles.get(1, 0.0)    # -60° front, +25° back
        l_arm_x = angles.get(4, 0.0)    # +25° up, -5° down
        l_arm_y = angles.get(2, 0.0)    # +60° front, -25° back

        # 1. Base Platform
        base_y = cy + 130
        self.canvas.create_oval(cx - 90, base_y - 15, cx + 90, base_y + 15, fill="#1e293b", outline="#38bdf8", width=2)
        self.canvas.create_oval(cx - 60, base_y - 8, cx + 60, base_y + 8, fill="#0f172a", outline="#0284c7", width=1)

        # Waist rotation compass ring
        pan_rad = math.radians(-waist_pan)  # Inverted for visual compass
        nx = cx + math.sin(pan_rad) * 45
        ny = base_y + math.cos(pan_rad) * 10
        self.canvas.create_line(cx, base_y, nx, ny, fill="#f59e0b", width=3, arrow="last")
        self.canvas.create_text(cx, base_y + 24, text=f"WAIST: {waist_pan:+.1f}°", font=("Consolas", 8, "bold"), fill="#f59e0b")

        # 2. Torso
        torso_w, torso_h = 100, 110
        torso_x1 = cx - torso_w // 2
        torso_y1 = cy - 10
        torso_x2 = cx + torso_w // 2
        torso_y2 = torso_y1 + torso_h

        # Body twist perspective offset
        twist_offset = int(waist_pan * 0.25)
        self.canvas.create_rectangle(
            torso_x1 + twist_offset,
            torso_y1,
            torso_x2 + twist_offset,
            torso_y2,
            fill="#1e293b",
            outline="#475569",
            width=2,
        )
        # Chest power arc / LED badge
        self.canvas.create_oval(
            cx - 16 + twist_offset,
            torso_y1 + 25,
            cx + 16 + twist_offset,
            torso_y1 + 57,
            fill="#0369a1",
            outline="#38bdf8",
            width=2,
        )
        self.canvas.create_oval(
            cx - 6 + twist_offset,
            torso_y1 + 35,
            cx + 6 + twist_offset,
            torso_y1 + 47,
            fill="#38bdf8",
            outline="",
        )

        # 3. Arms (Right Arm - Viewer's Left, Left Arm - Viewer's Right)
        # Right Arm (CH 3 X lift, CH 1 Y reach)
        r_shoulder_x = torso_x1 + twist_offset + 5
        r_shoulder_y = torso_y1 + 18
        self._draw_arm_front(
            r_shoulder_x,
            r_shoulder_y,
            lift_deg=r_arm_x,       # -25° is UP, +5° is DOWN
            reach_deg=r_arm_y,      # -60° is FRONT, +25° is BACK
            is_left=False,
        )

        # Left Arm (CH 4 X lift, CH 2 Y reach)
        l_shoulder_x = torso_x2 + twist_offset - 5
        l_shoulder_y = torso_y1 + 18
        self._draw_arm_front(
            l_shoulder_x,
            l_shoulder_y,
            lift_deg=l_arm_x,       # +25° is UP, -5° is DOWN
            reach_deg=l_arm_y,      # +60° is FRONT, -25° is BACK
            is_left=True,
        )

        # 4. Neck linkage
        neck_y1 = torso_y1 - 16
        self.canvas.create_rectangle(
            cx - 12 + twist_offset,
            neck_y1,
            cx + 12 + twist_offset,
            torso_y1,
            fill="#334155",
            outline="#475569",
        )

        # 5. Head (Channels 0 Tilt: -15° UP, +15° DOWN)
        # Tilt shifts head center vertically and distorts perspective
        tilt_y_shift = int(head_tilt * 2.2)  # Down when positive, up when negative
        head_cx = cx + twist_offset
        head_cy = neck_y1 - 42 + tilt_y_shift

        # Cute rounded robot head housing
        head_w, head_h = 110, 76
        self.canvas.create_rectangle(
            head_cx - head_w // 2,
            head_cy - head_h // 2,
            head_cx + head_w // 2,
            head_cy + head_h // 2,
            fill="#334155",
            outline="#38bdf8",
            width=2,
        )

        # Screen Visor (Black curved glass)
        visor_w, visor_h = 88, 52
        self.canvas.create_rectangle(
            head_cx - visor_w // 2,
            head_cy - visor_h // 2,
            head_cx + visor_w // 2,
            head_cy + visor_h // 2,
            fill="#020617",
            outline="#0ea5e9",
            width=1,
        )

        # Expressive Eyes
        self._draw_eyes(head_cx, head_cy, expr, head_tilt)

        # Head tilt angle label
        self.canvas.create_text(
            head_cx,
            head_cy - head_h // 2 - 12,
            text=f"HEAD TILT: {head_tilt:+.1f}°",
            font=("Consolas", 8, "bold"),
            fill="#38bdf8",
        )

    def _draw_arm_front(self, sx: int, sy: int, lift_deg: float, reach_deg: float, is_left: bool) -> None:
        """Draw 2-DOF arm with elevation (X) and forward/backward reach perspective (Y)."""
        # Shoulder Joint
        self.canvas.create_oval(sx - 7, sy - 7, sx + 7, sy + 7, fill="#64748b", outline="#cbd5e1", width=1)

        # Direction signs:
        # Left Arm: +25° lift is UP, -5° is DOWN
        # Right Arm: -25° lift is UP, +5° is DOWN
        if is_left:
            norm_lift = lift_deg / 25.0  # +1.0 = full up, -0.2 = down
            norm_reach = reach_deg / 60.0  # +1.0 = full front
            base_angle = math.radians(65.0 - norm_lift * 85.0)
            side_mult = 1.0
        else:
            norm_lift = -lift_deg / 25.0  # +1.0 = full up, -0.2 = down
            norm_reach = -reach_deg / 60.0  # +1.0 = full front
            base_angle = math.radians(115.0 + norm_lift * 85.0)
            side_mult = -1.0

        arm_len = 54
        # Forward reach adds foreshortening / forward angle
        elbow_x = sx + math.cos(base_angle) * (arm_len * 0.55)
        elbow_y = sy + math.sin(base_angle) * (arm_len * 0.55)

        # Hand reaches forward (simulated 3D depth by growing hand size when forward)
        hand_scale = 1.0 + norm_reach * 0.35
        hand_x = elbow_x + math.cos(base_angle + side_mult * 0.2) * (arm_len * 0.5)
        hand_y = elbow_y + math.sin(base_angle + side_mult * 0.2) * (arm_len * 0.5)

        # Upper arm
        self.canvas.create_line(sx, sy, elbow_x, elbow_y, fill="#94a3b8", width=7, capstyle="round")
        # Forearm
        self.canvas.create_line(elbow_x, elbow_y, hand_x, hand_y, fill="#cbd5e1", width=5, capstyle="round")

        # Hand / Gripper (glowing cyan)
        hr = int(7 * hand_scale)
        hand_color = "#38bdf8" if norm_reach >= 0 else "#64748b"
        self.canvas.create_oval(hand_x - hr, hand_y - hr, hand_x + hr, hand_y + hr, fill=hand_color, outline="#ffffff", width=1)

    def _draw_eyes(self, hx: int, hy: int, expr: str, tilt: float) -> None:
        """Render cute glowing expressive eyes inside the visor."""
        eye_y = hy + int(tilt * 0.4)
        eye_color = "#38bdf8"

        if expr in ("happy", "celebrate"):
            # Happy smiling eye curves ^ ^
            self.canvas.create_arc(hx - 28, eye_y - 12, hx - 8, eye_y + 8, start=0, extent=180, style="arc", outline=eye_color, width=3)
            self.canvas.create_arc(hx + 8, eye_y - 12, hx + 28, eye_y + 8, start=0, extent=180, style="arc", outline=eye_color, width=3)
        elif expr == "excited":
            # Big sparkling oval eyes
            self.canvas.create_oval(hx - 26, eye_y - 14, hx - 10, eye_y + 14, fill=eye_color, outline="#ffffff", width=1)
            self.canvas.create_oval(hx + 10, eye_y - 14, hx + 26, eye_y + 14, fill=eye_color, outline="#ffffff", width=1)
            self.canvas.create_oval(hx - 22, eye_y - 10, hx - 16, eye_y - 4, fill="#ffffff", outline="")
            self.canvas.create_oval(hx + 14, eye_y - 10, hx + 20, eye_y - 4, fill="#ffffff", outline="")
        elif expr == "thinking":
            # Eyes looking upward / tilted
            self.canvas.create_oval(hx - 24, eye_y - 16, hx - 10, eye_y + 4, fill=eye_color, outline="")
            self.canvas.create_oval(hx + 12, eye_y - 16, hx + 26, eye_y + 4, fill=eye_color, outline="")
        elif expr == "sleep":
            # Sleeping resting lines - -
            self.canvas.create_line(hx - 26, eye_y, hx - 10, eye_y, fill=eye_color, width=3)
            self.canvas.create_line(hx + 10, eye_y, hx + 26, eye_y, fill=eye_color, width=3)
        elif expr == "bored":
            # Half-closed eyelids
            self.canvas.create_arc(hx - 26, eye_y - 10, hx - 10, eye_y + 10, start=180, extent=180, style="arc", outline=eye_color, width=3)
            self.canvas.create_arc(hx + 10, eye_y - 10, hx + 26, eye_y + 10, start=180, extent=180, style="arc", outline=eye_color, width=3)
        else:
            # Default pill eyes
            self.canvas.create_oval(hx - 25, eye_y - 12, hx - 11, eye_y + 12, fill=eye_color, outline="")
            self.canvas.create_oval(hx + 11, eye_y - 12, hx + 25, eye_y + 12, fill=eye_color, outline="")

    def _draw_side_view(self, cx: int, cy: int, angles: Dict[int, float]) -> None:
        """Side Profile showing true Pitch / Front-Back reach of Head & Arms."""
        head_tilt = angles.get(0, 0.0)  # -15° UP, +15° DOWN
        # Average arm reach for side view or draw both
        r_reach = angles.get(1, 0.0)    # -60° FRONT, +25° BACK
        l_reach = angles.get(2, 0.0)    # +60° FRONT, -25° BACK

        # Base
        base_y = cy + 130
        self.canvas.create_rectangle(cx - 50, base_y - 10, cx + 50, base_y + 10, fill="#1e293b", outline="#475569", width=2)
        self.canvas.create_text(cx, base_y + 24, text="SIDE PROFILE", font=("Consolas", 8, "bold"), fill="#64748b")

        # Torso side profile
        torso_y1 = cy - 10
        torso_y2 = torso_y1 + 110
        self.canvas.create_rectangle(cx - 30, torso_y1, cx + 30, torso_y2, fill="#1e293b", outline="#334155", width=2)

        # Neck
        neck_y1 = torso_y1 - 16
        self.canvas.create_rectangle(cx - 8, neck_y1, cx + 8, torso_y1, fill="#334155", outline="#475569")

        # Head Side Profile with true tilt rotation!
        head_cx = cx
        head_cy = neck_y1 - 38
        tilt_rad = math.radians(head_tilt)  # Positive = pitch down, negative = pitch up

        # Rotate head profile box
        hw, hh = 60, 48
        cos_t = math.cos(tilt_rad)
        sin_t = math.sin(tilt_rad)

        pts = [(-hw / 2, -hh / 2), (hw / 2, -hh / 2), (hw / 2, hh / 2), (-hw / 2, hh / 2)]
        rot_pts = []
        for px, py in pts:
            rx = head_cx + (px * cos_t - py * sin_t)
            ry = head_cy + (px * sin_t + py * cos_t)
            rot_pts.extend([rx, ry])

        self.canvas.create_polygon(rot_pts, fill="#334155", outline="#38bdf8", width=2)

        # Nose / Visor front direction (Faces Left in side profile)
        visor_front_x = head_cx - (hw / 2 * cos_t)
        visor_front_y = head_cy - (hw / 2 * sin_t)
        self.canvas.create_line(visor_front_x, visor_front_y, visor_front_x - 12 * cos_t, visor_front_y - 12 * sin_t, fill="#0ea5e9", width=4)

        # Arm Reach Lines in Side Profile
        shoulder_x = cx
        shoulder_y = torso_y1 + 20

        # Right Arm (Viewer sees reach: -60° is front / left in profile)
        r_angle = math.radians(90.0 + (r_reach / 60.0) * 60.0)
        r_hand_x = shoulder_x - math.cos(r_angle) * 55
        r_hand_y = shoulder_y + math.sin(r_angle) * 55
        self.canvas.create_line(shoulder_x, shoulder_y, r_hand_x, r_hand_y, fill="#f59e0b", width=4, capstyle="round")
        self.canvas.create_oval(r_hand_x - 4, r_hand_y - 4, r_hand_x + 4, r_hand_y + 4, fill="#f59e0b", outline="")

        # Left Arm (Viewer sees reach: +60° is front / left in profile)
        l_angle = math.radians(90.0 + (l_reach / 60.0) * 60.0)
        l_hand_x = shoulder_x - math.cos(l_angle) * 55
        l_hand_y = shoulder_y + math.sin(l_angle) * 55
        self.canvas.create_line(shoulder_x, shoulder_y, l_hand_x, l_hand_y, fill="#38bdf8", width=3, capstyle="round", dash=(3, 2))
        self.canvas.create_oval(l_hand_x - 4, l_hand_y - 4, l_hand_x + 4, l_hand_y + 4, fill="#38bdf8", outline="")

        # Legend
        self.canvas.create_text(cx, torso_y2 - 25, text="— Right Arm (Y)", font=("Consolas", 8), fill="#f59e0b")
        self.canvas.create_text(cx, torso_y2 - 10, text="--- Left Arm (Y)", font=("Consolas", 8), fill="#38bdf8")

    def run(self) -> None:
        self.root.mainloop()


# =============================================================================
# Web Dashboard Mode (HTML5 Canvas + REST API)
# =============================================================================

_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LUMI Robot Motion Visualizer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
        body { background: #0f172a; color: #f8fafc; padding: 20px; }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        h1 { color: #38bdf8; font-size: 1.5rem; }
        .badge { background: #1e293b; color: #22c55e; padding: 6px 14px; border-radius: 9999px; font-weight: bold; font-size: 0.85rem; border: 1px solid #22c55e; }
        .grid { display: grid; grid-template-columns: 1fr 380px; gap: 20px; }
        .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; }
        canvas { background: #020617; border-radius: 8px; width: 100%; height: 500px; display: block; border: 1px solid #1e293b; }
        .btn-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 10px; }
        button { background: #334155; color: #f8fafc; border: none; padding: 8px 10px; border-radius: 6px; font-weight: bold; cursor: pointer; transition: all 0.15s; font-size: 0.85rem; }
        button:hover { background: #38bdf8; color: #0f172a; }
        .slider-group { margin-top: 12px; }
        .slider-row { margin-bottom: 8px; }
        .slider-label { display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8; margin-bottom: 2px; }
        input[type="range"] { width: 100%; accent-color: #38bdf8; }
        .danger-btn { background: #ef4444; color: white; }
        .danger-btn:hover { background: #dc2626; color: white; }
        .success-btn { background: #10b981; color: white; }
        .success-btn:hover { background: #059669; color: white; }
        .gauges { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
        .gauge-box { background: #0f172a; padding: 8px 12px; border-radius: 6px; font-size: 0.8rem; border-left: 3px solid #38bdf8; }
        .gauge-box span { color: #38bdf8; font-weight: bold; float: right; }
    </style>
</head>
<body>
    <header>
        <h1>🤖 LUMI ROBOT DUAL-PROJECTION SIMULATOR</h1>
        <div id="statusBadge" class="badge">STATUS: CONNECTED</div>
    </header>
    <div class="grid">
        <div>
            <div class="card">
                <canvas id="simCanvas" width="800" height="500"></canvas>
                <div class="gauges" id="gaugesGrid"></div>
            </div>
        </div>
        <div>
            <div class="card" style="margin-bottom: 16px;">
                <h3>🎭 Expressive Gestures</h3>
                <div class="btn-grid">
                    <button onclick="triggerGesture('greet')">👋 Greet</button>
                    <button onclick="triggerGesture('wave')">🙋 Wave</button>
                    <button onclick="triggerGesture('happy')">😄 Happy</button>
                    <button onclick="triggerGesture('celebrate')">🎉 Celebrate</button>
                    <button onclick="triggerGesture('dance')">🕺 Dance</button>
                    <button onclick="triggerGesture('thinking')">🤔 Thinking</button>
                    <button onclick="triggerGesture('curious')">🧐 Curious</button>
                    <button onclick="triggerGesture('excited')">⚡ Excited</button>
                    <button onclick="triggerGesture('sleep')">😴 Sleep</button>
                    <button onclick="triggerGesture('bored')">😐 Bored</button>
                    <button onclick="triggerGesture('home')" style="grid-column: span 2;">🏠 Soft Home (0°)</button>
                </div>
            </div>
            <div class="card" style="margin-bottom: 16px;">
                <h3>🎛️ Manual Joint Sliders</h3>
                <div class="slider-group" id="slidersContainer"></div>
            </div>
            <div class="card">
                <h3>⚡ Power-Cut & Recovery Test</h3>
                <div style="display: flex; gap: 8px; margin-top: 10px;">
                    <button class="danger-btn" style="flex: 1;" onclick="powerCut()">⚡ Cut Power</button>
                    <button class="success-btn" style="flex: 1;" onclick="rebootHome()">🔄 Reboot & Soft Home</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');

        const channelsSpec = [
            { ch: 0, name: "Head Tilt", min: -15, max: 15, unit: "[-15° .. +15°]" },
            { ch: 5, name: "Body Waist", min: -90, max: 90, unit: "[-90° .. +90°]" },
            { ch: 1, name: "Right Arm Y", min: -60, max: 25, unit: "[-60° .. +25°]" },
            { ch: 3, name: "Right Arm X", min: -25, max: 5, unit: "[-25° .. +5°]" },
            { ch: 2, name: "Left Arm Y", min: -25, max: 60, unit: "[-25° .. +60°]" },
            { ch: 4, name: "Left Arm X", min: -5, max: 25, unit: "[-5° .. +25°]" }
        ];

        // Build Gauges & Sliders
        const gaugesGrid = document.getElementById('gaugesGrid');
        const slidersContainer = document.getElementById('slidersContainer');
        channelsSpec.forEach(spec => {
            gaugesGrid.innerHTML += `<div class="gauge-box">${spec.name} <span id="gauge_${spec.ch}">0.0°</span></div>`;
            slidersContainer.innerHTML += `
                <div class="slider-row">
                    <div class="slider-label"><span>CH ${spec.ch}: ${spec.name}</span><span id="sval_${spec.ch}">0.0°</span></div>
                    <input type="range" id="slide_${spec.ch}" min="${spec.min}" max="${spec.max}" step="1" value="0" oninput="onSlide(${spec.ch}, this.value)">
                </div>
            `;
        });

        function triggerGesture(name) { fetch('/api/gesture?name=' + name, { method: 'POST' }); }
        function onSlide(ch, val) {
            document.getElementById('sval_' + ch).innerText = val + '°';
            fetch(`/api/servo?ch=${ch}&angle=${val}`, { method: 'POST' });
        }
        function powerCut() { fetch('/api/power_cut', { method: 'POST' }); }
        function rebootHome() { fetch('/api/reboot_home', { method: 'POST' }); }

        let state = { angles: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 }, gesture: 'IDLE', expression: 'happy' };

        async function pollState() {
            try {
                const res = await fetch('/api/state');
                state = await res.json();
                document.getElementById('statusBadge').innerText = 'STATUS: ' + (state.gesture !== 'IDLE' ? state.gesture : 'READY');
                for (let ch in state.angles) {
                    const val = Number(state.angles[ch]).toFixed(1);
                    const gEl = document.getElementById('gauge_' + ch);
                    if (gEl) gEl.innerText = val + '°';
                }
            } catch(e) {}
        }
        setInterval(pollState, 100);

        function render() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const w = canvas.width, h = canvas.height;

            // Background Grid
            ctx.strokeStyle = '#0b1120'; ctx.lineWidth = 1;
            for(let x=0; x<w; x+=40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
            for(let y=0; y<h; y+=40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

            // Split line
            const splitX = w * 0.65;
            ctx.strokeStyle = '#1e293b'; ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.moveTo(splitX, 20); ctx.lineTo(splitX, h-20); ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = '#64748b'; ctx.font = 'bold 12px monospace';
            ctx.fillText('FRONT VIEW (DUAL ARMS & SCREEN)', splitX / 2 - 100, 25);
            ctx.fillText('SIDE PROFILE (PITCH & REACH)', splitX + (w - splitX)/2 - 90, 25);

            drawFrontView(splitX / 2, h * 0.55, state.angles, state.expression);
            drawSideView(splitX + (w - splitX)/2, h * 0.55, state.angles);

            requestAnimationFrame(render);
        }

        function drawFrontView(cx, cy, angles, expr) {
            const head_tilt = angles[0] || 0;
            const waist_pan = angles[5] || 0;
            const r_arm_x = angles[3] || 0;
            const r_arm_y = angles[1] || 0;
            const l_arm_x = angles[4] || 0;
            const l_arm_y = angles[2] || 0;

            // Base & Waist Compass
            ctx.fillStyle = '#1e293b'; ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.ellipse(cx, cy + 120, 80, 14, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();

            const panRad = (-waist_pan * Math.PI) / 180;
            ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 3;
            ctx.beginPath(); ctx.moveTo(cx, cy + 120);
            ctx.lineTo(cx + Math.sin(panRad) * 45, cy + 120 + Math.cos(panRad) * 10); ctx.stroke();

            // Torso
            const twist = waist_pan * 0.25;
            ctx.fillStyle = '#1e293b'; ctx.strokeStyle = '#475569';
            ctx.fillRect(cx - 50 + twist, cy - 20, 100, 100);
            ctx.strokeRect(cx - 50 + twist, cy - 20, 100, 100);

            // Arc Reactor
            ctx.fillStyle = '#0284c7'; ctx.beginPath(); ctx.arc(cx + twist, cy + 25, 16, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = '#38bdf8'; ctx.beginPath(); ctx.arc(cx + twist, cy + 25, 6, 0, Math.PI * 2); ctx.fill();

            // Arms
            drawArmFront(cx - 45 + twist, cy, r_arm_x, r_arm_y, false);
            drawArmFront(cx + 45 + twist, cy, l_arm_x, l_arm_y, true);

            // Head
            const headY = cy - 65 + (head_tilt * 2.0);
            ctx.fillStyle = '#334155'; ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 2;
            ctx.fillRect(cx - 55 + twist, headY - 35, 110, 70);
            ctx.strokeRect(cx - 55 + twist, headY - 35, 110, 70);

            // Visor
            ctx.fillStyle = '#020617'; ctx.fillRect(cx - 45 + twist, headY - 25, 90, 50);

            // Eyes
            ctx.fillStyle = '#38bdf8'; ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 3;
            if (expr === 'happy' || expr === 'celebrate') {
                ctx.beginPath(); ctx.arc(cx - 20 + twist, headY, 8, Math.PI, 0); ctx.stroke();
                ctx.beginPath(); ctx.arc(cx + 20 + twist, headY, 8, Math.PI, 0); ctx.stroke();
            } else if (expr === 'sleep') {
                ctx.beginPath(); ctx.moveTo(cx - 30 + twist, headY); ctx.lineTo(cx - 10 + twist, headY); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(cx + 10 + twist, headY); ctx.lineTo(cx + 30 + twist, headY); ctx.stroke();
            } else {
                ctx.beginPath(); ctx.ellipse(cx - 20 + twist, headY, 7, 12, 0, 0, Math.PI * 2); ctx.fill();
                ctx.beginPath(); ctx.ellipse(cx + 20 + twist, headY, 7, 12, 0, 0, Math.PI * 2); ctx.fill();
            }
        }

        function drawArmFront(sx, sy, lift, reach, isLeft) {
            ctx.fillStyle = '#64748b'; ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2); ctx.fill();
            let baseAngle = isLeft ? (65 - (lift / 25) * 85) : (115 + (-lift / 25) * 85);
            let rad = (baseAngle * Math.PI) / 180;
            let ex = sx + Math.cos(rad) * 30, ey = sy + Math.sin(rad) * 30;
            let hx = ex + Math.cos(rad) * 25, hy = ey + Math.sin(rad) * 25;

            ctx.strokeStyle = '#94a3b8'; ctx.lineWidth = 6; ctx.lineCap = 'round';
            ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
            ctx.strokeStyle = '#cbd5e1'; ctx.lineWidth = 4;
            ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(hx, hy); ctx.stroke();

            ctx.fillStyle = reach >= 0 ? '#38bdf8' : '#64748b';
            ctx.beginPath(); ctx.arc(hx, hy, 7, 0, Math.PI * 2); ctx.fill();
        }

        function drawSideView(cx, cy, angles) {
            const head_tilt = angles[0] || 0;
            const r_reach = angles[1] || 0;
            const l_reach = angles[2] || 0;

            ctx.fillStyle = '#1e293b'; ctx.fillRect(cx - 40, cy + 115, 80, 15);
            ctx.fillRect(cx - 25, cy - 20, 50, 100);

            // Head Tilt
            const headY = cy - 60;
            const tiltRad = (head_tilt * Math.PI) / 180;
            ctx.save();
            ctx.translate(cx, headY);
            ctx.rotate(tiltRad);
            ctx.fillStyle = '#334155'; ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 2;
            ctx.fillRect(-28, -22, 56, 44);
            ctx.strokeRect(-28, -22, 56, 44);
            // Visor line
            ctx.strokeStyle = '#0ea5e9'; ctx.lineWidth = 4;
            ctx.beginPath(); ctx.moveTo(-28, -5); ctx.lineTo(-38, -5); ctx.stroke();
            ctx.restore();

            // Right Arm Y reach
            const rAng = Math.PI / 2 + (r_reach / 60) * (Math.PI / 3);
            ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 4;
            ctx.beginPath(); ctx.moveTo(cx, cy - 5);
            ctx.lineTo(cx - Math.cos(rAng) * 50, cy - 5 + Math.sin(rAng) * 50); ctx.stroke();

            // Left Arm Y reach
            const lAng = Math.PI / 2 + (l_reach / 60) * (Math.PI / 3);
            ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 3; ctx.setLineDash([3, 2]);
            ctx.beginPath(); ctx.moveTo(cx, cy - 5);
            ctx.lineTo(cx - Math.cos(lAng) * 50, cy - 5 + Math.sin(lAng) * 50); ctx.stroke();
            ctx.setLineDash([]);
        }

        render();
    </script>
</body>
</html>
"""


class WebSimulatorServer:
    """Lightweight HTTP server serving the interactive visualizer."""

    def __init__(self, backend: SimulationBackend, port: int = 8080) -> None:
        self.backend = backend
        self.port = port
        self.server: Optional[HTTPServer] = None

    def start(self) -> None:
        backend = self.backend

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass  # Suppress noisy console logs

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(_HTML_PAGE.encode("utf-8"))
                elif parsed.path == "/api/state":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    angles = backend.get_angles()
                    with backend._lock:
                        payload = {
                            "angles": angles,
                            "gesture": backend.active_gesture,
                            "expression": backend.current_expression,
                            "logs": backend.logs[-5:],
                        }
                    self.wfile.write(json.dumps(payload).encode("utf-8"))
                else:
                    self.send_error(404)

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)

                if parsed.path == "/api/gesture":
                    name = query.get("name", [""])[0]
                    backend.trigger_gesture(name)
                    self._send_ok()
                elif parsed.path == "/api/servo":
                    ch = int(query.get("ch", [0])[0])
                    angle = float(query.get("angle", [0.0])[0])
                    backend.set_single_joint(ch, angle, duration_s=0.25)
                    self._send_ok()
                elif parsed.path == "/api/power_cut":
                    backend.simulate_power_cut()
                    self._send_ok()
                elif parsed.path == "/api/reboot_home":
                    backend.simulate_reboot_recovery()
                    self._send_ok()
                else:
                    self.send_error(404)

            def _send_ok(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')

        self.server = HTTPServer(("0.0.0.0", self.port), Handler)
        local_ip = "127.0.0.1"
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            pass

        print(f"\n=======================================================")
        print(f"🚀 LUMI Web Motion Simulator running at:")
        print(f"   Local:   http://localhost:{self.port}/")
        print(f"   Network: http://{local_ip}:{self.port}/  (Open this in your browser!)")
        print(f"=======================================================\n")
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            pass


def run_terminal_menu(backend: SimulationBackend) -> None:
    """Interactive command-line movement controller and demo runner."""
    while True:
        mode_str = "REAL PCA9685 HARDWARE" if backend.is_real_hardware else "VIRTUAL SIMULATION"
        print("\n=======================================================")
        print("🤖 LUMI ROBOT MOVEMENT CONTROLLER & TESTER")
        print(f"   Actuation Mode: {mode_str}")
        print("=======================================================")
        print("Select an action:")
        print("  [1] ▶ Run Full Automatic Movement Demo")
        print("  [2] Head Tilt Test (-15° Up -> +15° Down -> 0° Center)")
        print("  [3] Body Waist Rotation (-45° Right -> +45° Left -> 0° Center)")
        print("  [4] Right Arm Test (Lift + Wave)")
        print("  [5] Left Arm Test (Lift + Wave)")
        print("  [6] Gesture: Greet (Tilt + Right Wave)")
        print("  [7] Gesture: Happy (Both Arms Up + Nod)")
        print("  [8] Gesture: Dance (Waist Sway + Alternating Arms)")
        print("  [9] Gesture: Celebrate (Both Arms Up + Excited Nod)")
        print("  [0] Home All Joints (0.0° Safe Resting Position)")
        print("  [w] Start Web Simulation Server (Port 8080)")
        print("  [q] Quit")
        print("-------------------------------------------------------")
        try:
            choice = input("Enter choice (default [1]): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting LUMI movement controller.")
            break

        if not choice or choice == "1":
            backend.run_full_demo()
        elif choice == "2":
            print("Moving Head Tilt...")
            backend.head.look_up(15.0, duration_s=0.35)
            time.sleep(0.4)
            backend.head.look_down(15.0, duration_s=0.35)
            time.sleep(0.4)
            backend.head.look_center(duration_s=0.3)
        elif choice == "3":
            print("Moving Body Waist...")
            backend.head.look_right(45.0, duration_s=0.4)
            time.sleep(0.4)
            backend.head.look_left(45.0, duration_s=0.4)
            time.sleep(0.4)
            backend.head.look_center(duration_s=0.3)
        elif choice == "4":
            print("Testing Right Arm Wave...")
            backend.arms.wave_right(count=2)
        elif choice == "5":
            print("Testing Left Arm Wave...")
            backend.arms.wave_left(count=2)
        elif choice == "6":
            backend.trigger_gesture("greet")
            time.sleep(1.5)
        elif choice == "7":
            backend.trigger_gesture("happy")
            time.sleep(1.5)
        elif choice == "8":
            backend.trigger_gesture("dance")
            time.sleep(2.0)
        elif choice == "9":
            backend.trigger_gesture("celebrate")
            time.sleep(1.8)
        elif choice == "0":
            print("Homing all channels...")
            backend.home_all(duration_s=0.4)
        elif choice == "w":
            server = WebSimulatorServer(backend, port=8080)
            server.start()
            break
        elif choice in ("q", "exit"):
            break
        else:
            print("Invalid selection. Please try again.")


def main() -> None:
    parser = argparse.ArgumentParser(description="LUMI Robot Motion Visual Simulator")
    parser.add_argument("--demo", action="store_true", help="Run full automated movement demonstration and exit")
    parser.add_argument("--cli", action="store_true", help="Run interactive terminal menu")
    parser.add_argument("--web", action="store_true", help="Launch Web Browser Dashboard")
    parser.add_argument("--gui", action="store_true", help="Launch Desktop Tkinter GUI")
    parser.add_argument("--mock", action="store_true", help="Force software simulation mock driver")
    parser.add_argument("--port", type=int, default=8080, help="Web port (default: 8080)")
    # Also support positional arguments like 'python simulate_movement.py demo'
    parser.add_argument("mode", nargs="?", default="", help="Optional mode: demo, web, cli, gui")
    args = parser.parse_args()

    backend = SimulationBackend(force_mock=args.mock)

    # 1. Direct Demo mode
    if args.demo or args.mode.lower() == "demo":
        backend.run_full_demo()
        return

    # 2. Web mode
    if args.web or args.mode.lower() == "web":
        server = WebSimulatorServer(backend, port=args.port)
        server.start()
        return

    # 3. CLI mode
    if args.cli or args.mode.lower() == "cli":
        run_terminal_menu(backend)
        return

    # 4. Explicit GUI mode
    if args.gui or args.mode.lower() == "gui":
        try:
            app = DesktopSimulatorGUI(backend)
            app.run()
            return
        except Exception as e:
            print(f"Notice: GUI could not be opened ({e}). Falling back to Terminal Menu.")
            run_terminal_menu(backend)
            return

    # 5. Default auto-detection mode:
    # If on desktop OS with DISPLAY, try GUI; otherwise fall back to interactive CLI menu
    gui_available = False
    if sys.platform == "win32":
        gui_available = True
    elif "DISPLAY" in os.environ and os.environ["DISPLAY"]:
        gui_available = True

    if gui_available:
        try:
            app = DesktopSimulatorGUI(backend)
            app.run()
            return
        except Exception as e:
            print(f"Notice: Desktop GUI could not be initialized ({e}).")

    # Headless / SSH terminal fallback: Run interactive terminal menu
    run_terminal_menu(backend)


if __name__ == "__main__":
    main()
