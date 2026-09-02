"""Real-Time Procedural Dual-Eye Renderer for GC9A01 240x240 LCDs.

Incorporates the exact geometry, pill/capsule shapes, and animations from boot/Eyes.cpp & Emotion.cpp:
- Screen: 240x240
- Geometry: Eye Width=40px, Height=70px, Distance=30px
- Center: Left Eye X=70, Right Eye X=170, Y=115
- Animations: Idle, Blink (height squash -> rounded horizontal slit), Wink, Sleep, Happy (^ ^), Curious, Angry
"""

from __future__ import annotations

import math
import random
import threading
import time
from typing import Any, Optional, Tuple

from ..core.logger import get_logger
from ..hardware.base import DisplayBackendBase
from .expressions import EXPRESSIONS, ExpressionConfig

logger = get_logger("eyes.renderer")

try:
    from PIL import Image, ImageDraw  # type: ignore
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


class EyeRenderer:
    """Dedicated render engine generating synchronized emotional eye graphics at 30 FPS."""

    def __init__(
        self,
        display_backend: DisplayBackendBase,
        width: int = 240,
        height: int = 240,
        target_fps: int = 30,
        single_display_both_eyes: bool = True,
    ) -> None:
        self.display = display_backend
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.single_display_both_eyes = single_display_both_eyes

        # Geometry derived from boot/Eyes.cpp
        self.w_eye = 40
        self.h_eye = 70
        self.x_eye_left = (self.width - 30 * 2 - self.w_eye) // 2  # 70
        self.x_eye_right = self.width - self.x_eye_left             # 170
        self.y_eye_center = 30 + (self.height - self.h_eye) // 2    # 115

        self.current_expr = EXPRESSIONS["neutral"]
        self.target_expr = EXPRESSIONS["neutral"]
        self.is_winking = False
        self.is_sleeping = False

        # Dynamic Gaze tracking offsets (-30.0 to +30.0)
        self.gaze_x: float = 0.0
        self.gaze_y: float = 0.0

        # Natural Blinking parameters
        self._is_blinking = False
        self._next_blink_time = time.time() + random.uniform(2.5, 5.5)
        self._blink_start_time = 0.0
        self._blink_duration = 0.18

        self._smooth_gaze_x = 0.0
        self._smooth_gaze_y = 0.0
        self._saccade_x = 0.0
        self._saccade_y = 0.0
        self._next_saccade_time = 0.0

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    def start(self) -> None:
        """Start the background 30 FPS rendering loop."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self.display.initialize()
            self._thread = threading.Thread(
                target=self._render_loop,
                name="LumiEyeRenderer-30FPS",
                daemon=True,
            )
            self._thread.start()
            mode_str = "single-display dual-eye" if self.single_display_both_eyes else "dual-display"
            logger.info(f"EyeRenderer started (30 FPS, {mode_str} pipeline).")

    def stop(self) -> None:
        """Stop rendering loop cleanly."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            if self._thread:
                self._thread.join(timeout=1.0)
            self.display.clear()
            logger.info("EyeRenderer stopped.")

    def set_expression(self, expression_name: str, transition_speed: float = 0.2) -> None:
        """Request high-level expression change (e.g. 'happy', 'curious', 'thinking', 'wink', 'sleep')."""
        with self._lock:
            name_lower = expression_name.lower()
            if name_lower == "wink":
                self.is_winking = True
                self.is_sleeping = False
                return
            elif name_lower == "sleepy" or name_lower == "sleep":
                self.is_sleeping = True
                self.is_winking = False
            else:
                self.is_sleeping = False
                self.is_winking = False

            expr = EXPRESSIONS.get(name_lower)
            if not expr:
                expr = EXPRESSIONS.get("neutral")
            if expr:
                self.target_expr = expr
            logger.debug(f"Eye expression set to '{expression_name}'")

    def set_gaze(self, gaze_x: float, gaze_y: float) -> None:
        """Set normalized gaze vector (-1.0 to +1.0)."""
        with self._lock:
            self.gaze_x = max(-1.0, min(1.0, gaze_x)) * 25.0
            self.gaze_y = max(-1.0, min(1.0, gaze_y)) * 18.0

    def trigger_blink(self) -> None:
        """Trigger an instantaneous organic blink."""
        with self._lock:
            self._is_blinking = True
            self._blink_start_time = time.time()

    def play_animation(self, image_path: str, duration: float = 3.0) -> None:
        """Temporarily replace the eyes with an image or GIF animation."""
        if not _HAS_PIL:
            logger.warning("PIL not installed, cannot play animation.")
            return
            
        try:
            from PIL import Image, ImageSequence
            img = Image.open(image_path)
            
            # Extract frames if GIF, else just one frame
            frames = []
            for frame in ImageSequence.Iterator(img):
                frame = frame.convert("RGB")
                # Resize to fit screen (240x240 usually)
                frame = frame.resize((self.width, self.height), Image.Resampling.LANCZOS)
                frames.append(frame)
                
            if not frames:
                return
                
            self._animation_frames = frames
            self._animation_end_time = time.time() + duration
            self._animation_frame_idx = 0
            # Get GIF duration per frame in seconds, default to 100ms
            self._animation_frame_time = img.info.get('duration', 100) / 1000.0
            if self._animation_frame_time <= 0:
                self._animation_frame_time = 0.1
            self._last_frame_time = time.time()
            logger.info(f"Playing animation {image_path} ({len(frames)} frames) for {duration}s")
        except Exception as e:
            logger.error(f"Failed to load animation {image_path}: {e}")

    def show_procedural_animal(self, animal: str, duration: float = 4.0) -> None:
        """Trigger a programmatic procedural animation of an animal."""
        self._active_animal = animal
        self._active_animal_start = time.time()
        self._active_animal_end = time.time() + duration
        logger.info(f"Playing procedural animation for {animal}")

    def _draw_animal_frame(self, animal: str, elapsed: float):
        from PIL import Image, ImageDraw
        import math
        try:
            from ..assets.animal_registry import ANIMALS
        except ImportError:
            ANIMALS = {}
            
        img = Image.new('RGB', (self.width, self.height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        animal_lower = animal.lower()
        animal_data = ANIMALS.get(animal_lower, {'template': 'bear'})
        template = animal_data.get('template', 'bear')
        
        # Robot's primary neon color
        eye_color = getattr(self, 'eye_color', (0, 255, 255))
        
        # Smooth transition effect: Scale up features at start, scale down at end
        duration = getattr(self, '_animal_duration', 3.0)
        time_left = max(0, duration - elapsed)
        
        if elapsed < 0.3:
            morph_scale = elapsed / 0.3
        elif time_left < 0.3:
            morph_scale = time_left / 0.3
        else:
            morph_scale = 1.0
            
        # Add slight elastic bounce to features appearing
        morph_scale = math.sin(morph_scale * math.pi / 2)
        
        # Scaling factors for 240x240 screen
        sx = (self.width / 240.0)
        sy = (self.height / 240.0)
        
        def scale(coords):
            return [c * sx if i % 2 == 0 else c * sy for i, c in enumerate(coords)]
            
        # Breathe and motion
        breathe = math.sin(elapsed * 4) * 2
        mouth_open = max(0, math.sin(elapsed * 12)) * 10
        
        # --- BASE EYES (Always present, maintaining the illusion) ---
        # The standard eye ovals
        eye_w, eye_h = 30, 60
        lx, rx = 70, 170
        y = 120 + breathe
        
        # Modify eyes slightly based on template
        if template == 'reptile':
            # Slit pupils for reptiles
            draw.ellipse(scale([lx - eye_w/2, y - eye_h/2, lx + eye_w/2, y + eye_h/2]), fill=eye_color)
            draw.ellipse(scale([rx - eye_w/2, y - eye_h/2, rx + eye_w/2, y + eye_h/2]), fill=eye_color)
            draw.rectangle(scale([lx - 3, y - eye_h/2, lx + 3, y + eye_h/2]), fill=(0,0,0))
            draw.rectangle(scale([rx - 3, y - eye_h/2, rx + 3, y + eye_h/2]), fill=(0,0,0))
        elif animal_data.get('panda_eyes'):
            # Big dark patches around eyes for pandas
            draw.ellipse(scale([lx - 40, y - 50, lx + 40, y + 50]), fill=(30, 30, 30))
            draw.ellipse(scale([rx - 40, y - 50, rx + 40, y + 50]), fill=(30, 30, 30))
            draw.ellipse(scale([lx - eye_w/2, y - eye_h/2, lx + eye_w/2, y + eye_h/2]), fill=eye_color)
            draw.ellipse(scale([rx - eye_w/2, y - eye_h/2, rx + eye_w/2, y + eye_h/2]), fill=eye_color)
        else:
            # Standard filled eyes
            draw.ellipse(scale([lx - eye_w/2, y - eye_h/2, lx + eye_w/2, y + eye_h/2]), fill=eye_color)
            draw.ellipse(scale([rx - eye_w/2, y - eye_h/2, rx + eye_w/2, y + eye_h/2]), fill=eye_color)

        # --- NEON FEATURES (Line-art drawn around the eyes) ---
        lw = int(4 * sx) # Line width
        
        if template == 'feline':
            # Cat ears
            ear_h = 40 * morph_scale
            draw.line(scale([lx-20, y-eye_h/2, lx-30, y-eye_h/2-ear_h, lx+10, y-eye_h/2]), fill=eye_color, width=lw)
            draw.line(scale([rx+20, y-eye_h/2, rx+30, y-eye_h/2-ear_h, rx-10, y-eye_h/2]), fill=eye_color, width=lw)
            # Whiskers
            w_w = 40 * morph_scale
            w_move = math.sin(elapsed * 10) * 5
            for wy in [y, y+15]:
                draw.line(scale([lx-20, wy, lx-20-w_w, wy+w_move]), fill=eye_color, width=lw)
                draw.line(scale([rx+20, wy, rx+20+w_w, wy+w_move]), fill=eye_color, width=lw)
            # Tiny mouth
            draw.arc(scale([110, y+20, 120, y+30+mouth_open]), 0, 180, fill=eye_color, width=lw)
            draw.arc(scale([120, y+20, 130, y+30+mouth_open]), 0, 180, fill=eye_color, width=lw)
            
            if animal_data.get('mane'):
                # Lion mane (dashed circular rays)
                for i in range(12):
                    ang = i * (math.pi / 6) + (elapsed * 0.5)
                    r1 = 70 + (1 - morph_scale) * 50
                    r2 = 90 * morph_scale
                    mx1 = 120 + math.cos(ang) * r1
                    my1 = 120 + math.sin(ang) * r1
                    mx2 = 120 + math.cos(ang) * r2
                    my2 = 120 + math.sin(ang) * r2
                    if morph_scale > 0.1:
                        draw.line(scale([mx1, my1, mx2, my2]), fill=eye_color, width=lw)

        elif template == 'canine':
            # Floppy ears
            ear_h = 50 * morph_scale
            flap = math.sin(elapsed * 10) * 10
            draw.line(scale([lx-20, y-eye_h/2, lx-40-flap, y-eye_h/2+ear_h]), fill=eye_color, width=lw)
            draw.line(scale([rx+20, y-eye_h/2, rx+40+flap, y-eye_h/2+ear_h]), fill=eye_color, width=lw)
            # Dog nose & mouth
            draw.ellipse(scale([110, y+10, 130, y+25]), fill=eye_color)
            draw.line(scale([120, y+25, 120, y+35]), fill=eye_color, width=lw)
            draw.arc(scale([100, y+15, 120, y+40+mouth_open]), 0, 180, fill=eye_color, width=lw)
            draw.arc(scale([120, y+15, 140, y+40+mouth_open]), 0, 180, fill=eye_color, width=lw)
            # Tongue if open
            if mouth_open > 5:
                draw.ellipse(scale([115, y+35, 125, y+45+mouth_open]), fill=eye_color)

        elif template == 'bird':
            # Beak in center
            beak_w = 20 * morph_scale
            beak_h = 15 * morph_scale
            chirp = (mouth_open / 2) * morph_scale
            draw.polygon(scale([120-beak_w, y+10, 120+beak_w, y+10, 120, y+10+beak_h]), fill=eye_color)
            if chirp > 0:
                draw.polygon(scale([120-beak_w+5, y+12+chirp, 120+beak_w-5, y+12+chirp, 120, y+12+beak_h+chirp]), fill=eye_color)
            # Little feather tuft on top
            draw.line(scale([120, y-40, 115, y-55*morph_scale]), fill=eye_color, width=lw)
            draw.line(scale([120, y-40, 125, y-55*morph_scale]), fill=eye_color, width=lw)

        elif template == 'ungulate':
            # Horns / Ears
            horn_h = 40 * morph_scale
            if animal_lower in ['cow', 'bull', 'buffalo', 'goat']:
                draw.arc(scale([lx-20, y-eye_h/2-horn_h, lx+20, y-eye_h/2]), 90, 270, fill=eye_color, width=lw)
                draw.arc(scale([rx-20, y-eye_h/2-horn_h, rx+20, y-eye_h/2]), -90, 90, fill=eye_color, width=lw)
            else: # Horse/Deer ears
                draw.line(scale([lx-10, y-eye_h/2, lx-30, y-eye_h/2-horn_h, lx, y-eye_h/2-10]), fill=eye_color, width=lw)
                draw.line(scale([rx+10, y-eye_h/2, rx+30, y-eye_h/2-horn_h, rx, y-eye_h/2-10]), fill=eye_color, width=lw)
            # Big snout line
            snout_w = 60 * morph_scale
            chew = math.sin(elapsed * 5) * 5
            draw.arc(scale([120-snout_w+chew, y+30, 120+snout_w+chew, y+60]), 0, 180, fill=eye_color, width=lw)
            # Elephant trunk
            if animal_data.get('trunk'):
                draw.line(scale([120, y+20, 120+math.sin(elapsed*3)*20, y+80*morph_scale]), fill=eye_color, width=lw*2)

        elif template == 'rodent':
            # Big round/tall ears
            long_ears = animal_data.get('long_ears', False)
            ear_h = (60 if long_ears else 30) * morph_scale
            ear_w = 20 if long_ears else 30
            draw.ellipse(scale([lx-ear_w, y-eye_h/2-ear_h, lx+ear_w, y-eye_h/2]), outline=eye_color, width=lw)
            draw.ellipse(scale([rx-ear_w, y-eye_h/2-ear_h, rx+ear_w, y-eye_h/2]), outline=eye_color, width=lw)
            # Buck teeth
            draw.rectangle(scale([112, y+20, 119, y+30+mouth_open]), outline=eye_color, width=lw-1)
            draw.rectangle(scale([121, y+20, 128, y+30+mouth_open]), outline=eye_color, width=lw-1)

        elif template == 'reptile':
            # Flicking tongue
            if math.sin(elapsed * 10) > 0.6 and morph_scale > 0.8:
                ty = y+30
                draw.line(scale([120, y+10, 120, ty+20]), fill=eye_color, width=lw)
                draw.line(scale([120, ty+20, 110, ty+30]), fill=eye_color, width=lw)
                draw.line(scale([120, ty+20, 130, ty+30]), fill=eye_color, width=lw)

        elif template == 'bear':
            # Small round ears
            ear_s = 25 * morph_scale
            draw.arc(scale([lx-ear_s, y-eye_h/2-ear_s, lx+ear_s, y-eye_h/2+ear_s]), 180, 360, fill=eye_color, width=lw)
            draw.arc(scale([rx-ear_s, y-eye_h/2-ear_s, rx+ear_s, y-eye_h/2+ear_s]), 180, 360, fill=eye_color, width=lw)
            # Snout
            draw.ellipse(scale([100, y+15, 140, y+35]), outline=eye_color, width=lw)
            draw.ellipse(scale([115, y+20, 125, y+25]), fill=eye_color)

        elif template == 'aquatic':
            # Bubbles rising
            for i, bx in enumerate([80, 160, 120]):
                speed = 40 + i*10
                offset = i * 20
                by = 240 - ((elapsed * speed + offset) % 240)
                if morph_scale > 0.5:
                    draw.ellipse(scale([bx, by, bx+10, by+10]), outline=eye_color, width=lw-1)
            # Side fins
            fin_w = 30 * morph_scale
            flap = math.sin(elapsed * 5) * 10
            draw.polygon(scale([lx-20, y, lx-20-fin_w, y-10-flap, lx-20-fin_w, y+10+flap]), fill=eye_color)
            draw.polygon(scale([rx+20, y, rx+20+fin_w, y-10-flap, rx+20+fin_w, y+10+flap]), fill=eye_color)

        return img

    def _render_loop(self) -> None:
        # Initialize animation state
        self._animation_frames = []
        self._animation_end_time = None
        self._animation_frame_idx = 0
        self._animation_frame_time = 0.1
        self._last_frame_time = 0
        
        self._active_animal = None
        self._active_animal_end = 0
        self._active_animal_start = 0
        
        while self._running:
            loop_start = time.time()
            try:
                now = time.time()
                
                # --- Procedural Animal Override ---
                if self._active_animal and now < self._active_animal_end:
                    elapsed = now - self._active_animal_start
                    frame = self._draw_animal_frame(self._active_animal, elapsed)
                    if self.single_display_both_eyes:
                        self.display.draw_eyes(frame, None)
                    else:
                        self.display.draw_eyes(frame, frame)
                        
                    elapsed_loop = time.time() - loop_start
                    sleep_dur = max(0.01, self.frame_time - elapsed_loop)
                    time.sleep(sleep_dur)
                    continue
                else:
                    self._active_animal = None
                
                # --- GIF Animation Override ---
                if self._animation_end_time and now < self._animation_end_time:
                    if self._animation_frames:
                        if now - self._last_frame_time >= self._animation_frame_time:
                            self._animation_frame_idx = (self._animation_frame_idx + 1) % len(self._animation_frames)
                            self._last_frame_time = now
                        
                        frame = self._animation_frames[self._animation_frame_idx]
                        if self.single_display_both_eyes:
                            self.display.draw_eyes(frame, None)
                        else:
                            self.display.draw_eyes(frame, frame)
                    
                    # Regulate frame rate and skip procedural eyes
                    elapsed_loop = time.time() - loop_start
                    sleep_dur = self.frame_time - elapsed_loop
                    if sleep_dur > 0.001:
                        time.sleep(sleep_dur)
                    continue
                else:
                    self._animation_end_time = None
                    self._animation_frames = []

                # --- Procedural Eyes Logic ---
                # Process Blinking Logic
                if not self._is_blinking and not self.is_sleeping and now >= self._next_blink_time:
                    self.trigger_blink()

                # Process Saccades (Micro-jitter)
                if now >= self._next_saccade_time and not self.is_sleeping:
                    self._saccade_x = random.uniform(-1.5, 1.5)
                    self._saccade_y = random.uniform(-1.5, 1.5)
                    self._next_saccade_time = now + random.uniform(0.1, 0.5)
                elif self.is_sleeping:
                    self._saccade_x = 0.0
                    self._saccade_y = 0.0

                # Smooth Gaze Lerp with safety clamping (-35 to +35)
                target_gx = max(-35.0, min(35.0, self.gaze_x + self._saccade_x))
                target_gy = max(-25.0, min(25.0, self.gaze_y + self._saccade_y))
                self._smooth_gaze_x += (target_gx - self._smooth_gaze_x) * 0.35
                self._smooth_gaze_y += (target_gy - self._smooth_gaze_y) * 0.35

                blink_cover = 0.0
                if self.is_sleeping:
                    blink_cover = 1.0
                elif self._is_blinking:
                    elapsed = now - self._blink_start_time
                    if elapsed < self._blink_duration:
                        progress = max(0.0, min(1.0, elapsed / self._blink_duration))
                        # Sinusoidal blink trajectory: 0 -> 1 -> 0
                        blink_cover = math.sin(progress * math.pi)
                    else:
                        self._is_blinking = False
                        self._next_blink_time = now + random.uniform(2.5, 5.5)

                # Render frames
                if self.single_display_both_eyes:
                    single_frame = self._draw_both_eyes_single_frame(blink_cover)
                    self.display.draw_eyes(single_frame, None)
                else:
                    left_frame = self._draw_single_eye(is_left=True, blink_cover=blink_cover)
                    right_frame = self._draw_single_eye(is_left=False, blink_cover=blink_cover)
                    self.display.draw_eyes(left_frame, right_frame)
            except Exception as e:
                logger.debug(f"EyeRenderer frame error: {e}")

            # Regulate frame rate (30 FPS)
            elapsed_loop = time.time() - loop_start
            sleep_dur = self.frame_time - elapsed_loop
            if sleep_dur > 0.001:
                time.sleep(sleep_dur)

    def _draw_eye_pill(
        self,
        draw: Any,
        cx: float,
        cy: float,
        w_radius: float,
        h_radius: float,
        color: Tuple[int, int, int],
        is_happy: bool = False,
        is_closed: bool = False,
    ) -> None:
        """Draws the capsule/pill eye or closed slit based on boot/Eyes.cpp."""
        w_radius = max(2.0, w_radius)
        h_radius = max(0.0, h_radius)

        if is_closed or h_radius <= 4.0:
            # Rounded horizontal slit
            slit_w = max(10.0, w_radius * 1.8)
            slit_h = 6.0
            draw.rounded_rectangle(
                [cx - slit_w / 2.0, cy - slit_h / 2.0, cx + slit_w / 2.0, cy + slit_h / 2.0],
                radius=3,
                fill=color,
            )
            return

        if is_happy:
            # Happy upward curved inverted-U shape (^ ^)
            arc_w = max(12.0, w_radius * 1.8)
            arc_h = 24.0
            draw.arc(
                [cx - arc_w / 2.0, cy - arc_h, cx + arc_w / 2.0, cy + arc_h / 2.0],
                start=200,
                end=340,
                fill=color,
                width=8,
            )
            return

        # Regular open capsule / ellipse eye
        x0 = cx - w_radius
        y0 = cy - h_radius
        x1 = cx + w_radius
        y1 = cy + h_radius

        if x1 > x0 and y1 > y0:
            # Outer Pill / Ellipse
            draw.ellipse([x0, y0, x1, y1], fill=color)

            # Specular Highlight (Cute reflection dot)
            hl_r = max(1.0, min(w_radius, h_radius) * 0.28)
            hl_x = cx + w_radius * 0.28
            hl_y = cy - h_radius * 0.32
            draw.ellipse([hl_x - hl_r, hl_y - hl_r, hl_x + hl_r, hl_y + hl_r], fill=(255, 255, 255))

    def _draw_both_eyes_single_frame(self, blink_cover: float = 0.0) -> Any:
        """Render both Left and Right eyes side-by-side onto a single 240x240 display buffer."""
        expr = self.target_expr
        if _HAS_PIL:
            img = Image.new("RGB", (self.width, self.height), (0, 0, 0))
            draw = ImageDraw.Draw(img)

            # Colors & Dimensions from Eyes.cpp
            color = expr.iris_color_rgb
            base_w = self.w_eye / 2.0  # radius X = 20
            base_h = self.h_eye / 2.0  # radius Y = 35

            # Dynamic squash factor
            left_blink = blink_cover
            right_blink = blink_cover
            if self.is_winking:
                left_blink = 1.0
                right_blink = 0.0

            left_h = max(0.0, base_h * (1.0 - left_blink))
            right_h = max(0.0, base_h * (1.0 - right_blink))

            is_happy = expr.name.lower() in ["happy", "ecstatic"]

            # Eye Centers with Gaze offset
            cy = self.y_eye_center + self._smooth_gaze_y
            cx_left = self.x_eye_left + self._smooth_gaze_x
            cx_right = self.x_eye_right + self._smooth_gaze_x

            # Draw Left Eye
            self._draw_eye_pill(
                draw, cx_left, cy, base_w, left_h, color, is_happy=is_happy, is_closed=(left_blink >= 0.85)
            )

            # Draw Right Eye
            self._draw_eye_pill(
                draw, cx_right, cy, base_w, right_h, color, is_happy=is_happy, is_closed=(right_blink >= 0.85)
            )

            return img
        else:
            return {
                "mode": "single_display_both_eyes",
                "expression": expr.name,
                "blink": blink_cover,
            }

    def _draw_single_eye(self, is_left: bool, blink_cover: float = 0.0) -> Any:
        """Render a single full-screen 240x240 round eye buffer for dual separate displays."""
        expr = self.target_expr
        cx = self.width / 2.0 + self._smooth_gaze_x
        cy = self.height / 2.0 + self._smooth_gaze_y
        r_w = self.w_eye * 1.2
        r_h = max(0.0, (self.h_eye * 1.2) * (1.0 - blink_cover))

        if _HAS_PIL:
            img = Image.new("RGB", (self.width, self.height), (0, 0, 0))
            draw = ImageDraw.Draw(img)
            self._draw_eye_pill(
                draw, cx, cy, r_w, r_h, expr.iris_color_rgb, is_closed=(blink_cover >= 0.85)
            )
            return img
        else:
            return {
                "is_left": is_left,
                "expression": expr.name,
                "center": (cx, cy),
                "blink": blink_cover,
            }
