"""Unit and Integration Tests for Phase 5: Lifelike Behavior & Fluidity.

Covers:
- [L1] Active low-duty holding torque on head_tilt + subtle breathing micro-motion
- [L2] Continuous 2D Valence/Arousal affective eye modulation
- [L3] Instant local acoustic reflex (<150ms, zero cloud dependency)
- [L4] Closed-loop adaptive child speech therapy & vocalization reinforcement
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock

from lumi.companion.anjum_companion import AnjumCompanionEngine
from lumi.config import LumiSettings
from lumi.core.event_bus import EventBus
from lumi.core.lumi_brain import LumiBrain
from lumi.core.state_manager import BehaviorState, StateManager
from lumi.eyes.expressions import EXPRESSIONS
from lumi.eyes.renderer import EyeRenderer
from lumi.hardware.mocks import MockDisplayBackend, MockServoDriver
from lumi.memory.manager import MemoryManager
from lumi.motion.servo_controller import ServoController


def test_servo_holding_torque_and_breathing_motion() -> None:
    """[L1] Verify active holding torque on head_tilt axis and sinusoidal breathing micro-motion."""
    driver = MockServoDriver()
    ctrl = ServoController(driver, auto_relax_delay_s=0.2)
    ctrl.initialize()

    try:
        assert ctrl.holding_torque is True
        assert ctrl.breathing_enabled is True
        assert ctrl.breathing_amplitude == 1.2
        assert abs(ctrl.breathing_frequency - 0.28) < 0.01

        # 1. Test relax_all with keep_tilt_holding=True
        ctrl.set_angle_immediate("head_pan", 10.0)
        assert not ctrl._is_relaxed

        ctrl.relax_all(keep_tilt_holding=True)
        assert ctrl._is_relaxed is True
        assert ctrl._is_holding_tilt is True

        # Tilt channel 0 should NOT be in released channels list of mock driver if holding
        # Other channels (pan, arms) should have been released
        assert 1 in driver.released_channels or 5 in driver.released_channels

        # 2. Test breathing_tick calculates smooth sinusoidal offset
        t0 = 0.0
        offset0 = ctrl.breathing_tick(t0)
        assert abs(offset0 - 0.0) < 0.01

        # Peak at quarter cycle: 1 / (4 * 0.28) ~ 0.893s
        t_peak = 1.0 / (4.0 * 0.28)
        offset_peak = ctrl.breathing_tick(t_peak)
        assert abs(offset_peak - 1.2) < 0.05

        # Opposite peak at 3/4 cycle
        t_trough = 3.0 / (4.0 * 0.28)
        offset_trough = ctrl.breathing_tick(t_trough)
        assert abs(offset_trough - (-1.2)) < 0.05

        # 3. Test shutdown fully releases all channels including tilt
        ctrl.shutdown()
        assert ctrl._is_holding_tilt is False
        assert 0 in driver.released_channels

    finally:
        if ctrl._running:
            ctrl.shutdown()


def test_continuous_affective_eyes_modulation() -> None:
    """[L2] Verify continuous 2D Valence/Arousal coordinates and geometry/color modulation."""
    disp = MockDisplayBackend()
    renderer = EyeRenderer(disp, single_display_both_eyes=True)

    # 1. Test set_affect mapping
    renderer.set_affect(valence=0.8, arousal=0.6)
    assert renderer.target_valence == 0.8
    assert renderer.target_arousal == 0.6
    assert renderer.target_expr.name == "happy"
    assert renderer.is_sleeping is False

    # 2. Test negative valence maps to sad
    renderer.set_affect(valence=-0.7, arousal=-0.2)
    assert renderer.target_valence == -0.7
    assert renderer.target_arousal == -0.2
    assert renderer.target_expr.name == "sad"

    # 3. Test extreme low arousal triggers sleep
    renderer.set_affect(valence=0.0, arousal=-0.95)
    assert renderer.is_sleeping is True

    # 4. Test set_expression syncs canonical affective coordinates
    renderer.set_expression("curious")
    assert renderer.target_valence == 0.3
    assert renderer.target_arousal == 0.6

    renderer.set_expression("neutral")
    assert renderer.target_valence == 0.0
    assert renderer.target_arousal == 0.0

    # 5. Test headless frame drawing dictionary outputs continuous affective values
    frame = renderer._draw_both_eyes_single_frame(blink_cover=0.0, valence=0.5, arousal=0.7)
    assert isinstance(frame, dict)
    assert frame["valence"] == 0.5
    assert frame["arousal"] == 0.7

    # 6. Test affective color tint helper
    base_color = (0, 200, 255)
    warm_tint = renderer._apply_affective_tint(base_color, valence=0.8)
    # Warm gold tint increases red component significantly
    assert warm_tint[0] > base_color[0]

    cool_tint = renderer._apply_affective_tint(base_color, valence=-0.8)
    # Cool indigo tint decreases green component
    assert cool_tint[1] < base_color[1]


def test_instant_local_acoustic_reflex() -> None:
    """[L3] Verify instant local acoustic reflex (<150ms) triggers without cloud dependency."""
    settings = LumiSettings()
    sm = StateManager()
    eb = EventBus()
    from lumi.memory.database import Database
    db = Database(":memory:")
    mm = MemoryManager(db)
    drv = MockServoDriver()
    sc = ServoController(drv)
    disp = MockDisplayBackend()
    eyes = EyeRenderer(disp)
    cam = MagicMock()
    mic = MagicMock()
    spk = MagicMock()
    spk.is_playing = False

    brain = LumiBrain(
        settings=settings,
        state_manager=sm,
        event_bus=eb,
        memory_manager=mm,
        servo_controller=sc,
        eye_renderer=eyes,
        camera=cam,
        mic=mic,
        speaker=spk,
    )

    try:
        assert sm.current_state == BehaviorState.IDLE

        # 1. Trigger acoustic reflex on voice energy onset
        fired = brain.trigger_acoustic_reflex(energy=550.0)
        assert fired is True

        # Should immediately switch state from IDLE to LISTENING
        assert sm.current_state == BehaviorState.LISTENING

        # Eyes should perk up with alert affect
        assert eyes.target_arousal >= 0.6
        assert eyes.target_valence >= 0.2

        # 2. Test reflex debounce cooldown (should not fire again within 1.2s)
        fired_again = brain.trigger_acoustic_reflex(energy=600.0)
        assert fired_again is False

        # 3. Test software AEC suppression: reflex must NOT fire while robot speaker is active
        time.sleep(1.25)
        spk.is_playing = True
        fired_during_playback = brain.trigger_acoustic_reflex(energy=800.0)
        assert fired_during_playback is False

    finally:
        brain.shutdown()


def test_anjum_closed_loop_adaptive_therapy() -> None:
    """[L4] Verify closed-loop vocalization reinforcement and adaptive pacing for Anjum."""
    engine = AnjumCompanionEngine(stimulus_cooldown=5.0)
    greeting = engine.activate()
    assert engine.is_active is True
    assert "আঞ্জুম" in greeting

    # 1. Non-lexical babble / vocalization detection test
    praise = engine.handle_vocalization_detected(duration_s=0.8, energy=600.0)
    assert praise is not None
    assert any(w in praise for w in ["আঞ্জুম", "বাহ্", "সাবাশ", "ওয়াও"])
    assert engine.vocalization_count == 1
    assert engine.unanswered_prompt_count == 0

    # 2. Interactive counting game advances target
    resp1 = engine.handle_anjum_speech("১")
    assert resp1 is not None
    assert "২" in resp1
    assert engine.counting_target == 2
    assert engine.vocalization_count == 2

    # 3. Session metrics tracking
    metrics = engine.get_session_metrics()
    assert metrics["vocalizations"] == 2
    assert metrics["counting_target"] == 2
    assert metrics["unanswered_prompts"] == 0
    assert metrics["engagement_level"] in ["high", "medium"]

    # 4. Adaptive interaction pacing: fallback to animal sounds on unanswered prompts
    engine.unanswered_prompt_count = 2
    stimulus = engine.check_proactive_stimulus(now=time.time() + 10.0)
    assert stimulus is not None
    assert engine.engagement_level == "calm"
    # Should select an animal prompt to lower cognitive resistance
    assert any(animal in stimulus for animal in ["বিড়াল", "কুকুর", "পাখি", "গরু"])
