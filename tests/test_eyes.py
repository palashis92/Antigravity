"""Unit tests for procedural EyeRenderer and expressions."""

import time
from lumi.eyes.expressions import EXPRESSIONS, ExpressionConfig
from lumi.eyes.renderer import EyeRenderer
from lumi.hardware.mocks import MockDisplayBackend


def test_expression_configs() -> None:
    assert "neutral" in EXPRESSIONS
    assert "happy" in EXPRESSIONS
    assert "thinking" in EXPRESSIONS
    assert EXPRESSIONS["happy"].lower_lid_cover > 0.4


def test_eye_renderer_lifecycle() -> None:
    backend = MockDisplayBackend()
    renderer = EyeRenderer(display_backend=backend, target_fps=30)
    renderer.start()

    renderer.set_expression("happy")
    assert renderer.target_expr.name == "happy"

    renderer.set_gaze(0.5, -0.5)
    assert renderer.gaze_x > 0.0

    renderer.trigger_blink()
    time.sleep(0.1)
    renderer.stop()
