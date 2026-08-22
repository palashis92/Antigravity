"""Master test runner for the complete LUMI AI Companion Robot test suite."""

import sys
import traceback
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tests.test_ai as tai
import tests.test_config as tc
import tests.test_database as td
import tests.test_event_bus as teb
import tests.test_eyes as te
import tests.test_memory as tm
import tests.test_motion as tmot
import tests.test_state_manager as tsm
import tests.test_vision as tv


class DummyMonkeypatch:
    def setenv(self, key, value):
        import os
        os.environ[key] = value


def run_test(name, fn, *args):
    try:
        fn(*args)
        print(f"  [PASS] {name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 65)
    print("Running Full LUMI AI Companion Robot Test Suite")
    print("=" * 65)

    total = 0
    passed = 0

    tests_to_run = [
        # Config & Environment
        ("test_default_settings_load", tc.test_default_settings_load, ()),
        ("test_env_overrides", tc.test_env_overrides, (DummyMonkeypatch(),)),
        # Database & Memory
        ("test_in_memory_database", td.test_in_memory_database, ()),
        ("test_insert_and_query", td.test_insert_and_query, ()),
        ("test_person_lifecycle", tm.test_person_lifecycle, ()),
        ("test_privacy_consent_enforcement", tm.test_privacy_consent_enforcement, ()),
        ("test_right_to_be_forgotten", tm.test_right_to_be_forgotten, ()),
        ("test_reminders_and_due_check", tm.test_reminders_and_due_check, ()),
        ("test_conversation_history", tm.test_conversation_history, ()),
        # State Machine & Event Bus
        ("test_initial_state", tsm.test_initial_state, ()),
        ("test_valid_transitions", tsm.test_valid_transitions, ()),
        ("test_invalid_transition_rejected", tsm.test_invalid_transition_rejected, ()),
        ("test_state_change_listener", tsm.test_state_change_listener, ()),
        ("test_sync_event_publishing", teb.test_sync_event_publishing, ()),
        ("test_topic_isolation", teb.test_topic_isolation, ()),
        ("test_wildcard_subscription", teb.test_wildcard_subscription, ()),
        ("test_async_event_publishing", teb.test_async_event_publishing, ()),
        # Motion & Kinematics
        ("test_cubic_easing", tmot.test_cubic_easing, ()),
        ("test_servo_controller_limits_and_interpolation", tmot.test_servo_controller_limits_and_interpolation, ()),
        ("test_head_and_arms", tmot.test_head_and_arms, ()),
        ("test_gestures_execution", tmot.test_gestures_execution, ()),
        # Procedural Eye Displays
        ("test_expression_configs", te.test_expression_configs, ()),
        ("test_eye_renderer_lifecycle", te.test_eye_renderer_lifecycle, ()),
        # Vision Subsystems
        ("test_plant_disease_detection", tv.test_plant_disease_detection, ()),
        ("test_chess_vision_fen", tv.test_chess_vision_fen, ()),
        ("test_face_recognition", tv.test_face_recognition, ()),
        # AI & Conversation
        ("test_tool_registry", tai.test_tool_registry, ()),
        ("test_conversation_engine_responses", tai.test_conversation_engine_responses, ()),
    ]

    for name, fn, args in tests_to_run:
        total += 1
        if run_test(name, fn, *args):
            passed += 1

    print("=" * 65)
    print(f"Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 65)

    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
