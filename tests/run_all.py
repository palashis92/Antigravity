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
import tests.test_anjum_mode as tanjum
import tests.test_audio as taud
import tests.test_identity_pipeline as tip
import tests.test_improvements as timp
import tests.test_phase1_hardware_telemetry as tp1
import tests.test_phase2_audio_session as tp2
import tests.test_phase3_vision_perception as tp3
import tests.test_phase4_architecture_fsm as tp4
import tests.test_phase5_lifelike_behavior as tp5


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
        ("test_ground_truth_channel_mappings_and_clamps", tmot.test_ground_truth_channel_mappings_and_clamps, ()),
        ("test_enhanced_arms_and_gestures", tmot.test_enhanced_arms_and_gestures, ()),
        ("test_multi_joint_simultaneous_and_power_cut_recovery", tmot.test_multi_joint_simultaneous_and_power_cut_recovery, ()),
        # Procedural Eye Displays
        ("test_expression_configs", te.test_expression_configs, ()),
        ("test_eye_renderer_lifecycle", te.test_eye_renderer_lifecycle, ()),
        # Vision & Face Recognition Subsystems
        ("test_plant_disease_detection", tv.test_plant_disease_detection, ()),
        ("test_chess_vision_fen", tv.test_chess_vision_fen, ()),
        ("test_face_recognition", tv.test_face_recognition, ()),
        ("test_person_model_age_and_multi_embedding", tip.test_person_model_age_and_multi_embedding, ()),
        ("test_test_case_a_and_b_recognition_and_isolation", tip.test_test_case_a_and_b_recognition_and_isolation, ()),
        ("test_test_case_c_unknown_person_no_hallucinated_name", tip.test_test_case_c_unknown_person_no_hallucinated_name, ()),
        ("test_test_case_d_face_and_name_never_cross_associated", tip.test_test_case_d_face_and_name_never_cross_associated, ()),
        ("test_pending_face_timestamp_freshness", tip.test_pending_face_timestamp_freshness, ()),
        # AI & Conversation
        ("test_tool_registry", tai.test_tool_registry, ()),
        ("test_conversation_engine_responses", tai.test_conversation_engine_responses, ()),
        # Anjum Companion & Speech-Therapy Mode
        ("test_anjum_state_transitions", tanjum.test_anjum_state_transitions, ()),
        ("test_companion_activation_and_deactivation", tanjum.test_companion_activation_and_deactivation, ()),
        ("test_absence_timeout", tanjum.test_absence_timeout, ()),
        ("test_proactive_stimulus_cooldown", tanjum.test_proactive_stimulus_cooldown, ()),
        ("test_counting_game_bengali_digits", tanjum.test_counting_game_bengali_digits, ()),
        ("test_counting_game_english_digits", tanjum.test_counting_game_english_digits, ()),
        ("test_identity_and_praise_speech", tanjum.test_identity_and_praise_speech, ()),
        # Audio & Proximity Filtering
        ("test_proximity_filter_silence", taud.test_proximity_filter_silence, ()),
        ("test_proximity_filter_single_speaker", taud.test_proximity_filter_single_speaker, ()),
        ("test_proximity_filter_overlap_near_field_priority", taud.test_proximity_filter_overlap_near_field_priority, ()),
        ("test_proximity_filter_cooldown", taud.test_proximity_filter_cooldown, ()),
        ("test_speaker_identifier_lifecycle", taud.test_speaker_identifier_lifecycle, ()),
        ("test_speaker_identifier_cosine_similarity", taud.test_speaker_identifier_cosine_similarity, ()),
        ("test_doa_body_orientation_mapping", taud.test_doa_body_orientation_mapping, ()),
        # LUMI System Improvements (Auto-Relax, Speaker Stop, Software AEC, Async Tools)
        ("test_servo_auto_relax_lifecycle", timp.test_servo_auto_relax_lifecycle, ()),
        ("test_speaker_interface_stop_stream_and_shutdown", timp.test_speaker_interface_stop_stream_and_shutdown, ()),
        ("test_gemini_live_software_aec", timp.test_gemini_live_software_aec, ()),
        ("test_gemini_live_async_tool_execution", timp.test_gemini_live_async_tool_execution, ()),
        ("test_learned_rules_store_and_adaptation", timp.test_learned_rules_store_and_adaptation, ()),
        ("test_camera_backend_close_and_face_service_cache", timp.test_camera_backend_close_and_face_service_cache, ()),
        ("test_greeting_cooldown_and_temporal_context", timp.test_greeting_cooldown_and_temporal_context, ()),
        ("test_memory_turn_recording_and_palash_fallback", timp.test_memory_turn_recording_and_palash_fallback, ()),
        ("test_tool_memorize_and_recall_palash_identity", timp.test_tool_memorize_and_recall_palash_identity, ()),
        ("test_spatial_audio_clean_downmix", timp.test_spatial_audio_clean_downmix, ()),
        ("test_proactive_recall_identity_query", timp.test_proactive_recall_identity_query, ()),
        ("test_owner_mizan_and_person_specific_memory_isolation", timp.test_owner_mizan_and_person_specific_memory_isolation, ()),
        ("test_silence_command_and_audio_suppression", timp.test_silence_command_and_audio_suppression, ()),
        ("test_conversation_context_retention_in_setup", timp.test_conversation_context_retention_in_setup, ()),
        ("test_unknown_greeting_suppression_and_sticky_active_person", timp.test_unknown_greeting_suppression_and_sticky_active_person, ()),
        ("test_bilingual_owner_matching_and_memorize_person", timp.test_bilingual_owner_matching_and_memorize_person, ()),
        ("test_owner_auto_enroll_and_persistent_identity", timp.test_owner_auto_enroll_and_persistent_identity, ()),
        ("test_universal_continuous_face_learning_and_conversational_intro", timp.test_universal_continuous_face_learning_and_conversational_intro, ()),
        ("test_setup_prompt_injects_owner_facts", timp.test_setup_prompt_injects_owner_facts, ()),
        ("test_camera_interface_micro_caching", timp.test_camera_interface_micro_caching, ()),
        ("test_mem0_deterministic_bengali_fact_extraction", timp.test_mem0_deterministic_bengali_fact_extraction, ()),
        ("test_display_driver_single_display_ce0_only", tp1.test_display_driver_single_display_ce0_only, ()),
        ("test_display_driver_dual_display_flag", tp1.test_display_driver_dual_display_flag, ()),
        ("test_speaker_wm8960_detection", tp1.test_speaker_wm8960_detection, ()),
        ("test_mic_wm8960_detection", tp1.test_mic_wm8960_detection, ()),
        ("test_servo_slew_rate_limiter", tp1.test_servo_slew_rate_limiter, ()),
        ("test_telemetry_logger_non_blocking", tp1.test_telemetry_logger_non_blocking, ()),
        ("test_mono_to_stereo_interleaving", tp1.test_mono_to_stereo_interleaving, ()),
        # Phase 2: Audio & Session Stability
        ("test_turn_arbiter_silence_mode", tp2.test_turn_arbiter_silence_mode, ()),
        ("test_turn_arbiter_speaker_echo_ducking", tp2.test_turn_arbiter_speaker_echo_ducking, ()),
        ("test_turn_arbiter_dialogue_window_and_ambient_rejection", tp2.test_turn_arbiter_dialogue_window_and_ambient_rejection, ()),
        ("test_gemini_live_awake_gating", tp2.test_gemini_live_awake_gating, ()),
        ("test_gemini_live_inject_context_wakes_dialogue", tp2.test_gemini_live_inject_context_wakes_dialogue, ()),
        ("test_gemini_live_barge_in_telemetry", tp2.test_gemini_live_barge_in_telemetry, ()),
        # Phase 3: Vision & Perception Throughput
        ("test_face_recognition_track_caching_throughput", tp3.test_face_recognition_track_caching_throughput, ()),
        ("test_target_lost_search_hysteresis_and_smooth_pan", tp3.test_target_lost_search_hysteresis_and_smooth_pan, ()),
        ("test_anjum_mode_three_frame_confirmation_and_adult_exit", tp3.test_anjum_mode_three_frame_confirmation_and_adult_exit, ()),
        # Phase 4: Architecture & FSM Robustness
        ("test_state_manager_watchdog_recovers_stuck_transient_state", tp4.test_state_manager_watchdog_recovers_stuck_transient_state, ()),
        ("test_state_manager_watchdog_ignores_stable_states", tp4.test_state_manager_watchdog_ignores_stable_states, ()),
        ("test_persona_segregation_reset_dialogue_state", tp4.test_persona_segregation_reset_dialogue_state, ()),
        ("test_learned_rules_sqlite_persistence", tp4.test_learned_rules_sqlite_persistence, ()),
        ("test_listening_state_transitions_to_greeting_and_observing", tp4.test_listening_state_transitions_to_greeting_and_observing, ()),
        ("test_mem0_cloud_empty_turn_guarded", tp4.test_mem0_cloud_empty_turn_guarded, ()),
        # Phase 5: Lifelike Behavior & Fluidity
        ("test_servo_holding_torque_and_breathing_motion", tp5.test_servo_holding_torque_and_breathing_motion, ()),
        ("test_continuous_affective_eyes_modulation", tp5.test_continuous_affective_eyes_modulation, ()),
        ("test_instant_local_acoustic_reflex", tp5.test_instant_local_acoustic_reflex, ()),
        ("test_anjum_closed_loop_adaptive_therapy", tp5.test_anjum_closed_loop_adaptive_therapy, ()),
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
