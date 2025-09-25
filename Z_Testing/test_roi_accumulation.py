#!/usr/bin/env python3
"""
Test script for Enhanced ROI-Based Wood Detection and Grading Accumulation System

This test verifies the new functionality:
1. Defect accumulation during ROI overlap sessions
2. Session-based defect data tracking across multiple frames
3. Processing accumulated defects when wood exits ROI
4. Integration with SS-EN 1611-1 grading system
5. Performance tracking and error handling during accumulation periods
6. Multiple wood pieces and session timeout handling

Author: Kilo Code
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple
import unittest
from unittest.mock import Mock, MagicMock

# Import modules
from modules.roi_module import (
    ROIManager, OverlapDetector, ROIBasedWorkflowManager,
    ROISession, ROISessionStatus, ROIModule
)
from modules.detection_module import DetectionModule
from modules.grading_module import determine_surface_grade, convert_grade_to_arduino_command
from modules.wood_detection_module import WoodDetectionResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockArduinoModule:
    """Mock Arduino module for testing"""
    def __init__(self):
        self.connected = True
        self.commands_sent = []

    def is_connected(self):
        return self.connected

    def send_grade_command(self, command):
        self.commands_sent.append(command)
        logger.info(f"Mock Arduino: Sent command '{command}'")
        return True

class MockDetectionModule:
    """Mock detection module for testing"""
    def __init__(self):
        self.defect_results = {}

    def detect_defects_in_full_frame(self, frame, camera_name):
        """Return mock defect detection results"""
        # Simulate different defect patterns
        if "no_defects" in camera_name:
            defects = {}
            measurements = []
        elif "few_defects" in camera_name:
            defects = {"Unsound_Knot": 2}
            measurements = [("Unsound_Knot", 15.0, 7.5), ("Unsound_Knot", 12.0, 6.0)]
        else:  # many_defects
            defects = {"Unsound_Knot": 5, "Sound_Knot": 3}
            measurements = [
                ("Unsound_Knot", 25.0, 12.5), ("Unsound_Knot", 30.0, 15.0),
                ("Sound_Knot", 18.0, 9.0), ("Sound_Knot", 22.0, 11.0)
            ]

        # Create annotated frame (just copy for mock)
        annotated_frame = frame.copy()
        return annotated_frame, defects, measurements

class TestROIAccumulationSystem(unittest.TestCase):
    """Test cases for the enhanced ROI accumulation system"""

    def setUp(self):
        """Set up test fixtures"""
        self.camera_name = "top"
        self.roi_id = "inspection_zone"

        # Create mock modules
        self.mock_detection = MockDetectionModule()
        self.mock_grading = Mock()  # We'll use real grading functions
        self.mock_arduino = MockArduinoModule()

        # Create ROI manager
        self.roi_manager = ROIManager()

        # Create workflow manager
        self.workflow_manager = ROIBasedWorkflowManager(
            self.mock_detection, self.mock_grading, self.mock_arduino
        )

        # Create test frame
        self.test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Define test ROI
        self.roi_coords = (100, 100, 500, 300)
        self.roi_manager.define_roi(self.camera_name, self.roi_id, self.roi_coords)

    def test_session_creation_and_tracking(self):
        """Test session creation and basic tracking"""
        logger.info("Testing session creation and tracking...")

        # Start a session
        wood_piece_id = "test_wood_001"
        session_id = self.workflow_manager.start_roi_session(
            self.camera_name, self.roi_id, wood_piece_id
        )

        self.assertNotEqual(session_id, "")
        self.assertIn(session_id, self.workflow_manager.active_sessions)

        session = self.workflow_manager.active_sessions[session_id]
        self.assertEqual(session.camera_name, self.camera_name)
        self.assertEqual(session.roi_id, self.roi_id)
        self.assertEqual(session.wood_piece_id, wood_piece_id)
        self.assertEqual(session.status, ROISessionStatus.ACTIVE)

        logger.info(f"✓ Session {session_id} created successfully")

    def test_defect_accumulation(self):
        """Test defect accumulation across multiple frames"""
        logger.info("Testing defect accumulation...")

        # Start session
        session_id = self.workflow_manager.start_roi_session(self.camera_name, self.roi_id)
        session = self.workflow_manager.active_sessions[session_id]

        # Simulate multiple frames with defects
        wood_detection = {
            'bbox': (150, 150, 400, 250),
            'confidence': 0.85
        }

        # Frame 1: Few defects
        defects1 = {"Unsound_Knot": 2}
        measurements1 = [("Unsound_Knot", 15.0, 7.5), ("Unsound_Knot", 12.0, 6.0)]

        success1 = self.workflow_manager.accumulate_defects(
            session_id, defects1, wood_detection, measurements1, 0.1
        )
        self.assertTrue(success1)

        # Frame 2: More defects
        defects2 = {"Unsound_Knot": 3, "Sound_Knot": 1}
        measurements2 = [
            ("Unsound_Knot", 18.0, 9.0), ("Unsound_Knot", 22.0, 11.0),
            ("Unsound_Knot", 16.0, 8.0), ("Sound_Knot", 14.0, 7.0)
        ]

        success2 = self.workflow_manager.accumulate_defects(
            session_id, defects2, wood_detection, measurements2, 0.12
        )
        self.assertTrue(success2)

        # Check accumulation
        self.assertEqual(session.frame_count, 2)
        self.assertEqual(len(session.accumulated_defect_measurements), 6)  # 2 + 4 measurements

        # Check total defects
        results = session.get_accumulated_results()
        self.assertEqual(results['total_defects']['Unsound_Knot'], 5)  # 2 + 3
        self.assertEqual(results['total_defects']['Sound_Knot'], 1)

        logger.info(f"✓ Accumulated {session.frame_count} frames with {sum(results['total_defects'].values())} total defects")

    def test_grading_integration(self):
        """Test integration with SS-EN 1611-1 grading system"""
        logger.info("Testing grading integration...")

        # Start session and accumulate defects
        session_id = self.workflow_manager.start_roi_session(self.camera_name, self.roi_id)
        session = self.workflow_manager.active_sessions[session_id]

        # Add defects that should result in G2-2 grade
        wood_detection = {'bbox': (150, 150, 400, 250), 'confidence': 0.85}
        defects = {"Unsound_Knot": 4}  # Should trigger G2-2 due to count > 2
        measurements = [("Unsound_Knot", 20.0, 10.0)] * 4

        self.workflow_manager.accumulate_defects(
            session_id, defects, wood_detection, measurements, 0.1
        )

        # End session and check grading
        results = self.workflow_manager.end_roi_session(session_id, "test_completion")

        self.assertIsNotNone(results)
        self.assertIn('grading_results', results)

        grading_results = results['grading_results']
        self.assertIsNotNone(grading_results)
        self.assertIn('grade', grading_results)
        self.assertEqual(grading_results['total_defects'], 4)

        # Verify Arduino command was sent
        self.assertTrue(len(self.mock_arduino.commands_sent) > 0)

        logger.info(f"✓ Grading completed: Grade {grading_results['grade']}, Arduino commands: {self.mock_arduino.commands_sent}")

    def test_multiple_wood_pieces(self):
        """Test handling multiple wood pieces simultaneously"""
        logger.info("Testing multiple wood pieces...")

        # Start multiple sessions
        wood_pieces = ["wood_001", "wood_002", "wood_003"]
        session_ids = []

        for wood_piece in wood_pieces:
            session_id = self.workflow_manager.start_roi_session(
                self.camera_name, self.roi_id, wood_piece
            )
            self.assertNotEqual(session_id, "")
            session_ids.append(session_id)

        # Verify all sessions are active
        active_sessions = self.workflow_manager.get_active_sessions_for_camera(self.camera_name)
        self.assertEqual(len(active_sessions), 3)

        # Verify wood piece tracking
        for i, wood_piece in enumerate(wood_pieces):
            session = self.workflow_manager.get_session_by_wood_piece(wood_piece)
            self.assertIsNotNone(session)
            self.assertEqual(session.session_id, session_ids[i])

        logger.info(f"✓ Managing {len(session_ids)} concurrent wood piece sessions")

    def test_session_timeout_handling(self):
        """Test session timeout handling"""
        logger.info("Testing session timeout handling...")

        # Temporarily reduce timeout for testing
        original_timeout = self.workflow_manager.session_timeout
        self.workflow_manager.session_timeout = 2.0  # 2 seconds

        # Start session
        session_id = self.workflow_manager.start_roi_session(self.camera_name, self.roi_id)
        self.assertIn(session_id, self.workflow_manager.active_sessions)

        # Wait for timeout
        time.sleep(3)

        # Force cleanup check (normally done by background thread)
        current_time = time.time()
        expired_sessions = []

        for sid, session in self.workflow_manager.active_sessions.items():
            if current_time - session.start_time > self.workflow_manager.session_timeout:
                expired_sessions.append(sid)

        # Simulate cleanup
        for sid in expired_sessions:
            self.workflow_manager.end_roi_session(sid, "timeout")

        # Verify session was ended due to timeout
        self.assertNotIn(session_id, self.workflow_manager.active_sessions)

        # Restore original timeout
        self.workflow_manager.session_timeout = original_timeout

        logger.info("✓ Session timeout handling working correctly")

    def test_performance_tracking(self):
        """Test performance tracking during accumulation"""
        logger.info("Testing performance tracking...")

        # Start session and accumulate with timing
        session_id = self.workflow_manager.start_roi_session(self.camera_name, self.roi_id)
        session = self.workflow_manager.active_sessions[session_id]

        # Accumulate multiple frames
        for i in range(5):
            wood_detection = {'bbox': (150, 150, 400, 250), 'confidence': 0.8 + i * 0.03}
            defects = {"Unsound_Knot": i + 1}
            measurements = [("Unsound_Knot", 15.0 + i, 7.5 + i * 0.5)]

            self.workflow_manager.accumulate_defects(
                session_id, defects, wood_detection, measurements, 0.1 + i * 0.01
            )

        # End session and check performance metrics
        results = self.workflow_manager.end_roi_session(session_id)

        perf_metrics = results.get('performance_metrics', {})
        self.assertIn('avg_processing_time', perf_metrics)
        self.assertIn('avg_detection_confidence', perf_metrics)
        self.assertGreater(perf_metrics['avg_processing_time'], 0)
        self.assertGreater(perf_metrics['avg_detection_confidence'], 0)

        logger.info(f"✓ Performance tracking: avg_time={perf_metrics['avg_processing_time']:.3f}s, "
                   f"avg_confidence={perf_metrics['avg_detection_confidence']:.3f}")

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        logger.info("Testing error handling and recovery...")

        # Test invalid session accumulation
        success = self.workflow_manager.accumulate_defects(
            "invalid_session_id", {}, {}, [], 0.1
        )
        self.assertFalse(success)

        # Test session stats with errors
        stats = self.workflow_manager.get_session_stats()
        self.assertIn('error_counts', stats)
        self.assertIn('recent_errors', stats)

        # Verify error was tracked
        self.assertGreater(len(stats['recent_errors']), 0)

        logger.info(f"✓ Error handling working: {len(stats['recent_errors'])} errors tracked")

    def test_complete_workflow_integration(self):
        """Test complete workflow from wood detection to grading"""
        logger.info("Testing complete workflow integration...")

        # Create ROI module for integration test
        roi_module = ROIModule()
        roi_module.initialize_workflow_manager(
            self.mock_detection, self.mock_grading, self.mock_arduino
        )

        # Process a frame (mock wood detection)
        results = roi_module.process_frame(self.test_frame, self.camera_name)

        # Verify processing completed without errors
        self.assertNotIn('error', results)
        self.assertIsInstance(results['wood_detections'], list)
        self.assertIsInstance(results['overlaps'], dict)
        self.assertIsInstance(results['session_events'], list)

        logger.info("✓ Complete workflow integration successful")

def run_performance_benchmark():
    """Run performance benchmark for the accumulation system"""
    logger.info("Running performance benchmark...")

    # Create system components
    roi_manager = ROIManager()
    workflow_manager = ROIBasedWorkflowManager(MockDetectionModule(), Mock(), MockArduinoModule())

    # Define ROI
    roi_manager.define_roi("top", "bench_zone", (100, 100, 500, 300))

    # Benchmark session operations
    start_time = time.time()

    sessions_created = 0
    defects_accumulated = 0

    # Create multiple sessions
    for i in range(10):
        session_id = workflow_manager.start_roi_session("top", "bench_zone", f"bench_wood_{i}")
        if session_id:
            sessions_created += 1

            # Accumulate defects in each session
            for frame in range(5):
                defects = {"Unsound_Knot": frame + 1}
                measurements = [("Unsound_Knot", 15.0 + frame, 7.5 + frame * 0.5)]
                wood_detection = {'bbox': (150, 150, 400, 250), 'confidence': 0.85}

                if workflow_manager.accumulate_defects(session_id, defects, wood_detection, measurements, 0.1):
                    defects_accumulated += 1

            # End session
            workflow_manager.end_roi_session(session_id)

    total_time = time.time() - start_time

    logger.info("Performance Benchmark Results:")
    logger.info(f"  Sessions created: {sessions_created}")
    logger.info(f"  Defects accumulated: {defects_accumulated}")
    logger.info(f"  Total time: {total_time:.3f}s")
    logger.info(f"  Operations/second: {(sessions_created * 5 + sessions_created) / total_time:.2f}")

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Starting ROI Accumulation System Tests...")

    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Run performance benchmark
    logger.info("\n" + "="*60)
    run_performance_benchmark()

    logger.info("All tests completed!")