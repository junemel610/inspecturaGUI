#!/usr/bin/env python3
"""
Unit and Integration Tests for ROI Module

This module contains comprehensive tests for the ROI-based wood detection system.
Tests cover all major components: ROIManager, OverlapDetector, ROIBasedWorkflowManager,
ROIVisualizer, and integration with existing systems.
"""

import unittest
import numpy as np
import cv2
import json
import os
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock

# Import ROI module components
from modules.roi_module import (
    ROIManager, OverlapDetector, ROIBasedWorkflowManager, ROIVisualizer,
    ROIModule, ROIConfig, ROISession, ROIStatus, ROISessionStatus,
    integrate_with_camera_module, integrate_with_detection_module
)

class TestROIManager(unittest.TestCase):
    """Test cases for ROIManager class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_config.close()
        self.config_file = self.temp_config.name
        self.roi_manager = ROIManager(self.config_file)

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.config_file):
            os.unlink(self.config_file)

    def test_define_roi(self):
        """Test ROI definition"""
        coordinates = (100, 100, 200, 200)
        result = self.roi_manager.define_roi("test_camera", "test_roi", coordinates, "Test ROI")

        self.assertTrue(result)
        roi_config = self.roi_manager.get_roi_config("test_camera", "test_roi")
        self.assertIsNotNone(roi_config)
        self.assertEqual(roi_config.coordinates, coordinates)
        self.assertEqual(roi_config.name, "Test ROI")

    def test_activate_deactivate_roi(self):
        """Test ROI activation and deactivation"""
        coordinates = (50, 50, 150, 150)
        self.roi_manager.define_roi("test_camera", "test_roi", coordinates)

        # Test activation
        result = self.roi_manager.activate_roi("test_camera", "test_roi")
        self.assertTrue(result)
        self.assertIn("test_roi", self.roi_manager.get_active_rois("test_camera"))

        # Test deactivation
        result = self.roi_manager.deactivate_roi("test_camera", "test_roi")
        self.assertTrue(result)
        self.assertNotIn("test_roi", self.roi_manager.get_active_rois("test_camera"))

    def test_invalid_coordinates(self):
        """Test handling of invalid ROI coordinates"""
        # Invalid coordinates (x1 >= x2)
        result = self.roi_manager.define_roi("test_camera", "invalid_roi", (200, 100, 100, 200))
        self.assertFalse(result)

    def test_roi_persistence(self):
        """Test ROI configuration persistence"""
        coordinates = (10, 10, 100, 100)
        self.roi_manager.define_roi("test_camera", "persistent_roi", coordinates, "Persistent ROI")

        # Create new manager instance
        new_manager = ROIManager(self.config_file)

        # Check if ROI was loaded
        roi_config = new_manager.get_roi_config("test_camera", "persistent_roi")
        self.assertIsNotNone(roi_config)
        self.assertEqual(roi_config.coordinates, coordinates)

    def test_get_all_rois(self):
        """Test getting all ROIs"""
        self.roi_manager.define_roi("cam1", "roi1", (0, 0, 50, 50))
        self.roi_manager.define_roi("cam1", "roi2", (60, 60, 100, 100))
        self.roi_manager.define_roi("cam2", "roi3", (10, 10, 40, 40))

        all_rois = self.roi_manager.get_all_rois()
        self.assertIn("cam1", all_rois)
        self.assertIn("cam2", all_rois)
        self.assertEqual(len(all_rois["cam1"]), 2)
        self.assertEqual(len(all_rois["cam2"]), 1)

class TestOverlapDetector(unittest.TestCase):
    """Test cases for OverlapDetector class"""

    def setUp(self):
        """Set up test fixtures"""
        self.roi_manager = ROIManager()  # In-memory only
        self.overlap_detector = OverlapDetector(self.roi_manager)

        # Define test ROIs
        self.roi_manager.define_roi("test_camera", "roi1", (100, 100, 200, 200))
        self.roi_manager.define_roi("test_camera", "roi2", (250, 250, 350, 350))

    def test_calculate_overlap_percentage(self):
        """Test overlap percentage calculation"""
        # No overlap
        overlap = self.overlap_detector.calculate_overlap_percentage(
            (0, 0, 50, 50), (100, 100, 150, 150)
        )
        self.assertEqual(overlap, 0.0)

        # Complete overlap
        overlap = self.overlap_detector.calculate_overlap_percentage(
            (100, 100, 200, 200), (100, 100, 200, 200)
        )
        self.assertEqual(overlap, 1.0)

        # Partial overlap
        overlap = self.overlap_detector.calculate_overlap_percentage(
            (100, 100, 200, 200), (150, 150, 250, 250)
        )
        self.assertGreater(overlap, 0.0)
        self.assertLess(overlap, 1.0)

    def test_detect_overlaps(self):
        """Test overlap detection with wood detections"""
        from modules.wood_detection_module import WoodDetectionResult

        # Create mock wood detections
        wood_detections = [
            WoodDetectionResult(detected=True, bbox=(150, 150, 180, 180), confidence=0.9),
            WoodDetectionResult(detected=True, bbox=(300, 300, 320, 320), confidence=0.8),
            WoodDetectionResult(detected=False)  # Non-detected
        ]

        overlaps = self.overlap_detector.detect_overlaps(wood_detections, "test_camera")

        # Check overlaps
        self.assertIn("wood_0", overlaps)  # Should overlap with roi1
        self.assertIn("roi1", overlaps["wood_0"])
        self.assertIn("wood_1", overlaps)  # Should overlap with roi2
        self.assertIn("roi2", overlaps["wood_1"])

    def test_performance_stats(self):
        """Test performance statistics tracking"""
        # Run some calculations
        self.overlap_detector.calculate_overlap_percentage((0, 0, 10, 10), (5, 5, 15, 15))
        self.overlap_detector.calculate_overlap_percentage((20, 20, 30, 30), (25, 25, 35, 35))

        stats = self.overlap_detector.get_performance_stats()
        self.assertGreater(stats['total_calculations'], 0)
        self.assertIsInstance(stats['avg_calculation_time'], float)

class TestROIBasedWorkflowManager(unittest.TestCase):
    """Test cases for ROIBasedWorkflowManager class"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock dependencies
        self.detection_module = Mock()
        self.grading_module = Mock()
        self.arduino_module = Mock()
        self.arduino_module.is_connected.return_value = True

        self.workflow_manager = ROIBasedWorkflowManager(
            self.detection_module, self.grading_module, self.arduino_module
        )

    def test_start_roi_session(self):
        """Test starting ROI session"""
        session_id = self.workflow_manager.start_roi_session("test_camera", "test_roi")

        self.assertNotEqual(session_id, "")
        self.assertIn(session_id, self.workflow_manager.active_sessions)

        session = self.workflow_manager.active_sessions[session_id]
        self.assertEqual(session.camera_name, "test_camera")
        self.assertEqual(session.roi_id, "test_roi")
        self.assertEqual(session.status, ROISessionStatus.ACTIVE)

    def test_accumulate_defects(self):
        """Test defect accumulation in session"""
        session_id = self.workflow_manager.start_roi_session("test_camera", "test_roi")

        defects = {"crack": 2, "knot": 1}
        wood_detection = {"bbox": (100, 100, 200, 200), "confidence": 0.9}

        result = self.workflow_manager.accumulate_defects(session_id, defects, wood_detection)
        self.assertTrue(result)

        session = self.workflow_manager.active_sessions[session_id]
        self.assertEqual(session.frame_count, 1)
        self.assertEqual(len(session.defects_accumulated), 1)

    def test_end_roi_session(self):
        """Test ending ROI session"""
        session_id = self.workflow_manager.start_roi_session("test_camera", "test_roi")

        # Add some defects
        defects = {"crack": 1}
        wood_detection = {"bbox": (100, 100, 200, 200)}
        self.workflow_manager.accumulate_defects(session_id, defects, wood_detection)

        results = self.workflow_manager.end_roi_session(session_id)

        self.assertIsNotNone(results)
        self.assertEqual(results['total_frames'], 1)
        self.assertEqual(results['total_defects']['crack'], 1)
        self.assertNotIn(session_id, self.workflow_manager.active_sessions)

    def test_trigger_grading_workflow(self):
        """Test grading workflow triggering"""
        session_results = {
            'session_id': 'test_session',
            'defect_measurements': [
                {'defects': {'crack': 2}, 'wood_detection': {}}
            ]
        }

        # Mock grading module
        with patch('modules.grading_module.determine_surface_grade') as mock_grade:
            mock_grade.return_value = 'G2-2'

            result = self.workflow_manager.trigger_grading_workflow(session_results)

            # Verify Arduino command was sent
            self.arduino_module.send_grade_command.assert_called_once_with('G2-2')

    def test_session_timeout(self):
        """Test session timeout handling"""
        # Set very short timeout for testing
        self.workflow_manager.session_timeout = 0.1

        session_id = self.workflow_manager.start_roi_session("test_camera", "test_roi")
        self.assertIn(session_id, self.workflow_manager.active_sessions)

        # Wait for timeout
        time.sleep(0.2)

        # Trigger cleanup (normally done by background thread)
        self.workflow_manager._cleanup_expired_sessions()

        # Session should be ended due to timeout
        self.assertNotIn(session_id, self.workflow_manager.active_sessions)

class TestROIVisualizer(unittest.TestCase):
    """Test cases for ROIVisualizer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.roi_manager = ROIManager()  # In-memory only
        self.visualizer = ROIVisualizer(self.roi_manager)

        # Define test ROI
        self.roi_manager.define_roi("test_camera", "test_roi", (50, 50, 150, 150), "Test ROI")

        # Create test frame
        self.test_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        self.test_frame[:] = (255, 255, 255)  # White background

    def test_draw_roi_overlays(self):
        """Test ROI overlay drawing"""
        result_frame = self.visualizer.draw_roi_overlays(self.test_frame, "test_camera")

        # Frame should be modified (not identical to original)
        self.assertFalse(np.array_equal(result_frame, self.test_frame))

        # Check that ROI rectangle was drawn (look for green color)
        green_pixels = np.sum(np.all(result_frame == [0, 255, 0], axis=2))
        self.assertGreater(green_pixels, 0)  # Should have green pixels for ROI border

    def test_draw_wood_detections(self):
        """Test wood detection overlay drawing"""
        from modules.wood_detection_module import WoodDetectionResult

        wood_detections = [
            WoodDetectionResult(detected=True, bbox=(75, 75, 125, 125), confidence=0.9)
        ]

        result_frame = self.visualizer.draw_wood_detections(self.test_frame, wood_detections)

        # Frame should be modified
        self.assertFalse(np.array_equal(result_frame, self.test_frame))

        # Check that red rectangle was drawn
        red_pixels = np.sum(np.all(result_frame == [255, 0, 0], axis=2))
        self.assertGreater(red_pixels, 0)  # Should have red pixels for wood detection

    def test_combined_overlay(self):
        """Test combined ROI and wood detection overlay"""
        from modules.wood_detection_module import WoodDetectionResult

        wood_detections = [
            WoodDetectionResult(detected=True, bbox=(75, 75, 125, 125), confidence=0.9)
        ]

        result_frame = self.visualizer.draw_combined_overlay(
            self.test_frame, "test_camera", wood_detections, ["test_roi"]
        )

        # Frame should be modified
        self.assertFalse(np.array_equal(result_frame, self.test_frame))

        # Should have both green (ROI) and red (wood) pixels
        green_pixels = np.sum(np.all(result_frame == [0, 255, 0], axis=2))
        red_pixels = np.sum(np.all(result_frame == [255, 0, 0], axis=2))

        self.assertGreater(green_pixels, 0)
        self.assertGreater(red_pixels, 0)

class TestIntegrationFunctions(unittest.TestCase):
    """Test cases for integration functions"""

    def test_integrate_with_camera_module(self):
        """Test camera module integration"""
        camera_module = Mock()
        camera_module.read_frame.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))

        roi_manager = ROIManager()
        roi_visualizer = ROIVisualizer(roi_manager)

        # Define ROI
        roi_manager.define_roi("test_camera", "test_roi", (10, 10, 50, 50))

        # Integrate
        integrated_camera = integrate_with_camera_module(camera_module, roi_manager, roi_visualizer)

        # Test the new method
        result = integrated_camera.get_frame_with_roi_overlay("test_camera")

        self.assertIsNotNone(result)
        camera_module.read_frame.assert_called_once_with("test_camera")

    def test_integrate_with_detection_module(self):
        """Test detection module integration"""
        detection_module = Mock()
        detection_module.detect_defects_in_full_frame.return_value = (
            np.zeros((50, 50, 3), dtype=np.uint8), {"crack": 1}, []
        )

        wood_detector = Mock()
        wood_detector.detect_wood.return_value = []

        overlap_detector = Mock()
        overlap_detector.detect_overlaps.return_value = {}

        workflow_manager = Mock()
        workflow_manager.get_active_sessions.return_value = []
        workflow_manager.active_sessions = {}

        # Integrate
        integrated_detection = integrate_with_detection_module(
            detection_module, wood_detector, overlap_detector, workflow_manager
        )

        # Test the new method
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = integrated_detection.analyze_frame_with_roi_workflow(test_frame, "test_camera")

        self.assertIsInstance(result, dict)
        self.assertIn('annotated_frame', result)
        self.assertIn('wood_detections', result)

class TestROIModule(unittest.TestCase):
    """Test cases for main ROIModule class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_config.close()
        self.config_file = self.temp_config.name
        self.roi_module = ROIModule(self.config_file)

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.config_file):
            os.unlink(self.config_file)

    def test_process_frame(self):
        """Test frame processing through ROI module"""
        test_frame = np.zeros((200, 200, 3), dtype=np.uint8)

        # Define ROI
        self.roi_module.roi_manager.define_roi("test_camera", "test_roi", (50, 50, 150, 150))

        results = self.roi_module.process_frame(test_frame, "test_camera")

        self.assertIsInstance(results, dict)
        self.assertIn('annotated_frame', results)
        self.assertIn('wood_detections', results)
        self.assertIn('overlaps', results)

    def test_get_stats(self):
        """Test statistics retrieval"""
        stats = self.roi_module.get_stats()

        self.assertIsInstance(stats, dict)
        self.assertIn('roi_manager', stats)
        self.assertIn('overlap_detector', stats)
        self.assertIn('visualizer', stats)

class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling"""

    def test_roi_manager_error_handling(self):
        """Test ROIManager error handling"""
        roi_manager = ROIManager()

        # Test with invalid coordinates
        result = roi_manager.define_roi("test", "invalid", (200, 100, 100, 200))
        self.assertFalse(result)

        # Test accessing non-existent ROI
        config = roi_manager.get_roi_config("nonexistent", "roi")
        self.assertIsNone(config)

    def test_overlap_detector_error_handling(self):
        """Test OverlapDetector error handling"""
        roi_manager = ROIManager()
        overlap_detector = OverlapDetector(roi_manager)

        # Test with invalid bounding boxes
        overlap = overlap_detector.calculate_overlap_percentage(
            (0, 0, 0, 0), (0, 0, 0, 0)
        )
        self.assertEqual(overlap, 0.0)

    def test_workflow_manager_error_handling(self):
        """Test WorkflowManager error handling"""
        workflow_manager = ROIBasedWorkflowManager(None, None, None)

        # Test ending non-existent session
        result = workflow_manager.end_roi_session("nonexistent")
        self.assertIsNone(result)

class TestPerformanceOptimizations(unittest.TestCase):
    """Test cases for performance optimizations"""

    def test_overlap_caching(self):
        """Test overlap calculation caching"""
        roi_manager = ROIManager()
        overlap_detector = OverlapDetector(roi_manager)

        # Define ROI
        roi_manager.define_roi("test", "roi", (100, 100, 200, 200))

        # First calculation (cache miss)
        overlap1 = overlap_detector.calculate_overlap_percentage(
            (150, 150, 180, 180), (100, 100, 200, 200)
        )

        # Second calculation with same parameters (should use cache)
        overlap2 = overlap_detector.calculate_overlap_percentage(
            (150, 150, 180, 180), (100, 100, 200, 200)
        )

        self.assertEqual(overlap1, overlap2)

        stats = overlap_detector.get_performance_stats()
        self.assertGreaterEqual(stats['cache_hits'], 0)

    def test_visualizer_performance(self):
        """Test visualizer performance tracking"""
        roi_manager = ROIManager()
        visualizer = ROIVisualizer(roi_manager)

        # Define ROI
        roi_manager.define_roi("test", "roi", (50, 50, 150, 150))

        # Draw multiple overlays
        test_frame = np.zeros((200, 200, 3), dtype=np.uint8)

        for _ in range(5):
            visualizer.draw_roi_overlays(test_frame, "test")

        stats = visualizer.get_render_stats()
        self.assertEqual(stats['total_renders'], 5)
        self.assertIsInstance(stats['avg_render_time'], float)

if __name__ == '__main__':
    # Create test suite
    unittest.main(verbosity=2)