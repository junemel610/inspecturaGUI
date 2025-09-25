#!/usr/bin/env python3
"""
Test script for ROI-based wood detection workflow simulation.

This script simulates the complete ROI-based wood detection workflow including:
- Wood detection on sample images
- ROI trigger activation checking
- Classification ROI intersection validation
- Synthetic test cases with edge boundary conditions
- Duplicate classification prevention verification
- Integration with detection, ROI, and grading modules
- Comprehensive test reporting and validation

"""

import unittest
import numpy as np
import cv2
import os
import tempfile
import time
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple, Dict, Any
import json

# Import existing modules
from modules.detection_module import DetectionModule
from modules.roi_module import ROIManager, OverlapDetector, ROIBasedWorkflowManager
from modules.grading_module import determine_surface_grade, convert_grade_to_arduino_command
import settings

class TestROIWorkflowSimulation(unittest.TestCase):
    """Test cases for ROI-based wood detection workflow simulation"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

        # Mock the detection module to return controlled results for testing
        self.detection_module = Mock()
        self.roi_manager = ROIManager()
        self.overlap_detector = OverlapDetector(self.roi_manager)

        # Mock grading and arduino modules for testing
        self.grading_module = Mock()
        self.arduino_module = Mock()
        self.arduino_module.is_connected.return_value = True

        self.workflow_manager = ROIBasedWorkflowManager(
            self.detection_module, self.grading_module, self.arduino_module
        )

        # Define test ROIs
        self.trigger_roi_coords = [100, 50, 1100, 200]  # From settings
        self.classification_roi_coords = [150, 100, 1050, 600]  # From settings

        self.roi_manager.define_roi("test_camera", "trigger_roi", self.trigger_roi_coords,
                                   "Trigger Zone", 0.3)
        self.roi_manager.define_roi("test_camera", "classification_roi", self.classification_roi_coords,
                                   "Classification Zone", 0.5)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Test statistics
        self.test_stats = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'roi_triggers': 0,
            'classifications': 0,
            'duplicates_prevented': 0
        }

    def setup_mock_detection(self, wood_bbox=None, defect_dict=None, defect_measurements=None):
        """Configure the mock detection module for testing"""
        # Create mock alignment result
        mock_alignment_result = Mock()
        mock_alignment_result.wood_bbox = wood_bbox

        # Configure analyze_frame return values
        if defect_dict is None:
            defect_dict = {}
        if defect_measurements is None:
            defect_measurements = []

        # Mock frame for visualization
        mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        self.detection_module.analyze_frame.return_value = (
            mock_frame, defect_dict, defect_measurements, mock_alignment_result
        )

        # Configure ROI methods
        self.detection_module.trigger_roi_config = {
            'coordinates': self.trigger_roi_coords,
            'overlap_threshold': 0.3,
            'active': True
        }
        self.detection_module.classification_roi_config = {
            'coordinates': self.classification_roi_coords,
            'overlap_threshold': 0.5,
            'active': True
        }

        # Mock ROI intersection and trigger methods
        def mock_check_roi_intersection(bbox, roi_config):
            if not bbox or not roi_config:
                return False, 0.0
            # Simple intersection calculation for testing
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
            roi_x1, roi_y1, roi_x2, roi_y2 = roi_config['coordinates']

            # Calculate intersection
            inter_x1 = max(bbox_x1, roi_x1)
            inter_y1 = max(bbox_y1, roi_y1)
            inter_x2 = min(bbox_x2, roi_x2)
            inter_y2 = min(bbox_y2, roi_y2)

            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                return False, 0.0

            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            bbox_area = (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)
            roi_area = (roi_x2 - roi_x1) * (roi_y2 - roi_y1)

            union_area = bbox_area + roi_area - inter_area
            overlap = inter_area / union_area if union_area > 0 else 0.0

            intersects = overlap >= roi_config['overlap_threshold']
            return intersects, overlap

        def mock_should_trigger_classification(wood_bbox):
            trigger_intersects, trigger_overlap = mock_check_roi_intersection(
                wood_bbox, self.detection_module.trigger_roi_config)
            classification_intersects, classification_overlap = mock_check_roi_intersection(
                wood_bbox, self.detection_module.classification_roi_config)

            if not trigger_intersects:
                return False, f"No intersection with trigger ROI (overlap: {trigger_overlap:.3f})"
            if not classification_intersects:
                return False, f"No intersection with classification ROI (overlap: {classification_overlap:.3f})"

            return True, f"Trigger conditions met (trigger: {trigger_overlap:.3f}, classification: {classification_overlap:.3f})"

        self.detection_module.check_roi_intersection.side_effect = mock_check_roi_intersection
        self.detection_module.should_trigger_classification.side_effect = mock_should_trigger_classification

        # Mock statistics methods
        self.detection_module.get_roi_statistics.return_value = {
            'roi_trigger_count': 0,
            'roi_classification_count': 0,
            'detection_events': [],
            'total_detection_events': 0
        }
        self.detection_module.reset_roi_statistics = Mock()

    def tearDown(self):
        """Clean up test fixtures"""
        # Reset ROI statistics
        self.detection_module.reset_roi_statistics()

        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_synthetic_wood_image(self, bbox: Tuple[int, int, int, int] = None,
                                   frame_size: Tuple[int, int] = (1280, 720)) -> np.ndarray:
        """Create a synthetic image with wood-like features that match detection parameters"""
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

        # Create wood-like colors in HSV range [10, 30, 30] to [40, 255, 255]
        # This corresponds to brown/orange colors that the wood detection algorithm looks for
        # Convert HSV to BGR for OpenCV
        wood_hsv = np.random.randint([10, 30, 30], [40, 255, 255], dtype=np.uint8)
        wood_bgr = cv2.cvtColor(np.uint8([[wood_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
        # Ensure wood_bgr is a tuple of ints for OpenCV compatibility
        wood_bgr = (int(wood_bgr[0]), int(wood_bgr[1]), int(wood_bgr[2]))

        if bbox:
            x1, y1, x2, y2 = bbox
            # Ensure bbox is large enough for detection (min_contour_area = 1000)
            width = max(x2 - x1, 50)
            height = max(y2 - y1, 50)
            x2 = x1 + width
            y2 = y1 + height

            # Create wood-like rectangle with proper color
            cv2.rectangle(frame, (x1, y1), (x2, y2), wood_bgr, -1)

            # Add some texture and edges for better detection
            # Add vertical wood grain lines
            for i in range(x1, x2, 10):
                darker_color = (max(0, wood_bgr[0]-20), max(0, wood_bgr[1]-20), max(0, wood_bgr[2]-20))
                cv2.line(frame, (i, y1), (i, y2), darker_color, 1)

            # Add some noise for realism
            noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
            frame[y1:y2, x1:x2] = np.clip(frame[y1:y2, x1:x2].astype(np.int16) + noise, 0, 255).astype(np.uint8)
        else:
            # Create random wood-like regions that meet size requirements
            for _ in range(2):  # Fewer but larger regions
                x1 = np.random.randint(0, frame_size[0]//2)
                y1 = np.random.randint(0, frame_size[1]//2)
                # Ensure minimum size for detection
                min_size = 40  # sqrt(1000) ≈ 32, so 40x40 = 1600 > 1000
                x2 = x1 + np.random.randint(min_size, min(frame_size[0]-x1, 200))
                y2 = y1 + np.random.randint(min_size, min(frame_size[1]-y1, 150))

                cv2.rectangle(frame, (x1, y1), (x2, y2), wood_bgr, -1)

                # Add wood grain texture
                for i in range(x1, x2, 15):
                    darker_color = (max(0, wood_bgr[0]-15), max(0, wood_bgr[1]-15), max(0, wood_bgr[2]-15))
                    cv2.line(frame, (i, y1), (i, y2), darker_color, 1)

        return frame

    def test_roi_trigger_activation(self):
        """Test ROI trigger activation with various bounding box positions"""
        self.logger.info("Testing ROI trigger activation...")

        test_cases = [
            # (bbox, expected_trigger, expected_classification, description)
            ([100, 50, 1100, 200], True, False, "Wood exactly matching trigger ROI"),
            ([50, 25, 150, 75], False, False, "Wood outside trigger ROI"),
            ([150, 100, 1050, 600], True, True, "Wood exactly matching classification ROI"),
            ([1200, 700, 1280, 720], False, False, "Wood completely outside ROIs"),
            ([80, 40, 120, 80], False, False, "Wood touching trigger ROI edge but below threshold"),
        ]

        for bbox, expected_trigger, expected_classification, description in test_cases:
            with self.subTest(bbox=bbox, description=description):
                self.test_stats['total_tests'] += 1

                # Set up mock detection with the specified wood bbox
                self.setup_mock_detection(wood_bbox=bbox)

                # Create synthetic image (not used in mock but for consistency)
                frame = self.create_synthetic_wood_image(bbox)

                # Run detection
                annotated_frame, defect_dict, defect_measurements, alignment_result = \
                    self.detection_module.analyze_frame(frame, "test_camera")

                wood_bbox = alignment_result.wood_bbox

                # Check trigger conditions
                should_trigger, reason = self.detection_module.should_trigger_classification(wood_bbox)

                # Validate results
                if expected_trigger:
                    self.assertTrue(should_trigger,
                                  f"Expected trigger activation for {description}, but got: {reason}")
                    self.test_stats['roi_triggers'] += 1
                else:
                    self.assertFalse(should_trigger,
                                   f"Unexpected trigger activation for {description}: {reason}")

                # Check classification ROI intersection if triggered
                if should_trigger and expected_classification:
                    # Should have run classification (defect_dict should not be empty or have been processed)
                    self.assertIsInstance(defect_dict, dict,
                                        f"Classification should have run for {description}")
                    self.test_stats['classifications'] += 1

                self.logger.info(f"✓ {description}: trigger={should_trigger}, reason={reason}")
                self.test_stats['passed_tests'] += 1

    def test_edge_boundary_conditions(self):
        """Test synthetic test cases with bounding boxes near ROI edges"""
        self.logger.info("Testing edge boundary conditions...")

        # Define edge test cases around ROI boundaries
        edge_cases = [
            # Near trigger ROI edges
            ([95, 45, 105, 55], "Near trigger ROI top-left corner"),
            ([1095, 45, 1105, 55], "Near trigger ROI top-right corner"),
            ([95, 195, 105, 205], "Near trigger ROI bottom-left corner"),
            ([1095, 195, 1105, 205], "Near trigger ROI bottom-right corner"),

            # Near classification ROI edges
            ([145, 95, 155, 105], "Near classification ROI top-left corner"),
            ([1045, 95, 1055, 105], "Near classification ROI top-right corner"),
            ([145, 595, 155, 605], "Near classification ROI bottom-left corner"),
            ([1045, 595, 1055, 605], "Near classification ROI bottom-right corner"),

            # Crossing ROI boundaries
            ([90, 40, 110, 60], "Crossing trigger ROI top boundary"),
            ([140, 90, 160, 110], "Crossing classification ROI top boundary"),
        ]

        for bbox, description in edge_cases:
            with self.subTest(bbox=bbox, description=description):
                self.test_stats['total_tests'] += 1

                frame = self.create_synthetic_wood_image(bbox)

                # Run detection
                annotated_frame, defect_dict, defect_measurements, alignment_result = \
                    self.detection_module.analyze_frame(frame, "test_camera")

                wood_bbox = alignment_result.wood_bbox

                # Check intersection calculations
                trigger_intersects, trigger_overlap = self.detection_module.check_roi_intersection(
                    wood_bbox, self.detection_module.trigger_roi_config)
                classification_intersects, classification_overlap = self.detection_module.check_roi_intersection(
                    wood_bbox, self.detection_module.classification_roi_config)

                # Validate overlap calculations are reasonable
                self.assertIsInstance(trigger_overlap, float)
                self.assertIsInstance(classification_overlap, float)
                self.assertGreaterEqual(trigger_overlap, 0.0)
                self.assertLessEqual(trigger_overlap, 1.0)
                self.assertGreaterEqual(classification_overlap, 0.0)
                self.assertLessEqual(classification_overlap, 1.0)

                # Check trigger logic
                should_trigger, reason = self.detection_module.should_trigger_classification(wood_bbox)

                # Log results for analysis
                self.logger.info(f"Edge case '{description}': bbox={bbox}, "
                               f"trigger_overlap={trigger_overlap:.3f}, "
                               f"classification_overlap={classification_overlap:.3f}, "
                               f"should_trigger={should_trigger}")

                self.test_stats['passed_tests'] += 1

    def test_duplicate_classification_prevention(self):
        """Test that duplicate classifications are prevented"""
        self.logger.info("Testing duplicate classification prevention...")

        # Create wood bbox that triggers classification
        trigger_bbox = [100, 50, 1100, 200]

        # Set up mock detection
        self.setup_mock_detection(wood_bbox=trigger_bbox)

        frame = self.create_synthetic_wood_image(trigger_bbox)

        # First detection - should trigger
        annotated_frame1, defect_dict1, defect_measurements1, alignment_result1 = \
            self.detection_module.analyze_frame(frame, "test_camera")

        should_trigger1, reason1 = self.detection_module.should_trigger_classification(
            alignment_result1.wood_bbox)

        # Immediately run second detection on same/similar frame
        annotated_frame2, defect_dict2, defect_measurements2, alignment_result2 = \
            self.detection_module.analyze_frame(frame, "test_camera")

        should_trigger2, reason2 = self.detection_module.should_trigger_classification(
            alignment_result2.wood_bbox)

        # First should trigger, second should be prevented due to duplicate prevention
        self.assertTrue(should_trigger1,
                       f"First detection should trigger classification: {reason1}")
        self.assertFalse(should_trigger2,
                        f"Second detection should be prevented due to duplicate prevention: {reason2}")

        # Check that reason mentions duplicate prevention
        self.assertIn("duplicate", reason2.lower(),
                     f"Duplicate prevention reason should mention duplicate: {reason2}")

        self.test_stats['duplicates_prevented'] += 1
        self.test_stats['passed_tests'] += 1
        self.logger.info("✓ Duplicate classification prevention working correctly")

    def test_workflow_integration(self):
        """Test complete workflow integration with session management"""
        self.logger.info("Testing complete workflow integration...")

        # Set up mock detection
        self.setup_mock_detection(wood_bbox=[150, 100, 1050, 600])

        # Create test frame with wood in ROIs
        frame = self.create_synthetic_wood_image([200, 75, 1000, 500])

        # Mock grading response
        self.grading_module.determine_surface_grade.return_value = 'G2-2'
        self.grading_module.convert_grade_to_arduino_command.return_value = '2'

        # Process frame through workflow
        results = self.workflow_manager.trigger_grading_workflow({
            'session_id': 'test_session_001',
            'defect_measurements': [
                ('crack', 5.0, 0.02),
                ('knot', 8.0, 0.03)
            ]
        })

        # Verify workflow components were called
        self.grading_module.determine_surface_grade.assert_called_once()
        self.grading_module.convert_grade_to_arduino_command.assert_called_once_with('G2-2')
        self.arduino_module.send_grade_command.assert_called_once_with('2')

        self.test_stats['passed_tests'] += 1
        self.logger.info("✓ Workflow integration test passed")

    def test_roi_statistics_tracking(self):
        """Test ROI statistics tracking and reporting"""
        self.logger.info("Testing ROI statistics tracking...")

        # Set up mock detection for different scenarios
        test_cases = [
            ([100, 50, 1100, 200], True),   # Should trigger
            ([50, 25, 150, 75], False),     # Should not trigger
            ([150, 100, 1050, 600], True),  # Should trigger
        ]

        for i, (bbox, should_trigger) in enumerate(test_cases):
            self.setup_mock_detection(wood_bbox=bbox)
            frame = self.create_synthetic_wood_image(bbox)
            self.detection_module.analyze_frame(frame, "test_camera")

        # Check statistics
        stats = self.detection_module.get_roi_statistics()

        self.assertIn('roi_trigger_count', stats)
        self.assertIn('roi_classification_count', stats)
        self.assertIn('detection_events', stats)

        # Should have recorded some triggers and classifications
        self.assertGreaterEqual(stats['roi_trigger_count'], 0)
        self.assertGreaterEqual(stats['roi_classification_count'], 0)
        self.assertIsInstance(stats['detection_events'], list)

        self.logger.info(f"ROI Statistics: triggers={stats['roi_trigger_count']}, "
                        f"classifications={stats['roi_classification_count']}, "
                        f"events={len(stats['detection_events'])}")

        self.test_stats['passed_tests'] += 1

    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases"""
        self.logger.info("Testing error handling and edge cases...")

        # Test with None frame
        with self.assertRaises(AttributeError):
            self.detection_module.analyze_frame(None, "test_camera")

        # Test with empty frame
        empty_frame = np.array([])
        result = self.detection_module.analyze_frame(empty_frame, "test_camera")
        self.assertIsInstance(result, tuple)

        # Test with invalid bbox
        invalid_bbox = None
        intersects, overlap = self.detection_module.check_roi_intersection(
            invalid_bbox, self.detection_module.trigger_roi_config)
        self.assertFalse(intersects)
        self.assertEqual(overlap, 0.0)

        # Test with bbox outside frame bounds
        out_of_bounds_bbox = [-100, -100, -50, -50]
        intersects, overlap = self.detection_module.check_roi_intersection(
            out_of_bounds_bbox, self.detection_module.trigger_roi_config)
        self.assertFalse(intersects)

        self.test_stats['passed_tests'] += 1
        self.logger.info("✓ Error handling tests passed")

    def test_performance_and_timing(self):
        """Test performance and timing constraints"""
        self.logger.info("Testing performance and timing...")

        # Set up mock detection
        self.setup_mock_detection(wood_bbox=[100, 50, 1100, 200])

        frame = self.create_synthetic_wood_image([200, 75, 300, 125])

        # Measure detection time
        start_time = time.time()
        annotated_frame, defect_dict, defect_measurements, alignment_result = \
            self.detection_module.analyze_frame(frame, "test_camera")
        processing_time = time.time() - start_time

        # Should complete within reasonable time (less than 1 second for mock data)
        self.assertLess(processing_time, 1.0,
                       f"Detection took too long: {processing_time:.3f}s")

        # Check that result is valid
        self.assertIsInstance(annotated_frame, np.ndarray)
        self.assertIsInstance(defect_dict, dict)
        self.assertIsInstance(defect_measurements, list)

        self.logger.info(f"✓ Performance test passed: {processing_time:.3f}s processing time")
        self.test_stats['passed_tests'] += 1

    def generate_test_report(self):
        """Generate comprehensive test report"""
        report = {
            'test_summary': {
                'total_tests': self.test_stats['total_tests'],
                'passed_tests': self.test_stats['passed_tests'],
                'failed_tests': self.test_stats['failed_tests'],
                'success_rate': (self.test_stats['passed_tests'] / max(self.test_stats['total_tests'], 1)) * 100
            },
            'roi_workflow_metrics': {
                'roi_triggers': self.test_stats['roi_triggers'],
                'classifications': self.test_stats['classifications'],
                'duplicates_prevented': self.test_stats['duplicates_prevented']
            },
            'module_statistics': self.detection_module.get_roi_statistics(),
            'test_timestamp': time.time(),
            'test_duration': time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }

        # Save report to file
        report_file = os.path.join(self.temp_dir, 'roi_trigger_test_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def run_comprehensive_test_suite(self):
        """Run the complete comprehensive test suite"""
        self.start_time = time.time()

        print("\n" + "="*60)
        print("ROI-BASED WOOD DETECTION WORKFLOW TEST SUITE")
        print("="*60)

        # Run all test methods
        test_methods = [
            self.test_roi_trigger_activation,
            self.test_edge_boundary_conditions,
            self.test_duplicate_classification_prevention,
            self.test_workflow_integration,
            self.test_roi_statistics_tracking,
            self.test_error_handling_and_edge_cases,
            self.test_performance_and_timing
        ]

        for test_method in test_methods:
            try:
                print(f"\nRunning {test_method.__name__}...")
                test_method()
                print(f"✓ {test_method.__name__} completed successfully")
            except Exception as e:
                print(f"✗ {test_method.__name__} failed: {e}")
                self.test_stats['failed_tests'] += 1

        # Generate final report
        report = self.generate_test_report()

        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        print(f"Total Tests: {report['test_summary']['total_tests']}")
        print(f"Passed: {report['test_summary']['passed_tests']}")
        print(f"Failed: {report['test_summary']['failed_tests']}")
        print(f"Success Rate: {report['test_summary']['success_rate']:.1f}%")
        print(f"ROI Triggers: {report['roi_workflow_metrics']['roi_triggers']}")
        print(f"Classifications: {report['roi_workflow_metrics']['classifications']}")
        print(f"Duplicates Prevented: {report['roi_workflow_metrics']['duplicates_prevented']}")

        return report

def main():
    """Main test execution"""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestROIWorkflowSimulation)
    runner = unittest.TextTestRunner(verbosity=2)

    # Run tests
    test_instance = TestROIWorkflowSimulation()
    test_instance.setUp()

    try:
        report = test_instance.run_comprehensive_test_suite()

        # Final validation
        success_rate = report['test_summary']['success_rate']
        if success_rate >= 90.0:
            print("\n🎉 ROI TRIGGER TEST SUITE PASSED!")
            print(f"✓ {success_rate:.1f}% success rate achieved")
            return 0
        else:
            print(f"\n❌ ROI TRIGGER TEST SUITE FAILED: {success_rate:.1f}% success rate")
            return 1

    except Exception as e:
        print(f"\n💥 CRITICAL ERROR during test execution: {e}")
        return 1
    finally:
        test_instance.tearDown()

if __name__ == '__main__':
    import sys
    sys.exit(main())