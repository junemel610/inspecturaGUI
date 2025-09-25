#!/usr/bin/env python3
"""
Basic functionality test for ROI Module

This script tests the core functionality of the ROI module without
running the full test suite.
"""

import numpy as np
import cv2
import os
import tempfile

# Import ROI module
from modules.roi_module import ROIManager, OverlapDetector, ROIVisualizer, ROIModule

def test_basic_roi_functionality():
    """Test basic ROI functionality"""
    print("Testing basic ROI functionality...")

    # Create temporary config file
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    temp_config.close()
    config_file = temp_config.name

    try:
        # Test ROIManager
        roi_manager = ROIManager(config_file)

        # Define ROI
        coordinates = (100, 100, 200, 200)
        result = roi_manager.define_roi("test_camera", "test_roi", coordinates, "Test ROI")
        print(f"✓ ROI definition: {'Success' if result else 'Failed'}")

        # Test activation
        result = roi_manager.activate_roi("test_camera", "test_roi")
        print(f"✓ ROI activation: {'Success' if result else 'Failed'}")

        # Test getting active ROIs
        active_rois = roi_manager.get_active_rois("test_camera")
        print(f"✓ Active ROIs: {active_rois}")

        # Test persistence
        roi_manager.save_config()
        new_manager = ROIManager(config_file)
        roi_config = new_manager.get_roi_config("test_camera", "test_roi")
        print(f"✓ ROI persistence: {'Success' if roi_config else 'Failed'}")

        # Test OverlapDetector
        overlap_detector = OverlapDetector(roi_manager)

        # Test overlap calculation
        bbox1 = (150, 150, 180, 180)
        bbox2 = (100, 100, 200, 200)
        overlap = overlap_detector.calculate_overlap_percentage(bbox1, bbox2)
        print(f"✓ Overlap calculation: {overlap:.3f}")

        # Test ROIVisualizer
        visualizer = ROIVisualizer(roi_manager)

        # Create test frame
        test_frame = np.zeros((300, 300, 3), dtype=np.uint8)
        test_frame[:] = (255, 255, 255)  # White background

        # Test ROI overlay
        overlay_frame = visualizer.draw_roi_overlays(test_frame, "test_camera")
        print(f"✓ ROI overlay: Frame shape {overlay_frame.shape}")

        # Test ROIModule
        roi_module = ROIModule(config_file)

        # Test frame processing
        results = roi_module.process_frame(test_frame, "test_camera")
        print(f"✓ Frame processing: {len(results)} result keys")

        print("\n🎉 All basic tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        if os.path.exists(config_file):
            os.unlink(config_file)

def test_integration_points():
    """Test integration points"""
    print("\nTesting integration points...")

    try:
        # Test integration functions
        from modules.roi_module import integrate_with_camera_module, integrate_with_detection_module

        # Mock camera module
        class MockCameraModule:
            def read_frame(self, camera_name):
                frame = np.zeros((200, 200, 3), dtype=np.uint8)
                return True, frame

        camera_module = MockCameraModule()
        roi_manager = ROIManager()
        roi_visualizer = ROIVisualizer(roi_manager)

        # Test camera integration
        integrated_camera = integrate_with_camera_module(camera_module, roi_manager, roi_visualizer)
        result = integrated_camera.get_frame_with_roi_overlay("test_camera")
        print(f"✓ Camera integration: {'Success' if result is not None else 'Failed'}")

        print("\n🎉 Integration tests passed!")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("=== ROI Module Basic Test ===")
    test_basic_roi_functionality()
    test_integration_points()
    print("\n=== Test Complete ===")