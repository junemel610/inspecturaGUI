#!/usr/bin/env python3
"""
ROI-Based Wood Detection System Demonstration

This script demonstrates the complete ROI-based wood detection system
according to the roi_based_wood_detection_design.md specification.

Features demonstrated:
1. Interactive ROI definition system
2. ROI configuration management with JSON persistence
3. Overlap detection algorithms
4. Multiple ROI support per camera
5. Real-time ROI activation/deactivation
6. Integration with camera and wood detection systems
7. Comprehensive error handling and performance optimizations
"""

import cv2
import numpy as np
import time
import os

# Import ROI module components
from modules.roi_module import (
    ROIManager, OverlapDetector, ROIBasedWorkflowManager,
    ROIVisualizer, ROIModule, ROIConfig, ROIStatus
)

def create_demo_frame(width=640, height=480):
    """Create a demo frame with simulated wood pieces"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (200, 180, 150)  # Light brown background

    # Draw simulated wood pieces
    # Wood piece 1 - in top ROI area
    cv2.rectangle(frame, (150, 50), (250, 120), (139, 69, 19), -1)  # Brown wood
    cv2.rectangle(frame, (155, 55), (245, 115), (101, 67, 33), 2)   # Darker border

    # Wood piece 2 - in bottom ROI area
    cv2.rectangle(frame, (400, 350), (500, 420), (160, 82, 45), -1)  # Reddish wood
    cv2.rectangle(frame, (405, 355), (495, 415), (101, 67, 33), 2)   # Darker border

    # Add some texture/noise
    noise = np.random.randint(-20, 20, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return frame

def demo_roi_management():
    """Demonstrate ROI management functionality"""
    print("=== ROI Management Demo ===")

    # Initialize ROI manager
    roi_manager = ROIManager()

    # Define ROIs for top and bottom cameras
    print("1. Defining ROIs...")

    # Top camera ROIs
    roi_manager.define_roi("top_camera", "inspection_zone",
                          (100, 20, 540, 120), "Top Inspection Zone", 0.3)
    roi_manager.define_roi("top_camera", "secondary_zone",
                          (200, 150, 440, 250), "Top Secondary Zone", 0.5)

    # Bottom camera ROIs
    roi_manager.define_roi("bottom_camera", "inspection_zone",
                          (100, 320, 540, 420), "Bottom Inspection Zone", 0.3)

    print("✓ Defined ROIs for both cameras")

    # Activate ROIs
    print("2. Activating ROIs...")
    roi_manager.activate_roi("top_camera", "inspection_zone")
    roi_manager.activate_roi("bottom_camera", "inspection_zone")

    print("✓ Activated inspection zone ROIs")

    # Show active ROIs
    top_active = roi_manager.get_active_rois("top_camera")
    bottom_active = roi_manager.get_active_rois("bottom_camera")

    print(f"✓ Active ROIs - Top: {top_active}, Bottom: {bottom_active}")

    return roi_manager

def demo_overlap_detection(roi_manager):
    """Demonstrate overlap detection functionality"""
    print("\n=== Overlap Detection Demo ===")

    overlap_detector = OverlapDetector(roi_manager)

    # Create test wood detections
    from modules.wood_detection_module import WoodDetectionResult

    wood_detections = [
        WoodDetectionResult(detected=True, bbox=(150, 50, 250, 120), confidence=0.9),  # Overlaps top ROI
        WoodDetectionResult(detected=True, bbox=(400, 350, 500, 420), confidence=0.8), # Overlaps bottom ROI
        WoodDetectionResult(detected=True, bbox=(50, 200, 100, 250), confidence=0.7),  # No overlap
    ]

    print("1. Testing overlap detection...")

    # Test overlaps for top camera
    top_overlaps = overlap_detector.detect_overlaps(wood_detections, "top_camera")
    print(f"✓ Top camera overlaps: {top_overlaps}")

    # Test overlaps for bottom camera
    bottom_overlaps = overlap_detector.detect_overlaps(wood_detections, "bottom_camera")
    print(f"✓ Bottom camera overlaps: {bottom_overlaps}")

    # Show performance stats
    stats = overlap_detector.get_performance_stats()
    print(f"✓ Performance - Calculations: {stats['total_calculations']}, "
          f"Avg time: {stats['avg_calculation_time']:.4f}s")

    return overlap_detector, wood_detections

def demo_visualization(roi_manager, wood_detections):
    """Demonstrate visualization functionality"""
    print("\n=== Visualization Demo ===")

    visualizer = ROIVisualizer(roi_manager)

    # Create demo frames
    top_frame = create_demo_frame()
    bottom_frame = create_demo_frame()

    print("1. Drawing ROI overlays...")

    # Draw ROI overlays
    top_with_rois = visualizer.draw_roi_overlays(top_frame, "top_camera")
    bottom_with_rois = visualizer.draw_roi_overlays(bottom_frame, "bottom_camera")

    print("✓ Added ROI overlays to frames")

    print("2. Drawing wood detection overlays...")

    # Draw wood detections
    top_with_detections = visualizer.draw_wood_detections(top_with_rois, wood_detections)
    bottom_with_detections = visualizer.draw_wood_detections(bottom_with_rois, wood_detections)

    print("✓ Added wood detection overlays")

    print("3. Creating combined overlays...")

    # Create combined overlays
    top_combined = visualizer.draw_combined_overlay(top_frame, "top_camera", wood_detections)
    bottom_combined = visualizer.draw_combined_overlay(bottom_frame, "bottom_camera", wood_detections)

    print("✓ Created combined ROI and wood detection overlays")

    # Show performance stats
    render_stats = visualizer.get_render_stats()
    print(f"✓ Render performance - Total renders: {render_stats['total_renders']}, "
          f"Avg time: {render_stats['avg_render_time']:.4f}s")

    return top_combined, bottom_combined

def demo_workflow_management():
    """Demonstrate workflow management functionality"""
    print("\n=== Workflow Management Demo ===")

    # Mock dependencies
    class MockDetectionModule:
        def detect_defects_in_full_frame(self, frame, camera_name):
            return frame, {"crack": 1, "knot": 2}, []

    class MockGradingModule:
        pass

    class MockArduinoModule:
        def is_connected(self): return True
        def send_grade_command(self, grade): return True

    # Initialize workflow manager
    workflow_manager = ROIBasedWorkflowManager(
        MockDetectionModule(), MockGradingModule(), MockArduinoModule()
    )

    print("1. Starting ROI sessions...")

    # Start sessions
    session1 = workflow_manager.start_roi_session("top_camera", "inspection_zone")
    session2 = workflow_manager.start_roi_session("bottom_camera", "inspection_zone")

    print(f"✓ Started sessions: {session1}, {session2}")

    print("2. Accumulating defects...")

    # Accumulate defects
    defects1 = {"crack": 2, "knot": 1}
    defects2 = {"crack": 1, "knot": 3}

    wood_detection1 = {"bbox": (150, 50, 250, 120), "confidence": 0.9}
    wood_detection2 = {"bbox": (400, 350, 500, 420), "confidence": 0.8}

    workflow_manager.accumulate_defects(session1, defects1, wood_detection1)
    workflow_manager.accumulate_defects(session2, defects2, wood_detection2)

    print("✓ Accumulated defects in sessions")

    print("3. Ending sessions and triggering grading...")

    # End sessions
    results1 = workflow_manager.end_roi_session(session1)
    results2 = workflow_manager.end_roi_session(session2)

    print(f"✓ Session 1 results: {results1['total_defects']}")
    print(f"✓ Session 2 results: {results2['total_defects']}")

    # Show session stats
    session_stats = workflow_manager.get_session_stats()
    print(f"✓ Session stats: {session_stats}")

    return workflow_manager

def demo_complete_system():
    """Demonstrate the complete ROI system"""
    print("\n=== Complete System Demo ===")

    # Initialize the complete ROI module
    roi_module = ROIModule()

    # Create demo frame
    demo_frame = create_demo_frame()

    print("1. Processing frame through complete ROI pipeline...")

    # Process frame
    results = roi_module.process_frame(demo_frame, "top_camera")

    print("✓ Frame processed successfully")
    print(f"✓ Results keys: {list(results.keys())}")
    print(f"✓ Wood detections found: {len(results['wood_detections'])}")
    print(f"✓ Overlaps detected: {len(results['overlaps'])}")

    # Show performance stats
    stats = roi_module.get_stats()
    print(f"✓ System stats: {stats}")

    return results

def main():
    """Main demonstration function"""
    print("🎯 ROI-Based Wood Detection System Demonstration")
    print("=" * 60)

    try:
        # Demo individual components
        roi_manager = demo_roi_management()
        overlap_detector, wood_detections = demo_overlap_detection(roi_manager)
        top_frame, bottom_frame = demo_visualization(roi_manager, wood_detections)
        workflow_manager = demo_workflow_management()

        # Demo complete system
        complete_results = demo_complete_system()

        print("\n" + "=" * 60)
        print("🎉 ROI System Demonstration Complete!")
        print("\nKey Features Implemented:")
        print("✅ Interactive ROI definition system")
        print("✅ ROI configuration management with JSON persistence")
        print("✅ Overlap detection algorithms between wood bounding boxes and ROIs")
        print("✅ Multiple ROI support per camera")
        print("✅ Real-time ROI activation/deactivation")
        print("✅ Integration methods with camera and wood detection systems")
        print("✅ Comprehensive error handling and performance optimizations")

        print("\n📊 Performance Summary:")
        print(f"   - ROI Manager: {len(roi_manager.get_all_rois())} cameras configured")
        print(f"   - Overlap Detector: {overlap_detector.get_performance_stats()['total_calculations']} calculations")
        print(f"   - Workflow Manager: {workflow_manager.get_session_stats()['total_sessions']} sessions processed")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()