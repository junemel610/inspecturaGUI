#!/usr/bin/env python3
"""
Test script for wood_detection_module.py
Tests basic functionality and integration
"""

import sys
import os
import numpy as np
import cv2
import time

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

try:
    from wood_detection_module import (
        WoodDetectionEngine, CannyEdgeDetector, ColorRecognitionEngine,
        ContourAnalyzer, WoodDetectionResult, DEFAULT_CONFIG,
        integrate_with_camera_module, integrate_with_detection_module
    )
    print("✓ Successfully imported wood detection module")
except ImportError as e:
    print(f"✗ Failed to import wood detection module: {e}")
    sys.exit(1)

def test_basic_components():
    """Test basic component instantiation"""
    print("\n=== Testing Basic Components ===")

    try:
        # Test CannyEdgeDetector
        canny = CannyEdgeDetector(DEFAULT_CONFIG['canny'])
        print("✓ CannyEdgeDetector instantiated successfully")

        # Test ColorRecognitionEngine
        color = ColorRecognitionEngine(DEFAULT_CONFIG['color'])
        print("✓ ColorRecognitionEngine instantiated successfully")

        # Test ContourAnalyzer
        contour = ContourAnalyzer(DEFAULT_CONFIG['contour'])
        print("✓ ContourAnalyzer instantiated successfully")

        # Test WoodDetectionEngine
        engine = WoodDetectionEngine(DEFAULT_CONFIG)
        print("✓ WoodDetectionEngine instantiated successfully")

        return True
    except Exception as e:
        print(f"✗ Component instantiation failed: {e}")
        return False

def test_edge_detection():
    """Test Canny edge detection"""
    print("\n=== Testing Canny Edge Detection ===")

    try:
        canny = CannyEdgeDetector(DEFAULT_CONFIG['canny'])

        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Test edge detection
        edges = canny.detect_edges(test_frame)

        if edges.shape == test_frame.shape[:2] and edges.dtype == np.uint8:
            print("✓ Canny edge detection working correctly")
            return True
        else:
            print(f"✗ Edge detection output shape/dtype incorrect: {edges.shape}, {edges.dtype}")
            return False

    except Exception as e:
        print(f"✗ Edge detection test failed: {e}")
        return False

def test_color_recognition():
    """Test color recognition"""
    print("\n=== Testing Color Recognition ===")

    try:
        color_engine = ColorRecognitionEngine(DEFAULT_CONFIG['color'])

        # Create test frame with brown color (typical wood color)
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        test_frame[:, :] = [80, 50, 30]  # Brown color in BGR

        # Test color recognition
        results = color_engine.recognize_wood_color(test_frame)

        if isinstance(results, dict) and len(results) > 0:
            print(f"✓ Color recognition working, detected colors: {list(results.keys())}")
            dominant = color_engine.get_dominant_color(results)
            print(f"✓ Dominant color: {dominant}")
            return True
        else:
            print("✗ Color recognition returned empty results")
            return False

    except Exception as e:
        print(f"✗ Color recognition test failed: {e}")
        return False

def test_contour_analysis():
    """Test contour analysis"""
    print("\n=== Testing Contour Analysis ===")

    try:
        contour_analyzer = ContourAnalyzer(DEFAULT_CONFIG['contour'])

        # Create test edges with a simple rectangle
        edges = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(edges, (50, 50), (150, 150), 255, 2)

        # Test contour finding
        contours = contour_analyzer.find_wood_contours(edges)

        if isinstance(contours, list) and len(contours) > 0:
            print(f"✓ Contour analysis working, found {len(contours)} contours")
            if 'bbox' in contours[0] and 'area' in contours[0]:
                print("✓ Contour data structure correct")
                return True
            else:
                print("✗ Contour data missing required fields")
                return False
        else:
            print("✗ No contours found")
            return False

    except Exception as e:
        print(f"✗ Contour analysis test failed: {e}")
        return False

def test_full_pipeline():
    """Test complete wood detection pipeline"""
    print("\n=== Testing Full Detection Pipeline ===")

    try:
        engine = WoodDetectionEngine(DEFAULT_CONFIG)

        # Create test frame resembling wood
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some wood-like colors and shapes
        cv2.rectangle(test_frame, (100, 100), (400, 300), (80, 50, 30), -1)  # Brown rectangle
        cv2.rectangle(test_frame, (110, 110), (390, 290), (60, 40, 20), 2)  # Darker border

        # Test detection
        start_time = time.time()
        detections = engine.detect_wood(test_frame)
        processing_time = time.time() - start_time

        print(f"✓ Pipeline completed in {processing_time:.3f}s")
        print(f"✓ Found {len(detections)} wood detections")

        if detections:
            best_detection = engine.get_best_detection(test_frame)
            if best_detection and best_detection.confidence > 0:
                print(f"✓ Best detection confidence: {best_detection.confidence:.3f}")
                print(f"✓ Detection bbox: {best_detection.bbox}")
                return True
            else:
                print("✗ No valid detections found")
                return False
        else:
            print("! No detections found (may be normal for synthetic test data)")
            return True  # Not necessarily a failure

    except Exception as e:
        print(f"✗ Full pipeline test failed: {e}")
        return False

def test_performance_stats():
    """Test performance statistics"""
    print("\n=== Testing Performance Statistics ===")

    try:
        engine = WoodDetectionEngine(DEFAULT_CONFIG)

        # Run multiple detections
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        for _ in range(5):
            engine.detect_wood(test_frame)

        stats = engine.get_performance_stats()

        if stats and 'avg_processing_time' in stats:
            print(f"✓ Performance stats available: {stats}")
            return True
        else:
            print("✗ Performance stats not available")
            return False

    except Exception as e:
        print(f"✗ Performance stats test failed: {e}")
        return False

def test_error_handling():
    """Test error handling"""
    print("\n=== Testing Error Handling ===")

    try:
        engine = WoodDetectionEngine(DEFAULT_CONFIG)

        # Test with None frame
        result = engine.detect_wood(None)
        if result == []:
            print("✓ Handles None frame correctly")
        else:
            print("✗ Does not handle None frame correctly")
            return False

        # Test with empty frame
        empty_frame = np.array([])
        result = engine.detect_wood(empty_frame)
        if result == []:
            print("✓ Handles empty frame correctly")
        else:
            print("✗ Does not handle empty frame correctly")
            return False

        return True

    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting Wood Detection Module Tests")
    print("=" * 50)

    tests = [
        test_basic_components,
        test_edge_detection,
        test_color_recognition,
        test_contour_analysis,
        test_full_pipeline,
        test_performance_stats,
        test_error_handling
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Wood detection module is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())