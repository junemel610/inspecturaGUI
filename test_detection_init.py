#!/usr/bin/env python3
"""
Test script to initialize DetectionModule and see diagnostic logs
"""
import sys
import os

# Add the modules directory to the path
sys.path.append('modules')

from detection_module import DetectionModule

def test_detection_init():
    print("=== TESTING DETECTION MODULE INITIALIZATION ===")

    try:
        # Initialize detection module
        detection = DetectionModule(dev_mode=True)
        print("DetectionModule initialized successfully")

        # Check model status
        if detection.ultralytics_wood_model is not None:
            print("Ultralytics YOLO model loaded successfully")
        elif detection.onnx_wood_session is not None:
            print("ONNX model loaded successfully")
        else:
            print("No YOLO/ONNX model available")

        if detection.wood_model is not None:
            print("DeGirum wood model loaded successfully")
        else:
            print("DeGirum wood model not available")

    except Exception as e:
        print(f"Error initializing DetectionModule: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detection_init()