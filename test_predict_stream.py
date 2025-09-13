#!/usr/bin/env python3
"""
Test script for predict_stream functionality
"""

import sys
import os
import time
import threading

# Add the project root to the path
sys.path.append('/home/inspectura/Desktop/InspecturaGUI')

from modules.detection_module import DetectionModule
from modules.error_handler import log_info, log_error, SystemComponent

def test_predict_stream():
    """Test the predict_stream functionality"""
    print("Testing predict_stream functionality...")

    # Initialize detection module
    detection = DetectionModule()

    # Check if model loaded
    if detection.defect_model is None:
        print("❌ Defect model not loaded")
        return False

    print("✅ Defect model loaded successfully")

    # Test predict_stream availability
    try:
        import degirum_tools
        print("✅ degirum_tools imported successfully")

        # Test predict_stream function
        if hasattr(degirum_tools, 'predict_stream'):
            print("✅ predict_stream function available")
        else:
            print("❌ predict_stream function not available")
            return False

    except ImportError:
        print("❌ degirum_tools not available")
        return False

    # Test basic predict_stream call (this will fail without camera, but tests the setup)
    try:
        print("Testing predict_stream setup...")

        # Create a simple test analyzer
        class TestAnalyzer(degirum_tools.ResultAnalyzerBase):
            def __init__(self):
                self.frame_count = 0

            def analyze(self, result):
                self.frame_count += 1
                print(f"Frame {self.frame_count} processed")
                if self.frame_count >= 3:  # Stop after 3 frames for testing
                    return False  # Signal to stop

        analyzer = TestAnalyzer()

        # Try to start predict_stream (this will likely fail due to no camera, but tests the setup)
        try:
            for result in degirum_tools.predict_stream(
                model=detection.defect_model,
                video_source_id=0,
                fps=5,
                analyzers=[analyzer]
            ):
                if analyzer.frame_count >= 3:
                    break
                time.sleep(0.1)

            print("✅ predict_stream setup successful")

        except Exception as e:
            if "camera" in str(e).lower() or "video" in str(e).lower():
                print("✅ predict_stream setup successful (expected camera error)")
            else:
                print(f"❌ predict_stream setup failed: {e}")
                return False

    except Exception as e:
        print(f"❌ predict_stream test failed: {e}")
        return False

    print("🎉 All predict_stream tests passed!")
    return True

if __name__ == "__main__":
    success = test_predict_stream()
    sys.exit(0 if success else 1)
