#!/usr/bin/env python3
"""
Comprehensive test for IR-triggered predict_stream workflow
"""

import sys
import os
import time
import threading

# Add the project root to the path
sys.path.append('/home/inspectura/Desktop/InspecturaGUI')

from modules.detection_module import DetectionModule
from modules.error_handler import log_info, log_error, SystemComponent

class MockWoodSortingApp:
    """Mock version of WoodSortingApp for testing predict_stream workflow"""

    def __init__(self):
        self.dev_mode = False
        self.current_mode = "IDLE"
        self.live_detection_var = False
        self.auto_grade_var = False
        self.predict_stream_active = False
        self.auto_detection_active = False
        self.ir_triggered = False
        self.wood_confirmed = False
        self.predict_stream_results = []
        self.current_defects = {"top": {"defects": {}, "defect_list": []}}
        self.latest_annotated_frame = None

        # Initialize detection module
        self.detection_module = DetectionModule()

        # Import required modules
        try:
            import degirum_tools
            self.DEGIRUM_TOOLS_AVAILABLE = True
        except ImportError:
            self.DEGIRUM_TOOLS_AVAILABLE = False

    def set_trigger_mode(self):
        """Set system to trigger mode"""
        self.current_mode = "TRIGGER"
        print("✅ System set to TRIGGER mode")

    def handle_ir_beam_broken(self):
        """Handle IR beam broken event - start predict_stream continuous inference"""
        print("📡 IR beam broken - starting predict_stream continuous inference")

        # Only respond to IR triggers in TRIGGER mode
        if self.current_mode == "TRIGGER":
            if not self.auto_detection_active:
                print("✅ TRIGGER MODE: Starting predict_stream inference...")

                # Start predict_stream continuous inference
                self.start_predict_stream_inference()

                # Set the live detection variables
                self.live_detection_var = True
                self.auto_grade_var = True

                # Update states
                self.ir_triggered = True
                self.wood_confirmed = False
                self.auto_detection_active = True

                print("✅ IR TRIGGERED - Predict Stream Active!")
                return True
            else:
                print("⚠️ IR beam broken but detection already active")
                return False
        else:
            print(f"❌ IR beam broken received but system is in {self.current_mode} mode - ignoring trigger")
            return False

    def start_predict_stream_inference(self):
        """Start continuous inference using predict_stream when IR beam is broken"""
        try:
            if not self.DEGIRUM_TOOLS_AVAILABLE:
                print("❌ degirum_tools not available - cannot start predict_stream")
                return False

            if self.predict_stream_active:
                print("⚠️ Predict stream already active")
                return False

            print("🚀 Starting predict_stream continuous inference")

            # Reset results collection
            self.predict_stream_results = []

            # Start predict_stream in a separate thread
            self.predict_stream_active = True
            self.predict_stream_thread = threading.Thread(target=self._run_predict_stream_inference)
            self.predict_stream_thread.daemon = True
            self.predict_stream_thread.start()

            print("✅ Predict stream inference started successfully")
            return True

        except Exception as e:
            print(f"❌ Error starting predict_stream inference: {str(e)}")
            self.predict_stream_active = False
            return False

    def _run_predict_stream_inference(self):
        """Run the predict_stream inference loop in a separate thread"""
        try:
            import degirum_tools

            # Get the defect model from detection module
            model = self.detection_module.defect_model
            if model is None:
                print("❌ Defect model not available for predict_stream")
                return

            print("🔄 Starting predict_stream inference loop")

            # Create a proper analyzer class that inherits from ResultAnalyzerBase
            class DefectAnalyzer(degirum_tools.ResultAnalyzerBase):
                def __init__(self, gui_instance):
                    self.gui = gui_instance
                    self.frame_count = 0

                def analyze(self, result):
                    """Analyze the inference result and update GUI state"""
                    try:
                        self.frame_count += 1

                        # Process the inference result
                        detections = result.results if hasattr(result, 'results') else []

                        frame_defects = {}
                        frame_defect_measurements = []

                        for det in detections:
                            confidence = det.get('confidence', 0)
                            if confidence < self.gui.detection_module.defect_confidence_threshold:
                                continue

                            model_label = det['label']
                            # Simple defect counting for test
                            if model_label in frame_defects:
                                frame_defects[model_label] += 1
                            else:
                                frame_defects[model_label] = 1

                        # Store frame results
                        frame_result = {
                            'frame_id': self.frame_count,
                            'defects': frame_defects,
                            'timestamp': time.time()
                        }

                        self.gui.predict_stream_results.append(frame_result)

                        # Store the annotated frame for GUI display
                        if hasattr(result, 'image_overlay'):
                            self.gui.latest_annotated_frame = result.image_overlay

                        print(f"📊 Processed frame {self.frame_count} - defects: {frame_defects}")

                        # Stop after 5 frames for testing
                        if self.frame_count >= 5:
                            self.gui.predict_stream_active = False
                            return False

                    except Exception as e:
                        print(f"❌ Error in DefectAnalyzer.analyze: {str(e)}")

                def annotate(self, result, image):
                    """Add annotations to the image - return the annotated image"""
                    return result.image_overlay if hasattr(result, 'image_overlay') else image

            # Create analyzer instance
            analyzer = DefectAnalyzer(self)

            # Use predict_stream with camera index
            camera_index = 0
            print(f"📹 Starting predict_stream with camera index {camera_index}")

            try:
                for result in degirum_tools.predict_stream(
                    model=model,
                    video_source_id=camera_index,
                    fps=10,  # 10 FPS for testing
                    analyzers=[analyzer]
                ):
                    # Check if we should stop
                    if not self.predict_stream_active:
                        print("🛑 Predict stream stopping due to flag")
                        break

                    # Small delay to prevent overwhelming the system
                    time.sleep(0.01)

            except Exception as e:
                print(f"❌ Error in predict_stream loop: {str(e)}")

            print(f"✅ Predict stream stopped after {analyzer.frame_count} frames")

        except Exception as e:
            print(f"❌ Error in predict_stream thread: {str(e)}")
        finally:
            self.predict_stream_active = False

def test_ir_triggered_workflow():
    """Test the complete IR-triggered predict_stream workflow"""
    print("🧪 Testing IR-triggered predict_stream workflow...")
    print("=" * 60)

    # Create mock app
    app = MockWoodSortingApp()

    # Step 1: Check initial state
    print("📋 Step 1: Initial state check")
    print(f"   Mode: {app.current_mode}")
    print(f"   Live detection: {app.live_detection_var}")
    print(f"   Predict stream active: {app.predict_stream_active}")
    print(f"   Model loaded: {app.detection_module.defect_model is not None}")
    print()

    # Step 2: Set to trigger mode
    print("📋 Step 2: Setting to TRIGGER mode")
    app.set_trigger_mode()
    print(f"   Mode: {app.current_mode}")
    print()

    # Step 3: Simulate IR beam break
    print("📋 Step 3: Simulating IR beam break (IR: 0)")
    success = app.handle_ir_beam_broken()
    if success:
        print("   ✅ IR beam break handled successfully")
        print(f"   Live detection: {app.live_detection_var}")
        print(f"   Predict stream active: {app.predict_stream_active}")
        print(f"   Auto detection active: {app.auto_detection_active}")
    else:
        print("   ❌ IR beam break failed")
        return False
    print()

    # Step 4: Wait for predict_stream to process frames
    print("📋 Step 4: Waiting for predict_stream to process frames...")
    timeout = 15  # 15 seconds timeout
    start_time = time.time()

    while app.predict_stream_active and (time.time() - start_time) < timeout:
        time.sleep(1)
        print(f"   Waiting... ({int(time.time() - start_time)}s)")

    if app.predict_stream_active:
        print("   ⚠️ Predict stream still active after timeout")
        app.predict_stream_active = False
        time.sleep(1)  # Give thread time to stop
    else:
        print("   ✅ Predict stream completed")

    print()

    # Step 5: Check results
    print("📋 Step 5: Checking results")
    print(f"   Results collected: {len(app.predict_stream_results)}")
    if app.predict_stream_results:
        print("   Sample results:")
        for i, result in enumerate(app.predict_stream_results[:3]):
            print(f"     Frame {result['frame_id']}: {result['defects']}")

    print()

    # Step 6: Final status
    print("📋 Step 6: Final status")
    print(f"   Mode: {app.current_mode}")
    print(f"   Live detection: {app.live_detection_var}")
    print(f"   Predict stream active: {app.predict_stream_active}")
    print(f"   IR triggered: {app.ir_triggered}")
    print(f"   Auto detection active: {app.auto_detection_active}")

    print()
    print("🎉 IR-triggered predict_stream workflow test completed!")
    return len(app.predict_stream_results) > 0

if __name__ == "__main__":
    success = test_ir_triggered_workflow()
    print()
    if success:
        print("✅ TEST PASSED: IR-triggered predict_stream workflow working correctly")
    else:
        print("❌ TEST FAILED: Issues with IR-triggered predict_stream workflow")
    sys.exit(0 if success else 1)
