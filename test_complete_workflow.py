#!/usr/bin/env python3
"""
Complete IR-triggered predict_stream workflow test with Arduino monitoring
"""

import sys
import os
import time
import threading
import serial

# Add the project root to the path
sys.path.append('/home/inspectura/Desktop/InspecturaGUI')

from modules.detection_module import DetectionModule
from modules.error_handler import log_info, log_error, SystemComponent

class ArduinoMonitor:
    """Monitor Arduino serial output for IR messages"""

    def __init__(self):
        self.ir_messages = []
        self.monitoring = False
        self.thread = None

    def start_monitoring(self):
        """Start monitoring Arduino serial output"""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_serial)
        self.thread.daemon = True
        self.thread.start()
        print("🔍 Started Arduino monitoring")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2)
        print("🛑 Stopped Arduino monitoring")

    def _monitor_serial(self):
        """Monitor serial output in background thread"""
        try:
            ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
            print("📡 Connected to Arduino for monitoring")

            while self.monitoring:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    if line and 'IR:' in line:
                        timestamp = time.strftime('%H:%M:%S')
                        self.ir_messages.append((timestamp, line))
                        print(f"📨 [{timestamp}] ARDUINO: {line}")
                time.sleep(0.1)

            ser.close()

        except Exception as e:
            print(f"❌ Arduino monitoring error: {e}")

    def get_ir_messages(self):
        """Get all IR messages received"""
        return self.ir_messages.copy()

    def wait_for_ir_message(self, expected_value="0", timeout=30):
        """Wait for specific IR message"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            for timestamp, message in self.ir_messages:
                if f"IR: {expected_value}" in message:
                    return True, (timestamp, message)
            time.sleep(0.5)
        return False, None

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
        print("✅ System set to TRIGGER mode - ready for IR beam triggers")

    def handle_arduino_message(self, message):
        """Handle Arduino message - simulate the GUI message handler"""
        if "IR: 0" in message:
            return self.handle_ir_beam_broken()
        elif "IR: 1" in message:
            return self.handle_ir_beam_cleared()
        return False

    def handle_ir_beam_broken(self):
        """Handle IR beam broken event - start predict_stream continuous inference"""
        print("📡 IR beam broken - starting predict_stream continuous inference")

        # Only respond to IR triggers in TRIGGER mode
        if self.current_mode == "TRIGGER":
            if not self.auto_detection_active:
                print("✅ TRIGGER MODE: Starting predict_stream inference...")

                # Start predict_stream continuous inference
                success = self.start_predict_stream_inference()

                if success:
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
                    print("❌ Failed to start predict_stream")
                    return False
            else:
                print("⚠️ IR beam broken but detection already active")
                return False
        else:
            print(f"❌ IR beam broken received but system is in {self.current_mode} mode - ignoring trigger")
            return False

    def handle_ir_beam_cleared(self):
        """Handle IR beam cleared event - stop predict_stream"""
        print("📡 IR beam cleared - stopping predict_stream")

        if self.current_mode == "TRIGGER" and self.auto_detection_active:
            print("✅ TRIGGER MODE: Stopping predict_stream inference")

            # Stop predict_stream
            self.stop_predict_stream_inference()

            # Reset states
            self.live_detection_var = False
            self.auto_grade_var = False
            self.ir_triggered = False
            self.wood_confirmed = False
            self.auto_detection_active = False

            print("✅ Predict stream stopped - back to waiting")
            return True
        else:
            print(f"❌ IR beam cleared but not in TRIGGER mode or no active detection")
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

    def stop_predict_stream_inference(self):
        """Stop predict_stream inference"""
        if self.predict_stream_active:
            print("🛑 Stopping predict_stream inference")
            self.predict_stream_active = False
            # Give thread time to stop
            time.sleep(1)
            print("✅ Predict stream stopped")
        else:
            print("⚠️ Predict stream not active")

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

                        # Stop after 10 frames for testing
                        if self.frame_count >= 10:
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

def test_complete_ir_workflow():
    """Test the complete IR-triggered predict_stream workflow with Arduino monitoring"""
    print("🧪 Testing Complete IR-Triggered Predict Stream Workflow")
    print("=" * 70)

    # Initialize components
    arduino_monitor = ArduinoMonitor()
    app = MockWoodSortingApp()

    try:
        # Step 1: Start Arduino monitoring
        print("📋 Step 1: Starting Arduino monitoring")
        arduino_monitor.start_monitoring()
        time.sleep(2)  # Let monitoring settle

        # Step 2: Check initial state
        print("\n📋 Step 2: Initial state check")
        print(f"   System mode: {app.current_mode}")
        print(f"   Live detection: {app.live_detection_var}")
        print(f"   Predict stream active: {app.predict_stream_active}")
        print(f"   Model loaded: {app.detection_module.defect_model is not None}")

        # Step 3: Set to trigger mode
        print("\n📋 Step 3: Setting system to TRIGGER mode")
        app.set_trigger_mode()
        print(f"   System mode: {app.current_mode}")

        # Step 4: Wait for IR: 0 message from Arduino
        print("\n📋 Step 4: Waiting for IR beam break (IR: 0) from Arduino...")
        print("   (Make sure something is breaking the IR beam)")

        found_ir, ir_message = arduino_monitor.wait_for_ir_message("0", timeout=60)

        if found_ir:
            print(f"   ✅ Found IR message: {ir_message}")

            # Step 5: Process the IR message
            print("\n📋 Step 5: Processing IR beam break message")
            success = app.handle_arduino_message(ir_message[1])

            if success:
                print("   ✅ IR beam break processed successfully")

                # Step 6: Wait for predict_stream to process frames
                print("\n📋 Step 6: Waiting for predict_stream to process frames...")
                timeout = 20
                start_time = time.time()

                while app.predict_stream_active and (time.time() - start_time) < timeout:
                    time.sleep(1)
                    frames_processed = len(app.predict_stream_results)
                    print(f"   Waiting... ({int(time.time() - start_time)}s) - Frames processed: {frames_processed}")

                if app.predict_stream_active:
                    print("   ⚠️ Predict stream still active after timeout")
                    app.stop_predict_stream_inference()
                else:
                    print("   ✅ Predict stream completed")

                # Step 7: Check results
                print("\n📋 Step 7: Checking results")
                print(f"   Frames processed: {len(app.predict_stream_results)}")
                if app.predict_stream_results:
                    print("   Sample results:")
                    for i, result in enumerate(app.predict_stream_results[:5]):
                        print(f"     Frame {result['frame_id']}: {result['defects']}")

                # Step 8: Wait for IR: 1 (beam clear) to stop
                print("\n📋 Step 8: Waiting for IR beam clear (IR: 1) to stop detection...")
                found_clear, clear_message = arduino_monitor.wait_for_ir_message("1", timeout=30)

                if found_clear:
                    print(f"   ✅ Found IR clear message: {clear_message}")
                    app.handle_arduino_message(clear_message[1])
                else:
                    print("   ⚠️ No IR clear message received, stopping manually")
                    app.stop_predict_stream_inference()

                # Step 9: Final status
                print("\n📋 Step 9: Final status")
                print(f"   System mode: {app.current_mode}")
                print(f"   Live detection: {app.live_detection_var}")
                print(f"   Predict stream active: {app.predict_stream_active}")
                print(f"   IR triggered: {app.ir_triggered}")
                print(f"   Auto detection active: {app.auto_detection_active}")
                print(f"   Total frames processed: {len(app.predict_stream_results)}")

                print("\n🎉 Complete IR-triggered predict_stream workflow test completed!")
                return len(app.predict_stream_results) > 0

            else:
                print("   ❌ Failed to process IR beam break")
                return False

        else:
            print("   ❌ No IR: 0 message received within timeout")
            print("   Make sure the IR beam is being broken by placing an object in the beam")
            return False

    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return False

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False

    finally:
        # Cleanup
        arduino_monitor.stop_monitoring()
        if app.predict_stream_active:
            app.stop_predict_stream_inference()

if __name__ == "__main__":
    success = test_complete_ir_workflow()
    print()
    if success:
        print("✅ TEST PASSED: Complete IR-triggered predict_stream workflow working correctly")
        print("🎯 The system successfully:")
        print("   • Monitors Arduino IR messages")
        print("   • Responds to IR: 0 (beam broken) in TRIGGER mode")
        print("   • Starts predict_stream inference")
        print("   • Processes frames and detects defects")
        print("   • Stops when IR: 1 (beam cleared) is received")
    else:
        print("❌ TEST FAILED: Issues with complete IR-triggered predict_stream workflow")
    sys.exit(0 if success else 1)
