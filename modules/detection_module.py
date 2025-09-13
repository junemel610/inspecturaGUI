print("=== DETECTION_MODULE LOADED ===")
import cv2
import numpy as np
from modules.utils_module import calculate_defect_size, map_model_output_to_standard
# from modules.alignment_module import AlignmentModule, AlignmentResult, AlignmentStatus

try:
    import degirum as dg
    import degirum_tools
    DEGIRUM_AVAILABLE = True
except ImportError:
    print("Warning: 'degirum' module not found. Running in mock mode for detection.")
    DEGIRUM_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print("DEBUG: onnxruntime imported successfully")
except ImportError as e:
    print(f"Warning: 'onnxruntime' module not found. ONNX models will not be available. Error: {e}")
    ONNX_AVAILABLE = False
    print("DEBUG: onnxruntime import failed")

# Check for ultralytics (required for YOLOv8/YOLOE models)
try:
    import ultralytics
    ULTRALYTICS_AVAILABLE = True
    print("DEBUG: ultralytics imported successfully")
except ImportError as e:
    print(f"Warning: 'ultralytics' module not found. YOLOv8/YOLOE models may not load properly. Error: {e}")
    ULTRALYTICS_AVAILABLE = False
    print("DEBUG: ultralytics import failed")

class DetectionModule:
    def __init__(self, dev_mode=False, inference_host_address="@local", config=None):
        print("DEBUG: DetectionModule __init__ called")
        self.dev_mode = dev_mode
        self.inference_host_address = inference_host_address
        self.config = config

        # Model paths for defect detection only
        # COMMENTED OUT: Wood detection model - focusing on defect detection only
        # self.wood_model_path = "/home/inspectura/Desktop/InspecturaGUI/models/Wood_Plank--640x640_quant_hailort_hailo8_2/Wood_Plank--640x640_quant_hailort_hailo8_2.hef"

        # Defect detection model path (focus model)
        self.defect_model_path = "/home/inspectura/Desktop/InspecturaGUI/models/UpdatedDefects--640x640_quant_hailort_hailo8_1/UpdatedDefects--640x640_quant_hailort_hailo8_1.hef"
        self.defect_model_zoo_url = "/home/inspectura/Desktop/InspecturaGUI/models/UpdatedDefects--640x640_quant_hailort_hailo8_1"
        self.defect_model_name = "UpdatedDefects--640x640_quant_hailort_hailo8_1"

        # ONNX model path (commented out)
        # self.onnx_wood_model_path = "models/Wood_Detection/320yoloe-11s-seg.onnx"

        # Model instances
        # COMMENTED OUT: Wood model - focusing on defect detection only
        # self.wood_model = None
        self.defect_model = None
        self.onnx_wood_session = None
        self.ultralytics_wood_model = None

        # Detection thresholds
        self.wood_confidence_threshold = 0.3  # Reasonable threshold for wood detection
        self.defect_confidence_threshold = 0.5

        # Load models in both dev and production modes
        self.load_models()

    def load_models(self):
        """Load both the wood detection and defect detection models."""
        print("DEBUG: load_models() called")

        # Wood detection model loading commented out - focusing on defect detection only
        # print(f"DEBUG: ULTRALYTICS_AVAILABLE = {ULTRALYTICS_AVAILABLE}")
        # if ULTRALYTICS_AVAILABLE:
        #     print(f"DEBUG: Loading YOLO model with ultralytics from: {self.onnx_wood_model_path}")
        #     try:
        #         import os
        #         if os.path.exists(self.onnx_wood_model_path):
        #             print(f"DEBUG: Model file exists at path: {self.onnx_wood_model_path}")
        #             file_size = os.path.getsize(self.onnx_wood_model_path)
        #             print(f"DEBUG: Model file size: {file_size} bytes")

        #             from ultralytics import YOLO
        #             print(f"DEBUG: Attempting to load YOLO model from: {self.onnx_wood_model_path}")
        #             self.ultralytics_wood_model = YOLO(self.onnx_wood_model_path, task='segment')
        #             print("Ultralytics YOLO wood detection model loaded successfully.")
    
        #             # Test the model with a dummy frame to catch ONNX Runtime issues early
        #             try:
        #                 import numpy as np
        #                 dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        #                 test_results = self.ultralytics_wood_model.predict(dummy_frame, conf=0.5, verbose=False)
        #                 print("DEBUG: YOLO model test prediction successful")
        #             except Exception as test_e:
        #                 print(f"ERROR: YOLO model test prediction failed: {test_e}")
        #                 print("DEBUG: Disabling YOLO model due to runtime issues")
        #                 self.ultralytics_wood_model = None
        #         else:
        #             print(f"ERROR: Model file NOT found at path: {self.onnx_wood_model_path}")
        #             self.ultralytics_wood_model = None
        #     except Exception as e:
        #         print(f"ERROR: Failed to load YOLO model with ultralytics: {e}")
        #         print(f"DEBUG: Exception type: {type(e).__name__}")
        #         import traceback
        #         print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        #         self.ultralytics_wood_model = None
        # else:
        #     print("ERROR: Ultralytics not available. Please install with: pip install ultralytics")
        #     self.ultralytics_wood_model = None

        # Load DeGirum defect detection model only (wood detection commented out)
        if DEGIRUM_AVAILABLE:
            try:
                # COMMENTED OUT: Wood detection model - focusing on defect detection only
                # self.wood_model = dg.load_model(
                #     model_name=self.wood_model_path,
                #     inference_host_address=self.inference_host_address
                # )
                # print("DeGirum wood detection model loaded successfully.")

                # Load defect detection model
                try:
                    self.defect_model = dg.load_model(
                        model_name=self.defect_model_name,
                        inference_host_address=self.inference_host_address,
                        zoo_url=self.defect_model_zoo_url
                    )
                    print("DeGirum defect detection model loaded successfully.")
                except Exception as model_error:
                    print(f"Failed to load model with dg.load_model: {model_error}")
                    # Try alternative loading method for HEF files
                    try:
                        # Try loading directly as a local model
                        import os
                        if os.path.exists(self.defect_model_path):
                            print(f"Model file exists at {self.defect_model_path}, trying alternative loading...")
                            # For now, create a mock model for testing
                            self.defect_model = None
                            print("WARNING: Using mock model for testing - replace with proper HEF loading")
                        else:
                            print(f"Model file not found at {self.defect_model_path}")
                            self.defect_model = None
                    except Exception as alt_error:
                        print(f"Alternative loading also failed: {alt_error}")
                        self.defect_model = None

            except Exception as e:
                print(f"Error loading DeGirum models: {e}")
                # self.wood_model = None
                self.defect_model = None
        else:
            print("DeGirum not available, defect detection will not be available.")

    # COMMENTED OUT: Wood detection method - focusing on defect detection only
    # def detect_wood_presence(self, frame):
    #     """
    #     Stage 1: Detect if wood is present in the frame using the wood detection model.
    #     Returns (wood_detected, confidence, wood_bbox)
    #     """
    #     print(f"DEBUG: Starting wood detection on frame shape: {frame.shape}")
    #     print(f"DEBUG: ultralytics_wood_model: {self.ultralytics_wood_model}")
    #     print(f"DEBUG: onnx_wood_session: {self.onnx_wood_session}")
    #     print(f"DEBUG: wood_model: {self.wood_model}")

    #     # COMMENTED OUT: ONNX/YOLO wood detection - focusing on DeGirum only
    #     # if self.ultralytics_wood_model is not None or self.onnx_wood_session is not None:
    #     #     print("DEBUG: Trying YOLO/ONNX model first")
    #     #     try:
    #     #         result = self._detect_wood_with_onnx(frame)
    #     #         print(f"DEBUG: YOLO/ONNX result: {result}")
    #     #         return result
    #     #     except Exception as e:
    #     #         print(f"Error in YOLO/ONNX wood detection: {e}")
    #     #         print("DEBUG: Falling back due to YOLO/ONNX error")

    #     # Focus on DeGirum model for wood detection
    #     if self.wood_model is not None:
    #         print("DEBUG: Using DeGirum model for wood detection")
    #         try:
    #             result = self._detect_wood_with_degirum(frame)
    #             print(f"DEBUG: DeGirum result: {result}")
    #             return result
    #         except Exception as e:
    #         print(f"Error in DeGirum wood detection: {e}")
    #         print("DEBUG: Falling back due to DeGirum error")

    #     # Final fallback to basic detection
    #     print("DEBUG: No ML models available, using fallback detection")
    #     result = self._fallback_wood_detection(frame)
    #     print(f"DEBUG: Fallback detection result: {result}")
    #     return result

    # STUB: Simple replacement for detect_wood_presence to avoid GUI errors
    def detect_wood_presence(self, frame):
        """Stub method - wood detection commented out, always return wood detected"""
        print("DEBUG: detect_wood_presence called but wood detection is commented out")
        # Return dummy values to prevent GUI errors
        return True, 0.9, [100, 100, 500, 300]

    # REMOVED: All wood detection methods - focusing on defect detection only
    # def _detect_wood_with_onnx(self, frame):
    # def _detect_wood_with_degirum(self, frame):
    # def _fallback_wood_detection(self, frame):
    # def detect_defects_in_wood_region(self, frame, wood_bbox, camera_name="top"):
        """
        Stage 2: Detect defects within the identified wood region.
        Returns (annotated_frame, defect_dict, defect_measurements)
        """
        if self.defect_model is None:
            print("Defect detection model not available")
            return frame, {}, []
        
        try:
            # Crop frame to wood region if bbox is provided
            if wood_bbox:
                x1, y1, x2, y2 = wood_bbox
                # Add some padding around the wood region
                padding = 20
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)
                
                wood_region = frame[y1:y2, x1:x2]
            else:
                # Use full frame if no wood bbox
                wood_region = frame
                x1, y1 = 0, 0
            
            # Run defect detection on wood region
            inference_result = self.defect_model(wood_region)
            
            # Get annotated frame for the wood region
            annotated_region = inference_result.image_overlay
            
            # Create full annotated frame
            annotated_frame = frame.copy()
            if wood_bbox:
                # Place the annotated region back into the full frame
                annotated_frame[y1:y2, x1:x2] = annotated_region
                # Note: Wood bounding box will be drawn at the end of analyze_frame
            else:
                annotated_frame = annotated_region
            
            # Process defect detections
            final_defect_dict = {}
            defect_measurements = []
            detections = inference_result.results
            
            print(f"DEBUG: Processing {len(detections)} defect detections")
            
            for det in detections:
                confidence = det.get('confidence', 0)
                print(f"DEBUG: Defect detection - confidence: {confidence:.3f}, threshold: {self.defect_confidence_threshold}")
                
                if confidence < self.defect_confidence_threshold:
                    print(f"DEBUG: Skipping low confidence detection: {confidence:.3f}")
                    continue
                
                model_label = det['label']
                print(f"DEBUG: Raw model label: '{model_label}'")
                standard_defect_type = map_model_output_to_standard(model_label)
                print(f"DEBUG: Mapped to standard type: '{standard_defect_type}'")
                
                # Adjust bbox coordinates if we cropped the frame
                bbox = det['bbox'].copy()
                if wood_bbox:
                    bbox[0] += x1  # Adjust x1
                    bbox[1] += y1  # Adjust y1
                    bbox[2] += x1  # Adjust x2
                    bbox[3] += y1  # Adjust y2
                
                bbox_info = {'bbox': bbox}
                size_mm, percentage = calculate_defect_size(bbox_info, camera_name)
                
                # Store measurements for grading
                defect_measurements.append((standard_defect_type, size_mm, percentage))
                
                # Count defects by type
                if standard_defect_type in final_defect_dict:
                    final_defect_dict[standard_defect_type] += 1
                else:
                    final_defect_dict[standard_defect_type] = 1
            
            return annotated_frame, final_defect_dict, defect_measurements
            
        except Exception as e:
            print(f"Error during defect detection on {camera_name} camera: {e}")
            return frame, {}, []

    def detect_defects_in_full_frame(self, frame, camera_name="top"):
        """
        Detect defects on the full frame (no wood region cropping)
        Returns (annotated_frame, defect_dict, defect_measurements)
        """
        if self.defect_model is None:
            print("Defect detection model not available")
            return frame, {}, []
        
        try:
            # Run defect detection on full frame
            inference_result = self.defect_model(frame)
            
            # Get annotated frame
            annotated_frame = inference_result.image_overlay
            
            # Process defect detections
            final_defect_dict = {}
            defect_measurements = []
            detections = inference_result.results
            
            print(f"DEBUG: Processing {len(detections)} defect detections on full frame")
            
            for det in detections:
                confidence = det.get('confidence', 0)
                print(f"DEBUG: Defect detection - confidence: {confidence:.3f}, threshold: {self.defect_confidence_threshold}")
                
                if confidence < self.defect_confidence_threshold:
                    print(f"DEBUG: Skipping low confidence detection: {confidence:.3f}")
                    continue
                
                model_label = det['label']
                print(f"DEBUG: Raw model label: '{model_label}'")
                standard_defect_type = map_model_output_to_standard(model_label)
                print(f"DEBUG: Mapped to standard type: '{standard_defect_type}'")
                
                # Adjust bbox coordinates (no cropping, so coordinates are already correct)
                bbox = det['bbox'].copy()
                
                bbox_info = {'bbox': bbox}
                size_mm, percentage = calculate_defect_size(bbox_info, camera_name)
                
                # Store measurements for grading
                defect_measurements.append((standard_defect_type, size_mm, percentage))
                
                # Count defects by type
                if standard_defect_type in final_defect_dict:
                    final_defect_dict[standard_defect_type] += 1
                else:
                    final_defect_dict[standard_defect_type] = 1
            
            print(f"DEBUG: Final defect counts: {final_defect_dict}")
            return annotated_frame, final_defect_dict, defect_measurements
            
        except Exception as e:
            print(f"Error during defect detection on full frame for {camera_name} camera: {e}")
            return frame, {}, []

    def analyze_frame(self, frame, camera_name="top"):
        """
        Main analysis function - FOCUS ON DEFECT DETECTION ONLY:
        Skip wood detection, go directly to defect detection on full frame
        """
        print(f"DEBUG: analyze_frame - Focusing on defect detection only (wood detection commented out)")
        
        # COMMENTED OUT: Wood detection - focusing on defect detection only
        # Stage 1: Wood detection
        # wood_detected, wood_confidence, wood_bbox = self.detect_wood_presence(frame)
        # Store confidence for visualization
        # self.last_wood_confidence = wood_confidence

        # Debug logging for bounding box
        # print(f"DEBUG: analyze_frame - wood_detected: {wood_detected}, wood_bbox: {wood_bbox}")
        # if wood_detected and wood_bbox:
        #     print(f"DEBUG: Wood detected with bbox: {wood_bbox}")
        # elif wood_detected:
        #     print("DEBUG: Wood detected but no bbox returned")
        # else:
        #     print("DEBUG: No wood detected")

        # if not wood_detected:
        #     # No wood detected, return original frame with alignment overlay
        #     print("DEBUG: analyze_frame - No wood detected path")
        #     annotated_frame = frame.copy()
        #     cv2.putText(annotated_frame, "No Wood Detected", (50, 50),
        #                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #     # Add alignment overlay for reference
        #     alignment_result = self.alignment_module.check_wood_alignment(frame, None)
        #     annotated_frame = self.alignment_module.draw_alignment_overlay(annotated_frame, alignment_result)

        #     return annotated_frame, {}, [], alignment_result

        # # Stage 2: Alignment checking
        # alignment_result = self.alignment_module.check_wood_alignment(frame, wood_bbox)

        # # Stage 3: Defect detection in wood region (only if aligned)
        # if alignment_result.status == AlignmentStatus.ALIGNED:
        #     print("DEBUG: analyze_frame - Aligned path")
        #     annotated_frame, defect_dict, defect_measurements = self.detect_defects_in_wood_region(frame, wood_bbox, camera_name)
        # else:
        #     # Wood detected but misaligned - still show wood region but skip defect detection
        #     print("DEBUG: analyze_frame - Misaligned path")
        #     annotated_frame = frame.copy()
        #     if wood_bbox:
        #         x1, y1, x2, y2 = wood_bbox
        #         # Enhanced visualization for misaligned wood - use RED for better visibility
        #         cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Thicker RED border
        #         # Add corner markers for misaligned wood
        #         corner_size = 25
        #         # Top-left corner
        #         cv2.line(annotated_frame, (x1, y1), (x1 + corner_size, y1), (0, 0, 255), 4)
        #         cv2.line(annotated_frame, (x1, y1), (x1, y1 + corner_size), (0, 0, 255), 4)
        #         # Top-right corner
        #         cv2.line(annotated_frame, (x2, y1), (x2 - corner_size, y1), (0, 0, 255), 4)
        #         cv2.line(annotated_frame, (x2, y1), (x2, y1 + corner_size), (0, 0, 255), 4)
        #         # Bottom-left corner
        #         cv2.line(annotated_frame, (x1, y2), (x1 + corner_size, y2), (0, 0, 255), 4)
        #         cv2.line(annotated_frame, (x1, y2), (x1, y2 - corner_size), (0, 0, 255), 4)
        #         # Bottom-right corner
        #         cv2.line(annotated_frame, (x2, y2), (x2 - corner_size, y2), (0, 0, 255), 4)
        #         cv2.line(annotated_frame, (x2, y2), (x2, y2 - corner_size), (0, 0, 255), 4)

        #     # Add alignment overlay
        #     annotated_frame = self.alignment_module.draw_alignment_overlay(annotated_frame, alignment_result)
        #     return annotated_frame, {}, [], alignment_result

        # FOCUS: Go directly to defect detection on full frame
        print("DEBUG: analyze_frame - Direct defect detection on full frame")
        annotated_frame, defect_dict, defect_measurements = self.detect_defects_in_full_frame(frame, camera_name)

        # Create a dummy alignment result for compatibility
        from enum import Enum
        class DummyAlignmentStatus(Enum):
            ALIGNED = "aligned"

        class DummyAlignmentResult:
            def __init__(self):
                self.status = DummyAlignmentStatus.ALIGNED
                self.top_overlap_percent = 1.0
                self.bottom_overlap_percent = 1.0
                self.wood_bbox = None
                self.confidence_score = 1.0
                self.details = {"message": "Full-frame defect detection - alignment not required"}

        alignment_result = DummyAlignmentResult()

        return annotated_frame, defect_dict, defect_measurements, alignment_result

    def detect_wood(self, frame):
        """
        Enhanced wood detection using the wood detection model.
        Falls back to visual detection if model is not available.
        Returns True if wood is detected, False otherwise.
        """
        wood_detected, confidence, _ = self.detect_wood_presence(frame)
        return wood_detected

    # REMOVED: All wood detection methods - focusing on defect detection only
    # def _detect_wood_by_color(self, frame):
    # def _detect_wood_by_texture(self, frame):
    # def _detect_wood_by_shape(self, frame):
    # def detect_wood_triggered_by_ir(self, frame, ir_triggered=False):
