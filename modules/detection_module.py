print("=== DETECTION_MODULE LOADED ===")
import cv2
import numpy as np
from modules.utils_module import calculate_defect_size, map_model_output_to_standard
from modules.alignment_module import AlignmentModule, AlignmentResult, AlignmentStatus

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

        # Model paths for two-stage detection
        self.wood_model_path = "/home/inspectura/Desktop/wood_sorting_app/models/Wood_Plank--640x640_quant_hailort_hailo8_2/Wood_Plank--640x640_quant_hailort_hailo8_2/Wood_Plank--640x640_quant_hailort_hailo8_2.hef"
        self.defect_model_path = "/home/inspectura/Desktop/wood_sorting_app/models/Defect_Detection--640x640_quant_hailort_hailo8_1/Defect_Detection--640x640_quant_hailort_hailo8_1/Defect_Detection--640x640_quant_hailort_hailo8_1.hef"

        # ONNX model path
        self.onnx_wood_model_path = "models/Wood_Detection/320yoloe-11s-seg.onnx"

        # Model instances
        self.wood_model = None
        self.defect_model = None
        self.onnx_wood_session = None
        self.ultralytics_wood_model = None

        # Detection thresholds
        self.wood_confidence_threshold = 0.3  # Reasonable threshold for wood detection
        self.defect_confidence_threshold = 0.5

        # Initialize alignment module with proper config conversion
        if config is not None:
            # Convert SimpleConfig object to dictionary for AlignmentModule
            config_dict = {
                'alignment': {
                    'top_roi_margin_percent': getattr(config.alignment, 'top_roi_margin_percent', 0.15),
                    'bottom_roi_margin_percent': getattr(config.alignment, 'bottom_roi_margin_percent', 0.15),
                    'min_overlap_threshold': getattr(config.alignment, 'min_overlap_threshold', 0.6),
                    'alignment_tolerance_percent': getattr(config.alignment, 'alignment_tolerance_percent', 0.1),
                    'enable_alignment_visualization': getattr(config.alignment, 'enable_alignment_visualization', True),
                    'roi_display_color': getattr(config.alignment, 'roi_display_color', (255, 255, 0)),
                    'aligned_color': getattr(config.alignment, 'aligned_color', (0, 255, 0)),
                    'misaligned_color': getattr(config.alignment, 'misaligned_color', (0, 0, 255))
                }
            }
            self.alignment_module = AlignmentModule(config_dict)
        else:
            # Fallback to default config
            self.alignment_module = AlignmentModule({})

        # Load models in both dev and production modes
        self.load_models()

    def load_models(self):
        """Load both the wood detection and defect detection models."""
        print("DEBUG: load_models() called")

        # Load wood detection model using ultralytics (simplified approach)
        print(f"DEBUG: ULTRALYTICS_AVAILABLE = {ULTRALYTICS_AVAILABLE}")
        if ULTRALYTICS_AVAILABLE:
            print(f"DEBUG: Loading YOLO model with ultralytics from: {self.onnx_wood_model_path}")
            try:
                import os
                if os.path.exists(self.onnx_wood_model_path):
                    print(f"DEBUG: Model file exists at path: {self.onnx_wood_model_path}")
                    file_size = os.path.getsize(self.onnx_wood_model_path)
                    print(f"DEBUG: Model file size: {file_size} bytes")

                    from ultralytics import YOLO
                    print(f"DEBUG: Attempting to load YOLO model from: {self.onnx_wood_model_path}")
                    self.ultralytics_wood_model = YOLO(self.onnx_wood_model_path, task='segment')
                    print("Ultralytics YOLO wood detection model loaded successfully.")
    
                    # Test the model with a dummy frame to catch ONNX Runtime issues early
                    try:
                        import numpy as np
                        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                        test_results = self.ultralytics_wood_model.predict(dummy_frame, conf=0.5, verbose=False)
                        print("DEBUG: YOLO model test prediction successful")
                    except Exception as test_e:
                        print(f"ERROR: YOLO model test prediction failed: {test_e}")
                        print("DEBUG: Disabling YOLO model due to runtime issues")
                        self.ultralytics_wood_model = None
                else:
                    print(f"ERROR: Model file NOT found at path: {self.onnx_wood_model_path}")
                    self.ultralytics_wood_model = None
            except Exception as e:
                print(f"ERROR: Failed to load YOLO model with ultralytics: {e}")
                print(f"DEBUG: Exception type: {type(e).__name__}")
                import traceback
                print(f"DEBUG: Full traceback: {traceback.format_exc()}")
                self.ultralytics_wood_model = None
        else:
            print("ERROR: Ultralytics not available. Please install with: pip install ultralytics")
            self.ultralytics_wood_model = None

        # Load DeGirum models (fallback)
        if DEGIRUM_AVAILABLE:
            try:
                # Load wood detection model
                self.wood_model = dg.load_model(
                    model_name=self.wood_model_path,
                    inference_host_address=self.inference_host_address
                )
                print("DeGirum wood detection model loaded successfully.")

                # Load defect detection model
                self.defect_model = dg.load_model(
                    model_name=self.defect_model_path,
                    inference_host_address=self.inference_host_address
                )
                print("DeGirum defect detection model loaded successfully.")

            except Exception as e:
                print(f"Error loading DeGirum models: {e}")
                self.wood_model = None
                self.defect_model = None
        else:
            print("DeGirum not available, using ONNX models only.")

    def detect_wood_presence(self, frame):
        """
        Stage 1: Detect if wood is present in the frame using the wood detection model.
        Returns (wood_detected, confidence, wood_bbox)
        """
        print(f"DEBUG: Starting wood detection on frame shape: {frame.shape}")
        print(f"DEBUG: ultralytics_wood_model: {self.ultralytics_wood_model}")
        print(f"DEBUG: onnx_wood_session: {self.onnx_wood_session}")
        print(f"DEBUG: wood_model: {self.wood_model}")

        # Try ultralytics or ONNX model first (works in both dev and prod modes)
        if self.ultralytics_wood_model is not None or self.onnx_wood_session is not None:
            print("DEBUG: Trying YOLO/ONNX model first")
            try:
                result = self._detect_wood_with_onnx(frame)
                print(f"DEBUG: YOLO/ONNX result: {result}")
                return result
            except Exception as e:
                print(f"Error in YOLO/ONNX wood detection: {e}")
                print("DEBUG: Falling back due to YOLO/ONNX error")

        # Fallback to DeGirum model (works in both dev and prod modes)
        if self.wood_model is not None:
            print("DEBUG: Trying DeGirum model")
            try:
                result = self._detect_wood_with_degirum(frame)
                print(f"DEBUG: DeGirum result: {result}")
                return result
            except Exception as e:
                print(f"Error in DeGirum wood detection: {e}")
                print("DEBUG: Falling back due to DeGirum error")

        # Final fallback to basic detection
        print("DEBUG: No ML models available, using fallback detection")
        result = self._fallback_wood_detection(frame)
        print(f"DEBUG: Fallback detection result: {result}")
        return result

    def _detect_wood_with_onnx(self, frame):
        """Detect wood using YOLOE model (simplified ultralytics approach)"""
        if self.ultralytics_wood_model is not None:
            print("DEBUG: Using ultralytics YOLO model for detection")
            try:
                # Run YOLOE model on the frame (matching your test code)
                results = self.ultralytics_wood_model.predict(frame, conf=self.wood_confidence_threshold, verbose=False)

                if len(results) > 0 and len(results[0].boxes) > 0:
                    # Get the first result (assuming single image)
                    result = results[0]

                    # Find best wood detection
                    best_confidence = 0
                    best_bbox = None
                    best_class_id = 0

                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])

                        # Accept classes that might represent wood
                        wood_classes = list(range(20))  # Accept classes 0-19 for now
                        if confidence > best_confidence and confidence > self.wood_confidence_threshold and class_id in wood_classes:
                            best_confidence = confidence
                            best_class_id = class_id
                            # Convert tensor bbox to list [x1, y1, x2, y2]
                            bbox_tensor = box.xyxy[0]
                            best_bbox = [int(bbox_tensor[0]), int(bbox_tensor[1]), int(bbox_tensor[2]), int(bbox_tensor[3])]

                    if best_bbox:
                        print(f"YOLOE: Wood detected (class {best_class_id}) with confidence: {best_confidence:.3f}, bbox: {best_bbox}")
                        return True, best_confidence, best_bbox
                    else:
                        print("YOLOE: No valid wood detections found")
                        return False, 0.0, None
                else:
                    print("YOLOE: No detections found")
                    return False, 0.0, None

            except Exception as e:
                print(f"Error in YOLOE detection (likely ONNX Runtime issue): {e}")
                print("DEBUG: Disabling YOLO model and falling back to alternative detection methods")
                # Disable the model so future calls will use fallback
                self.ultralytics_wood_model = None
                return False, 0.0, None

        print("DEBUG: No YOLOE model available for detection")
        return False, 0.0, None

    def _detect_wood_with_degirum(self, frame):
        """Detect wood using DeGirum model (original implementation)"""
        # Run wood detection inference
        inference_result = self.wood_model(frame)
        detections = inference_result.results

        # Find the best wood detection
        best_wood_detection = None
        best_confidence = 0

        for det in detections:
            confidence = det.get('confidence', 0)
            if confidence > best_confidence and confidence > self.wood_confidence_threshold:
                best_confidence = confidence
                best_wood_detection = det

        if best_wood_detection:
            wood_bbox = best_wood_detection['bbox']
            print(f"DeGirum: Wood detected with confidence: {best_confidence:.3f}, bbox: {wood_bbox}")
            print(f"DEBUG: DeGirum detection successful, returning bbox: {wood_bbox}")
            return True, best_confidence, wood_bbox
        else:
            print("DeGirum: No wood detected")
            print("DEBUG: DeGirum returned no detections")
            return False, 0.0, None


    def _fallback_wood_detection(self, frame):
        """Fallback wood detection using basic computer vision"""
        try:
            print("DEBUG: Running fallback wood detection")
            # Use the existing color/texture/shape based detection
            wood_confidence_color = self._detect_wood_by_color(frame)
            wood_confidence_texture = self._detect_wood_by_texture(frame)
            wood_confidence_shape = self._detect_wood_by_shape(frame)

            combined_confidence = (
                0.4 * wood_confidence_color +
                0.3 * wood_confidence_texture +
                0.3 * wood_confidence_shape
            )

            # Lower threshold for fallback detection to ensure we get some detection
            is_wood = combined_confidence > 0.15

            # Estimate wood bbox for fallback (center region)
            h, w = frame.shape[:2]
            wood_bbox = [w//4, h//4, 3*w//4, 3*h//4] if is_wood else None

            if is_wood:
                print(f"Fallback: Wood detected with confidence: {combined_confidence:.3f}, bbox: {wood_bbox}")
            else:
                print("Fallback: No wood detected")

            return is_wood, combined_confidence, wood_bbox

        except Exception as e:
            print(f"Error in fallback wood detection: {e}")
            # Return a basic detection if all else fails
            h, w = frame.shape[:2]
            return True, 0.5, [w//4, h//4, 3*w//4, 3*h//4]
    def detect_defects_in_wood_region(self, frame, wood_bbox, camera_name="top"):
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
            
            for det in detections:
                confidence = det.get('confidence', 0)
                if confidence < self.defect_confidence_threshold:
                    continue
                
                model_label = det['label']
                standard_defect_type = map_model_output_to_standard(model_label)
                
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

    def analyze_frame(self, frame, camera_name="top"):
        """
        Main analysis function implementing three-stage detection:
        1. Detect wood presence
        2. Check wood alignment
        3. If wood detected and aligned, analyze for defects
        """
        # Stage 1: Wood detection
        wood_detected, wood_confidence, wood_bbox = self.detect_wood_presence(frame)
        # Store confidence for visualization
        self.last_wood_confidence = wood_confidence

        # Debug logging for bounding box
        print(f"DEBUG: analyze_frame - wood_detected: {wood_detected}, wood_bbox: {wood_bbox}")
        if wood_detected and wood_bbox:
            print(f"DEBUG: Wood detected with bbox: {wood_bbox}")
        elif wood_detected:
            print("DEBUG: Wood detected but no bbox returned")
        else:
            print("DEBUG: No wood detected")

        if not wood_detected:
            # No wood detected, return original frame with alignment overlay
            print("DEBUG: analyze_frame - No wood detected path")
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "No Wood Detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Add alignment overlay for reference
            alignment_result = self.alignment_module.check_wood_alignment(frame, None)
            annotated_frame = self.alignment_module.draw_alignment_overlay(annotated_frame, alignment_result)

            return annotated_frame, {}, [], alignment_result

        # Stage 2: Alignment checking
        alignment_result = self.alignment_module.check_wood_alignment(frame, wood_bbox)

        # Stage 3: Defect detection in wood region (only if aligned)
        if alignment_result.status == AlignmentStatus.ALIGNED:
            print("DEBUG: analyze_frame - Aligned path")
            annotated_frame, defect_dict, defect_measurements = self.detect_defects_in_wood_region(frame, wood_bbox, camera_name)
        else:
            # Wood detected but misaligned - still show wood region but skip defect detection
            print("DEBUG: analyze_frame - Misaligned path")
            annotated_frame = frame.copy()
            if wood_bbox:
                x1, y1, x2, y2 = wood_bbox
                # Enhanced visualization for misaligned wood - use RED for better visibility
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Thicker RED border
                # Add corner markers for misaligned wood
                corner_size = 25
                # Top-left corner
                cv2.line(annotated_frame, (x1, y1), (x1 + corner_size, y1), (0, 0, 255), 4)
                cv2.line(annotated_frame, (x1, y1), (x1, y1 + corner_size), (0, 0, 255), 4)
                # Top-right corner
                cv2.line(annotated_frame, (x2, y1), (x2 - corner_size, y1), (0, 0, 255), 4)
                cv2.line(annotated_frame, (x2, y1), (x2, y1 + corner_size), (0, 0, 255), 4)
                # Bottom-left corner
                cv2.line(annotated_frame, (x1, y2), (x1 + corner_size, y2), (0, 0, 255), 4)
                cv2.line(annotated_frame, (x1, y2), (x1, y2 - corner_size), (0, 0, 255), 4)
                # Bottom-right corner
                cv2.line(annotated_frame, (x2, y2), (x2 - corner_size, y2), (0, 0, 255), 4)
                cv2.line(annotated_frame, (x2, y2), (x2, y2 - corner_size), (0, 0, 255), 4)

                # Enhanced text for misaligned wood
                cv2.putText(annotated_frame, "WOOD MISALIGNED",
                            (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.putText(annotated_frame, "Adjust Position",
                            (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                print(f"DEBUG: Drew RED bounding box for misaligned wood: [{x1}, {y1}, {x2}, {y2}]")
            defect_dict = {}
            defect_measurements = []

        # Add alignment overlay
        annotated_frame = self.alignment_module.draw_alignment_overlay(annotated_frame, alignment_result)

        # Ensure wood bounding box is always drawn if wood was detected
        print(f"DEBUG: About to draw final bounding box. wood_detected={wood_detected}, wood_bbox={wood_bbox}")
        if wood_detected and wood_bbox:
            print(f"DEBUG: Drawing final bounding box for detected wood")
            # Draw wood bounding box regardless of alignment status
            x1, y1, x2, y2 = wood_bbox
            print(f"DEBUG: Bbox coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            print(f"DEBUG: Frame shape: {frame.shape}")

            # Validate bbox coordinates
            valid_coords = x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]
            print(f"DEBUG: Bbox coordinates valid: {valid_coords}")

            if valid_coords:
                if alignment_result.status == AlignmentStatus.ALIGNED:
                    # Green for aligned
                    box_color = (0, 255, 0)
                    label_text = "WOOD DETECTED"
                else:
                    # RED for misaligned (more visible)
                    box_color = (0, 0, 255)
                    label_text = "WOOD MISALIGNED"

                print(f"DEBUG: Drawing bbox with color {box_color} at coordinates ({x1},{y1}) to ({x2},{y2})")

                # Draw enhanced bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 5)
                print(f"DEBUG: Drew rectangle")

                # Add corner markers
                corner_size = 25
                cv2.line(annotated_frame, (x1, y1), (x1 + corner_size, y1), box_color, 4)
                cv2.line(annotated_frame, (x1, y1), (x1, y1 + corner_size), box_color, 4)
                cv2.line(annotated_frame, (x2, y1), (x2 - corner_size, y1), box_color, 4)
                cv2.line(annotated_frame, (x2, y1), (x2, y1 + corner_size), box_color, 4)
                cv2.line(annotated_frame, (x1, y2), (x1 + corner_size, y2), box_color, 4)
                cv2.line(annotated_frame, (x1, y2), (x1, y2 - corner_size), box_color, 4)
                cv2.line(annotated_frame, (x2, y2), (x2 - corner_size, y2), box_color, 4)
                cv2.line(annotated_frame, (x2, y2), (x2, y2 - corner_size), box_color, 4)
                print(f"DEBUG: Drew corner markers")

                # Add text label
                cv2.putText(annotated_frame, label_text, (x1 + 10, y1 + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, 3)
                print(f"DEBUG: Drew text label: {label_text}")

                # Add confidence if available
                if hasattr(self, 'last_wood_confidence'):
                    confidence_text = f"Conf: {self.last_wood_confidence:.2f}"
                    cv2.putText(annotated_frame, confidence_text, (x1 + 10, y1 + 55),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                    print(f"DEBUG: Drew confidence text: {confidence_text}")

                print(f"DEBUG: Successfully drew bbox and text on frame")
            else:
                print(f"DEBUG: Invalid bbox coordinates: {wood_bbox}, frame shape: {frame.shape}")
        else:
            print(f"DEBUG: Not drawing bbox - wood_detected={wood_detected}, wood_bbox={wood_bbox}")

        return annotated_frame, defect_dict, defect_measurements, alignment_result

    def detect_wood(self, frame):
        """
        Enhanced wood detection using the wood detection model.
        Falls back to visual detection if model is not available.
        Returns True if wood is detected, False otherwise.
        """
        wood_detected, confidence, _ = self.detect_wood_presence(frame)
        return wood_detected

    def _detect_wood_by_color(self, frame):
        """Detect wood using HSV color segmentation"""
        try:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define multiple wood color ranges to handle different wood types
            wood_ranges = [
                # Light wood (pine, birch)
                ([8, 50, 50], [25, 255, 255]),
                # Medium wood (oak, maple) 
                ([10, 40, 40], [20, 200, 200]),
                # Dark wood (walnut, mahogany)
                ([5, 30, 30], [15, 150, 180])
            ]
            
            combined_mask = None
            for lower, upper in wood_ranges:
                mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Clean up mask with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Calculate percentage of wood-like pixels
            wood_pixel_count = cv2.countNonZero(combined_mask)
            total_pixels = frame.shape[0] * frame.shape[1]
            wood_percentage = (wood_pixel_count / total_pixels) * 100
            
            # Return confidence (normalized to 0-1)
            return min(wood_percentage / 20.0, 1.0)  # 20% wood pixels = 100% confidence
            
        except Exception as e:
            print(f"Error in color-based wood detection: {e}")
            return 0.0

    def _detect_wood_by_texture(self, frame):
        """Detect wood using basic texture analysis"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Calculate texture using standard deviation in local neighborhoods
            kernel_size = 15
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            
            # Calculate local standard deviation (texture measure)
            mean = cv2.blur(blurred.astype(np.float32), (kernel_size, kernel_size))
            sqr_mean = cv2.blur((blurred.astype(np.float32))**2, (kernel_size, kernel_size))
            texture_variance = sqr_mean - mean**2
            texture_std = np.sqrt(np.maximum(texture_variance, 0))
            
            # Wood typically has moderate texture (not too smooth, not too rough)
            # Calculate confidence based on texture distribution
            texture_mean = np.mean(texture_std)
            texture_confidence = 0.0
            
            # Optimal texture range for wood (adjust based on testing)
            if 10 < texture_mean < 40:
                texture_confidence = 1.0 - abs(texture_mean - 25) / 15.0
            
            return max(0.0, min(1.0, texture_confidence))
            
        except Exception as e:
            print(f"Error in texture-based wood detection: {e}")
            return 0.0

    def _detect_wood_by_shape(self, frame):
        """Detect wood using contour and shape analysis"""
        try:
            # Convert to grayscale and apply edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0
            
            # Analyze largest contours for rectangular/wood-like shapes
            frame_area = frame.shape[0] * frame.shape[1]
            shape_confidence = 0.0
            
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                area = cv2.contourArea(contour)
                
                # Skip very small contours
                if area < frame_area * 0.05:
                    continue
                
                # Calculate contour properties
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                # Aspect ratio analysis
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                # Wood planks typically have certain aspect ratios
                # Adjust these ranges based on your conveyor setup
                if 0.3 < aspect_ratio < 5.0:  # Not too square, not too thin
                    # Calculate rectangularity (how close to rectangle)
                    rect_area = w * h
                    rectangularity = area / rect_area
                    
                    if rectangularity > 0.6:  # Reasonably rectangular
                        shape_confidence = max(shape_confidence, rectangularity)
            
            return min(1.0, shape_confidence)
            
        except Exception as e:
            print(f"Error in shape-based wood detection: {e}")
            return 0.0

    def detect_wood_triggered_by_ir(self, frame, ir_triggered=False):
        """
        Wood detection specifically for IR-triggered scenarios.
        This is called when the IR sensor detects an object.
        Uses the wood detection model for accurate classification.
        """
        if not ir_triggered:
            return False, "No IR trigger"
        
        wood_detected, confidence, wood_bbox = self.detect_wood_presence(frame)
        
        if wood_detected:
            return True, f"Wood confirmed with confidence: {confidence:.3f}"
        else:
            return False, "Non-wood object detected by IR sensor"
