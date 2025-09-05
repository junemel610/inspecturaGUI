import cv2
import numpy as np
from modules.utils_module import calculate_defect_size, map_model_output_to_standard

try:
    import degirum as dg
    import degirum_tools
    DEGIRUM_AVAILABLE = True
except ImportError:
    print("Warning: 'degirum' module not found. Running in mock mode for detection.")
    DEGIRUM_AVAILABLE = False

class DetectionModule:
    def __init__(self, dev_mode=False, inference_host_address="@local"):
        self.dev_mode = dev_mode
        self.inference_host_address = inference_host_address
        
        # Model paths for two-stage detection
        self.wood_model_path = "/home/inspectura/Desktop/wood_sorting_app/models/Wood_Plank--640x640_quant_hailort_hailo8_2/Wood_Plank--640x640_quant_hailort_hailo8_2/Wood_Plank--640x640_quant_hailort_hailo8_2.hef"
        self.defect_model_path = "/home/inspectura/Desktop/wood_sorting_app/models/Defect_Detection--640x640_quant_hailort_hailo8_1/Defect_Detection--640x640_quant_hailort_hailo8_1/Defect_Detection--640x640_quant_hailort_hailo8_1.hef"
        
        # Model instances
        self.wood_model = None
        self.defect_model = None
        
        # Detection thresholds
        self.wood_confidence_threshold = 0.6
        self.defect_confidence_threshold = 0.7
        
        if not self.dev_mode and DEGIRUM_AVAILABLE:
            self.load_models()
        elif self.dev_mode:
            print("DetectionModule: Running in development mode, DeGirum models will be mocked.")

    def load_models(self):
        """Load both the wood detection and defect detection models."""
        if not DEGIRUM_AVAILABLE:
            print("DeGirum not available, cannot load models.")
            return
        
        try:
            # Load wood detection model
            self.wood_model = dg.load_model(
                model_name=self.wood_model_path,
                inference_host_address=self.inference_host_address
            )
            print("Wood detection model loaded successfully.")
            
            # Load defect detection model
            self.defect_model = dg.load_model(
                model_name=self.defect_model_path,
                inference_host_address=self.inference_host_address
            )
            print("Defect detection model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading DeGirum models: {e}")
            self.wood_model = None
            self.defect_model = None

    def detect_wood_presence(self, frame):
        """
        Stage 1: Detect if wood is present in the frame using the wood detection model.
        Returns (wood_detected, confidence, wood_bbox)
        """
        if self.dev_mode:
            # Mock wood detection for development
            return True, 0.9, [100, 100, 500, 400]
        
        if self.wood_model is None:
            print("Wood detection model not available, falling back to basic detection")
            return self._fallback_wood_detection(frame)
        
        try:
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
                print(f"Wood detected with confidence: {best_confidence:.3f}")
                return True, best_confidence, wood_bbox
            else:
                print("No wood detected by model")
                return False, 0.0, None
                
        except Exception as e:
            print(f"Error in wood detection: {e}")
            return self._fallback_wood_detection(frame)
    
    def _fallback_wood_detection(self, frame):
        """Fallback wood detection using basic computer vision"""
        try:
            # Use the existing color/texture/shape based detection
            wood_confidence_color = self._detect_wood_by_color(frame)
            wood_confidence_texture = self._detect_wood_by_texture(frame)
            wood_confidence_shape = self._detect_wood_by_shape(frame)
            
            combined_confidence = (
                0.4 * wood_confidence_color +
                0.3 * wood_confidence_texture +
                0.3 * wood_confidence_shape
            )
            
            is_wood = combined_confidence > 0.3
            
            # Estimate wood bbox for fallback (center region)
            h, w = frame.shape[:2]
            wood_bbox = [w//4, h//4, 3*w//4, 3*h//4] if is_wood else None
            
            return is_wood, combined_confidence, wood_bbox
            
        except Exception as e:
            print(f"Error in fallback wood detection: {e}")
            return False, 0.0, None
    def detect_defects_in_wood_region(self, frame, wood_bbox, camera_name="top"):
        """
        Stage 2: Detect defects within the identified wood region.
        Returns (annotated_frame, defect_dict, defect_measurements)
        """
        if self.dev_mode:
            # Mock defect detection for development
            h, w, _ = frame.shape
            annotated_frame = frame.copy()
            if wood_bbox:
                x1, y1, x2, y2 = wood_bbox
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, "Wood Region", (x1 + 10, y1 + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add mock defect
                defect_x = x1 + (x2 - x1) // 3
                defect_y = y1 + (y2 - y1) // 3
                cv2.rectangle(annotated_frame, (defect_x, defect_y), (defect_x + 50, defect_y + 30), (0, 0, 255), 2)
                cv2.putText(annotated_frame, "Mock Defect", (defect_x + 5, defect_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            return annotated_frame, {"Unsound_Knot": 1}, [("Unsound_Knot", 15.0, 8.0)]

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
                # Draw wood boundary
                cv2.rectangle(annotated_frame, (wood_bbox[0], wood_bbox[1]), 
                             (wood_bbox[2], wood_bbox[3]), (0, 255, 0), 2)
                cv2.putText(annotated_frame, "Wood Region", 
                           (wood_bbox[0] + 10, wood_bbox[1] + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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
        Main analysis function implementing two-stage detection:
        1. Detect wood presence
        2. If wood detected, analyze for defects
        """
        # Stage 1: Wood detection
        wood_detected, wood_confidence, wood_bbox = self.detect_wood_presence(frame)
        
        if not wood_detected:
            # No wood detected, return original frame
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "No Wood Detected", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return annotated_frame, {}, []
        
        # Stage 2: Defect detection in wood region
        return self.detect_defects_in_wood_region(frame, wood_bbox, camera_name)

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
