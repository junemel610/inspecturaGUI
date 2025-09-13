import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)

class WoodDetectionResult:
    """Result container for wood detection operations"""
    def __init__(self, detected: bool, bbox: Optional[Tuple] = None,
                 confidence: float = 0.0, features: Dict = None):
        self.detected = detected
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.features = features or {}

class CannyEdgeDetector:
    """Canny edge detection optimized for wood boundary detection"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.lower_threshold = self.config.get('lower_threshold', 50)
        self.upper_threshold = self.config.get('upper_threshold', 150)
        self.aperture_size = self.config.get('aperture_size', 3)
        self.gaussian_blur_ksize = self.config.get('gaussian_blur_ksize', (5, 5))
        self.gaussian_blur_sigma = self.config.get('gaussian_blur_sigma', 0)

    def detect_edges(self, frame: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection optimized for wood detection"""
        try:
            if frame is None or frame.size == 0:
                logger.warning("Invalid frame provided to CannyEdgeDetector")
                return np.zeros((720, 1280), dtype=np.uint8)

            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, self.gaussian_blur_ksize, self.gaussian_blur_sigma)

            # Apply Canny edge detection
            edges = cv2.Canny(blurred, self.lower_threshold, self.upper_threshold,
                            apertureSize=self.aperture_size)

            return edges

        except Exception as e:
            logger.error(f"Error in Canny edge detection: {e}")
            return np.zeros_like(frame) if frame is not None else np.zeros((720, 1280), dtype=np.uint8)

    def update_thresholds(self, lower: int, upper: int):
        """Update Canny thresholds dynamically"""
        self.lower_threshold = max(0, min(lower, 255))
        self.upper_threshold = max(0, min(upper, 255))
        if self.lower_threshold >= self.upper_threshold:
            self.upper_threshold = min(self.lower_threshold + 50, 255)

class ColorRecognitionEngine:
    """HSV color recognition for wood type identification"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.color_ranges = self.config.get('color_ranges', self._get_default_color_ranges())
        self.hsv_ranges = self._convert_to_hsv_ranges()

    def _get_default_color_ranges(self) -> Dict:
        """Get default wood color ranges in RGB"""
        return {
            "wood_brown": [[60, 30, 10], [120, 80, 60]],
            "wood_red": [[80, 20, 20], [150, 70, 70]],
            "wood_yellow": [[100, 80, 40], [180, 140, 100]],
            "wood_gray": [[40, 40, 40], [100, 100, 100]]
        }

    def _convert_to_hsv_ranges(self) -> Dict:
        """Convert RGB color ranges to HSV"""
        hsv_ranges = {}
        for color_name, rgb_range in self.color_ranges.items():
            try:
                # Convert RGB to HSV
                lower_rgb = np.array(rgb_range[0], dtype=np.uint8)
                upper_rgb = np.array(rgb_range[1], dtype=np.uint8)

                # Convert to HSV
                lower_hsv = cv2.cvtColor(np.array([[lower_rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0][0]
                upper_hsv = cv2.cvtColor(np.array([[upper_rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0][0]

                hsv_ranges[color_name] = [lower_hsv, upper_hsv]
            except Exception as e:
                logger.warning(f"Failed to convert color range for {color_name}: {e}")
                # Fallback to default HSV ranges
                hsv_ranges[color_name] = [np.array([0, 50, 50]), np.array([30, 255, 255])]

        return hsv_ranges

    def recognize_wood_color(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
        """Analyze color features within masked region"""
        try:
            if frame is None or frame.size == 0:
                return {}

            # Convert to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            results = {}
            for color_name, (lower, upper) in self.hsv_ranges.items():
                try:
                    # Create color mask
                    color_mask = cv2.inRange(hsv, lower, upper)

                    # Apply region mask if provided
                    if mask is not None:
                        combined_mask = cv2.bitwise_and(color_mask, mask)
                    else:
                        combined_mask = color_mask

                    # Calculate color coverage
                    color_pixels = cv2.countNonZero(combined_mask)
                    total_pixels = cv2.countNonZero(mask) if mask is not None else (frame.shape[0] * frame.shape[1])

                    if total_pixels > 0:
                        coverage = color_pixels / total_pixels
                        results[color_name] = coverage
                    else:
                        results[color_name] = 0.0

                except Exception as e:
                    logger.warning(f"Error analyzing color {color_name}: {e}")
                    results[color_name] = 0.0

            return results

        except Exception as e:
            logger.error(f"Error in color recognition: {e}")
            return {}

    def get_dominant_color(self, color_results: Dict) -> str:
        """Get the dominant wood color from analysis results"""
        if not color_results:
            return "unknown"

        # Find color with highest coverage
        dominant_color = max(color_results.items(), key=lambda x: x[1])
        return dominant_color[0] if dominant_color[1] > 0.1 else "unknown"

class ContourAnalyzer:
    """Contour analysis and filtering for wood objects"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.min_area = self.config.get('min_contour_area', 1000)
        self.max_area = self.config.get('max_contour_area', 500000)
        self.approximation_method = cv2.CHAIN_APPROX_SIMPLE
        self.retrieval_mode = cv2.RETR_EXTERNAL

    def find_wood_contours(self, edges: np.ndarray) -> List[Dict]:
        """Find and filter contours for wood objects"""
        try:
            if edges is None or edges.size == 0:
                return []

            # Find contours
            contours, hierarchy = cv2.findContours(
                edges.copy(), self.retrieval_mode, self.approximation_method
            )

            wood_detections = []

            for contour in contours:
                try:
                    area = cv2.contourArea(contour)

                    # Filter by area
                    if not (self.min_area <= area <= self.max_area):
                        continue

                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    bbox = (x, y, x + w, y + h)

                    # Calculate additional features
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                    # Calculate aspect ratio
                    aspect_ratio = w / h if h > 0 else 0

                    # Calculate solidity (area / convex hull area)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0

                    detection = {
                        'bbox': bbox,
                        'area': area,
                        'perimeter': perimeter,
                        'circularity': circularity,
                        'aspect_ratio': aspect_ratio,
                        'solidity': solidity,
                        'contour': contour
                    }

                    wood_detections.append(detection)

                except Exception as e:
                    logger.warning(f"Error processing contour: {e}")
                    continue

            # Sort by area (largest first)
            wood_detections.sort(key=lambda x: x['area'], reverse=True)

            return wood_detections

        except Exception as e:
            logger.error(f"Error in contour analysis: {e}")
            return []

    def filter_contours_by_shape(self, detections: List[Dict],
                                min_circularity: float = 0.1,
                                max_aspect_ratio: float = 5.0) -> List[Dict]:
        """Filter contours based on shape characteristics"""
        filtered = []

        for detection in detections:
            try:
                circularity = detection.get('circularity', 0)
                aspect_ratio = detection.get('aspect_ratio', 0)
                solidity = detection.get('solidity', 0)

                # Apply shape filters
                if (circularity >= min_circularity and
                    aspect_ratio <= max_aspect_ratio and
                    solidity >= 0.7):  # Solidity threshold for wood-like shapes
                    filtered.append(detection)

            except Exception as e:
                logger.warning(f"Error filtering contour: {e}")
                continue

        return filtered

class WoodDetectionEngine:
    """Main wood detection pipeline combining edge detection, color recognition, and contour analysis"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)

        # Initialize components
        self.canny_detector = CannyEdgeDetector(self.config.get('canny', {}))
        self.color_recognizer = ColorRecognitionEngine(self.config.get('color', {}))
        self.contour_analyzer = ContourAnalyzer(self.config.get('contour', {}))

        # Performance tracking
        self.processing_times = []
        self.detection_history = []

    def detect_wood(self, frame: np.ndarray) -> List[WoodDetectionResult]:
        """Main wood detection pipeline"""
        start_time = time.time()

        try:
            if frame is None or frame.size == 0:
                logger.warning("Invalid frame provided to wood detection")
                return []

            # Step 1: Apply Canny edge detection
            edges = self.canny_detector.detect_edges(frame)

            # Step 2: Find contours
            contours = self.contour_analyzer.find_wood_contours(edges)

            # Step 3: Filter contours by shape
            filtered_contours = self.contour_analyzer.filter_contours_by_shape(contours)

            # Step 4: Analyze each potential wood region
            wood_detections = []
            for contour_data in filtered_contours:
                try:
                    # Extract region of interest
                    bbox = contour_data['bbox']
                    x1, y1, x2, y2 = bbox

                    # Create mask for the contour
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask, [contour_data['contour']], -1, 255, -1)

                    # Analyze color within the masked region
                    color_results = self.color_recognizer.recognize_wood_color(frame, mask)

                    # Calculate confidence score
                    confidence = self._calculate_confidence(contour_data, color_results)

                    # Create detection result
                    if confidence >= self.confidence_threshold:
                        result = WoodDetectionResult(
                            detected=True,
                            bbox=bbox,
                            confidence=confidence,
                            features={
                                'contour_data': contour_data,
                                'color_analysis': color_results,
                                'dominant_color': self.color_recognizer.get_dominant_color(color_results)
                            }
                        )
                        wood_detections.append(result)

                except Exception as e:
                    logger.warning(f"Error processing contour: {e}")
                    continue

            # Track performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times = self.processing_times[-100:]

            # Track detection history
            self.detection_history.append({
                'timestamp': time.time(),
                'detections': len(wood_detections),
                'processing_time': processing_time
            })
            if len(self.detection_history) > 50:
                self.detection_history = self.detection_history[-50:]

            return wood_detections

        except Exception as e:
            logger.error(f"Error in wood detection pipeline: {e}")
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            return []

    def _calculate_confidence(self, contour_data: Dict, color_results: Dict) -> float:
        """Calculate confidence score for wood detection"""
        try:
            confidence = 0.0

            # Shape-based confidence (40% weight)
            area = contour_data.get('area', 0)
            solidity = contour_data.get('solidity', 0)
            circularity = contour_data.get('circularity', 0)
            aspect_ratio = contour_data.get('aspect_ratio', 1)

            # Ideal wood shape characteristics
            shape_score = 0
            if 5000 <= area <= 200000:  # Reasonable size range
                shape_score += 0.3
            if solidity >= 0.8:  # High solidity for wood
                shape_score += 0.3
            if 0.1 <= circularity <= 0.8:  # Not too circular, not too elongated
                shape_score += 0.2
            if 0.5 <= aspect_ratio <= 3.0:  # Reasonable aspect ratio
                shape_score += 0.2

            confidence += shape_score * 0.4

            # Color-based confidence (60% weight)
            if color_results:
                total_color_coverage = sum(color_results.values())
                max_color_coverage = max(color_results.values()) if color_results else 0

                # High coverage of wood-like colors
                if total_color_coverage >= 0.3:
                    confidence += 0.4
                elif total_color_coverage >= 0.1:
                    confidence += 0.2

                # Strong presence of dominant wood color
                if max_color_coverage >= 0.2:
                    confidence += 0.2

            return min(confidence, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.0

    def get_best_detection(self, frame: np.ndarray) -> Optional[WoodDetectionResult]:
        """Get the best wood detection from the frame"""
        detections = self.detect_wood(frame)
        if not detections:
            return None

        # Return detection with highest confidence
        return max(detections, key=lambda x: x.confidence)

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.processing_times:
            return {}

        return {
            'avg_processing_time': sum(self.processing_times) / len(self.processing_times),
            'min_processing_time': min(self.processing_times),
            'max_processing_time': max(self.processing_times),
            'total_detections': len(self.detection_history),
            'avg_detections_per_frame': sum(h['detections'] for h in self.detection_history) / len(self.detection_history) if self.detection_history else 0
        }

    def update_config(self, new_config: Dict):
        """Update configuration and reinitialize components"""
        self.config.update(new_config)

        # Update component configs
        if 'canny' in new_config:
            self.canny_detector = CannyEdgeDetector(self.config.get('canny', {}))
        if 'color' in new_config:
            self.color_recognizer = ColorRecognitionEngine(self.config.get('color', {}))
        if 'contour' in new_config:
            self.contour_analyzer = ContourAnalyzer(self.config.get('contour', {}))

        if 'confidence_threshold' in new_config:
            self.confidence_threshold = new_config['confidence_threshold']

# Integration methods for existing systems
def integrate_with_camera_module(camera_module, wood_detector: WoodDetectionEngine):
    """Integration method for camera module"""
    def detect_wood_in_frame(camera_name: str) -> Optional[WoodDetectionResult]:
        """Detect wood in a frame from specified camera"""
        try:
            success, frame = camera_module.read_frame(camera_name)
            if not success or frame is None:
                return None

            detections = wood_detector.detect_wood(frame)
            return wood_detector.get_best_detection(frame) if detections else None

        except Exception as e:
            logger.error(f"Error detecting wood in camera {camera_name}: {e}")
            return None

    # Add method to camera module
    camera_module.detect_wood_in_frame = detect_wood_in_frame
    return camera_module

def integrate_with_detection_module(detection_module, wood_detector: WoodDetectionEngine):
    """Integration method for detection module"""
    def enhanced_analyze_frame(frame, camera_name="top"):
        """Enhanced frame analysis with wood detection"""
        try:
            # First detect wood
            wood_detection = wood_detector.get_best_detection(frame)

            if wood_detection and wood_detection.detected:
                # Use existing defect detection within wood region
                x1, y1, x2, y2 = wood_detection.bbox
                wood_region = frame[y1:y2, x1:x2]

                # Get defect detection results
                annotated_region, defect_dict, defect_measurements = detection_module.detect_defects_in_full_frame(wood_region, camera_name)

                # Adjust coordinates back to full frame
                adjusted_defects = {}
                for defect_type, count in defect_dict.items():
                    adjusted_defects[defect_type] = count

                # Create full frame annotation
                annotated_frame = frame.copy()
                annotated_frame[y1:y2, x1:x2] = annotated_region

                # Draw wood bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                return annotated_frame, adjusted_defects, defect_measurements
            else:
                # No wood detected, use full frame detection
                return detection_module.detect_defects_in_full_frame(frame, camera_name)

        except Exception as e:
            logger.error(f"Error in enhanced frame analysis: {e}")
            return detection_module.detect_defects_in_full_frame(frame, camera_name)

    # Add method to detection module
    detection_module.enhanced_analyze_frame = enhanced_analyze_frame
    return detection_module

# Default configuration
DEFAULT_CONFIG = {
    'canny': {
        'lower_threshold': 50,
        'upper_threshold': 150,
        'aperture_size': 3,
        'gaussian_blur_ksize': (5, 5),
        'gaussian_blur_sigma': 0
    },
    'color': {
        'color_ranges': {
            "wood_brown": [[60, 30, 10], [120, 80, 60]],
            "wood_red": [[80, 20, 20], [150, 70, 70]],
            "wood_yellow": [[100, 80, 40], [180, 140, 100]],
            "wood_gray": [[40, 40, 40], [100, 100, 100]]
        }
    },
    'contour': {
        'min_contour_area': 1000,
        'max_contour_area': 500000
    },
    'confidence_threshold': 0.6
}