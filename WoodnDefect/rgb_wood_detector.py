#!/usr/bin/env python3
"""
Color-based Wood Detection with 4-Point Rectangle Detection
This module provides robust wood detection using:
1. RGB color analysis for wood tones
2. Contour detection for rectangular shapes
3. Automatic ROI generation
4. Adaptive thresholding
"""

import cv2
import numpy as np
import json
import subprocess
import time
import tkinter as tk
from typing import List, Tuple, Dict, Optional

# Global wood pallet width storage - single variable for current wood piece
WOOD_PALLET_WIDTH_MM = 0  # Global variable for current detected wood width

# Camera-specific calibration constants (matching testIR.py)
TOP_CAMERA_DISTANCE_CM = 28
BOTTOM_CAMERA_DISTANCE_CM = 27.5
TOP_CAMERA_PIXEL_TO_MM = 3.5  # Top camera: 2.96 pixels per mm
BOTTOM_CAMERA_PIXEL_TO_MM = 3.18  # Bottom camera: 3.18 pixels per mm

class ColorWoodDetector:
    def __init__(self):
        self.wood_color_profiles = {
            'top_panel': {
                'rgb_lower': np.array([160, 160, 160]),  # BGR
                'rgb_upper': np.array([225, 220, 210]),
                'name': 'Top Panel Wood'
            },
            'bottom_panel': {
                'rgb_lower': np.array([70, 70, 85]),  # BGR
                'rgb_upper': np.array([225, 220, 210]),
                'name': 'Bottom Panel Wood'
            }
        }
        
        # Detection parameters
        self.min_contour_area = 1000      # Increased for more reliable detection with tighter RGB ranges
        self.max_contour_area = 500000    # Slightly reduced for typical wood plank sizes
        self.min_aspect_ratio = 1.0       # Tightened for more rectangular wood shapes
        self.max_aspect_ratio = 10.0      # Reduced for more typical plank proportions
        self.contour_approximation = 0.025 # Slightly tighter for better shape approximation
        
        # Morphological operations
        self.morph_kernel_size = 11
        self.closing_iterations = 3
        self.opening_iterations = 2

        # Pixel to mm conversion parameters for width measurement
        self.pixel_per_mm_top = 2.96     # Placeholder: calibrate based on top camera distance (31cm)
        self.pixel_per_mm_bottom = 3.18  # Placeholder: calibrate based on bottom camera distance
        
        # Dynamic wood width storage - matches testIR.py functionality
        self.detected_wood_width_mm = {'top': 0, 'bottom': 0}
        self.wood_detection_results = {'top': None, 'bottom': None}
        self.dynamic_roi = {'top': None, 'bottom': None}

    def calculate_width_mm(self, bbox_pixels: int, camera: str = 'top') -> float:
        """Calculate width in mm from bounding box dimension in pixels using pixel_per_mm factors"""
        if camera == 'top':
            return bbox_pixels / self.pixel_per_mm_top
        elif camera == 'bottom':
            return bbox_pixels / self.pixel_per_mm_bottom
        else:
            raise ValueError("Camera must be 'top' or 'bottom'")

    def calibrate_pixel_to_mm(self, reference_object_width_px, reference_object_width_mm, camera_name="top"):
        """Calibrate the pixel-to-millimeter conversion factor for specific camera"""
        global TOP_CAMERA_PIXEL_TO_MM, BOTTOM_CAMERA_PIXEL_TO_MM
        
        conversion_factor = reference_object_width_mm / reference_object_width_px
        
        if camera_name == "top":
            TOP_CAMERA_PIXEL_TO_MM = conversion_factor
            self.pixel_per_mm_top = conversion_factor
            print(f"Calibrated TOP camera pixel-to-mm factor: {TOP_CAMERA_PIXEL_TO_MM}")
        else:  # bottom camera
            BOTTOM_CAMERA_PIXEL_TO_MM = conversion_factor
            self.pixel_per_mm_bottom = conversion_factor
            print(f"Calibrated BOTTOM camera pixel-to-mm factor: {BOTTOM_CAMERA_PIXEL_TO_MM}")
        
        return conversion_factor

    def calibrate_with_wood_pallet(self, wood_pallet_width_px_top, wood_pallet_width_px_bottom):
        """Auto-calibrate both cameras using the known wood pallet width"""
        global WOOD_PALLET_WIDTH_MM
        
        print(f"Auto-calibrating cameras with {WOOD_PALLET_WIDTH_MM}mm wood pallet...")

        top_factor = self.calibrate_pixel_to_mm(wood_pallet_width_px_top, WOOD_PALLET_WIDTH_MM, "top")
        bottom_factor = self.calibrate_pixel_to_mm(wood_pallet_width_px_bottom, WOOD_PALLET_WIDTH_MM, "bottom")
        
        print(f"Calibration complete:")
        print(f"  Top camera (28cm): {top_factor:.4f} mm/pixel")
        print(f"  Bottom camera (27.5cm): {bottom_factor:.4f} mm/pixel")
        
        return top_factor, bottom_factor

    def update_wood_width_dynamic(self, camera_name: str, wood_candidates: List[Dict]) -> float:
        """Update global wood width based on detected wood dimensions - matches testIR.py algorithm"""
        global WOOD_PALLET_WIDTH_MM
        
        if wood_candidates:
            candidate = wood_candidates[0]  # Use best candidate
            x, y, w, h = candidate['bbox']
            detected_width_mm = self.calculate_width_mm(h, camera_name)  # Use height (cross-section)
            
            # Update global wood height variable dynamically
            WOOD_PALLET_WIDTH_MM = detected_width_mm
            self.detected_wood_width_mm[camera_name] = detected_width_mm
            print(f"🎯 Dynamic wood height updated: {detected_width_mm:.1f}mm (from bbox {w}x{h}px, camera: {camera_name})")
            
            return detected_width_mm
        
        return 0.0

    def calculate_defect_size(self, detection_box, camera_name="top"):
        """Calculate defect size in mm and percentage from detection bounding box - matches testIR.py"""
        global WOOD_PALLET_WIDTH_MM, TOP_CAMERA_PIXEL_TO_MM, BOTTOM_CAMERA_PIXEL_TO_MM
        
        try:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = detection_box['bbox']

            # Calculate defect dimensions in pixels
            width_px = abs(x2 - x1)   # Horizontal dimension (across wood width)
            height_px = abs(y2 - y1) # Vertical dimension (along wood length)

            # For wood width measurement, use the horizontal dimension (width_px)
            # This matches rgb_wood_detector.py which uses bbox width (w) for width calculation
            defect_size_px = width_px

            # Use camera-specific conversion factor
            if camera_name == "top":
                pixel_to_mm = TOP_CAMERA_PIXEL_TO_MM
            else:  # bottom camera
                pixel_to_mm = BOTTOM_CAMERA_PIXEL_TO_MM

            # Prevent division by zero
            if pixel_to_mm <= 0:
                pixel_to_mm = 2.96 if camera_name == "top" else 3.18
                print(f"Warning: pixel_to_mm was zero, using default {pixel_to_mm}")

            # Convert to millimeters using division (pixels per mm factor)
            size_mm = defect_size_px / pixel_to_mm

            # Calculate percentage of actual wood pallet width
            if WOOD_PALLET_WIDTH_MM > 0:
                percentage = (size_mm / WOOD_PALLET_WIDTH_MM) * 100
            else:
                percentage = 0.0  # Avoid division by zero

            # Debug logging to understand bounding box sizes
            print(f"DEBUG [{camera_name}]: bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                  f"-> width_px={width_px:.1f}, height_px={height_px:.1f} "
                  f"-> defect_size_px={defect_size_px:.1f} -> size_mm={size_mm:.1f}")

            return size_mm, percentage

        except Exception as e:
            print(f"Error calculating defect size: {e}")
            # Return conservative values if calculation fails
            return 50.0, 35.0  # Assumes large defect for safety

    def analyze_image_colors(self, image_path: str) -> Dict:
        """Analyze the color composition of the captured image"""
        print(f"🎨 Analyzing colors in: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        analysis = {
            "image_size": f"{w}x{h}",
            "wood_profiles_detected": {},
            "dominant_colors": {},
            "recommendations": []
        }

        # Test each wood color profile
        for profile_name, profile in self.wood_color_profiles.items():
            mask = cv2.inRange(rgb, profile['rgb_lower'], profile['rgb_upper'])
            pixels_detected = cv2.countNonZero(mask)
            percentage = (pixels_detected / (h * w)) * 100

            analysis["wood_profiles_detected"][profile_name] = {
                "pixels": pixels_detected,
                "percentage": round(percentage, 2),
                "detected": percentage > 1.0  # Consider detected if >1% of image
            }

            if percentage > 1.0:
                print(f"  ✅ {profile['name']}: {percentage:.1f}% of image")
            else:
                print(f"  ❌ {profile['name']}: {percentage:.1f}% of image")

        # Find dominant colors in RGB
        rgb_flat = rgb.reshape(-1, 3)
        r_values = rgb_flat[:, 0]
        g_values = rgb_flat[:, 1]
        b_values = rgb_flat[:, 2]

        analysis["dominant_colors"] = {
            "red_mean": int(np.mean(r_values)),
            "red_std": int(np.std(r_values)),
            "green_mean": int(np.mean(g_values)),
            "blue_mean": int(np.mean(b_values))
        }
        
        # Generate recommendations
        best_profiles = []
        for name, data in analysis["wood_profiles_detected"].items():
            if data["detected"] and data["percentage"] > 5:
                best_profiles.append((name, data["percentage"]))
        
        if best_profiles:
            best_profiles.sort(key=lambda x: x[1], reverse=True)
            analysis["recommendations"].append(f"Use {best_profiles[0][0]} profile as primary detection method")
        else:
            analysis["recommendations"].append("Consider creating custom color profile for this wood type")
            analysis["recommendations"].append(f"Dominant RGB: R={analysis['dominant_colors']['red_mean']}, G={analysis['dominant_colors']['green_mean']}, B={analysis['dominant_colors']['blue_mean']}")
        
        return analysis
    
    def detect_document_style_edges(self, image: np.ndarray) -> np.ndarray:
        """Detect edges like a document scanner - find rectangular boundaries"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection with wider thresholds for better edge detection
        edges = cv2.Canny(blurred, 75, 200)

        # Dilate edges to make them more visible and connect broken segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)

        # Find contours in the edge image
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask from significant contours (like document scanning)
        edge_mask = np.zeros_like(edges)

        for contour in contours:
            area = cv2.contourArea(contour)
            # Only keep contours that are large enough to be potential wood boundaries
            if area > 1000:  # Minimum area threshold
                # Draw filled contour to create mask
                cv2.drawContours(edge_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Apply morphological operations to clean up the edge mask
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel_clean, iterations=2)
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, kernel_clean, iterations=1)

        return edge_mask

    def detect_wood_by_color(self, image: np.ndarray, profile_names: List[str] = None) -> Tuple[np.ndarray, List[Dict]]:
        """Detect wood using color-first approach with edge enhancement"""
        try:
            if profile_names is None:
                profile_names = list(self.wood_color_profiles.keys())

            # Validate input image
            if image is None or image.size == 0:
                print("❌ Error: Invalid input image for color detection")
                return np.zeros((100, 100), dtype=np.uint8), []

            # Step 1: Apply histogram equalization on V channel for better lighting compensation
            hsv_temp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_temp)
            v = cv2.equalizeHist(v)
            hsv_temp = cv2.merge([h, s, v])
            rgb = cv2.cvtColor(hsv_temp, cv2.COLOR_HSV2BGR)

            combined_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            detections = []

            print(f"🎨 Using profiles: {profile_names}")

            # Combine masks from selected profiles
            for profile_name in profile_names:
                if profile_name in self.wood_color_profiles:
                    profile = self.wood_color_profiles[profile_name]
                    mask = cv2.inRange(rgb, profile['rgb_lower'], profile['rgb_upper'])
                    mask_pixels = cv2.countNonZero(mask)
                    total_pixels = rgb.shape[0] * rgb.shape[1]
                    mask_percentage = (mask_pixels / total_pixels) * 100
                    print(f"  📊 {profile_name}: RGB range {profile['rgb_lower']} - {profile['rgb_upper']}, mask {mask_pixels} pixels ({mask_percentage:.1f}%)")
                    combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Step 2: Apply edge detection within the color mask to find wood boundaries
            # Convert color mask to find edges only within wood-colored regions
            color_mask_blurred = cv2.GaussianBlur(combined_mask, (5, 5), 0)
            color_edges = cv2.Canny(color_mask_blurred, 100, 200)

            # Dilate the edges to make them more visible in the mask
            kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            color_edges_dilated = cv2.dilate(color_edges, kernel_edge, iterations=1)

            # Combine the original color mask with edge information
            # This preserves the wood color regions but enhances boundaries
            enhanced_mask = cv2.bitwise_or(combined_mask, color_edges_dilated)

            edge_enhanced_pixels = cv2.countNonZero(enhanced_mask)
            edge_enhanced_percentage = (edge_enhanced_pixels / total_pixels) * 100
            print(f"🎨🔍 Color + Edge enhanced mask: {edge_enhanced_pixels} pixels ({edge_enhanced_percentage:.1f}%)")

            pre_morph_pixels = cv2.countNonZero(enhanced_mask)
            pre_morph_percentage = (pre_morph_pixels / total_pixels) * 100
            print(f"🔧 Pre-morph enhanced mask: {pre_morph_pixels} pixels ({pre_morph_percentage:.1f}%)")

            # Clean up mask with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
            enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, kernel, iterations=self.closing_iterations)
            enhanced_mask = cv2.dilate(enhanced_mask, kernel, iterations=1)
            enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_OPEN, kernel, iterations=self.opening_iterations)

            post_morph_pixels = cv2.countNonZero(enhanced_mask)
            post_morph_percentage = (post_morph_pixels / total_pixels) * 100
            print(f"🔧 Post-morph enhanced mask: {post_morph_pixels} pixels ({post_morph_percentage:.1f}%)")

            # Additional logging for dominant colors
            rgb_flat = rgb.reshape(-1, 3)
            r_values = rgb_flat[:, 0]
            g_values = rgb_flat[:, 1]
            b_values = rgb_flat[:, 2]
            print(f"🎨 Dominant RGB in image: R={int(np.mean(r_values)):.0f}±{int(np.std(r_values)):.0f}, G={int(np.mean(g_values)):.0f}, B={int(np.mean(b_values)):.0f}")

            return enhanced_mask, detections
            
        except Exception as e:
            print(f"❌ Error in color detection: {e}")
            # Return empty mask and detections on error
            return np.zeros(image.shape[:2], dtype=np.uint8), []

    def update_rgb_ranges_based_on_dominant_colors(self, rgb):
        """Dynamically adjust RGB ranges based on dominant colors in the image"""
        rgb_flat = rgb.reshape(-1, 3)
        r_mean = int(np.mean(rgb_flat[:, 0]))
        g_mean = int(np.mean(rgb_flat[:, 1]))
        b_mean = int(np.mean(rgb_flat[:, 2]))

        # Update profiles based on dominant colors
        self.wood_color_profiles['top_panel']['rgb_lower'] = np.array([max(0, r_mean - 30), max(0, g_mean - 30), max(0, b_mean - 30)])
        self.wood_color_profiles['top_panel']['rgb_upper'] = np.array([min(255, r_mean + 30), min(255, g_mean + 30), min(255, b_mean + 30)])
        self.wood_color_profiles['bottom_panel']['rgb_lower'] = np.array([max(0, r_mean - 30), max(0, g_mean - 30), max(0, b_mean - 30)])
        self.wood_color_profiles['bottom_panel']['rgb_upper'] = np.array([min(255, r_mean + 30), min(255, g_mean + 30), min(255, b_mean + 30)])
        print(f"🔧 Dynamically updated RGB ranges: R=[{r_mean-30}-{r_mean+30}], G=[{g_mean-30}-{g_mean+30}], B=[{b_mean-30}-{b_mean+30}]")
    
    def detect_rectangular_contours(self, mask: np.ndarray, camera: str = 'top') -> List[Dict]:
        """Detect rectangular contours that could be wood planks - focusing on center area"""
        try:
            if mask is None or mask.size == 0:
                print("❌ Error: Invalid mask for contour detection")
                return []
                
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"📐 Found {len(contours)} total contours")

            # Get mask dimensions for center focus
            mask_height, mask_width = mask.shape
            center_x, center_y = mask_width // 2, mask_height // 2
            
            # Define center region (middle 60% of the image)
            center_margin_x = int(mask_width * 0.2)  # 20% margin on each side
            center_margin_y = int(mask_height * 0.2)  # 20% margin on top/bottom
            center_region = {
                'x_min': center_margin_x,
                'x_max': mask_width - center_margin_x,
                'y_min': center_margin_y,
                'y_max': mask_height - center_margin_y
            }
            
            print(f"🎯 Center focus region: x=[{center_region['x_min']}-{center_region['x_max']}], y=[{center_region['y_min']}-{center_region['y_max']}]")

            wood_candidates = []
            rejected_area = 0
            rejected_aspect = 0
            rejected_center = 0

            for i, contour in enumerate(contours):
                try:
                    area = cv2.contourArea(contour)

                    # Filter by area
                    if area < self.min_contour_area or area > self.max_contour_area:
                        rejected_area += 1
                        print(f"  ❌ Contour {i}: area {area:.0f} out of range [{self.min_contour_area}, {self.max_contour_area}]")
                        continue

                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if contour center is in the center region
                    contour_center_x = x + w // 2
                    contour_center_y = y + h // 2
                    
                    if not (center_region['x_min'] <= contour_center_x <= center_region['x_max'] and 
                            center_region['y_min'] <= contour_center_y <= center_region['y_max']):
                        rejected_center += 1
                        print(f"  ❌ Contour {i}: center ({contour_center_x}, {contour_center_y}) outside focus region")
                        continue

                    # Filter by minimum size to prevent small detections
                    if camera == 'top':
                        min_height = 266
                        min_width = 100
                    elif camera == 'bottom':
                        min_height = 286
                        min_width = 100
                    else:
                        min_height = 100
                        min_width = 100

                    if h < min_height or w < min_width:
                        rejected_area += 1
                        print(f"  ❌ Contour {i}: size {w}x{h} too small for {camera} camera (min {min_width}x{min_height})")
                        continue

                    aspect_ratio = max(w, h) / min(w, h)

                    # Filter by aspect ratio (wood planks are typically rectangular)
                    if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                        rejected_aspect += 1
                        print(f"  ❌ Contour {i}: aspect {aspect_ratio:.2f} out of range [{self.min_aspect_ratio}, {self.max_aspect_ratio}]")
                        continue

                    # Approximate contour to polygon
                    epsilon = self.contour_approximation * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    # Calculate additional metrics
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0

                    # Get rotated rectangle for better angle detection
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)

                    confidence = self._calculate_wood_confidence(area, aspect_ratio, solidity, len(approx))

                    wood_candidate = {
                        'contour': contour,
                        'approx_points': approx,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'solidity': solidity,
                        'vertices': len(approx),
                        'rotated_rect': rect,
                        'corner_points': box,
                        'confidence': confidence
                    }

                    wood_candidates.append(wood_candidate)
                    print(f"  ✅ Contour {i}: area {area:.0f}, aspect {aspect_ratio:.2f}, solidity {solidity:.2f}, confidence {confidence:.2f}")
                        
                except Exception as contour_error:
                    print(f"  ❌ Error processing contour {i}: {contour_error}")
                    continue

            print(f"📊 Contour filtering: {len(contours)} total, {rejected_area} rejected by area, {rejected_aspect} by aspect, {rejected_center} rejected by center, {len(wood_candidates)} candidates")

            # Sort by confidence
            wood_candidates.sort(key=lambda x: x['confidence'], reverse=True)

            return wood_candidates
            
        except Exception as e:
            print(f"❌ Error in rectangular contour detection: {e}")
            return []
    
    def _calculate_wood_confidence(self, area: float, aspect_ratio: float, solidity: float, vertices: int) -> float:
        """Calculate confidence score for wood detection"""
        confidence = 0.0
        
        # Area score (larger is better, up to a point)
        if 10000 <= area <= 100000:
            confidence += 0.3
        elif area > 5000:
            confidence += 0.2
        
        # Aspect ratio score (rectangular is better)
        if 2.0 <= aspect_ratio <= 6.0:
            confidence += 0.3
        elif 1.5 <= aspect_ratio <= 8.0:
            confidence += 0.2
        
        # Solidity score (more solid shapes are better)
        if solidity > 0.7:
            confidence += 0.2
        elif solidity > 0.5:
            confidence += 0.1
        
        # Vertex count score (4-6 vertices for rectangular shapes)
        if vertices == 4:
            confidence += 0.2
        elif 4 <= vertices <= 6:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def generate_auto_roi(self, wood_candidates: List[Dict], image_shape: Tuple) -> Optional[Tuple[int, int, int, int]]:
        """Generate automatic ROI based on detected wood"""
        if not wood_candidates:
            return None
        
        # Use the highest confidence detection
        best_candidate = wood_candidates[0]
        x, y, w, h = best_candidate['bbox']
        
        # Add some padding around the detected wood
        padding_x = int(w * 0.1)  # 10% padding
        padding_y = int(h * 0.1)
        
        roi_x1 = max(0, x - padding_x)
        roi_y1 = max(0, y - padding_y)
        roi_x2 = min(image_shape[1], x + w + padding_x)
        roi_y2 = min(image_shape[0], y + h + padding_y)
        
        return (roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1)
    
    def detect_wood_comprehensive(self, image: np.ndarray, profile_names: List[str] = None, roi: Tuple[int, int, int, int] = None, camera: str = 'top') -> Dict:
        """Comprehensive wood detection combining color and shape analysis"""
        
        try:
            # Validate input image
            if image is None or image.size == 0:
                print("❌ Error: Invalid input image for comprehensive detection")
                return {
                    'wood_detected': False,
                    'wood_count': 0,
                    'wood_candidates': [],
                    'auto_roi': None,
                    'color_mask': np.zeros((100, 100), dtype=np.uint8),
                    'confidence': 0.0,
                    'texture_confidence': 0.0,
                    'error': 'Invalid input image'
                }

            print(f"🪵 Starting comprehensive wood detection on image shape: {image.shape}")

            # Step 1: Color-based detection with optional ROI
            # Use camera-specific profile if none specified
            if profile_names is None:
                if camera == 'top':
                    profile_names = ['top_panel']
                elif camera == 'bottom':
                    profile_names = ['bottom_panel']
                else:
                    profile_names = list(self.wood_color_profiles.keys())

            if roi is not None:
                x, y, w, h = roi
                cropped = image[y:y+h, x:x+w]
                color_mask_cropped, _ = self.detect_wood_by_color(cropped, profile_names)
                color_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                color_mask[y:y+h, x:x+w] = color_mask_cropped
            else:
                color_mask, _ = self.detect_wood_by_color(image, profile_names)

            mask_pixels = cv2.countNonZero(color_mask)
            total_pixels = image.shape[0] * image.shape[1]
            mask_percentage = (mask_pixels / total_pixels) * 100
            print(f"🎨 Color mask: {mask_pixels} pixels ({mask_percentage:.1f}%)")

            # Step 2: Find rectangular contours
            wood_candidates = self.detect_rectangular_contours(color_mask, camera)
            print(f"📐 Found {len(wood_candidates)} wood candidates after contour filtering")

            # Step 3: Generate automatic ROI
            auto_roi = self.generate_auto_roi(wood_candidates, image.shape)
            if auto_roi:
                print(f"🎯 Auto ROI generated: {auto_roi}")
            else:
                print("❌ No auto ROI generated (no candidates)")

            # Step 4: Integrate texture analysis for enhanced confidence
            texture_confidence = self._detect_wood_by_texture(image)
            combined_confidence = (wood_candidates[0]['confidence'] + texture_confidence) / 2 if wood_candidates else texture_confidence

            # Step 5: Create result
            result = {
                'wood_detected': len(wood_candidates) > 0,
                'wood_count': len(wood_candidates),
                'wood_candidates': wood_candidates,
                'auto_roi': auto_roi,
                'color_mask': color_mask,
                'confidence': combined_confidence,
                'texture_confidence': texture_confidence
            }
            
            # Step 6: Update dynamic wood width if wood is detected (matches testIR.py)
            if result['wood_detected']:
                detected_width = self.update_wood_width_dynamic(camera, wood_candidates)
                result['detected_width_mm'] = detected_width
                
                # Store wood detection results for later use
                self.wood_detection_results[camera] = result
                self.dynamic_roi[camera] = auto_roi
            else:
                # Clear results when no wood detected
                self.wood_detection_results[camera] = None
                self.dynamic_roi[camera] = None

            print(f"✅ Detection complete: wood_detected={result['wood_detected']}, count={result['wood_count']}, confidence={result['confidence']:.2f}")

            return result
            
        except Exception as e:
            print(f"❌ Error in comprehensive wood detection: {e}")
            return {
                'wood_detected': False,
                'wood_count': 0,
                'wood_candidates': [],
                'auto_roi': None,
                'color_mask': np.zeros(image.shape[:2] if image is not None else (100, 100), dtype=np.uint8),
                'confidence': 0.0,
                'texture_confidence': 0.0,
                'error': str(e)
            }
    
    def visualize_detection(self, image: np.ndarray, detection_result: Dict) -> np.ndarray:
        """Create visualization of wood detection results"""
        vis_image = image.copy()
        
        # Draw all wood candidates
        for i, candidate in enumerate(detection_result['wood_candidates']):
            # Draw bounding box
            x, y, w, h = candidate['bbox']
            color = (0, 255, 0) if i == 0 else (0, 255, 255)  # Best candidate in green, others in yellow
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Add confidence label
            label = f"Wood {i+1}: {candidate['confidence']:.2f}"
            cv2.putText(vis_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add metrics
            metrics = f"AR:{candidate['aspect_ratio']:.1f} S:{candidate['solidity']:.2f}"
            cv2.putText(vis_image, metrics, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw auto ROI
        if detection_result['auto_roi']:
            roi_x, roi_y, roi_w, roi_h = detection_result['auto_roi']
            cv2.rectangle(vis_image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 3)
            cv2.putText(vis_image, "AUTO ROI", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Add summary info
        summary = f"Wood Detected: {detection_result['wood_detected']} | Count: {detection_result['wood_count']} | Confidence: {detection_result['confidence']:.2f}"
        cv2.putText(vis_image, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image
    
    def draw_wood_detection_overlay(self, frame, camera_name):
        """Draw wood detection overlay similar to testIR.py"""
        overlay_frame = frame.copy()
        
        # Get stored wood detection results
        if hasattr(self, 'wood_detection_results') and self.wood_detection_results.get(camera_name):
            wood_detection = self.wood_detection_results[camera_name]
            
            # Draw all wood candidates
            for i, candidate in enumerate(wood_detection.get('wood_candidates', [])):
                x, y, w, h = candidate['bbox']
                confidence = candidate['confidence']
                
                # Use different colors for different candidates
                color = (0, 255, 0) if i == 0 else (0, 255, 255)  # Green for best, yellow for others
                
                # Draw bounding box
                cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), color, 2)
                
                # Add labels
                label = f"Wood {i+1}: {confidence:.2f}"
                cv2.putText(overlay_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Add width measurement for best candidate
                if i == 0:
                    width_mm = self.calculate_width_mm(h, camera_name)  # Use height for cross-section
                    width_label = f"Width: {width_mm:.1f}mm"
                    cv2.putText(overlay_frame, width_label, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw auto ROI if available
            if wood_detection.get('auto_roi'):
                roi_x, roi_y, roi_w, roi_h = wood_detection['auto_roi']
                cv2.rectangle(overlay_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 2)
                cv2.putText(overlay_frame, "AUTO ROI", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return overlay_frame

    def create_segment_visualization(self, frame, wood_detection_result, camera_name):
        """Create segment visualization for wood detection - matches testIR.py interface"""
        return self.draw_wood_detection_overlay(frame, camera_name)

    def detect_wood_presence(self, frame):
        color_conf = self._detect_wood_by_color(frame)
        texture_conf = self._detect_wood_by_texture(frame)
        shape_conf = self._detect_wood_by_shape(frame)
        
        # Combine confidences with weights (color most important for wood)
        combined_conf = (0.5 * color_conf + 0.3 * texture_conf + 0.2 * shape_conf)
        wood_detected = combined_conf > 0.3  # Lower threshold since multiple methods
        
        return wood_detected, combined_conf, {
            'color_confidence': color_conf,
            'texture_confidence': texture_conf,
            'shape_confidence': shape_conf
        }

    def detect_wood(self, frame):
        """
        Enhanced wood detection using the wood detection model.
        Falls back to visual detection if model is not available.
        Returns True if wood is detected, False otherwise.
        """
        wood_detected, confidence, _ = self.detect_wood_presence(frame)
        return wood_detected

    def _detect_wood_by_color(self, frame):
        """Detect wood using RGB color segmentation"""
        try:
            rgb_frame = frame

            # Use calibrated wood color profiles
            
            combined_mask = None
            for profile in self.wood_color_profiles.values():
                mask = cv2.inRange(rgb_frame, profile['rgb_lower'], profile['rgb_upper'])
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
            edges = cv2.Canny(gray, 100, 200)
            
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

class CameraHandler:
    def __init__(self):
        self.top_camera = None
        self.bottom_camera = None
        # Device paths for cameras (Rapoo for top, C922 for bottom)
        self.top_camera_devices = ['/dev/video0','/dev/video1', '/dev/video3']  # Rapoo Camera
        self.bottom_camera_devices = ['/dev/video2', '/dev/video4', '/dev/video5']  # C922 Pro Stream Webcam
        self.top_camera_index = 0 # Cam0
        self.bottom_camera_index = 2  # Cam2
        self.top_camera_device = None  # Will be set to successful device path
        self.bottom_camera_device = None  # Will be set to successful device path
        self.top_camera_settings = {
            'brightness': 0,
            'contrast': 32,
            'saturation': 64,
            'hue': 0,
            'exposure': -6,
            'white_balance': 4520,
            'gain': 0,
            'sharpness': 200,
            'backlight_compensation': 1
        }
        self.bottom_camera_settings = {
            'brightness': 110,
            'contrast': 125,
            'saturation': 125,
            'hue': 0,
            'exposure': -6,
            'white_balance': 4850,
            'gain': 0,
            'sharpness': 200,
            'backlight_compensation': 1
        }

    def _get_camera_device_info(self):
        """Get camera device information using v4l2-ctl to identify cameras by name"""
        try:
            print("🔍 Running v4l2-ctl --list-devices to detect cameras...")
            result = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True, timeout=5)

            # Print the raw output for visibility
            print("📋 v4l2-ctl output:")
            print(result.stdout)
            print("📋 End of v4l2-ctl output")

            # Parse output even if returncode != 0, as v4l2-ctl may return 1 but still provide device list
            devices = {}
            lines = result.stdout.strip().split('\n')
            current_device = None

            for line in lines:
                # Check for device paths before stripping (preserve tabs)
                if line.startswith('\t/dev/video'):
                    # This is a device path
                    if current_device:
                        device_path = line.strip()
                        devices[device_path] = current_device
                elif line.strip() and not line.startswith('\t'):
                    # This is a device name (not indented)
                    current_device = line.strip()

            print(f"📊 Parsed {len(devices)} video devices: {list(devices.keys())}")
            return devices
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            print(f"❌ Error getting camera device info: {e}")
            return {}

    def _identify_camera_by_name(self, device_path):
        """Identify camera type by device name"""
        device_info = self._get_camera_device_info()
        device_name = device_info.get(device_path, "").lower()

        if "c922" in device_name or "stream webcam" in device_name:
            return "C922"
        elif "rapoo" in device_name:
            return "Rapoo"
        else:
            return "Unknown"

    def _initialize_camera_with_devices(self, device_list, camera_name):
        """Try to initialize camera using specific device paths"""
        for device_path in device_list:
            try:
                print(f"Trying to open {camera_name} camera at {device_path}...")
                cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
                if cap.isOpened():
                    # Try to read a frame to ensure camera is working
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Disable autofocus for consistent focus
                        try:
                            subprocess.run(['v4l2-ctl', '-d', device_path, '-c', 'focus_automatic_continuous=0'],
                                          capture_output=True, timeout=2)
                            print(f"Disabled autofocus for {device_path}")
                        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
                            print(f"Warning: Could not disable autofocus for {device_path}")

                        camera_type = self._identify_camera_by_name(device_path)
                        print(f"Successfully opened {camera_name} camera at {device_path} (Type: {camera_type})")
                        return cap, device_path
                    else:
                        print(f"Camera at {device_path} opened but cannot read frames")
                        cap.release()
                else:
                    print(f"Failed to open camera at {device_path}")
            except Exception as e:
                print(f"Error opening camera at {device_path}: {e}")
                continue
        return None, None

    def initialize_cameras(self):
        try:
            self.top_camera = cv2.VideoCapture(self.top_camera_index)
            if not self.top_camera.isOpened():
                raise RuntimeError(f"Could not open top camera (Cam0 - index {self.top_camera_index})")
            self.bottom_camera = cv2.VideoCapture(self.bottom_camera_index)
            if not self.bottom_camera.isOpened():
                self.top_camera.release()
                raise RuntimeError(f"Could not open bottom camera (Cam2 - index {self.bottom_camera_index})")
            self.top_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.top_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.bottom_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.bottom_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._apply_camera_settings(self.top_camera, self.top_camera_settings)
            self._apply_camera_settings(self.bottom_camera, self.bottom_camera_settings)
            print("Cameras initialized successfully at 720p (1280x720)")
        except Exception as e:
            self.release_cameras()
            raise RuntimeError(f"Failed to initialize cameras: {str(e)}")

    def _apply_camera_settings(self, camera, settings):
        try:
            camera.set(cv2.CAP_PROP_BRIGHTNESS, settings['brightness'])
            camera.set(cv2.CAP_PROP_CONTRAST, settings['contrast'])
            camera.set(cv2.CAP_PROP_SATURATION, settings['saturation'])
            camera.set(cv2.CAP_PROP_HUE, settings['hue'])
            camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            camera.set(cv2.CAP_PROP_EXPOSURE, settings['exposure'])
            camera.set(cv2.CAP_PROP_AUTO_WB, 0)
            camera.set(cv2.CAP_PROP_WB_TEMPERATURE, settings['white_balance'])
            camera.set(cv2.CAP_PROP_GAIN, settings['gain'])
            if 'sharpness' in settings:
                camera.set(cv2.CAP_PROP_SHARPNESS, settings['sharpness'])
            if 'backlight_compensation' in settings:
                camera.set(cv2.CAP_PROP_BACKLIGHT, settings['backlight_compensation'])
        except Exception as e:
            print(f"Warning: Some camera settings may not be supported: {e}")

    def reconnect_cameras(self):
        """Attempt to reconnect cameras if they become disconnected"""
        print("🔌 Attempting to reconnect cameras...")
        print("   This is called automatically when camera disconnections are detected")

        # Release current cameras
        self.release_cameras()

        try:
            # Try dynamic reassignment first
            success = self._dynamic_reassign_cameras()
            if success:
                return True

            # Fallback to original device paths
            print("🔄 Dynamic reassignment failed, trying original device paths...")
            if self.top_camera_device:
                print(f"Trying to reconnect top camera at {self.top_camera_device}")
                self.top_camera = cv2.VideoCapture(self.top_camera_device, cv2.CAP_V4L2)
                if self.top_camera.isOpened():
                    ret, _ = self.top_camera.read()
                    if ret:
                        print(f"Reconnected top camera at {self.top_camera_device}")
                    else:
                        self.top_camera.release()
                        self.top_camera = None

            if self.bottom_camera_device:
                print(f"Trying to reconnect bottom camera at {self.bottom_camera_device}")
                self.bottom_camera = cv2.VideoCapture(self.bottom_camera_device, cv2.CAP_V4L2)
                if self.bottom_camera.isOpened():
                    ret, _ = self.bottom_camera.read()
                    if ret:
                        print(f"Reconnected bottom camera at {self.bottom_camera_device}")
                    else:
                        self.bottom_camera.release()
                        self.bottom_camera = None

            # Apply settings if cameras are connected
            if self.top_camera:
                self.top_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.top_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self._apply_camera_settings(self.top_camera, self.top_camera_settings)

            if self.bottom_camera:
                self.bottom_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.bottom_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self._apply_camera_settings(self.bottom_camera, self.bottom_camera_settings)

            if self.top_camera and self.bottom_camera:
                print("Camera reconnection successful")
                print(f"Top camera: {self.top_camera_device}")
                print(f"Bottom camera: {self.bottom_camera_device}")
                return True
            else:
                print("Camera reconnection failed - not all cameras reconnected")
                return False

        except Exception as e:
            print(f"Camera reconnection failed: {e}")
            return False

    def _dynamic_reassign_cameras(self):
        """Dynamically scan and reassign cameras based on device identification"""
        print("🔄 Performing dynamic camera reassignment...")
        print("   This happens at startup and whenever camera disconnections are detected")

        # Release any existing cameras first
        self.release_cameras()

        # Get all available video devices
        device_info = self._get_camera_device_info()
        available_devices = list(device_info.keys())

        if len(available_devices) < 2:
            print(f"❌ Only {len(available_devices)} devices available, need at least 2")
            return False

        # Identify camera types
        c922_devices = []
        rapoo_devices = []

        for device_path in available_devices:
            camera_type = self._identify_camera_by_name(device_path)
            if camera_type == "C922":
                c922_devices.append(device_path)
            elif camera_type == "Rapoo":
                rapoo_devices.append(device_path)

        print(f"📷 Found C922 devices: {c922_devices}")
        print(f"📷 Found Rapoo devices: {rapoo_devices}")

        # Assign cameras based on identification
        top_device = None
        bottom_device = None

        # Rapoo for top
        if rapoo_devices:
            for device in rapoo_devices:
                try:
                    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret:
                            top_device = device
                            cap.release()
                            break
                        cap.release()
                except:
                    continue

        # C922 for bottom
        if c922_devices:
            for device in c922_devices:
                try:
                    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret:
                            bottom_device = device
                            cap.release()
                            break
                        cap.release()
                except:
                    continue

        if top_device and bottom_device:
            # Successfully identified and assigned
            self.top_camera = cv2.VideoCapture(top_device, cv2.CAP_V4L2)
            self.bottom_camera = cv2.VideoCapture(bottom_device, cv2.CAP_V4L2)
            self.top_camera_device = top_device
            self.bottom_camera_device = bottom_device

            # Disable autofocus for reconnected cameras
            for device in [top_device, bottom_device]:
                try:
                    subprocess.run(['v4l2-ctl', '-d', device, '-c', 'focus_automatic_continuous=0'],
                                  capture_output=True, timeout=2)
                    print(f"Disabled autofocus for reconnected {device}")
                except (subprocess.SubprocessError, subprocess.TimeoutExpired):
                    print(f"Warning: Could not disable autofocus for reconnected {device}")

            # Apply settings
            self.top_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.top_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._apply_camera_settings(self.top_camera, self.top_camera_settings)

            self.bottom_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.bottom_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._apply_camera_settings(self.bottom_camera, self.bottom_camera_settings)

            print("Dynamic camera reassignment successful!")
            print(f"Top camera (Rapoo): {top_device}")
            print(f"Bottom camera (C922): {bottom_device}")
            return True
        else:
            print("Dynamic reassignment failed - could not identify both camera types")
            return False

    def check_camera_status(self):
        """Check if cameras are still connected and working"""
        try:
            top_ok = False
            bottom_ok = False

            if self.top_camera and self.top_camera.isOpened():
                # Try to read a frame multiple times to account for temporary failures
                for attempt in range(3):
                    ret, _ = self.top_camera.read()
                    if ret:
                        top_ok = True
                        break
                    else:
                        time.sleep(0.1)  # Short delay between retries
                if not top_ok:
                    print("⚠️ Top camera is not responding after retries")
            else:
                print("⚠️ Top camera is not opened")

            if not top_ok:
                print("🔌 Top camera disconnection detected")

            if self.bottom_camera and self.bottom_camera.isOpened():
                # Try to read a frame multiple times to account for temporary failures
                for attempt in range(3):
                    ret, _ = self.bottom_camera.read()
                    if ret:
                        bottom_ok = True
                        break
                    else:
                        time.sleep(0.1)  # Short delay between retries
                if not bottom_ok:
                    print("⚠️ Bottom camera is not responding after retries")
            else:
                print("⚠️ Bottom camera is not opened")

            if not bottom_ok:
                print("🔌 Bottom camera disconnection detected")

            # Log status periodically (not every check to avoid spam)
            if not (top_ok and bottom_ok):
                print(f"📊 Camera status check: Top={'✅ OK' if top_ok else '❌ FAIL'}, Bottom={'✅ OK' if bottom_ok else '❌ FAIL'}")

            return top_ok and bottom_ok
        except Exception as e:
            print(f"❌ Error checking camera status: {e}")
            return False

    def reassign_cameras_runtime(self):
        """Runtime method to dynamically reassign cameras - can be called from UI or automatically"""
        print("Runtime camera reassignment requested...")
        success = self._dynamic_reassign_cameras()
        if success:
            print("Runtime camera reassignment successful")
            # Update any UI elements if needed
            if hasattr(self, 'status_label'):
                # status_label is a Text widget, not a Label widget
                self.status_label.config(state=tk.NORMAL)
                self.status_label.delete(1.0, tk.END)
                self.status_label.insert(1.0, "Status: Cameras reassigned successfully")
                self.status_label.config(foreground="green", state=tk.DISABLED)
        else:
            print("Runtime camera reassignment failed")
            if hasattr(self, 'status_label'):
                # status_label is a Text widget, not a Label widget
                self.status_label.config(state=tk.NORMAL)
                self.status_label.delete(1.0, tk.END)
                self.status_label.insert(1.0, "Status: Camera reassignment failed")
                self.status_label.config(foreground="red", state=tk.DISABLED)
        return success

    def reassign_arduino_runtime(self):
        """Runtime method to dynamically reassign Arduino port - can be called from UI or automatically"""
        print("Runtime Arduino reassignment requested...")
        try:
            # Close current connection
            if hasattr(self, 'ser') and self.ser:
                self.ser.close()
                self.ser = None

            # Try to setup again
            self.setup_arduino()
            if self.ser and self.ser.is_open:
                print("Runtime Arduino reassignment successful")
                if hasattr(self, 'status_label'):
                    # status_label is a Text widget, not a Label widget
                    self.status_label.config(state=tk.NORMAL)
                    self.status_label.delete(1.0, tk.END)
                    self.status_label.insert(1.0, "Status: Arduino reassigned successfully")
                    self.status_label.config(foreground="green", state=tk.DISABLED)
                return True
            else:
                print("Runtime Arduino reassignment failed")
                if hasattr(self, 'status_label'):
                    # status_label is a Text widget, not a Label widget
                    self.status_label.config(state=tk.NORMAL)
                    self.status_label.delete(1.0, tk.END)
                    self.status_label.insert(1.0, "Status: Arduino reassignment failed")
                    self.status_label.config(foreground="red", state=tk.DISABLED)
                return False
        except Exception as e:
            print(f"Error during runtime Arduino reassignment: {e}")
            if hasattr(self, 'status_label'):
                # status_label is a Text widget, not a Label widget
                self.status_label.config(state=tk.NORMAL)
                self.status_label.delete(1.0, tk.END)
                self.status_label.insert(1.0, "Status: Arduino reassignment error")
                self.status_label.config(foreground="red", state=tk.DISABLED)
            return False

    def release_cameras(self):
        if self.top_camera:
            self.top_camera.release()
            self.top_camera = None
        if self.bottom_camera:
            self.bottom_camera.release()
            self.bottom_camera = None
        print("Cameras released")

def main():
    global WOOD_PALLET_WIDTH_MM  # Declare global at function start
    
    camera_handler = CameraHandler()
    camera_handler.initialize_cameras()
    detector = ColorWoodDetector()

    # Define ROIs following the same coordinates as testIR.py
    roi_top = (370, 0, 880-370, 720-0)  # x1=370, y1=0, width=510, height=720
    roi_bottom = (350, 0, 965-350, 720-0)  # x1=350, y1=0, width=615, height=720

    cap0 = camera_handler.top_camera
    cap2 = camera_handler.bottom_camera
    if not cap0 or not cap2:
        print("❌ Could not open cameras")
        return
    print("🎥 Starting live wood detection from video0 and video2")
    print("Press 'q' to quit")
    print("📋 Keyboard Controls:")
    print("  'q' - Quit application")
    print("  'd' - Toggle detection on/off")
    print("  'c' - Toggle color mask view")
    print("  's' - Save current frames")
    print("  'r' - Reset wood width calibration")
    print("  'w' - Show current wood width info")
    frame_count = 0
    show_mask = False  # Ensure show_mask is always defined before the loop
    detection_enabled = True  # Toggle for detection processing
    # Directory to save captured frames
    save_dir = "captured_frames"
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        while True:
            ret0, frame0 = cap0.read()
            ret2, frame2 = cap2.read()
            if not ret0 or not ret2:
                print("❌ Failed to read frames from cameras")
                break
            frame_count += 1

            frame2_flipped = cv2.flip(frame2, 1)

            if detection_enabled:
                try:
                    # Process frame from camera 0 (top) with ROI
                    detection_result0 = detector.detect_wood_comprehensive(frame0, roi=roi_top, camera='top')
                    wood_detected0 = detection_result0['wood_detected']
                    confidence0 = detection_result0['confidence']

                    # Update global wood width based on detection (matches testIR.py)
                    top_width_mm = None
                    if wood_detected0 and detection_result0['wood_candidates']:
                        candidate = detection_result0['wood_candidates'][0]
                        x, y, w, h = candidate['bbox']
                        # Use height (h) for cross-sectional width measurement like testIR.py
                        top_width_mm = detector.calculate_width_mm(h, 'top')
                        print(f"🎯 Top camera wood width: {top_width_mm:.1f}mm (from {w}x{h}px bbox)")
                except Exception as e:
                    print(f"❌ Error in top camera detection: {e}")
                    detection_result0 = {'wood_candidates': [], 'color_mask': np.zeros(frame0.shape[:2], dtype=np.uint8), 'wood_detected': False, 'wood_count': 0, 'confidence': 0.0}
                    wood_detected0 = False
                    confidence0 = 0.0
                    top_width_mm = None
            else:
                # Skip detection processing for top
                detection_result0 = {'wood_candidates': [], 'color_mask': np.zeros(frame0.shape[:2], dtype=np.uint8), 'wood_detected': False, 'wood_count': 0, 'confidence': 0.0}
                wood_detected0 = False
                confidence0 = 0.0
                top_width_mm = None

            # Check if bottom camera display should be enabled based on top camera detection
            bottom_display_enabled = wood_detected0

            if detection_enabled:
                try:
                    # Process frame from camera 2 (bottom) - only if top detected wood (hierarchy)
                    if wood_detected0:
                        detection_result2 = detector.detect_wood_comprehensive(frame2_flipped, roi=roi_bottom, camera='bottom')
                        wood_detected2 = detection_result2['wood_detected']
                        confidence2 = detection_result2['confidence']
                        bottom_width_mm = None  # Bottom camera measurements disabled
                    else:
                        # No top detection, skip bottom processing
                        detection_result2 = {'wood_candidates': [], 'color_mask': np.zeros(frame2_flipped.shape[:2], dtype=np.uint8), 'wood_detected': False, 'wood_count': 0, 'confidence': 0.0}
                        wood_detected2 = False
                        confidence2 = 0.0
                        bottom_width_mm = None
                except Exception as e:
                    print(f"❌ Error in bottom camera detection: {e}")
                    detection_result2 = {'wood_candidates': [], 'color_mask': np.zeros(frame2_flipped.shape[:2], dtype=np.uint8), 'wood_detected': False, 'wood_count': 0, 'confidence': 0.0}
                    wood_detected2 = False
                    confidence2 = 0.0
                    bottom_width_mm = None
            else:
                # Skip detection processing for bottom
                detection_result2 = {'wood_candidates': [], 'color_mask': np.zeros(frame2_flipped.shape[:2], dtype=np.uint8), 'wood_detected': False, 'wood_count': 0, 'confidence': 0.0}
                wood_detected2 = False
                confidence2 = 0.0
                bottom_width_mm = None


            # Draw ROI bounding boxes
            cv2.rectangle(frame0, (roi_top[0], roi_top[1]), (roi_top[0] + roi_top[2], roi_top[1] + roi_top[3]), (0, 255, 0), 2)
            cv2.rectangle(frame2_flipped, (roi_bottom[0], roi_bottom[1]), (roi_bottom[0] + roi_bottom[2], roi_bottom[1] + roi_bottom[3]), (0, 255, 0), 2)
            
            # Draw bounding box on frame 0 for the best candidate only
            if detection_result0['wood_candidates']:
                candidate = detection_result0['wood_candidates'][0]
                x, y, w, h = candidate['bbox']
                color = (0, 255, 0)
                cv2.rectangle(frame0, (x, y), (x + w, y + h), color, 2)

                # Add confidence and width label - using height for width measurement like testIR.py
                label = f"Wood: {candidate['confidence']:.2f} | Width: {top_width_mm:.1f}mm" if top_width_mm is not None else f"Wood: {candidate['confidence']:.2f}"
                cv2.putText(frame0, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Add global wood width indicator
                cv2.putText(frame0, f"Global Wood Width: {WOOD_PALLET_WIDTH_MM:.1f}mm", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Draw bounding box on frame 2 for the best candidate only (conditional on top camera confidence)
            if bottom_display_enabled and detection_result2['wood_candidates']:
                candidate = detection_result2['wood_candidates'][0]
                x, y, w, h = candidate['bbox']
                color = (0, 255, 0)
                cv2.rectangle(frame2_flipped, (x, y), (x + w, y + h), color, 2)

                # Add confidence label only (no width measurement)
                label = f"Wood: {candidate['confidence']:.2f}"
                cv2.putText(frame2_flipped, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            
            # Add text overlays
            # Camera 0
            if detection_enabled:
                status0 = "WOOD DETECTED" if wood_detected0 else "NO WOOD"
                color0 = (0, 255, 0) if wood_detected0 else (0, 0, 255)
            else:
                status0 = "DETECTION OFF"
                color0 = (0, 0, 255)
            cv2.putText(frame0, f"Camera 0 (Top): {status0}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color0, 2)
            cv2.putText(frame0, f"Count: {detection_result0['wood_count']} | Confidence: {confidence0:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame0, f"Frame: {frame_count}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Camera 2
            if detection_enabled:
                if bottom_display_enabled:
                    status2 = "WOOD DETECTED" if wood_detected2 else "NO WOOD"
                    color2 = (0, 255, 0) if wood_detected2 else (0, 0, 255)
                else:
                    status2 = "WAITING FOR TOP DETECTION"
                    color2 = (0, 165, 255)  # Orange color for waiting
            else:
                status2 = "DETECTION OFF"
                color2 = (0, 0, 255)
            cv2.putText(frame2, f"Camera 2 (Bottom): {status2}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color2, 2)
            cv2.putText(frame2, f"Count: {detection_result2['wood_count']} | Confidence: {confidence2:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame2, f"Frame: {frame_count}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Resize frames and masks to 480p for display
            display_width, display_height = 852, 480  # 480p (16:9)
            frame0_disp = cv2.resize(frame0, (display_width, display_height))
            frame2_disp = cv2.resize(frame2_flipped, (display_width, display_height))  # Already flipped
            mask0_disp = cv2.resize(detection_result0['color_mask'], (display_width, display_height))
            mask2_disp = cv2.resize(detection_result2['color_mask'], (display_width, display_height))  # Already flipped

            # No additional flip needed for bottom camera display

            # Stack frames side by side
            if show_mask:
                combined_disp = np.hstack((cv2.cvtColor(mask0_disp, cv2.COLOR_GRAY2BGR), cv2.cvtColor(mask2_disp, cv2.COLOR_GRAY2BGR)))
            else:
                combined_disp = np.hstack((frame0_disp, frame2_disp))

            cv2.imshow('Wood Detection (Top | Bottom)', combined_disp)

            # Print to console every 30 frames
            if frame_count % 30 == 0:
                width0_str = f", width={top_width_mm:.1f}mm" if top_width_mm is not None else ""
                print(f"Frame {frame_count}: Cam0={status0}({confidence0:.2f}{width0_str}), Cam2={status2}({confidence2:.2f})")

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                detection_enabled = not detection_enabled
                print(f"🔄 Detection {'enabled' if detection_enabled else 'disabled'}")
            elif key == ord('c'):
                show_mask = not show_mask
                if show_mask:
                    print("🎭 Mask view enabled (press 'C' again to toggle off)")
                else:
                    print("📷 Normal view enabled (press 'C' to toggle masks)")
            elif key == ord('s'):
                # Save both frames
                top_path = os.path.join(save_dir, f"TopPanel_{frame_count}.jpg")
                bottom_path = os.path.join(save_dir, f"BottomPanel_{frame_count}.jpg")
                cv2.imwrite(top_path, frame0)
                cv2.imwrite(bottom_path, frame2_flipped)
                print(f"💾 Saved frames: {top_path}, {bottom_path}")
            elif key == ord('r'):
                # Reset wood width calibration
                WOOD_PALLET_WIDTH_MM = 0
                detector.detected_wood_width_mm = {'top': 0, 'bottom': 0}
                print("🔄 Wood width calibration reset")
            elif key == ord('w'):
                # Show current wood width info
                print(f"📏 Current wood width: Global={WOOD_PALLET_WIDTH_MM:.1f}mm, Top={detector.detected_wood_width_mm.get('top', 0):.1f}mm, Bottom={detector.detected_wood_width_mm.get('bottom', 0):.1f}mm")
                
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        camera_handler.release_cameras()
        cv2.destroyAllWindows()
        print("📷 Cameras released")


if __name__ == "__main__":
    main()

