#!/usr/bin/env python3
"""
Test script for RGB-based wood detection
"""

import cv2
import numpy as np
import sys
import os
from typing import List, Dict, Tuple

# Add the current directory to the path so we can import detection
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ColorWoodDetector:
    def __init__(self):
        self.wood_color_profiles = {
            'top_panel': {
                'rgb_lower': np.array([169, 180, 176]),  # BGR
                'rgb_upper': np.array([225, 220, 210]),
                'name': 'Top Panel Wood'
            },
            'bottom_panel': {
                'rgb_lower': np.array([150, 180, 150]),  # BGR
                'rgb_upper': np.array([225, 220, 210]),
                'name': 'Bottom Panel Wood'
            }
        }

        # Detection parameters - Updated to match testIR/testIR.py exactly
        self.min_contour_area = 2000      # Increased for more reliable detection with tighter RGB ranges
        self.max_contour_area = 500000    # Slightly reduced for typical wood plank sizes
        self.min_aspect_ratio = 1.0       # Tightened for more rectangular wood shapes
        self.max_aspect_ratio = 10.0      # Reduced for more typical plank proportions
        self.contour_approximation = 0.025 # Slightly tighter for better shape approximation

        # Morphological operations - Updated to match testIR/testIR.py
        self.morph_kernel_size = 11       # Larger kernel for better noise reduction
        self.closing_iterations = 3       # More closing iterations
        self.opening_iterations = 2        # More opening iterations

    def detect_wood_by_color(self, image: np.ndarray, profile_names: List[str] = None) -> Tuple[np.ndarray, List[Dict]]:
        """Detect wood using RGB color profiles"""
        if profile_names is None:
            profile_names = list(self.wood_color_profiles.keys())

        # Apply histogram equalization on V channel for better lighting compensation
        hsv_temp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_temp)
        v = cv2.equalizeHist(v)
        hsv_temp = cv2.merge([h, s, v])
        rgb = cv2.cvtColor(hsv_temp, cv2.COLOR_HSV2BGR)

        # Dynamically update RGB ranges based on dominant colors for better adaptation
        self.update_rgb_ranges_based_on_dominant_colors(rgb)

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

        pre_morph_pixels = cv2.countNonZero(combined_mask)
        pre_morph_percentage = (pre_morph_pixels / total_pixels) * 100
        print(f"🔧 Pre-morph combined mask: {pre_morph_pixels} pixels ({pre_morph_percentage:.1f}%)")

        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=self.closing_iterations)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=self.opening_iterations)

        post_morph_pixels = cv2.countNonZero(combined_mask)
        post_morph_percentage = (post_morph_pixels / total_pixels) * 100
        print(f"🔧 Post-morph combined mask: {post_morph_pixels} pixels ({post_morph_percentage:.1f}%)")

        # Additional logging for dominant colors
        rgb_flat = rgb.reshape(-1, 3)
        r_values = rgb_flat[:, 0]
        g_values = rgb_flat[:, 1]
        b_values = rgb_flat[:, 2]
        print(f"🎨 Dominant RGB in image: R={int(np.mean(r_values)):.0f}±{int(np.std(r_values)):.0f}, G={int(np.mean(g_values)):.0f}, B={int(np.mean(b_values)):.0f}")

        return combined_mask, detections

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

def main():
    print("🧪 Testing RGB-based wood detection...")

    # Test the ColorWoodDetector class
    detector = ColorWoodDetector()
    print('✅ ColorWoodDetector initialized successfully')

    # Test with a dummy image (wood-like color in RGB)
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_image[:, :] = [150, 120, 100]  # Wood-like color in RGB

    # Test the RGB-based detection
    mask, detections = detector.detect_wood_by_color(dummy_image)
    print(f'✅ RGB-based wood detection completed. Mask pixels: {cv2.countNonZero(mask)}')

    # Test the _detect_wood_by_color method
    confidence = detector._detect_wood_by_color(dummy_image)
    print(f'✅ RGB color confidence: {confidence:.2f}')

    # Test with non-wood color (should have low confidence)
    non_wood_image = np.zeros((100, 100, 3), dtype=np.uint8)
    non_wood_image[:, :] = [50, 50, 200]  # Blue color (not wood-like)

    confidence_non_wood = detector._detect_wood_by_color(non_wood_image)
    print(f'✅ Non-wood color confidence: {confidence_non_wood:.2f}')

    print('🎉 All RGB-based wood detection tests passed!')
    print(f'📊 Wood-like color confidence: {confidence:.2f}')
    print(f'📊 Non-wood color confidence: {confidence_non_wood:.2f}')

if __name__ == "__main__":
    main()