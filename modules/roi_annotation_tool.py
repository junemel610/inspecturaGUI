#!/usr/bin/env python3
"""
ROI Annotation Tool
A Python tool for drawing bounding boxes on images and extracting coordinates.

Features:
- Load images from file or webcam
- Mouse-based rectangle drawing
- Real-time coordinate display
- Multiple ROI support
- Save/load ROI coordinates to JSON
- Keyboard shortcuts for common operations

Usage:
python roi_annotation_tool.py [image_path]

Keyboard Shortcuts:
- 'q' or 'ESC': Quit
- 's': Save ROIs to file
- 'l': Load ROIs from file
- 'c': Clear all ROIs
- 'd': Delete last ROI
- 'r': Reset image (clear all and reload)
- 'h': Show/hide help
- 'f': Toggle fullscreen
- Mouse: Click and drag to draw rectangles
"""

import cv2
import numpy as np
import json
import os
import sys
import argparse
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

class ROIAnnotationTool:
    def __init__(self, image_path=None, use_webcam=False):
        self.image_path = image_path
        self.use_webcam = use_webcam
        self.original_image = None
        self.display_image = None
        self.rois = []  # List of (x1, y1, x2, y2) tuples
        self.drawing = False
        self.start_point = None
        self.current_roi = None
        self.window_name = "ROI Annotation Tool"
        self.show_help = True
        self.fullscreen = False

        # Colors
        self.roi_color = (0, 255, 0)  # Green
        self.current_roi_color = (255, 0, 0)  # Red
        self.text_color = (255, 255, 255)  # White
        self.bg_color = (0, 0, 0)  # Black

        # Initialize
        self.load_image()
        self.setup_window()
        self.setup_mouse_callback()

    def load_image(self):
        """Load image from file or initialize webcam"""
        if self.use_webcam:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open webcam")
                sys.exit(1)
            ret, frame = self.cap.read()
            if ret:
                self.original_image = frame.copy()
            else:
                print("Error: Could not read frame from webcam")
                sys.exit(1)
        elif self.image_path:
            if not os.path.exists(self.image_path):
                print(f"Error: Image file '{self.image_path}' not found")
                sys.exit(1)
            self.original_image = cv2.imread(self.image_path)
            if self.original_image is None:
                print(f"Error: Could not load image '{self.image_path}'")
                sys.exit(1)
        else:
            # Create a blank image for demonstration
            self.original_image = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(self.original_image, "No image loaded", (400, 360),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        self.display_image = self.original_image.copy()
        self.update_display()

    def setup_window(self):
        """Setup OpenCV window"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

    def setup_mouse_callback(self):
        """Setup mouse callback for drawing"""
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_roi = (x, y, x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_roi = (self.start_point[0], self.start_point[1], x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                # Ensure proper rectangle coordinates
                x1, y1 = self.start_point
                x2, y2 = x, y

                # Normalize coordinates
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                # Only add if rectangle has minimum size
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    self.rois.append((x1, y1, x2, y2))
                    print(f"ROI {len(self.rois)} added: ({x1}, {y1}) to ({x2}, {y2})")
                else:
                    print("ROI too small, discarded")

                self.current_roi = None
                self.update_display()

    def update_display(self):
        """Update the display image with ROIs and information"""
        self.display_image = self.original_image.copy()

        # Draw existing ROIs
        for i, (x1, y1, x2, y2) in enumerate(self.rois):
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), self.roi_color, 2)
            # Label ROI
            cv2.putText(self.display_image, f"ROI {i+1}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.roi_color, 2)

        # Draw current ROI being drawn
        if self.current_roi:
            x1, y1, x2, y2 = self.current_roi
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), self.current_roi_color, 2)

        # Add information overlay
        self.add_info_overlay()

        # Show help if enabled
        if self.show_help:
            self.add_help_overlay()

    def add_info_overlay(self):
        """Add information overlay to the display"""
        height, width = self.display_image.shape[:2]

        # Background for info
        cv2.rectangle(self.display_image, (10, 10), (400, 120), self.bg_color, -1)
        cv2.rectangle(self.display_image, (10, 10), (400, 120), self.text_color, 1)

        # Info text
        info_lines = [
            f"ROIs: {len(self.rois)}",
            f"Image: {width}x{height}",
            f"Mouse: Draw rectangles",
            "Press 'h' for help"
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(self.display_image, line, (20, 35 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 1)

    def add_help_overlay(self):
        """Add help overlay"""
        height, width = self.display_image.shape[:2]

        # Semi-transparent background
        overlay = self.display_image.copy()
        cv2.rectangle(overlay, (width-350, height-250), (width-10, height-10),
                     (0, 0, 0), -1)
        cv2.addWeighted(self.display_image, 0.7, overlay, 0.3, 0, self.display_image)

        # Help text
        help_lines = [
            "HELP - Keyboard Shortcuts:",
            "q/ESC: Quit",
            "s: Save ROIs",
            "l: Load ROIs",
            "c: Clear all ROIs",
            "d: Delete last ROI",
            "r: Reset image",
            "f: Toggle fullscreen",
            "h: Hide help",
            "",
            "Mouse: Click & drag to draw"
        ]

        for i, line in enumerate(help_lines):
            cv2.putText(self.display_image, line, (width-340, height-230 + i*18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

    def save_rois(self):
        """Save ROIs to JSON file"""
        if not self.rois:
            print("No ROIs to save")
            return

        # Create filename based on image path or timestamp
        if self.image_path:
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            filename = f"{base_name}_rois.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rois_{timestamp}.json"

        # Prepare data
        data = {
            "image_path": self.image_path,
            "image_size": self.original_image.shape[:2] if self.original_image is not None else None,
            "rois": [{"id": i+1, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": x2-x1, "height": y2-y1}
                    for i, (x1, y1, x2, y2) in enumerate(self.rois)],
            "timestamp": datetime.now().isoformat()
        }

        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"ROIs saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving ROIs: {e}")
            return None

    def load_rois(self):
        """Load ROIs from JSON file"""
        # Use tkinter for file dialog
        root = tk.Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(
            title="Select ROI file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        root.destroy()

        if not filename:
            return

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.rois = [(roi['x1'], roi['y1'], roi['x2'], roi['y2'])
                        for roi in data.get('rois', [])]

            print(f"Loaded {len(self.rois)} ROIs from {filename}")
            self.update_display()

        except Exception as e:
            print(f"Error loading ROIs: {e}")

    def clear_rois(self):
        """Clear all ROIs"""
        self.rois = []
        self.update_display()
        print("All ROIs cleared")

    def delete_last_roi(self):
        """Delete the last ROI"""
        if self.rois:
            self.rois.pop()
            self.update_display()
            print("Last ROI deleted")
        else:
            print("No ROIs to delete")

    def reset_image(self):
        """Reset image and clear all ROIs"""
        self.rois = []
        if self.use_webcam:
            ret, frame = self.cap.read()
            if ret:
                self.original_image = frame.copy()
        self.update_display()
        print("Image reset")

    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN,
                                cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN,
                                cv2.WINDOW_NORMAL)

    def run(self):
        """Main loop"""
        print("ROI Annotation Tool started")
        print("Press 'h' for help, 'q' to quit")

        while True:
            # Update webcam frame if using webcam
            if self.use_webcam:
                ret, frame = self.cap.read()
                if ret:
                    self.original_image = frame.copy()
                    self.update_display()

            # Show image
            cv2.imshow(self.window_name, self.display_image)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('s'):
                self.save_rois()
            elif key == ord('l'):
                self.load_rois()
            elif key == ord('c'):
                self.clear_rois()
            elif key == ord('d'):
                self.delete_last_roi()
            elif key == ord('r'):
                self.reset_image()
            elif key == ord('h'):
                self.show_help = not self.show_help
                self.update_display()
            elif key == ord('f'):
                self.toggle_fullscreen()

        # Cleanup
        if self.use_webcam and hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="ROI Annotation Tool")
    parser.add_argument("image_path", nargs="?", help="Path to image file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam instead of image file")

    args = parser.parse_args()

    if not args.image_path and not args.webcam:
        print("Usage: python roi_annotation_tool.py [image_path] [--webcam]")
        print("Examples:")
        print("  python roi_annotation_tool.py image.jpg")
        print("  python roi_annotation_tool.py --webcam")
        sys.exit(1)

    tool = ROIAnnotationTool(image_path=args.image_path, use_webcam=args.webcam)
    tool.run()

if __name__ == "__main__":
    main()