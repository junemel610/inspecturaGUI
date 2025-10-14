import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import serial
import threading
import time
import queue
import degirum as dg
import degirum_tools
import json
import os
import subprocess
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import numpy as np
from typing import Dict, List, Tuple, Optional

# =============================================================================
# UI CONFIGURATION SECTION - Customize the user interface appearance and behavior
# =============================================================================

# ------------------------------------------------------------------------------
# WINDOW AND DISPLAY SETTINGS
# Adjust window sizing and display parameters
# ------------------------------------------------------------------------------
WINDOW_SCALE = 0.65              # Fraction of screen size for window (further reduced for smaller containers)
MIN_WINDOW_WIDTH = 800            # Minimum window width in pixels
MIN_WINDOW_HEIGHT = 600           # Minimum window height in pixels
ENABLE_FULLSCREEN_STARTUP = True # Automatically start in fullscreen mode (for kiosks/RPi)

# ------------------------------------------------------------------------------
# FONT SETTINGS
# Customize text appearance throughout the application
# ------------------------------------------------------------------------------
PRIMARY_FONT_FAMILY = "Helvetica"  # Primary font family for UI elements
BUTTON_FONT_FAMILY = "Helvetica"   # Font family for buttons
MONOSPACE_FONT_FAMILY = "Courier"  # Font family for code/logs

FONT_SIZE_DIVISOR = 80            # Divisor for responsive font sizing (base = max(8, min(12, screen_height / FONT_SIZE_DIVISOR)))
FONT_SIZE_BASE_MIN = 8             # Minimum base font size
FONT_SIZE_BASE_MAX = 12            # Maximum base font size

# ------------------------------------------------------------------------------
# COLOR THEME SETTINGS
# Customize the color scheme of the application
# ------------------------------------------------------------------------------
# Main application colors
BACKGROUND_COLOR = "#f0f0f0"       # Main window background color
FRAME_BACKGROUND_COLOR = "#ffffff" # Frame/panel background color
TEXT_COLOR = "#000000"            # Primary text color
SECONDARY_TEXT_COLOR = "#666666"  # Secondary/muted text color

# Button colors
BUTTON_BACKGROUND_COLOR = "#e0e0e0"  # Normal button background
BUTTON_ACTIVE_COLOR = "#d0d0d0"       # Button pressed/active color
BUTTON_TEXT_COLOR = "#000000"        # Button text color

# Status and grade colors
STATUS_READY_COLOR = "#28a745"      # Green for ready/active status
STATUS_WARNING_COLOR = "#ffc107"    # Yellow for warning status
STATUS_ERROR_COLOR = "#dc3545"      # Red for error status

GRADE_PERFECT_COLOR = "#228B22"     # Dark green for perfect grade
GRADE_GOOD_COLOR = "#32CD32"        # Green for good grade
GRADE_FAIR_COLOR = "#FFA500"        # Orange for fair grade
GRADE_POOR_COLOR = "#FF0000"        # Red for poor grade

# Detection overlay colors
DETECTION_BOX_COLOR = "#00FF00"     # Green for detection bounding boxes
ROI_OVERLAY_COLOR = "#FFFF00"       # Yellow for ROI overlay

# ------------------------------------------------------------------------------
# LAYOUT AND SPACING SETTINGS
# Adjust spacing, padding, and layout proportions
# ------------------------------------------------------------------------------
# Padding and margins (in pixels)
MAIN_PADDING = 5                   # Main window padding
FRAME_PADDING = 2                   # Frame/panel padding
CAMERA_FRAME_PADDING = 1            # Camera feed LabelFrame padding
ELEMENT_PADDING_X = 2              # Horizontal padding between elements
ELEMENT_PADDING_Y = 2              # Vertical padding between elements
LABEL_PADDING = 5                  # Label padding

# Grid layout weights (proportions)
CAMERA_FEEDS_WEIGHT = 0            # Weight for camera feeds section (relative to controls and stats)
CONTROLS_WEIGHT = 0                # Weight for controls section (compact)
STATS_WEIGHT = 1                   # Weight for statistics section
CAMERA_FEED_HEIGHT_WEIGHT = 0      # Weight for camera feeds row height (minimized for compact layout)

# Camera display settings
CAMERA_ASPECT_RATIO = "16:9"       # Target aspect ratio for camera displays
CAMERA_DISPLAY_MARGIN = -35          # Margin around camera displays (pixels)
CAMERA_FEED_MARGIN = 0             # White margin between and around camera feeds (pixels)

# ------------------------------------------------------------------------------
# UI BEHAVIOR SETTINGS
# Control interactive behavior and responsiveness
# ------------------------------------------------------------------------------
ENABLE_TOOLTIPS = True             # Show tooltips on hover
ENABLE_ANIMATIONS = False          # Enable UI animations (may impact performance)
AUTO_SCROLL_LOGS = True            # Automatically scroll logs to bottom
SCROLL_SENSITIVITY = 3             # Mouse wheel scroll sensitivity (lines per scroll)

# Update intervals (frames to skip between updates)
UI_UPDATE_SKIP = 3                 # Update UI elements every Nth frame
STATS_UPDATE_SKIP = 15             # Update statistics every Nth frame
LOG_UPDATE_SKIP = 10               # Update logs when no detection every Nth frame

# ------------------------------------------------------------------------------
# ADVANCED UI SETTINGS
# Fine-tune specific UI components
# ------------------------------------------------------------------------------
# Tabbed interface settings
STATS_TAB_HEIGHT = 200             # Minimum height for statistics tabs (pixels)
LOG_SCROLLABLE_HEIGHT = 200        # Height of scrollable log area (pixels)

# Status bar settings
STATUS_BAR_HEIGHT = 25             # Height of status bar (pixels)
STATUS_UPDATE_INTERVAL = 100       # Status update interval (milliseconds)

# Detection display settings
DETECTION_DETAILS_HEIGHT = 150     # Height of detection details panels (pixels)
MAX_DETECTION_ENTRIES = 50         # Maximum number of detection entries to keep in memory

# ------------------------------------------------------------------------------
# REGION OF INTEREST (ROI) SETTINGS
# Define areas to focus detection on (crop out irrelevant areas)
# Coordinates are in pixels: (x1, y1) to (x2, y2)
# ------------------------------------------------------------------------------
ROI_COORDINATES = {
    "top": {
        "x1": 370,  # Left boundary - exclude left equipment
        "y1": 0,   # Top boundary - exclude top area
        "x2": 880, # Right boundary - exclude right equipment
        "y2": 720   # Bottom boundary - focus on wood area
    },
    "bottom": {
        "x1": 350,    # Left boundary for bottom camera
        "y1": 0,      # Top boundary
        "x2": 965,   # Right boundary
        "y2": 720     # Bottom boundary
    },
    "wood_detection": {
        "x1": 100,  # Left boundary for wood detection
        "y1": 0,    # Top boundary
        "x2": 500,  # Right boundary for wood detection
        "y2": 300   # Bottom boundary for wood detection
    },
    "exit_wood": {
        "x1": 1175,  # Left boundary for exit wood ROI
        "y1": 0,    # Top boundary
        "x2": 1250, # Right boundary for exit wood ROI
        "y2": 720   # Bottom boundary for exit wood ROI
    }
}

# =============================================================================
# END OF UI CONFIGURATION SECTION
# =============================================================================

class CameraHandler:
    def __init__(self):
        self.top_camera = None
        self.bottom_camera = None
        # Device paths for cameras (Rapoo for top, C922 for bottom)
        self.top_camera_devices = ['/dev/video0','/dev/video1', '/dev/video3']  # Rapoo Camera
        self.bottom_camera_devices = ['/dev/video2', '/dev/video4', '/dev/video5']  # C922 Pro Stream Webcam
        self.top_camera_device = None  # Will be set to successful device path
        self.bottom_camera_device = None  # Will be set to successful device path
        self.top_camera_settings = {
            'brightness': 0,
            'contrast': 32,
            'saturation': 64,
            'hue': 0,
            'exposure': -6,
            'white_balance': 4520,
            'gain': 0
        }
        self.bottom_camera_settings = {
            'brightness': 110,
            'contrast': 125,
            'saturation': 125,
            'hue': 0,
            'exposure': -6,
            'white_balance': 4850,
            'gain': 0,
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
            # Try dynamic camera identification first
            success = self._dynamic_reassign_cameras()
            if success:
                print("Cameras initialized successfully using dynamic identification")
                return

            # Fallback to specific device paths
            print("Dynamic identification failed, trying specific device paths...")
            # Initialize top camera (Rapoo)
            self.top_camera, self.top_camera_device = self._initialize_camera_with_devices(
                self.top_camera_devices, "top")
            if self.top_camera is None:
                raise RuntimeError("Could not open top camera (Rapoo) on any device")

            # Initialize bottom camera (C922)
            self.bottom_camera, self.bottom_camera_device = self._initialize_camera_with_devices(
                self.bottom_camera_devices, "bottom")
            if self.bottom_camera is None:
                self.top_camera.release()
                raise RuntimeError("Could not open bottom camera (C922) on any device")

            # Set resolution and apply settings
            self.top_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.top_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.bottom_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.bottom_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            self._apply_camera_settings(self.top_camera, self.top_camera_settings)
            self._apply_camera_settings(self.bottom_camera, self.bottom_camera_settings)

            print("Cameras initialized successfully at 720p (1280x720)")
            print(f"Top camera (Rapoo): {self.top_camera_device}")
            print(f"Bottom camera (C922): {self.bottom_camera_device}")

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
            camera.set(cv2.CAP_PROP_SHARPNESS, settings['sharpness'])
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
            try:
                self.top_camera.release()
            except Exception as e:
                print(f"Error releasing top camera: {e}")
            self.top_camera = None
        if self.bottom_camera:
            try:
                self.bottom_camera.release()
            except Exception as e:
                print(f"Error releasing bottom camera: {e}")
            self.bottom_camera = None
        print("Cameras released")

# SS-EN 1611-1 Grading Standards Implementation (Revised)
# Grade constants - Individual grades
GRADE_G2_0 = "G2-0"
GRADE_G2_1 = "G2-1"
GRADE_G2_2 = "G2-2"
GRADE_G2_3 = "G2-3"
GRADE_G2_4 = "G2-4"

class SSEN1611_1_PineGrader_Final:
    """
    Implements the appearance grading logic for PINE timber.

    This version excludes:
    1. Spike/Splay Knot checks.
    2. Encased Knot checks (Treated as NOT PERMITTED for all grades).

    The final grade is determined by the worst (lowest) quality face.
    """

    # G2-0 is the highest quality (best), G2-4 is the lowest (worst)
    GRADES = ["G2-0", "G2-1", "G2-2", "G2-3", "G2-4"]

    # ONLY these three types of knots are considered for size/number
    KNOT_TYPES = ["Sound Knots", "Dead Knots", "Unsound/Missing Knots"]

    # --- KNOT SIZE LIMITS (A) ---
    # Note: Encased Knot limits have been removed from the array definition.
    # The order of the tuples must match the order of KNOT_TYPES.
    # (Sound Knots, Dead Knots, Unsound/Missing Knots)
    KNOT_SIZE_LIMITS = {
        # Unsound/Missing NOT PERMITTED (0.00, 0) in G2-0/G2-1
        "G2-0": [(0.10, 10), (0.10, 0), (0.00, 0)],
        "G2-1": [(0.10, 20), (0.10, 10), (0.00, 0)],
        "G2-2": [(0.10, 35), (0.10, 20), (0.10, 15)],
        "G2-3": [(0.10, 50), (0.10, 50), (0.10, 40)],
        "G2-4": [(1.00, 9999), (1.00, 9999), (1.00, 9999)]
    }

    # --- KNOT FREQUENCY LIMITS (C) ---
    # Limits are tuples: (Max Total Knots/m, Max Poor-Quality Knots/m)
    # Poor-Quality Knots are now ONLY Unsound/Missing.
    KNOT_NUMBER_LIMITS = {
        "G2-0": (2, 0), # Max 0 Poor-Quality (Unsound/Missing)
        "G2-1": (4, 0), # Max 0 Poor-Quality (Unsound/Missing) - was 1 for Encased, now 0
        "G2-2": (6, 2),
        "G2-3": (9999, 5),
        "G2-4": (9999, 9999)
    }

    def __init__(self, width_mm):
        self.original_width = width_mm

        # PINE Note B: Size Increase for pieces >= 180 mm wide.
        self.size_increase_mm = 0
        if width_mm >= 180:
            self.size_increase_mm = 10

    def get_max_allowed_size(self, grade, knot_type):
        """Calculates the maximum allowed size for a specific knot type and grade."""
        grade_limits = self.KNOT_SIZE_LIMITS[grade]

        try:
            knot_index = self.KNOT_TYPES.index(knot_type)
        except ValueError:
            # If the user tries to check for an excluded knot type (like "Encased Knots")
            return 0

        percent_limit, absolute_limit = grade_limits[knot_index]
        max_size = (percent_limit * self.original_width) + absolute_limit + self.size_increase_mm

        # For Unsound/Missing in G2-0 and G2-1, the limit is strictly 0.
        if percent_limit == 0.00 and absolute_limit == 0:
             return 0

        return round(max_size)

    def _check_size_compliance(self, knot_data_size):
        """Internal function to check size compliance for a single face."""
        passed_grades = []
        for grade in self.GRADES:
            grade_passed = True
            for knot_type, max_size_found in knot_data_size.items():

                # IMPORTANT: Skip any knot type that is not used in this grading scheme
                if knot_type not in self.KNOT_TYPES:
                    continue

                allowed_size = self.get_max_allowed_size(grade, knot_type)

                if allowed_size == 0 and max_size_found > 0:
                    # Fails due to NOT PERMITTED (Unsound/Missing in G2-0/G2-1, or if a zero size Encased was passed)
                    grade_passed = False
                    break
                elif max_size_found > allowed_size:
                    grade_passed = False
                    break

            if grade_passed:
                passed_grades.append(grade)
        return passed_grades

    def _check_number_compliance(self, knot_data_number):
        """Internal function to check number compliance for a single face."""
        passed_grades = []

        # Note C: Knot Total Number Increase for pieces > 225 mm wide.
        number_increase_factor = 1.0
        if self.original_width > 225:
            number_increase_factor = 1.5

        total_knots_found = knot_data_number.get('total', 0)
        poor_quality_knots_found = knot_data_number.get('unsound_only', 0) # Only counting Unsound/Missing now

        for grade in self.GRADES:
            total_limit, poor_limit = self.KNOT_NUMBER_LIMITS[grade]
            grade_passed = True

            # 1. Check Total Number Limit
            adjusted_total_limit = int(total_limit * number_increase_factor)
            if total_knots_found > adjusted_total_limit:
                grade_passed = False

            # 2. Check Poor-Quality Knots Limit (ONLY Unsound/Missing)
            if grade_passed:
                if poor_quality_knots_found > poor_limit:
                    grade_passed = False

            if grade_passed:
                passed_grades.append(grade)

        return passed_grades

    def _determine_single_face_grade(self, knot_data_size, knot_data_number):
        """Determines the single highest grade achieved by one face."""
        size_passed = self._check_size_compliance(knot_data_size)
        number_passed = self._check_number_compliance(knot_data_number)

        final_grades_passed = [g for g in self.GRADES if g in size_passed and g in number_passed]

        if not final_grades_passed:
            return "Fails G2-4"

        return final_grades_passed[0]

    def determine_final_grade_dual_face(self, top_face_data, bottom_face_data):
        """
        Determines the final grade by finding the worst grade between the two faces.
        :param top_face_data: Dict with keys 'size' and 'number' for the top face.
        :param bottom_face_data: Dict with keys 'size' and 'number' for the bottom face.
        :return: The lowest (worst) grade achieved by either face.
        """
        top_grade = self._determine_single_face_grade(
            top_face_data['size'],
            top_face_data['number']
        )
        bottom_grade = self._determine_single_face_grade(
            bottom_face_data['size'],
            bottom_face_data['number']
        )

        if top_grade == "Fails G2-4" or bottom_grade == "Fails G2-4":
            return "Fails G2-4 (One face failed integrity check)"

        # Get the index of the grade (higher index = worse grade)
        top_index = self.GRADES.index(top_grade)
        bottom_index = self.GRADES.index(bottom_grade)

        # The worst grade is the one with the higher index
        worst_index = max(top_index, bottom_index)

        return self.GRADES[worst_index]

    def convert_measurements_to_knot_data(self, measurements):
        """Convert defect measurements to knot data format expected by PineGrader."""
        knot_data_size = {}
        knot_data_number = {'total': 0, 'unsound_only': 0}

        # Initialize all knot types with 0
        for knot_type in self.KNOT_TYPES:
            knot_data_size[knot_type] = 0

        # Process measurements
        for defect_type, size_mm, percentage in measurements:
            # Map defect types to knot types
            if defect_type in ['Sound_Knot', 'Sound Knot', 'live_knot', 'live knot']:
                knot_type = 'Sound Knots'
            elif defect_type in ['Dead_Knot', 'Dead Knot', 'dead_knot', 'dead knot']:
                knot_type = 'Dead Knots'
            elif defect_type in ['Unsound_Knot', 'Unsound Knot', 'unsound_knot', 'unsound knot',
                                'Crack_Knot', 'Knot with Crack', 'crack_knot', 'knot with crack',
                                'Missing_Knot', 'Missing Knot', 'missing_knot', 'missing knot']:
                knot_type = 'Unsound/Missing Knots'
            else:
                # Default to Unsound for unknown types
                knot_type = 'Unsound/Missing Knots'

            # Update max size for this knot type
            if knot_type in knot_data_size:
                knot_data_size[knot_type] = max(knot_data_size[knot_type], size_mm)

            # Count total knots
            knot_data_number['total'] += 1

            # Count unsound knots
            if knot_type == 'Unsound/Missing Knots':
                knot_data_number['unsound_only'] += 1

        return knot_data_size, knot_data_number

    def determine_surface_grade(self, measurements):
        """Determine grade for a single surface using SS-EN 1611-1 PineGrader."""
        if not measurements:
            return "G2-0"  # Best grade if no defects

        # Convert measurements to knot data format
        knot_data_size, knot_data_number = self.convert_measurements_to_knot_data(measurements)

        # Get grade from PineGrader
        grade = self._determine_single_face_grade(knot_data_size, knot_data_number)

        # Handle "Fails G2-4" case
        if grade == "Fails G2-4":
            return "G2-4"

        return grade

    def determine_final_grade(self, top_measurements, bottom_measurements):
        """Determine final grade using dual-face grading with SS-EN 1611-1 PineGrader."""
        # Convert measurements to knot data format
        top_knot_data_size, top_knot_data_number = self.convert_measurements_to_knot_data(top_measurements)
        bottom_knot_data_size, bottom_knot_data_number = self.convert_measurements_to_knot_data(bottom_measurements)

        # Get final grade from dual-face grading
        final_grade = self.determine_final_grade_dual_face(
            {'size': top_knot_data_size, 'number': top_knot_data_number},
            {'size': bottom_knot_data_size, 'number': bottom_knot_data_number}
        )

        # Handle "Fails G2-4" case
        if "Fails G2-4" in final_grade:
            return "G2-4"

        return final_grade

# Camera-specific calibration based on your setup
# Top camera: 37cm distance, Bottom camera: 29cm distance
# Assuming 1280x720 resolution with typical camera FOV
TOP_CAMERA_DISTANCE_CM = 28
BOTTOM_CAMERA_DISTANCE_CM = 27.5

# Actual pixel-to-millimeter factors (measured)
TOP_CAMERA_PIXEL_TO_MM = 2.96  # Top camera: 2.96 pixels per mm
BOTTOM_CAMERA_PIXEL_TO_MM = 3.18  # Bottom camera: 3.18 pixels per mm

# Dynamic wood pallet width storage - single variable for current wood piece
WOOD_PALLET_WIDTH_MM = 0  # Global variable for current detected wood width

# SS-EN 1611-1 Grading constants for size limits: limit = (0.10 * wood_width) + constant
GRADING_CONSTANTS = {
    "Sound_Knot": {"G2-0": 10, "G2-1": 20, "G2-2": 35, "G2-3": 50},
    "Dead_Knot": {"G2-0": 0, "G2-1": 10, "G2-2": 20, "G2-3": 50},
    "Unsound_Knot": {"G2-2": 15, "G2-3": 40},  # Not permitted in G2-0, G2-1
    "Missing_Knot": {"G2-2": 15, "G2-3": 40},  # Same as Unsound_Knot
    "Crack_Knot": {"G2-2": 15, "G2-3": 40}  # Same as Unsound_Knot for "Knot with Crack"
}

# Knot count limits per meter for Dead and Unsound knots
KNOT_COUNT_LIMITS = {
    "G2-0": 0,
    "G2-1": 1,
    "G2-2": 2,
    "G2-3": 5,
    "G2-4": float('inf')
}




class DetectionDeduplicator:
    """Deduplicates detections based on spatial and temporal proximity for low FPS scenarios"""

    def __init__(self, spatial_threshold_mm=10.0, temporal_threshold_sec=0.5):
        self.spatial_threshold_mm = spatial_threshold_mm  # Max distance to consider same defect
        self.temporal_threshold_sec = temporal_threshold_sec  # Max time gap to group detections

    def deduplicate(self, detections):
        """
        Group detections by spatial-temporal proximity and return best detection from each group
        detections: list of detection dicts with keys: timestamp, defect_type, size_mm, percentage
        """
        if not detections:
            return []

        # Sort detections by timestamp
        sorted_detections = sorted(detections, key=lambda x: x['timestamp'])

        # Group detections into clusters
        clusters = []
        current_cluster = [sorted_detections[0]]

        for detection in sorted_detections[1:]:
            # Check if this detection belongs to the current cluster
            if self._should_merge_with_cluster(detection, current_cluster):
                current_cluster.append(detection)
            else:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = [detection]

        # Don't forget the last cluster
        if current_cluster:
            clusters.append(current_cluster)

        # For each cluster, select the best detection
        deduplicated = []
        for cluster in clusters:
            best_detection = self._select_best_detection(cluster)
            deduplicated.append(best_detection)

        return deduplicated

    def _should_merge_with_cluster(self, detection, cluster):
        """Check if detection should be merged with existing cluster"""
        # Check temporal proximity with the most recent detection in cluster
        last_detection = cluster[-1]
        time_diff = abs(self._timestamp_to_seconds(detection['timestamp']) -
                       self._timestamp_to_seconds(last_detection['timestamp']))

        if time_diff > self.temporal_threshold_sec:
            return False

        # Check spatial proximity with all detections in cluster
        for cluster_detection in cluster:
            if (detection['defect_type'] == cluster_detection['defect_type'] and
                abs(detection['size_mm'] - cluster_detection['size_mm']) <= self.spatial_threshold_mm):
                return True

        return False

    def _select_best_detection(self, cluster):
        """Select the best detection from a cluster (largest size, highest confidence)"""
        if len(cluster) == 1:
            return cluster[0]

        # Select detection with largest size (most conservative for grading)
        best_detection = max(cluster, key=lambda x: x['size_mm'])
        return best_detection

    def _timestamp_to_seconds(self, timestamp_str):
        """Convert ISO timestamp to seconds since epoch"""
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.timestamp()


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
            
            # Store the detected width for this camera
            self.detected_wood_width_mm[camera_name] = detected_width_mm
            
            # CRITICAL: Only use TOP CAMERA as the authoritative source for global wood width
            if camera_name == 'top':
                # TOP CAMERA: Update global wood width (authoritative source)
                WOOD_PALLET_WIDTH_MM = detected_width_mm
                print(f"🎯 Dynamic wood height updated: {detected_width_mm:.1f}mm (from bbox {w}x{h}px, TOP camera - AUTHORITATIVE)")
                print(f"🔗 Synchronization check: detected_width_mm={detected_width_mm:.1f}mm, WOOD_PALLET_WIDTH_MM={WOOD_PALLET_WIDTH_MM:.1f}mm, self.detected_wood_width_mm[{camera_name}]={self.detected_wood_width_mm[camera_name]:.1f}mm")
                
                # Validation: Ensure all variables are exactly equal to detected_width_mm
                self._validate_wood_width_sync(detected_width_mm, camera_name)
            else:
                # BOTTOM CAMERA: Only store locally, do NOT update global width
                print(f"📐 Bottom camera wood width detected: {detected_width_mm:.1f}mm (from bbox {w}x{h}px, camera: {camera_name}) - NOT used for global width")
                print(f"🔒 Global WOOD_PALLET_WIDTH_MM remains: {WOOD_PALLET_WIDTH_MM:.1f}mm (controlled by TOP camera only)")
            
            return detected_width_mm
        
        return 0.0

    def _validate_wood_width_sync(self, detected_width_mm: float, camera_name: str):
        """Validate that all wood width variables are synchronized with detected_width_mm (TOP CAMERA ONLY)"""
        global WOOD_PALLET_WIDTH_MM
        
        # Only validate synchronization for top camera since it's the authoritative source
        if camera_name != 'top':
            return
        
        sync_errors = []
        
        # Check WOOD_PALLET_WIDTH_MM synchronization
        if abs(WOOD_PALLET_WIDTH_MM - detected_width_mm) > 0.001:  # Allow tiny floating point differences
            sync_errors.append(f"WOOD_PALLET_WIDTH_MM={WOOD_PALLET_WIDTH_MM:.3f}mm != detected_width_mm={detected_width_mm:.3f}mm")
            # Force synchronization
            WOOD_PALLET_WIDTH_MM = detected_width_mm
            
        # Check self.detected_wood_width_mm synchronization for top camera
        if abs(self.detected_wood_width_mm[camera_name] - detected_width_mm) > 0.001:
            sync_errors.append(f"self.detected_wood_width_mm[{camera_name}]={self.detected_wood_width_mm[camera_name]:.3f}mm != detected_width_mm={detected_width_mm:.3f}mm")
            # Force synchronization
            self.detected_wood_width_mm[camera_name] = detected_width_mm
        
        if sync_errors:
            print(f"⚠️ Wood width synchronization errors detected and corrected:")
            for error in sync_errors:
                print(f"   - {error}")
            print(f"✅ All variables now synchronized to TOP camera detected_width_mm={detected_width_mm:.1f}mm")
        else:
            print(f"✅ Wood width synchronization validated: TOP camera controls all variables = {detected_width_mm:.1f}mm")

    def get_current_wood_width_mm(self) -> float:
        """Get the current authoritative wood width in mm"""
        global WOOD_PALLET_WIDTH_MM
        return WOOD_PALLET_WIDTH_MM

    def get_wood_width_for_grading(self, fallback_mm: float = 100.0) -> float:
        """Get wood width for grading calculations, with fallback if not detected"""
        global WOOD_PALLET_WIDTH_MM
        width = WOOD_PALLET_WIDTH_MM if WOOD_PALLET_WIDTH_MM > 0 else fallback_mm
        print(f"🔗 Wood width for grading: {width:.1f}mm (WOOD_PALLET_WIDTH_MM={WOOD_PALLET_WIDTH_MM:.1f}mm, fallback={fallback_mm:.1f}mm)")
        return width

    def report_wood_width_status(self, context: str = ""):
        """Report current status of all wood width variables - TOP CAMERA AUTHORITY"""
        global WOOD_PALLET_WIDTH_MM
        
        print(f"\n📊 WOOD WIDTH STATUS REPORT {f'({context})' if context else ''}")
        print(f"   � AUTHORITATIVE SOURCE: Global WOOD_PALLET_WIDTH_MM: {WOOD_PALLET_WIDTH_MM:.1f}mm")
        print(f"   📐 Top camera detected: {self.detected_wood_width_mm.get('top', 'N/A')}mm (CONTROLS GLOBAL)")
        print(f"   � Bottom camera detected: {self.detected_wood_width_mm.get('bottom', 'N/A')}mm (LOCAL ONLY)")
        print(f"   🔗 Authority chain: TOP camera detected_width_mm → WOOD_PALLET_WIDTH_MM → all grading calculations")
        print(f"   ⚠️  Bottom camera measurements are for reference only and DO NOT affect global variables\n")

    def calculate_defect_size(self, detection_box, camera_name="top"):
        """Calculate defect size in mm and percentage from detection bounding box - matches testIR.py"""
        global WOOD_PALLET_WIDTH_MM, TOP_CAMERA_PIXEL_TO_MM, BOTTOM_CAMERA_PIXEL_TO_MM
        
        try:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = detection_box['bbox']

            # Calculate defect dimensions in pixels
            width_px = abs(x2 - x1)   # Horizontal dimension (X-axis)
            height_px = abs(y2 - y1) # Vertical dimension (Y-axis)

            # For wood width measurement, use the vertical dimension (height_px) because camera is in landscape
            # Wood runs vertically in the camera view, so Y-axis represents wood width
            defect_size_px = height_px

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
                  f"-> defect_size_px={defect_size_px:.1f} (using Y-axis) -> size_mm={size_mm:.1f}")

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
                
                # Report wood width status to verify synchronization
                self.report_wood_width_status(f"after {camera} detection")
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
    
    def visualize_detection(self, image: np.ndarray, detection_result: Dict, output_path: str = None) -> np.ndarray:
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
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"💾 Visualization saved to: {output_path}")
        
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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wood Sorting Application")

        # Get screen dimensions for dynamic sizing
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Calculate window size based on configuration
        if ENABLE_FULLSCREEN_STARTUP:
            self.attributes("-fullscreen", True)
            self.is_fullscreen = True
            window_width = screen_width
            window_height = screen_height
        else:
            window_width = int(screen_width * WINDOW_SCALE)
            window_height = int(screen_height * WINDOW_SCALE)

            # Center the window on screen
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2

            # Set geometry with calculated dimensions
            self.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Make window resizable
        self.resizable(True, True)

        # Set minimum size to prevent too small windows
        self.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        # For Raspberry Pi - detect if running in fullscreen environment
        self.bind("<F11>", self.toggle_fullscreen)
        self.bind("<Escape>", self.exit_fullscreen)

        # Auto-fullscreen for Raspberry Pi (configurable)
        if ENABLE_FULLSCREEN_STARTUP:
            self.after(100, self.auto_fullscreen_rpi)

        # Calculate responsive font sizes based on screen size
        base_font_size = max(FONT_SIZE_BASE_MIN, min(FONT_SIZE_BASE_MAX, int(screen_height / FONT_SIZE_DIVISOR)))
        self.font_small = (PRIMARY_FONT_FAMILY, base_font_size - 1)
        self.font_normal = (PRIMARY_FONT_FAMILY, base_font_size)
        self.font_large = (PRIMARY_FONT_FAMILY, base_font_size + 2, "bold")
        self.font_button = (BUTTON_FONT_FAMILY, base_font_size, "bold")  # Button font

        # Configure styles for white margins and custom button colors
        style = ttk.Style()
        style.configure("White.TFrame", background="white")

        # Configure custom button style with colors from UI config
        style.configure("Custom.TButton",
                       background=BUTTON_BACKGROUND_COLOR,
                       foreground=BUTTON_TEXT_COLOR,
                       font=self.font_button)
        style.map("Custom.TButton",
                 background=[("active", BUTTON_ACTIVE_COLOR),
                           ("pressed", BUTTON_ACTIVE_COLOR)])

        # Helper method for updating status label (Text widget)
        def update_status_text(text, color=None):
            self.status_label.config(state=tk.NORMAL)
            self.status_label.delete(1.0, tk.END)
            self.status_label.insert(1.0, text)
            if color:
                self.status_label.config(foreground=color)
            self.status_label.config(state=tk.DISABLED)

        self.update_status_text = update_status_text

        # Create message queue for thread communication
        self.message_queue = queue.Queue()

        # Initialize variables that might be accessed early by message processing
        self.total_pieces_processed = 0
        self.session_start_time = time.time()  # Track session start time for statistics
        self.grade_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # Arduino commands: 1=G2-0, 2=G2-1, 3=G2-2, 4=G2-3, 5=G2-4
        self.report_generated = False
        self.last_report_path = None
        self.last_activity_time = time.time()
        self.live_stats = {"grade1": 0, "grade2": 0, "grade3": 0, "grade4": 0, "grade5": 0}  # Arduino commands
        self._shutting_down = False  # Flag to indicate shutdown in progress
        self.session_log = [] # New: Log for individual piece details
        self._camera_check_cooldown = 0  # Timestamp to skip camera checks after mode changes


        # Cache for preventing unnecessary UI updates
        self._last_detection_content = {"top": "", "bottom": ""}
        self._last_stats_content = ""
        self._user_scrolling = {"top": False, "bottom": False}
        self._user_scrolling_stats = False
        self._scroll_positions = {"top": 0.0, "bottom": 0.0}
        
        # Test case tracking system
        self.test_case_counter = 0
        self.current_test_case = None
        self.detection_log = []
        self.test_cases_data = {}

        # Disconnection popup flags
        self.camera_disconnected_popup_shown = False
        self.arduino_disconnected_popup_shown = False
        
        # Detection tracking variables
        self.detection_session_id = None
        self.piece_counter = 0
        self.current_piece_data = None
        
        # System mode tracking
        self.current_mode = "IDLE"  # Can be "IDLE", "TRIGGER", "CONTINUOUS", or "SCAN_PHASE"

        # SCAN_PHASE mode variables
        self.scan_phase_active = False
        self.current_wood_number = 0
        self.scan_session_data = {}
        self.captured_frames = {"top": [], "bottom": []}
        self.segment_defects = {"top": [], "bottom": []}
        self.scan_session_start_time = None
        self.scan_session_folder = None

        # Automatic detection state (triggered by IR beam)
        self.live_detection_var = tk.BooleanVar(value=False) # For live inference mode
        self.auto_grade_var = tk.BooleanVar(value=False) # For auto grading in live mode
        self._last_auto_grade_time = 0
        self._in_active_inference = False  # Flag to prevent UI conflicts during inference
        self.has_current_grade = False  # Track if we have a current grade to display

        self.auto_detection_active = False
        self.detection_frames = []  # Store frames during detection
        self.detection_session_data = {
            "start_time": None,
            "end_time": None,
            "total_detections": {"top": [], "bottom": []},
            "best_frames": {"top": None, "bottom": None},
            "final_grade": None
        }

        # Store all detections throughout the session for deduplication
        self.session_detections = {"top": [], "bottom": []}
        self.final_deduplicated_defects = {"top": [], "bottom": []}

        # Flag to control processed frame display in SCAN_PHASE
        self.displaying_processed_frame = False
        self.processed_frame_timer = None  # Timer for processed frame display duration

        # --- DeGirum Model and Camera Initialization ---
        # DeGirum Configuration
        self.inference_host_address = "@local"
        self.zoo_url = "/home/inspectura/Desktop/WoodSortingApplication/models/V2DefectCombined--640x640_quant_hailort_hailo8_1"
        self.model_name = "V2DefectCombined--640x640_quant_hailort_hailo8_1"
        
        # Load DeGirum model
        try:
            self.model = dg.load_model(
                model_name=self.model_name,
                inference_host_address=self.inference_host_address,
                zoo_url=self.zoo_url
            )
            print("DeGirum model loaded successfully.")
        except Exception as e:
            print(f"Error loading DeGirum model: {e}")
            messagebox.showerror("Model Error", f"Failed to load DeGirum model: {e}")
            self.model = None
        

        # Initialize detection deduplicator for low FPS scenarios
        self.deduplicator = DetectionDeduplicator(spatial_threshold_mm=15.0, temporal_threshold_sec=1.0)

        # Initialize Camera Handler with optimized settings
        self.camera_handler = CameraHandler()
        try:
            self.camera_handler.initialize_cameras()
            self.cap_top = self.camera_handler.top_camera
            self.cap_bottom = self.camera_handler.bottom_camera

            # Store camera resolution for display scaling
            self.camera_width = 1280
            self.camera_height = 720

            print("Camera handler initialized with optimized settings")
        except RuntimeError as e:
            print(f"Camera initialization failed: {e}")
            self.cap_top = None
            self.cap_bottom = None

        # Initialize RGB Wood Detector for dynamic ROI generation
        self.rgb_wood_detector = ColorWoodDetector()
        print("RGB Wood Detector initialized for dynamic ROI generation")

        # Initialize dynamic ROI storage
        self.dynamic_roi = {}
        self.detected_wood_width_mm = {"top": None, "bottom": None}

        # Store wood detection results for visualization
        self.wood_detection_results = {"top": None, "bottom": None}

        # Live detection tracking
        self.live_detections = {"top": {}, "bottom": {}}
        self.live_grades = {"top": "", "bottom": ""}

        # ROI (Region of Interest) settings
        self.roi_enabled = {"top": True, "bottom": True, "wood_detection": True, "exit_wood": True}  # Enable ROI for both cameras and wood detection
        self.roi_coordinates = ROI_COORDINATES.copy()

        # UI colors
        self.roi_overlay_color = ROI_OVERLAY_COLOR

        # Create canvases for camera feeds maintaining dynamic width and fixed height of 360 pixels
        self.canvas_width = screen_width // 2 - 25
        self.canvas_height = 360
        self.top_canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.top_canvas.place(x=25, y=25, width=self.canvas_width, height=self.canvas_height)

        self.bottom_canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.bottom_canvas.place(x=self.canvas_width + 50, y=25, width=self.canvas_width, height=self.canvas_height)

        # Initialize canvas images
        self._top_photo = None
        self._bottom_photo = None

        # Place control frames at specific positions
        # Status panel under top camera
        status_frame = ttk.LabelFrame(self, text="System Status", padding=FRAME_PADDING)
        status_frame.place(x=25, y=415, width=640, height=125)

        style = ttk.Style()
        frame_bg = style.lookup("TLabelFrame", "background") or "#f0f0f0"
        self.status_label = tk.Text(status_frame, font=self.font_normal, wrap=tk.WORD,
                                   height=3, width=25, state=tk.DISABLED, relief="flat",
                                   background=frame_bg)
        self.status_label.pack(pady=LABEL_PADDING, fill="x", expand=False)
        self.status_label.insert(1.0, "Status: Initializing...")

        # ROI panel under bottom camera
        roi_frame = ttk.LabelFrame(self, text="ROI", padding=FRAME_PADDING)
        roi_frame.place(x=675, y=415, width=250, height=125)

        self.roi_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(roi_frame, text="Top ROI", variable=self.roi_var,
                        command=self.toggle_roi).pack(anchor="w")

        self.bottom_roi_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(roi_frame, text="Bottom ROI", variable=self.bottom_roi_var,
                        command=self.toggle_bottom_roi).pack(anchor="w")

        # Conveyor Control (place next to ROI)
        control_frame = ttk.LabelFrame(self, text="Conveyor Control", padding=FRAME_PADDING)
        control_frame.place(x=935, y=415, width=655, height=125)

        tk.Button(control_frame, text="ON",
                  command=self.set_scan_mode, bg=BUTTON_BACKGROUND_COLOR,
                  fg=BUTTON_TEXT_COLOR, activebackground=BUTTON_ACTIVE_COLOR,
                  font=self.font_button, relief="raised", borderwidth=2).place(x=0, y=0, width=327, height=60)
        tk.Button(control_frame, text="OFF",
                  command=self.set_idle_mode, bg=BUTTON_BACKGROUND_COLOR,
                  fg=BUTTON_TEXT_COLOR, activebackground=BUTTON_ACTIVE_COLOR,
                  font=self.font_button, relief="raised", borderwidth=2).place(x=328, y=0, width=327, height=60)
        tk.Button(control_frame, text="CLOSE",
                  command=self.on_closing, bg="#ff4444",
                  fg="white", activebackground="#cc0000",
                  font=self.font_button, relief="raised", borderwidth=2).place(x=0, y=65, width=655, height=60)

        # Reports panel at fixed position (no overlap) - increased height for more buttons
        REPORT_W, REPORT_H = 300, 200
        REPORT_X, REPORT_Y = 1600, 415
        reports_frame = ttk.LabelFrame(self, text="Reports", padding=FRAME_PADDING)
        reports_frame.place(x=REPORT_X, y=REPORT_Y, width=REPORT_W, height=REPORT_H)

        self.log_status_label = ttk.Label(reports_frame, text="Log: Ready",
                                        foreground=STATUS_READY_COLOR, font=self.font_small)
        self.log_status_label.pack()

        tk.Button(reports_frame, text="Generate Report",
                  command=self.manual_generate_report, bg=BUTTON_BACKGROUND_COLOR,
                  fg=BUTTON_TEXT_COLOR, activebackground=BUTTON_ACTIVE_COLOR,
                  font=self.font_button, relief="raised", borderwidth=2).pack(pady=ELEMENT_PADDING_Y)

        tk.Button(reports_frame, text="View Session Folder",
                  command=self.view_session_folder, bg="#4444ff",
                  fg="white", activebackground="#0000cc",
                  font=self.font_button, relief="raised", borderwidth=2).pack(pady=ELEMENT_PADDING_Y)

        self.show_report_notification = tk.BooleanVar(value=True)
        ttk.Checkbutton(reports_frame, text="Notifications",
                       variable=self.show_report_notification).pack()

        self.last_report_label = ttk.Label(reports_frame, text="Last: None",
                                          font=self.font_small, wraplength=100)
        self.last_report_label.pack()


        # Statistics section full width at bottom
        stats_frame = ttk.LabelFrame(self, text="Statistics", padding=FRAME_PADDING)
        stats_frame.place(x=0, rely=1.0, relwidth=1, height=500, anchor="sw")

        # Create notebook for tabbed statistics
        self.stats_notebook = ttk.Notebook(stats_frame, height=STATS_TAB_HEIGHT - 40)  # Account for padding and tab headers
        self.stats_notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Tab 1: Grade Summary (Overview with Live Grading)
        grade_summary_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(grade_summary_tab, text="Grade Summary")


        # Live Grading Section (includes both individual grades and grade counts)
        live_grading_frame = ttk.LabelFrame(grade_summary_tab, text="Live Grading Results", padding="10")
        live_grading_frame.pack(fill="both", expand=True, pady=(10, 5), padx=10)

        # Configure the main frame for proper layout
        live_grading_frame.grid_rowconfigure(0, weight=0)  # Individual grades row
        live_grading_frame.grid_rowconfigure(1, weight=1)  # Grade counts row
        live_grading_frame.grid_columnconfigure(0, weight=1)

        # Row 0: Individual camera grades (horizontal layout)
        individual_grades_frame = ttk.Frame(live_grading_frame)
        individual_grades_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        # Configure individual grades frame
        individual_grades_frame.grid_columnconfigure(0, weight=3)  # Top camera
        individual_grades_frame.grid_columnconfigure(1, weight=3)  # Bottom camera
        individual_grades_frame.grid_columnconfigure(2, weight=4)  # Combined grade

        # Individual camera grades (horizontal layout)
        top_grade_container = ttk.Frame(individual_grades_frame)
        top_grade_container.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Label(top_grade_container, text="Top Camera:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.top_grade_label = ttk.Label(top_grade_container, text="No Wood Graded",
                                        foreground="gray", font=self.font_small)
        self.top_grade_label.pack(anchor="w")

        bottom_grade_container = ttk.Frame(individual_grades_frame)
        bottom_grade_container.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        ttk.Label(bottom_grade_container, text="Bottom Camera:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.bottom_grade_label = ttk.Label(bottom_grade_container, text="No Wood Graded",
                                           foreground="gray", font=self.font_small)
        self.bottom_grade_label.pack(anchor="w")

        # Combined grade (prominent display, takes more space)
        combined_container = ttk.Frame(individual_grades_frame)
        combined_container.grid(row=0, column=2, sticky="ew")
        ttk.Label(combined_container, text="Final Grade:", font=("Arial", 12, "bold")).pack(anchor="w")
        self.combined_grade_label = ttk.Label(combined_container, text="No Wood Graded",
                                             font=("Arial", 11, "bold"), foreground="gray")
        self.combined_grade_label.pack(anchor="w")

        # Row 1: Grade counts container with dynamic resizing
        grade_counts_container = ttk.Frame(live_grading_frame)
        grade_counts_container.grid(row=1, column=0, sticky="nsew")

        # Configure the grade counts container for dynamic resizing
        grade_counts_container.grid_rowconfigure(0, weight=1)
        grade_counts_container.grid_columnconfigure(0, weight=1)
        grade_counts_container.grid_columnconfigure(1, weight=1)
        grade_counts_container.grid_columnconfigure(2, weight=1)
        grade_counts_container.grid_columnconfigure(3, weight=1)
        grade_counts_container.grid_columnconfigure(4, weight=1)

        # Initialize stats labels with consistent spacing and fonts
        self.live_stats_labels = {}
        grade_info = [
            ("grade1", "G2-0", GRADE_PERFECT_COLOR),
            ("grade2", "G2-1", GRADE_GOOD_COLOR),
            ("grade3", "G2-2", GRADE_FAIR_COLOR),
            ("grade4", "G2-3", GRADE_FAIR_COLOR),
            ("grade5", "G2-4", GRADE_POOR_COLOR)
        ]

        for i, (grade_key, label_text, color) in enumerate(grade_info):
            # Create a container frame that can expand
            grade_container = ttk.Frame(grade_counts_container, relief="solid", borderwidth=2)
            grade_container.grid(row=0, column=i, sticky="nsew", padx=2, pady=5)
            grade_container.grid_columnconfigure(0, weight=1)
            grade_container.grid_rowconfigure(0, weight=1)
            grade_container.grid_rowconfigure(1, weight=1)

            # Create a consistent inner frame for better control
            inner_frame = ttk.Frame(grade_container)
            inner_frame.grid(row=0, column=0, sticky="nsew", rowspan=2)
            inner_frame.grid_columnconfigure(0, weight=1)
            inner_frame.grid_rowconfigure(0, weight=1)
            inner_frame.grid_rowconfigure(1, weight=1)

            # Title label with responsive font - centered
            title_label = ttk.Label(inner_frame, text=label_text, font=("Arial", 12, "bold"),
                                   justify="center", wraplength=80, anchor="center")
            title_label.grid(row=0, column=0, sticky="ew", padx=2, pady=(8, 2))

            # Count label with responsive font and consistent positioning - centered
            self.live_stats_labels[grade_key] = ttk.Label(inner_frame, text="0",
                                                          foreground=color, font=("Arial", 32, "bold"),
                                                          anchor="center")
            self.live_stats_labels[grade_key].grid(row=1, column=0, sticky="ew", padx=2, pady=(2, 8))

        # Tab 2: Grading Details
        grading_details_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(grading_details_tab, text="Grading Details")

        # Grading details content
        self.grading_details_frame = ttk.Frame(grading_details_tab)
        self.grading_details_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.update_grade_details_tab()

        # Tab 3: Performance Metrics
        performance_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(performance_tab, text="Performance")

        # Performance metrics content
        self.performance_frame = ttk.Frame(performance_tab)
        self.performance_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.update_performance_tab()
        
        # Tab 4: Recent Activity
        activity_tab = ttk.Frame(self.stats_notebook)
        self.stats_notebook.add(activity_tab, text="Recent Activity")

        # Main container with two sections
        activity_main_container = ttk.Frame(activity_tab)
        activity_main_container.pack(fill="both", expand=True, padx=5, pady=5)

        # Session Summary Section (wider, fixed height)
        self.session_summary_frame = ttk.LabelFrame(activity_main_container, text="Current Session Summary", padding="10")
        self.session_summary_frame.place(x=0, y=0, relwidth=1, height=70)

        # Processing Log Section (scrollable)
        log_container = ttk.LabelFrame(activity_main_container, text="Recent Processing Log", padding="5")
        log_container.place(x=0, y=70, relwidth=1, relheight=1)

        # Scrollable canvas for processing log
        log_canvas = tk.Canvas(log_container, height=LOG_SCROLLABLE_HEIGHT)  # Configurable height
        log_scrollbar = ttk.Scrollbar(log_container, orient="vertical", command=log_canvas.yview)
        self.processing_log_frame = ttk.Frame(log_canvas)

        self.processing_log_frame.bind(
            "<Configure>",
            lambda e: log_canvas.configure(scrollregion=log_canvas.bbox("all"))
        )

        def on_log_scroll(event):
            self._user_scrolling_log = True
            if hasattr(self, '_log_scroll_timer'):
                self.after_cancel(self._log_scroll_timer)
            self._log_scroll_timer = self.after(3000, lambda: setattr(self, '_user_scrolling_log', False))

        log_canvas.bind("<MouseWheel>", on_log_scroll)

        # Bind scrollbar interactions
        log_scrollbar.bind("<ButtonPress-1>", lambda e: setattr(self, '_user_scrolling_log', True))

        log_canvas.create_window((0, 0), window=self.processing_log_frame, anchor="nw")
        log_canvas.configure(yscrollcommand=log_scrollbar.set)

        log_canvas.place(x=0, y=0, relwidth=0.9, relheight=1)
        log_scrollbar.place(relx=0.9, y=0, relwidth=0.1, relheight=1)

        # Store canvas references for updates
        self.log_canvas = log_canvas
        self.activity_canvas = log_canvas  # Keep for compatibility

        # Initialize scroll state variables
        self._user_scrolling_log = False
        self._last_defects_files_count = 0  # Track number of defects files for change detection

        # Create simplified detection tracking (retain logic without complex UI)
        self.top_dashboard_widgets = self.create_simple_detection_tracker("top")
        self.bottom_dashboard_widgets = self.create_simple_detection_tracker("bottom")
        
        # Keep compatibility with existing code
        self.top_details = None
        self.bottom_details = None
        self.top_details_widgets = None
        self.bottom_details_widgets = None

        # Initialize live statistics display
        self.update_live_stats_display()

        # Initialize tab content
        self.update_grade_details_tab()
        self.update_performance_tab()
        self.update_recent_activity_tab()

        # Removed grading_status_label as requested

        # --- Arduino Communication ---
        self.setup_arduino()

        # Start the video feed update loop
        self.update_feeds()

        # Start processing messages from background threads
        self.process_message_queue()

        # --- Inactivity and Reporting ---
        self.check_inactivity()

        # Set the action for when the window is closed
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_live_grading_display(self):
        if self.current_mode == "IDLE":
            self.top_grade_label.config(text="No Wood Graded", foreground="gray")
            self.bottom_grade_label.config(text="No Wood Graded", foreground="gray")
            self.combined_grade_label.config(text="No Wood Graded", foreground="gray")
        else:
            # Show current grades if available
            top_grade = self.live_grades.get("top", "No Wood Graded")
            bottom_grade = self.live_grades.get("bottom", "No Wood Graded")
            if isinstance(top_grade, dict):
                top_text = top_grade.get("text", "No Wood Graded")
                top_color = top_grade.get("color", "gray")
            else:
                top_text = top_grade
                top_color = "gray"
            if isinstance(bottom_grade, dict):
                bottom_text = bottom_grade.get("text", "No Wood Graded")
                bottom_color = bottom_grade.get("color", "gray")
            else:
                bottom_text = bottom_grade
                bottom_color = "gray"
            self.top_grade_label.config(text=top_text, foreground=top_color)
            self.bottom_grade_label.config(text=bottom_text, foreground=bottom_color)
            combined_info = self.live_grades.get("combined")
            if combined_info:
                combined_text = combined_info.get("text", "No Wood Graded")
                combined_color = combined_info.get("color", "gray")
            else:
                combined_text = self.live_grades.get("combined", "No Wood Graded")
                combined_color = "gray"
            self.combined_grade_label.config(text=combined_text, foreground=combined_color)

    def calibrate_pixel_to_mm(self, reference_object_width_px, reference_object_width_mm, camera_name="top"):
        """Calibrate the pixel-to-millimeter conversion factor for specific camera"""
        global TOP_CAMERA_PIXEL_TO_MM, BOTTOM_CAMERA_PIXEL_TO_MM
        
        conversion_factor = reference_object_width_mm / reference_object_width_px
        
        if camera_name == "top":
            TOP_CAMERA_PIXEL_TO_MM = conversion_factor
            print(f"Calibrated TOP camera pixel-to-mm factor: {TOP_CAMERA_PIXEL_TO_MM}")
        else:  # bottom camera
            BOTTOM_CAMERA_PIXEL_TO_MM = conversion_factor
            print(f"Calibrated BOTTOM camera pixel-to-mm factor: {BOTTOM_CAMERA_PIXEL_TO_MM}")
        
        return conversion_factor

    def calibrate_with_wood_pallet(self, wood_pallet_width_px_top, wood_pallet_width_px_bottom):
        """Auto-calibrate both cameras using the known wood pallet width"""
        print(f"Auto-calibrating cameras with {WOOD_PALLET_WIDTH_MM}mm wood pallet...")

        top_factor = self.calibrate_pixel_to_mm(wood_pallet_width_px_top, WOOD_PALLET_WIDTH_MM, "top")
        bottom_factor = self.calibrate_pixel_to_mm(wood_pallet_width_px_bottom, WOOD_PALLET_WIDTH_MM, "bottom")
        
        print(f"Calibration complete:")
        print(f"  Top camera (37cm): {top_factor:.4f} mm/pixel")
        print(f"  Bottom camera (29cm): {bottom_factor:.4f} mm/pixel")
        
        return top_factor, bottom_factor

    def map_model_output_to_standard(self, model_label):
        """Map your model's output labels to standardized defect types"""
        # Mapping from your model's actual output labels to standard categories
        label_mapping = {
            # Your actual model outputs (case-insensitive)
            "dead knots": "Dead_Knot",
            "knots with crack": "Unsound_Knot",
            "live knots": "Sound_Knot",
            "missing knots": "Missing_Knot",
            # Variations and alternatives
            "dead_knots": "Dead_Knot",
            "knots_with_crack": "Crack_Knot",
            "live_knots": "Sound_Knot",
            "missing_knots": "Missing_Knot",
            # Legacy mappings for backward compatibility
            "sound_knots": "Sound_Knot",
            "dead_knots": "Dead_Knot",
            "unsound_knots": "Unsound_Knot",
            "sound knots": "Sound_Knot",
            "dead knots": "Dead_Knot",
            "unsound knots": "Unsound_Knot",
            "knot with crack": "Unsound_Knot",
            "live_knot": "Sound_Knot",
            "dead_knot": "Dead_Knot",
            "missing_knot": "Missing_Knot",
            "crack_knot": "Unsound_Knot",
            "live knot": "Sound_Knot",
            "knot with crack": "Unsound_Knot",
            # Generic fallback
            "knot": "Unsound_Knot"
        }

        # Normalize the label (lowercase, remove extra spaces)
        normalized_label = model_label.lower().strip().replace('_', ' ')

        # Return mapped label or default to unsound knot
        return label_mapping.get(normalized_label, "Unsound_Knot")

    def calculate_defect_size(self, detection_box, camera_name="top"):
        """Calculate defect size in mm and percentage from detection bounding box"""
        try:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = detection_box['bbox']

            # Calculate defect dimensions in pixels
            width_px = abs(x2 - x1)   # Horizontal dimension (X-axis)
            height_px = abs(y2 - y1) # Vertical dimension (Y-axis)

            # For wood width measurement, use the vertical dimension (height_px) because camera is in landscape
            # Wood runs vertically in the camera view, so Y-axis represents wood width
            defect_size_px = height_px

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
                  f"-> defect_size_px={defect_size_px:.1f} (using Y-axis) -> size_mm={size_mm:.1f}")

            return size_mm, percentage

        except Exception as e:
            print(f"Error calculating defect size: {e}")
            # Return conservative values if calculation fails
            return 50.0, 35.0  # Assumes large defect for safety

    def _convert_measurements_to_face_data(self, measurements):
        """Convert app measurements to PineGrader face data format"""
        knot_data_size = {}
        knot_data_number = {'total': 0, 'unsound_only': 0}

        # Map defect types to PineGrader format
        type_mapping = {
            "Sound_Knot": "Sound Knots",
            "Dead_Knot": "Dead Knots",
            "Unsound_Knot": "Unsound/Missing Knots",
            "Missing_Knot": "Unsound/Missing Knots",
            "Crack_Knot": "Unsound/Missing Knots"
        }

        for defect_type, size_mm, _ in measurements:
            mapped_type = type_mapping.get(defect_type, defect_type)
            if mapped_type not in knot_data_size:
                knot_data_size[mapped_type] = size_mm
            else:
                knot_data_size[mapped_type] = max(knot_data_size[mapped_type], size_mm)

            knot_data_number['total'] += 1
            if mapped_type == "Unsound/Missing Knots":
                knot_data_number['unsound_only'] += 1

        # Ensure all knot types are present
        for kt in ["Sound Knots", "Dead Knots", "Unsound/Missing Knots"]:
            if kt not in knot_data_size:
                knot_data_size[kt] = 0

        return {'size': knot_data_size, 'number': knot_data_number}

    def determine_surface_grade(self, defect_measurements):
        """
        Determine overall surface grade based on worst knot size and knot count.
        defect_measurements: list of tuples [("Sound_Knot", size_mm), ...]
        """
        if not defect_measurements:
            return GRADE_G2_0

        # Check if wood height has been measured
        if WOOD_PALLET_WIDTH_MM <= 0:
            return GRADE_G2_4  # Cannot grade without wood dimensions

        # 1. Grade based on the size of the worst individual knot
        worst_grade_by_size = "G2-0"
        grade_order = ["G2-0", "G2-1", "G2-2", "G2-3", "G2-4"]

        dead_or_unsound_count = 0

        for defect_type, defect_size_mm, _ in defect_measurements:
            # Tally knots for count constraint
            if defect_type in ["Dead_Knot", "Unsound_Knot"]:
                dead_or_unsound_count += 1

            # Get individual knot grade
            knot_grade = self.get_individual_knot_grade(defect_type, defect_size_mm, WOOD_PALLET_WIDTH_MM)

            # Check if this knot's grade is worse than the current worst
            if grade_order.index(knot_grade) > grade_order.index(worst_grade_by_size):
                worst_grade_by_size = knot_grade

        # 2. Grade based on the count of Dead and Unsound knots
        grade_by_count = "G2-0"
        if dead_or_unsound_count > 5:
            grade_by_count = "G2-4"
        elif dead_or_unsound_count > 2:
            grade_by_count = "G2-3"
        elif dead_or_unsound_count > 1:
            grade_by_count = "G2-2"
        elif dead_or_unsound_count > 0:
            grade_by_count = "G2-1"

        # 3. The final grade for the surface is the WORST of the two criteria
        final_grade_index = max(grade_order.index(worst_grade_by_size), grade_order.index(grade_by_count))
        final_grade = grade_order[final_grade_index]

        # 4. Return the actual worst grade found (G2-0, G2-1, G2-2, G2-3, or G2-4)
        return final_grade

    def determine_final_grade(self, top_grade, bottom_grade):
        """Determine final grade based on worst surface (SS-EN 1611-1 standard)"""
        grade_hierarchy = [GRADE_G2_0, GRADE_G2_1, GRADE_G2_2, GRADE_G2_3, GRADE_G2_4]

        # Handle None values (no detection)
        if top_grade is None:
            top_grade = GRADE_G2_0
        if bottom_grade is None:
            bottom_grade = GRADE_G2_0

        # Get indices for comparison
        top_index = grade_hierarchy.index(top_grade) if top_grade in grade_hierarchy else 0
        bottom_index = grade_hierarchy.index(bottom_grade) if bottom_grade in grade_hierarchy else 0

        # Return the worse grade (higher index)
        final_grade = grade_hierarchy[max(top_index, bottom_index)]

        print(f"Final grading: Top={top_grade}, Bottom={bottom_grade}, Final={final_grade}")
        return final_grade

    def convert_grade_to_arduino_command(self, standard_grade):
        """Convert SS-EN 1611-1 grade to Arduino sorting command"""
        # Map individual grades to sorting gates:
        grade_to_command = {
            GRADE_G2_0: 1,    # Perfect (G2-0) - Gate 1
            GRADE_G2_1: 2,    # Good (G2-1) - Gate 2
            GRADE_G2_2: 2,    # Fair (G2-2) - Gate 2
            GRADE_G2_3: 2,    # Poor (G2-3) - Gate 2
            GRADE_G2_4: 3     # Reject (G2-4) - Gate 3
        }

        return grade_to_command.get(standard_grade, 3)  # Default to worst gate if unknown

    def get_grade_color(self, grade):
        """Get color coding for grades"""
        color_map = {
            GRADE_G2_0: 'dark green',
            GRADE_G2_1: 'green',
            GRADE_G2_2: 'orange',
            GRADE_G2_3: 'red',
            GRADE_G2_4: 'dark red'
        }
        return color_map.get(grade, 'gray')

    def create_section(self, parent, title, col):
        section_frame = ttk.LabelFrame(parent, text=title, padding="10")
        section_frame.grid(row=0, column=col, sticky="nsew", padx=5, pady=5)
        
        # Fixed proportions to prevent shrinking
        section_frame.grid_rowconfigure(0, weight=3, minsize=200)  # Live feed gets most space with minimum
        section_frame.grid_rowconfigure(1, weight=0, minsize=150)  # Details get fixed space
        section_frame.grid_columnconfigure(0, weight=1, minsize=350)  # Minimum width

        # Live feed area - now takes up most of the space with minimum size
        live_feed_label = ttk.Label(section_frame, background="black", text="Live Feed")
        live_feed_label.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

        # Details area with proper wrapping, formatting and scrolling capability
        details_frame = ttk.Frame(section_frame)
        details_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        details_frame.grid_rowconfigure(0, weight=1)
        details_frame.grid_columnconfigure(0, weight=1)
        
        # Create scrollable text widget for details
        details_text = tk.Text(details_frame, wrap=tk.WORD, height=8, width=40, 
                              font=self.font_small, state=tk.DISABLED,
                              relief="sunken", borderwidth=1)
        details_scrollbar = ttk.Scrollbar(details_frame, orient="vertical", command=details_text.yview)
        details_text.configure(yscrollcommand=details_scrollbar.set)
        
        # Track user scrolling to prevent auto-updates during manual scrolling
        camera_name = "top" if col == 0 else "bottom"
        
        def on_scroll_start(event):
            self._user_scrolling[camera_name] = True
            # Clear the auto-scroll timer if it exists
            if hasattr(self, f'_scroll_timer_{camera_name}'):
                self.after_cancel(getattr(self, f'_scroll_timer_{camera_name}'))
        
        def on_scroll_end():
            # Allow updates again after 3 seconds of no scrolling
            timer_id = self.after(3000, lambda: self._user_scrolling.update({camera_name: False}))
            setattr(self, f'_scroll_timer_{camera_name}', timer_id)
        
        def on_scroll_event(event):
            on_scroll_start(event)
            on_scroll_end()
            # Store current scroll position
            self._scroll_positions[camera_name] = details_text.yview()[0]
        
        # Bind scroll events
        details_text.bind("<Button-4>", on_scroll_event)  # Mouse wheel up
        details_text.bind("<Button-5>", on_scroll_event)  # Mouse wheel down
        details_text.bind("<MouseWheel>", on_scroll_event)  # Windows mouse wheel
        details_scrollbar.bind("<ButtonPress-1>", on_scroll_start)
        details_scrollbar.bind("<B1-Motion>", on_scroll_event)
        
        details_text.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        details_scrollbar.grid(row=0, column=1, sticky="ns")
        
        details_frame.grid_columnconfigure(0, weight=1)
        details_frame.grid_columnconfigure(1, weight=0)

        return live_feed_label, None, details_text

    def create_detection_details_section(self, parent, title, camera_name):
        """Create an object-based detection details section that updates efficiently"""
        frame = ttk.LabelFrame(parent, text=title, padding="5")
        
        # Create canvas and scrollbar for scrolling
        canvas = tk.Canvas(frame, height=150)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind mouse wheel with camera-specific scrolling detection
        def _on_mousewheel(event):
            self._user_scrolling[camera_name] = True
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            # Reset scroll flag after 3 seconds
            timer_id = self.after(3000, lambda: self._user_scrolling.update({camera_name: False}))
        
        canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # Bind scrollbar interactions
        scrollbar.bind("<ButtonPress-1>", lambda e: self._user_scrolling.update({camera_name: True}))
        scrollbar.bind("<B1-Motion>", lambda e: self._user_scrolling.update({camera_name: True}))
        scrollbar.bind("<ButtonRelease-1>", lambda e: self.after(3000, lambda: self._user_scrolling.update({camera_name: False})))
        
        # Pack elements
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create the widget structure for detection details
        details_widgets = self.create_detection_widgets(scrollable_frame, camera_name)
        
        return frame, details_widgets

    def create_detection_widgets(self, parent, camera_name):
        """Create the widget structure for detection details"""
        widgets = {}
        
        # Header section
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill="x", pady=2)
        
        widgets['header_label'] = ttk.Label(header_frame, 
                                          text=f"SS-EN 1611-1 Grading ({camera_name.title()} Camera):",
                                          font=("Arial", 10, "bold"))
        widgets['header_label'].pack(anchor="w")
        
        # Calibration info section
        calib_frame = ttk.Frame(parent)
        calib_frame.pack(fill="x", pady=1)
        
        if camera_name == "top":
            calib_text = f"Distance: {TOP_CAMERA_DISTANCE_CM}cm, Factor: {TOP_CAMERA_PIXEL_TO_MM:.3f}mm/px"
        else:
            calib_text = f"Distance: {BOTTOM_CAMERA_DISTANCE_CM}cm, Factor: {BOTTOM_CAMERA_PIXEL_TO_MM:.3f}mm/px"
        
        widgets['calibration_label'] = ttk.Label(calib_frame, text=calib_text, font=self.font_small)
        widgets['calibration_label'].pack(anchor="w")
        
        widgets['wood_height_label'] = ttk.Label(calib_frame,
                                              text=f"Wood Height: {WOOD_PALLET_WIDTH_MM}mm",
                                              font=self.font_small)
        widgets['wood_width_label'].pack(anchor="w")
        
        # Separator
        separator1 = ttk.Separator(parent, orient="horizontal")
        separator1.pack(fill="x", pady=2)
        
        # Status section
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill="x", pady=2)
        
        widgets['status_label'] = ttk.Label(status_frame, 
                                          text="Status: Waiting for detection...",
                                          font=self.font_small, foreground="blue")
        widgets['status_label'].pack(anchor="w")
        
        widgets['defect_count_label'] = ttk.Label(status_frame, 
                                                text="No wood or defects detected",
                                                font=self.font_small)
        widgets['defect_count_label'].pack(anchor="w")
        
        # Defects container frame (will hold individual defect widgets)
        widgets['defects_container'] = ttk.Frame(parent)
        widgets['defects_container'].pack(fill="both", expand=True, pady=2)
        
        # Separator
        separator2 = ttk.Separator(parent, orient="horizontal")
        separator2.pack(fill="x", pady=2)
        
        # Grade summary section
        grade_frame = ttk.Frame(parent)
        grade_frame.pack(fill="x", pady=2)
        
        widgets['grade_label'] = ttk.Label(grade_frame, 
                                         text="Final Surface Grade: No detection",
                                         font=("Arial", 9, "bold"))
        widgets['grade_label'].pack(anchor="w")
        
        widgets['reasoning_label'] = ttk.Label(grade_frame, 
                                             text="Ready to analyze: Sound Knots, Unsound Knots",
                                             font=self.font_small, foreground="gray")
        widgets['reasoning_label'].pack(anchor="w")
        
    def create_tabbed_detection_details(self, parent, camera_name):
        """Create a tabbed interface for detection details - better for real-time updates"""
        notebook = ttk.Notebook(parent)
        
        # Tab 1: Current Detection
        current_tab = ttk.Frame(notebook)
        notebook.add(current_tab, text="Current")
        
        # Tab 2: Statistics
        stats_tab = ttk.Frame(notebook)
        notebook.add(stats_tab, text="Statistics")
        
        # Tab 3: History (last 5 detections)
        history_tab = ttk.Frame(notebook)
        notebook.add(history_tab, text="History")
        
        notebook.pack(fill="both", expand=True)
        
        # Create widgets for each tab
        current_widgets = self.create_current_detection_widgets(current_tab, camera_name)
        stats_widgets = self.create_stats_widgets(stats_tab, camera_name)
        history_widgets = self.create_history_widgets(history_tab, camera_name)
        
        return {
            'notebook': notebook,
            'current': current_widgets,
            'stats': stats_widgets,
            'history': history_widgets
        }

    def create_current_detection_widgets(self, parent, camera_name):
        """Create widgets for current detection - fixed layout, no scrolling"""
        widgets = {}
        
        # Header with camera info
        header_frame = ttk.LabelFrame(parent, text=f"{camera_name.title()} Camera - Current Detection", padding="5")
        header_frame.pack(fill="x", pady=2)
        
        # Status display (always visible)
        widgets['status_label'] = ttk.Label(header_frame, text="Status: Waiting...", 
                                          font=("Arial", 10, "bold"), foreground="blue")
        widgets['status_label'].pack(anchor="w")
        
        # Quick summary (defect count, grade)
        summary_frame = ttk.Frame(header_frame)
        summary_frame.pack(fill="x", pady=2)
        
        widgets['defect_count'] = ttk.Label(summary_frame, text="Defects: 0", font=("Arial", 9))
        widgets['defect_count'].pack(side="left")
        
        widgets['grade_display'] = ttk.Label(summary_frame, text="Grade: No detection", 
                                           font=("Arial", 9, "bold"))
        widgets['grade_display'].pack(side="right")
        
        # Most significant defect display (only show worst one)
        defect_frame = ttk.LabelFrame(parent, text="Most Significant Defect", padding="5")
        defect_frame.pack(fill="both", expand=True, pady=2)
        
        widgets['main_defect_type'] = ttk.Label(defect_frame, text="None", font=("Arial", 10))
        widgets['main_defect_type'].pack(anchor="w")
        
        widgets['main_defect_size'] = ttk.Label(defect_frame, text="", font=("Arial", 9))
        widgets['main_defect_size'].pack(anchor="w")
        
        widgets['main_defect_grade'] = ttk.Label(defect_frame, text="", font=("Arial", 9))
        widgets['main_defect_grade'].pack(anchor="w")
        
        return widgets

    def create_grid_detection_display(self, parent, camera_name):
        """Create a fixed grid layout for detection display - no scrolling needed"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Configure grid weights
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=0)  # Status row
        main_frame.grid_rowconfigure(1, weight=1)  # Content row
        
        widgets = {}
        
        # Row 0: Status Bar (always visible)
        status_frame = ttk.Frame(main_frame, relief="sunken", borderwidth=1)
        status_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        
        widgets['status_bar'] = ttk.Label(status_frame, 
                                        text=f"{camera_name.title()}: Waiting for detection...",
                                        font=("Arial", 9, "bold"), background="lightgray")
        widgets['status_bar'].pack(fill="x", padx=5, pady=2)
        
        # Row 1, Col 0: Current Detection Info
        detection_frame = ttk.LabelFrame(main_frame, text="Current Detection", padding="5")
        detection_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 2))
        
        widgets['defect_count_label'] = ttk.Label(detection_frame, text="Defects: 0", font=("Arial", 10))
        widgets['defect_count_label'].pack(anchor="w", pady=1)
        
        widgets['worst_defect_label'] = ttk.Label(detection_frame, text="Worst: None", font=("Arial", 9))
        widgets['worst_defect_label'].pack(anchor="w", pady=1)
        
        widgets['grade_label'] = ttk.Label(detection_frame, text="Grade: No detection", 
                                         font=("Arial", 10, "bold"))
        widgets['grade_label'].pack(anchor="w", pady=1)
        
        # Row 1, Col 1: Camera Calibration (static info)
        calib_frame = ttk.LabelFrame(main_frame, text="Camera Info", padding="5")
        calib_frame.grid(row=1, column=1, sticky="nsew", padx=(2, 0))
        
        if camera_name == "top":
            calib_text = f"Distance: {TOP_CAMERA_DISTANCE_CM}cm\nFactor: {TOP_CAMERA_PIXEL_TO_MM:.3f}mm/px"
        else:
            calib_text = f"Distance: {BOTTOM_CAMERA_DISTANCE_CM}cm\nFactor: {BOTTOM_CAMERA_PIXEL_TO_MM:.3f}mm/px"
        
        calib_label = ttk.Label(calib_frame, text=calib_text, font=("Arial", 9))
        calib_label.pack(anchor="w")
        
        wood_label = ttk.Label(calib_frame, text=f"Wood Height: {WOOD_PALLET_WIDTH_MM}mm",
                             font=("Arial", 9))
        wood_label.pack(anchor="w", pady=(5, 0))
        
        standard_label = ttk.Label(calib_frame, text="Standard: SS-EN 1611-1", 
                                 font=("Arial", 9), foreground="blue")
        standard_label.pack(anchor="w", pady=(5, 0))
        
        return widgets

    def create_simple_detection_tracker(self, camera_name):
        """Create a simplified detection tracker that retains all logic but without complex UI"""
        # This maintains the detection tracking logic without the complex UI components
        # The actual detection data is still processed and logged as before
        tracker = {
            'camera_name': camera_name,
            'last_detection_time': None,
            'current_defects': {},
            'current_measurements': [],
            'surface_grade': None,
            'detection_active': False
        }
        return tracker

    def update_dashboard_display(self, camera_name, defect_dict, measurements=None):
        """Update simplified dashboard display and log detailed defect data"""
        # Update the simple tracker
        tracker = getattr(self, f'{camera_name}_dashboard_widgets', None)
        if tracker:
            tracker['current_defects'] = defect_dict.copy() if defect_dict else {}
            tracker['current_measurements'] = measurements.copy() if measurements else []
            tracker['detection_active'] = bool(defect_dict and measurements)
            tracker['last_detection_time'] = time.time() if defect_dict else None
            
            if measurements and defect_dict:
                tracker['surface_grade'] = self.determine_surface_grade(measurements)
        
        # Continue with the existing detailed logging logic (this is retained)
        if measurements and defect_dict:
            surface_grade = self.determine_surface_grade(measurements)
            self.log_detection_details(camera_name, defect_dict, measurements, surface_grade)

    def create_dashboard_detection_display(self, parent, camera_name):
        """Compatibility method - creates the simple tracker"""
        return self.create_simple_detection_tracker(camera_name)

    def log_detection_details(self, camera_name, defect_dict, measurements, surface_grade):
        """Log detailed defect information for documentation and analysis"""
        import datetime
        import json
        
        # Create detection entry
        detection_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "camera": camera_name,
            "piece_number": getattr(self, 'piece_counter', 0),
            "test_case": getattr(self, 'current_test_case', 'N/A'),
            "total_defects": len(measurements),
            "final_grade": surface_grade,
            "defects": []
        }
        
        # Add camera calibration info
        detection_entry["camera_info"] = {
            "distance_cm": TOP_CAMERA_DISTANCE_CM if camera_name == "top" else BOTTOM_CAMERA_DISTANCE_CM,
            "pixel_to_mm": TOP_CAMERA_PIXEL_TO_MM if camera_name == "top" else BOTTOM_CAMERA_PIXEL_TO_MM,
            "wood_height_mm": WOOD_PALLET_WIDTH_MM
        }
        
        # Add individual defect details
        for i, (defect_type, size_mm, percentage) in enumerate(measurements, 1):
            individual_grade = self.get_individual_knot_grade(defect_type, size_mm, WOOD_PALLET_WIDTH_MM)
            
            defect_detail = {
                "defect_id": i,
                "type": defect_type,
                "size_mm": round(size_mm, 2),
                "percentage_of_width": round(percentage, 2),
                "individual_grade": individual_grade,
                "grading_standard": "SS-EN 1611-1"
            }
            
            # Add threshold information for new grading system
            constants = GRADING_CONSTANTS.get(defect_type, {})
            defect_detail["applied_threshold"] = f"Limit = (0.10 * {WOOD_PALLET_WIDTH_MM}mm) + constant"
            defect_detail["threshold_grade"] = individual_grade
            
            detection_entry["defects"].append(defect_detail)
        
        # Add grading reasoning
        total_defects = len(measurements)
        if total_defects > 6:
            detection_entry["grading_reason"] = "More than 6 defects detected - Automatic G2-4 (SS-EN 1611-1)"
        elif total_defects > 4:
            detection_entry["grading_reason"] = "More than 4 defects detected - Maximum G2-3 (SS-EN 1611-1)"
        elif total_defects > 2:
            detection_entry["grading_reason"] = "More than 2 defects detected - Maximum G2-2 (SS-EN 1611-1)"
        else:
            detection_entry["grading_reason"] = "≤2 defects detected - Based on individual grades (SS-EN 1611-1)"
        
        # Store in detection log
        self.detection_log.append(detection_entry)
        
        # Save to file for documentation
        self.save_detection_log(detection_entry)
        
        # Update piece counter
        self.piece_counter += 1

    def save_detection_log(self, detection_entry):
        """Save detection log entry to file for test case documentation"""
        import json
        import os
        from datetime import datetime, timedelta
        
        # Create logs directory if it doesn't exist
        log_dir = "detection_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create filename with date
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"detection_log_{date_str}.json")
        
        # Load existing log or create new one
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
            except:
                log_data = {"detections": []}
        else:
            log_data = {"detections": []}
        
        # Add new detection
        log_data["detections"].append(detection_entry)
        
        # Save updated log
        try:
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save detection log: {e}")

    def start_test_case(self, test_case_number):
        """Start a new test case for documentation"""
        self.test_case_counter = test_case_number
        self.current_test_case = f"TEST_CASE_{test_case_number:02d}"
        self.piece_counter = 0
        print(f"Started {self.current_test_case}")

    def export_test_case_summary(self, test_case_number):
        """Export summary of a specific test case for documentation"""
        import json
        from datetime import datetime, timedelta
        
        # Filter detections for this test case
        test_case_name = f"TEST_CASE_{test_case_number:02d}"
        test_detections = [d for d in self.detection_log if d.get("test_case") == test_case_name]
        
        if not test_detections:
            print(f"No detections found for {test_case_name}")
            return
        
        # Create summary
        summary = {
            "test_case": test_case_name,
            "export_timestamp": datetime.now().isoformat(),
            "total_pieces": len(test_detections),
            "grade_distribution": {},
            "defect_statistics": {},
            "camera_performance": {"top": 0, "bottom": 0},
            "detections": test_detections
        }
        
        # Calculate statistics
        for detection in test_detections:
            # Grade distribution
            grade = detection["final_grade"]
            summary["grade_distribution"][grade] = summary["grade_distribution"].get(grade, 0) + 1
            
            # Camera performance
            camera = detection["camera"]
            summary["camera_performance"][camera] += 1
            
            # Defect statistics
            for defect in detection["defects"]:
                defect_type = defect["type"]
                summary["defect_statistics"][defect_type] = summary["defect_statistics"].get(defect_type, 0) + 1
        
        # Save summary
        summary_file = f"TEST_CASE_{test_case_number:02d}_Summary.json"
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Test case summary exported to {summary_file}")
        except Exception as e:
            print(f"Failed to export summary: {e}")

    def update_detection_details_widgets(self, camera_name, defect_dict, measurements=None):
        """Update detection details using widget objects instead of text replacement"""
        
        # Check if user is currently scrolling - if so, don't update
        if self._user_scrolling.get(camera_name, False):
            return
        
        # Get the widgets for this camera
        if camera_name == "top":
            widgets = getattr(self, 'top_details_widgets', None)
        elif camera_name == "bottom":
            widgets = getattr(self, 'bottom_details_widgets', None)
        else:
            return
            
        if not widgets:
            return
        
        # Update status and defect count
        if defect_dict and measurements:
            total_defects = len(measurements)
            widgets['status_label'].config(text="Status: Active detection", foreground="green")
            widgets['defect_count_label'].config(text=f"Defects detected: {total_defects}")
            
            # Clear previous defect widgets
            for widget in widgets['defects_container'].winfo_children():
                widget.destroy()
            
            # Create individual defect widgets
            for i, (defect_type, size_mm, percentage) in enumerate(measurements, 1):
                defect_frame = ttk.LabelFrame(widgets['defects_container'], 
                                            text=f"Defect {i}: {defect_type.replace('_', ' ')}", 
                                            padding="3")
                defect_frame.pack(fill="x", pady=1)
                
                # Defect details
                individual_grade = self.get_individual_knot_grade(defect_type, size_mm, WOOD_PALLET_WIDTH_MM)
                
                size_label = ttk.Label(defect_frame, 
                                     text=f"Size: {size_mm:.1f}mm ({percentage:.1f}% of width)",
                                     font=self.font_small)
                size_label.pack(anchor="w")
                
                grade_color = self.get_grade_color(individual_grade)
                grade_label = ttk.Label(defect_frame, 
                                      text=f"Individual Grade: {individual_grade}",
                                      font=self.font_small, foreground=grade_color)
                grade_label.pack(anchor="w")
                
                # Show threshold info for new grading system
                constants = GRADING_CONSTANTS.get(defect_type, {})
                limit = (0.10 * WOOD_PALLET_WIDTH_MM) + constants.get(individual_grade, 0)
                threshold_text = f"Threshold: ≤{limit:.1f}mm (0.10*{WOOD_PALLET_WIDTH_MM} + {constants.get(individual_grade, 0)})"
                
                threshold_label = ttk.Label(defect_frame, text=threshold_text, 
                                          font=self.font_small, foreground="gray")
                threshold_label.pack(anchor="w")
            
            # Update final grade
            surface_grade = self.determine_surface_grade(measurements)
            grade_color = self.get_grade_color(surface_grade)
            widgets['grade_label'].config(text=f"Final Surface Grade: {surface_grade}", 
                                        foreground=grade_color)
            
            # Update reasoning
            if total_defects > 6:
                reasoning_text = "Reasoning: >6 defects = Automatic G2-4 (SS-EN 1611-1)"
            elif total_defects > 4:
                reasoning_text = "Reasoning: >4 defects = Maximum G2-3 (SS-EN 1611-1)"
            elif total_defects > 2:
                reasoning_text = "Reasoning: >2 defects = Maximum G2-2 (SS-EN 1611-1)"
            else:
                reasoning_text = "Reasoning: ≤2 defects = Use individual grades (SS-EN 1611-1)"
            
            widgets['reasoning_label'].config(text=reasoning_text, foreground="black")
            
        elif defect_dict:
            # Simple detection mode
            total_defects = sum(defect_dict.values())
            widgets['status_label'].config(text="Status: Simple detection mode", foreground="orange")
            widgets['defect_count_label'].config(text=f"Total defects: {total_defects}")
            
            # Clear previous defect widgets
            for widget in widgets['defects_container'].winfo_children():
                widget.destroy()
            
            # Show simple defect counts
            sorted_defects = sorted(defect_dict.items(), key=lambda x: x[1], reverse=True)
            for defect_type, count in sorted_defects:
                defect_frame = ttk.Frame(widgets['defects_container'])
                defect_frame.pack(fill="x", pady=1)
                
                formatted_name = defect_type.replace('_', ' ').title()
                defect_label = ttk.Label(defect_frame, 
                                       text=f"• {formatted_name}: {count} detected",
                                       font=self.font_small)
                defect_label.pack(anchor="w")
            
            widgets['grade_label'].config(text="Final Surface Grade: Simple mode (no size data)", 
                                        foreground="gray")
            widgets['reasoning_label'].config(text="Note: Size measurements not available in simple mode", 
                                            foreground="gray")
        else:
            # No detection
            widgets['status_label'].config(text="Status: Waiting for detection...", foreground="blue")
            widgets['defect_count_label'].config(text="No wood or defects detected")
            
            # Clear defect widgets
            for widget in widgets['defects_container'].winfo_children():
                widget.destroy()
            
            widgets['grade_label'].config(text="Final Surface Grade: No detection", foreground="gray")
            widgets['reasoning_label'].config(text="Ready to analyze: Sound Knots, Unsound Knots", 
                                            foreground="gray")

    def update_feeds(self):
        self.update_single_feed(self.cap_top, self.top_canvas, "top")
        self.update_single_feed(self.cap_bottom, self.bottom_canvas, "bottom")

        # Reduce update frequency for non-critical components to prevent UI lag
        # Only update every 15th frame (~4.4 FPS for dashboard updates) to reduce load
        if not hasattr(self, '_frame_counter'):
            self._frame_counter = 0

        self._frame_counter += 1
        if self._frame_counter % 15 == 0:
            # Check camera status and reconnect if needed (skip during cooldown after mode changes)
            if time.time() > self._camera_check_cooldown:
                if not self.camera_handler.check_camera_status():
                    print("Camera status check failed - attempting reconnection...")
                    if not self.camera_disconnected_popup_shown:
                        messagebox.showwarning("Camera Disconnection", "Camera disconnection detected. Attempting reconnection...")
                        self.camera_disconnected_popup_shown = True
                    if self.camera_handler.reassign_cameras_runtime():
                        # Update the cap references after successful reconnection
                        self.cap_top = self.camera_handler.top_camera
                        self.cap_bottom = self.camera_handler.bottom_camera
                        print("Camera reconnection successful during runtime")
                        messagebox.showinfo("Camera Reconnection", "Cameras have been reconnected successfully.")
                        self.camera_disconnected_popup_shown = False
                        if hasattr(self, 'status_label'):
                            # status_label is a Text widget, not a Label widget
                            self.status_label.config(state=tk.NORMAL)
                            self.status_label.delete(1.0, tk.END)
                            self.status_label.insert(1.0, "Status: Cameras reconnected")
                            self.status_label.config(foreground="green", state=tk.DISABLED)
                    else:
                        print("Camera reconnection failed during runtime")
                        if hasattr(self, 'status_label'):
                            # status_label is a Text widget, not a Label widget
                            self.status_label.config(state=tk.NORMAL)
                            self.status_label.delete(1.0, tk.END)
                            self.status_label.insert(1.0, "Status: Camera reconnection failed")
                            self.status_label.config(foreground="red", state=tk.DISABLED)

            # Update detection status
            self.update_detection_status_display()

            # Only update details if not in active inference to prevent interference
            if not getattr(self, '_in_active_inference', False):
                self.ensure_detection_details_updated()

        # Optimize for constant detection - update every 10ms for ~100 FPS
        self.after(10, self.update_feeds)

    def ensure_detection_details_updated(self):
        """Ensure detection details are showing current state, even when not actively detecting"""
        # Update detection details for both cameras using dashboard approach
        for camera_name in ["top", "bottom"]:
            # If automatic detection is off, make sure we show the waiting state
            if not self.auto_detection_active:
                self.update_dashboard_display(camera_name, {}, [])
            # If automatic detection is on but no recent detections, also update to show waiting state
            elif not hasattr(self, 'live_detections') or not self.live_detections.get(camera_name):
                self.update_dashboard_display(camera_name, {}, [])
        
        # Also update the live grading display and statistics
        self.update_live_grading_display()
        self.update_detailed_statistics()

    def update_detection_status_display(self):
        """Update status display based on current detection state"""
        if hasattr(self, 'status_label'):
            # Helper function to update Text widget properly
            def update_status_text(text, foreground="black"):
                self.status_label.config(state=tk.NORMAL)
                self.status_label.delete(1.0, tk.END)
                self.status_label.insert(1.0, text)
                if foreground != "black":
                    self.status_label.config(foreground=foreground)
                self.status_label.config(state=tk.DISABLED)

            if self.auto_detection_active:
                total_detections = (len(self.detection_session_data["total_detections"]["top"]) +
                                  len(self.detection_session_data["total_detections"]["bottom"]))
                update_status_text(f"Status: AUTO (IR) DETECTION ACTIVE 🔍 ({total_detections} detections)", "orange")
            elif self.live_detection_var.get():
                update_status_text(f"Status: {self.current_mode} MODE - Live Detection ACTIVE", "blue")
            elif self.current_mode == "IDLE":
                update_status_text("Status: IDLE MODE - System disabled, no operations", "gray")
            elif self.current_mode == "TRIGGER":
                update_status_text("Status: TRIGGER MODE - Waiting for IR beam trigger", "green")
            elif self.current_mode == "CONTINUOUS":
                update_status_text("Status: CONTINUOUS MODE - Live detection enabled", "blue")
            else:
                # Fallback for unknown states
                update_status_text(f"Status: {self.current_mode} MODE - Ready", "green")

    def toggle_live_detection_mode(self):
        """Handle toggling between IR trigger and live detection modes."""
        # Clear wood detection results when toggling live detection off
        if not self.live_detection_var.get():
            self.wood_detection_results = {"top": None, "bottom": None}
            self.dynamic_roi = {}
        self.update_detection_status_display()

    def toggle_roi(self):
        """Toggle ROI for top camera"""
        self.roi_enabled["top"] = self.roi_var.get()
        status = "enabled" if self.roi_enabled["top"] else "disabled"
        print(f"ROI for top camera {status}")

    def toggle_bottom_roi(self):
        """Toggle ROI for bottom camera"""
        self.roi_enabled["bottom"] = self.bottom_roi_var.get()
        status = "enabled" if self.roi_enabled["bottom"] else "disabled"
        print(f"ROI for bottom camera {status}")

    def start_automatic_detection(self):
        """Start automatic detection when IR beam detects object"""
        self.auto_detection_active = True
        self._in_active_inference = True  # Flag to prevent UI conflicts during inference
        self.detection_session_data = {
            "start_time": datetime.now(),
            "end_time": None,
            "total_detections": {"top": [], "bottom": []},
            "best_frames": {"top": None, "bottom": None},
            "final_grade": None
        }
        self.detection_frames = []


        # Clear trackers for new session
        self.trackers["top"].clear()
        self.trackers["bottom"].clear()

        # Clear session detections for new piece
        self.session_detections = {"top": [], "bottom": []}
        self.final_deduplicated_defects = {"top": [], "bottom": []}

        # Clear previous live detections
        self.live_detections = {"top": {}, "bottom": {}}
        self.live_grades = {"top": "Detecting...", "bottom": "Detecting..."}

        # Reset wood detection reporting flags for new session
        self._wood_reported = {'top': False, 'bottom': False}

        print("🔍 Automatic detection STARTED - Object detected by IR beam")
        print("🎯 Custom object trackers initialized for unique defect counting")
        self.log_action("Automatic detection started - IR beam triggered")

    def stop_automatic_detection_and_grade(self):
        """Stop automatic detection and send grade when object clears IR beam"""
        if not self.auto_detection_active:
            return

        self.auto_detection_active = False
        self._in_active_inference = False  # Clear inference flag to resume normal UI updates
        self.detection_session_data["end_time"] = datetime.now()

        # Collect ALL detections from the entire session (every frame during the detection period)
        all_measurements = []
        top_measurements = []
        bottom_measurements = []

        # Process all detections from top camera session data
        for detection_entry in self.detection_session_data["total_detections"]["top"]:
            measurements = detection_entry.get("measurements", [])
            for measurement in measurements:
                top_measurements.append(measurement)
                all_measurements.append(measurement)

        # Process all detections from bottom camera session data
        for detection_entry in self.detection_session_data["total_detections"]["bottom"]:
            measurements = detection_entry.get("measurements", [])
            for measurement in measurements:
                bottom_measurements.append(measurement)
                all_measurements.append(measurement)

        # STEP 4 & 5: List all Wood and Defect Details and Grade the wood based on the list and logic of the grading system
        print("\n" + "="*80)
        print("FINAL WOOD AND DEFECT ANALYSIS REPORT")
        print("="*80)

        # Report wood detection results
        print("\nWOOD DETECTION SUMMARY:")
        print("-" * 40)
        if hasattr(self, 'wood_detection_results') and self.wood_detection_results:
            for camera_name in ["top", "bottom"]:
                if camera_name in self.wood_detection_results and self.wood_detection_results[camera_name]:
                    detection = self.wood_detection_results[camera_name]
                    if detection.get('wood_detected', False):
                        candidates = detection.get('wood_candidates', [])
                        confidence = detection.get('confidence', 0)
                        print(f"📷 {camera_name.upper()} CAMERA:")
                        print(f"   ✓ Wood detected (confidence: {confidence:.2f})")
                        print(f"   ✓ {len(candidates)} wood piece(s) found:")
                        for i, candidate in enumerate(candidates, 1):
                            bbox = candidate['bbox']
                            area = candidate['area']
                            conf = candidate['confidence']
                            print(f"      {i}. Position: {bbox}, Area: {area:.0f}px, Confidence: {conf:.2f}")
                    else:
                        print(f"📷 {camera_name.upper()} CAMERA:")
                        print("   ✗ No wood detected")
        else:
            print("   No wood detection data available")

        # Report defect detection results (ALL detections from session)
        print("\nDEFECT DETECTION SUMMARY:")
        print("-" * 40)
        print(f"🎯 TOP CAMERA: {len(top_measurements)} defect(s) detected across {len(self.detection_session_data['total_detections']['top'])} frames")
        for i, (defect_type, size_mm, percentage) in enumerate(top_measurements, 1):
            print(f"   {i}. {defect_type} - Size: {size_mm:.1f}mm")

        print(f"🎯 BOTTOM CAMERA: {len(bottom_measurements)} defect(s) detected across {len(self.detection_session_data['total_detections']['bottom'])} frames")
        for i, (defect_type, size_mm, percentage) in enumerate(bottom_measurements, 1):
            print(f"   {i}. {defect_type} - Size: {size_mm:.1f}mm")

        # Determine final grades from tracked objects
        final_top_grade = self.determine_surface_grade(top_measurements)
        final_bottom_grade = self.determine_surface_grade(bottom_measurements)

        # Combine grades for final decision
        combined_grade = self.determine_final_grade(final_top_grade, final_bottom_grade)
        self.detection_session_data["final_grade"] = combined_grade

        print("\nGRADING ANALYSIS:")
        print("-" * 40)
        print(f"📊 Top Surface Grade: {final_top_grade}")
        print(f"📊 Bottom Surface Grade: {final_bottom_grade}")
        print(f"🏆 Final Combined Grade: {combined_grade}")

        # Show grading reasoning
        print("\nGRADING REASONING:")
        print("-" * 40)
        total_defects = len(all_measurements)
        if total_defects > 6:
            print("Reasoning: More than 6 defects detected - Automatic G2-4 (SS-EN 1611-1)")
        elif total_defects > 4:
            print("Reasoning: More than 4 defects detected - Maximum G2-3 (SS-EN 1611-1)")
        elif total_defects > 2:
            print("Reasoning: More than 2 defects detected - Maximum G2-2 (SS-EN 1611-1)")
        else:
            print("Reasoning: Based on individual defect sizes (SS-EN 1611-1)")

        print("="*80 + "\n")

        # Log the final grading with tracked objects
        self.finalize_grading(combined_grade, all_measurements)

        # Update live grading display
        self.live_grades["top"] = final_top_grade
        self.live_grades["bottom"] = final_bottom_grade
        self.update_live_grading_display()

        # Clear trackers for next piece
        self.trackers["top"].clear()
        self.trackers["bottom"].clear()

        # Clear detection data for next piece
        self.detection_frames = []
        self.session_detections = {"top": [], "bottom": []}
        self.final_deduplicated_defects = {"top": [], "bottom": []}

        # Clear wood detection results for next piece
        self.wood_detection_results = {"top": None, "bottom": None}
        self.dynamic_roi = {}

    def determine_final_grade_from_session(self, camera_name, detections_list):
        """Determine final grade from all detections collected during session"""
        if not detections_list:
            return GRADE_G2_0 # No defects found, perfect grade

        # Combine all defect counts from the session
        all_measurements = [m for d in detections_list for m in d.get("measurements", [])]

        # Use sophisticated grading if measurements available
        if all_measurements:
            return self.determine_surface_grade(all_measurements)
        else:
            # Fall back to simple grading if no measurements (should not happen in normal operation)
            combined_defects = {}
            for detection_data in detections_list:
                defect_dict = detection_data.get("defects", {})
                for defect_type, count in defect_dict.items():
                    combined_defects[defect_type] = combined_defects.get(defect_type, 0) + count

            grade_info = self.calculate_grade(combined_defects)
            # Map numeric grade back to standard grade text
            grade_map = {0: GRADE_G2_0, 1: GRADE_G2_0, 2: GRADE_G2_2, 3: GRADE_G2_4}
            return grade_map.get(grade_info.get('grade'), GRADE_G2_4)

    def save_detection_session(self):
        """Save the complete detection session data"""
        session_data = {
            "session_id": f"AUTO_{int(time.time())}",
            "timestamp": self.detection_session_data["start_time"].isoformat(),
            "duration_seconds": (self.detection_session_data["end_time"] - self.detection_session_data["start_time"]).total_seconds(),
            "detection_data": self.detection_session_data,
            "total_frames_captured": len(self.detection_frames),
            "trigger_method": "IR_BEAM_AUTOMATIC"
        }
        
        # Save to JSON log
        self.save_detection_log(session_data)
        
        # Save best frames as images if available
        if self.detection_session_data["best_frames"]["top"] is not None:
            self.save_detection_frame("top", self.detection_session_data["best_frames"]["top"])
        if self.detection_session_data["best_frames"]["bottom"] is not None:
            self.save_detection_frame("bottom", self.detection_session_data["best_frames"]["bottom"])

    def save_detection_frame(self, camera_name, frame):
        """Save a detection frame as image file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{camera_name}_{timestamp}.jpg"
            filepath = os.path.join("detection_frames", filename)
            
            # Create directory if it doesn't exist
            os.makedirs("detection_frames", exist_ok=True)
            
            # Convert from RGB back to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            cv2.imwrite(filepath, frame_bgr)
            print(f"📸 Saved detection frame: {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving detection frame: {e}")


    def apply_roi(self, frame, camera_name, custom_roi_coords=None):
        """Apply Region of Interest (ROI) to frame for focused detection"""
        # Use custom ROI if provided, otherwise check if ROI is enabled
        if custom_roi_coords:
            roi_coords = custom_roi_coords
        elif not self.roi_enabled.get(camera_name, False):
            return frame, None
        else:
            roi_coords = self.roi_coordinates.get(camera_name, {})

        if not roi_coords:
            return frame, None

        x1, y1 = roi_coords.get("x1", 0), roi_coords.get("y1", 0)
        x2, y2 = roi_coords.get("x2", frame.shape[1]), roi_coords.get("y2", frame.shape[0])

        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, frame.shape[1]))
        y1 = max(0, min(y1, frame.shape[0]))
        x2 = max(x1, min(x2, frame.shape[1]))
        y2 = max(y1, min(y2, frame.shape[0]))

        # Extract ROI
        roi_frame = frame[y1:y2, x1:x2]
        roi_info = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

        return roi_frame, roi_info

    def draw_roi_overlay(self, frame, camera_name):
        """Draw ROI rectangle overlay on frame for visualization"""
        if not self.roi_enabled.get(camera_name, False):
            return frame

        roi_coords = self.roi_coordinates.get(camera_name, {})
        if not roi_coords:
            return frame

        frame_copy = frame.copy()
        x1, y1 = roi_coords.get("x1", 0), roi_coords.get("y1", 0)
        x2, y2 = roi_coords.get("x2", frame.shape[1]), roi_coords.get("y2", frame.shape[0])

        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, frame.shape[1]))
        y1 = max(0, min(y1, frame.shape[0]))
        x2 = max(x1, min(x2, frame.shape[1]))
        y2 = max(y1, min(y2, frame.shape[0]))

        # Draw ROI rectangle (yellow border)
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 255), 3)

        # Add ROI label
        # cv2.putText(frame_copy, f"ROI - {camera_name.upper()}",
        #           (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return frame_copy

    def bbox_intersects_roi(self, bbox, camera_name):
        """Check if bounding box intersects with ROI"""
        if not self.roi_enabled.get(camera_name, False):
            return True  # No ROI means all detections count

        # Scale bbox from model coordinates (640x640) to original frame coordinates (1280x720)
        x1, y1, x2, y2 = bbox[:4]
        scale_x = 1280.0 / 640.0  # Original width / model width
        scale_y = 720.0 / 640.0   # Original height / model height

        x1_orig = x1 * scale_x
        y1_orig = y1 * scale_y
        x2_orig = x2 * scale_x
        y2_orig = y2 * scale_y

        roi_coords = self.roi_coordinates.get(camera_name, {})
        roi_x1 = roi_coords.get("x1", 0)
        roi_y1 = roi_coords.get("y1", 0)
        roi_x2 = roi_coords.get("x2", 1280)
        roi_y2 = roi_coords.get("y2", 720)

        # Check for intersection between scaled bounding box and ROI
        return not (x2_orig < roi_x1 or x1_orig > roi_x2 or y2_orig < roi_y1 or y1_orig > roi_y2)

    def draw_wood_detection_overlay(self, frame, camera_name):
        """Draw wood detection results overlay on frame for visualization"""
        # Only show wood detection overlay when live detection is active (beam blocked) or in scan mode
        if not self.live_detection_var.get() and self.current_mode != "SCAN_PHASE":
            return frame

        frame_copy = frame.copy()

        # Check if we have wood detection results
        if hasattr(self, 'wood_detection_results') and self.wood_detection_results:
            detection_result = self.wood_detection_results.get(camera_name)

            if detection_result and detection_result.get('wood_detected', False):
                # Wood detected - draw detection results
                wood_candidates = detection_result.get('wood_candidates', [])
                for i, candidate in enumerate(wood_candidates):
                    x, y, w, h = candidate['bbox']
                    confidence = candidate['confidence']

                    # Draw bounding box (green for wood detection)
                    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)

                    # Add wood detection label with confidence
                    label = f"Wood {i+1}: {confidence:.2f}"
                    cv2.putText(frame_copy, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Draw dynamic ROI if available
                if hasattr(self, 'dynamic_roi') and self.dynamic_roi and camera_name in self.dynamic_roi:
                    roi = self.dynamic_roi[camera_name]
                    if roi:
                        x, y, w, h = roi
                        # Draw dynamic ROI (blue border)
                        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(frame_copy, " ",
                                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Add wood detection summary
                wood_count = detection_result.get('wood_count', 0)
                confidence = detection_result.get('confidence', 0.0)
                summary_text = f"Wood: {wood_count} detected (conf: {confidence:.2f})"
                cv2.putText(frame_copy, summary_text, (10, frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # No wood detected - show clear message
                h, w = frame_copy.shape[:2]
                cv2.putText(frame_copy, "NO WOOD DETECTED", (w//2 - 150, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            # No detection results yet - show waiting message
            h, w = frame_copy.shape[:2]
            cv2.putText(frame_copy, "INITIALIZING WOOD DETECTION", (w//2 - 180, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 165, 0), 2)
            cv2.putText(frame_copy, "Please wait...",
                        (w//2 - 60, h//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

        return frame_copy

    def update_single_feed(self, cap, label, camera_name):
        ret, frame = cap.read()
        if ret:
            # Mirror the bottom camera horizontally from the start for consistent perspective
            if camera_name == "bottom":
                frame = cv2.flip(frame, 1)  # Horizontal flip

            # Skip detection processing if frame rate is too high
            if not hasattr(self, '_detection_frame_skip'):
                self._detection_frame_skip = {"top": 0, "bottom": 0}

            # Initialize memory management counter
            if not hasattr(self, '_memory_cleanup_counter'):
                self._memory_cleanup_counter = 0

            # Perform memory cleanup every 300 frames (~24 seconds at 125 FPS)
            self._memory_cleanup_counter += 1
            if self._memory_cleanup_counter % 300 == 0:
                import gc
                gc.collect()  # Force garbage collection
                
                # Clear cached dimensions periodically to handle window resizing
                if hasattr(self, '_label_dimensions'):
                    self._label_dimensions.clear()
                    
                print(f"Memory cleanup performed at frame {self._memory_cleanup_counter}")
            
            # Process detection based on automatic IR beam OR live detection toggle
            should_detect = self.auto_detection_active or (self.live_detection_var.get() and self.current_mode != "TRIGGER")

            # For bottom camera, only detect if top camera detected wood (synchronized detection)
            # But wood detection is done per camera, so bottom camera can detect wood independently
            # Defect detection on bottom camera only happens if wood was detected on bottom camera

            # Only perform wood detection when detection is active (not in idle mode)
            if should_detect:
                try:
                    # Always use wood_detection ROI (Yellow ROI) for wood detection to maintain hierarchy
                    wood_roi_coords = self.roi_coordinates.get("wood_detection", {})
                    if wood_roi_coords and self.roi_enabled.get("wood_detection", True):
                        x1, y1 = wood_roi_coords.get("x1", 0), wood_roi_coords.get("y1", 0)
                        x2, y2 = wood_roi_coords.get("x2", frame.shape[1]), wood_roi_coords.get("y2", frame.shape[0])
                        cropped_frame = frame[y1:y2, x1:x2]
                        wood_detection = self.rgb_wood_detector.detect_wood_comprehensive(cropped_frame, camera=camera_name)
                        # Adjust auto_roi coordinates back to full frame
                        if wood_detection.get('auto_roi'):
                            ax, ay, aw, ah = wood_detection['auto_roi']
                            wood_detection['auto_roi'] = (x1 + ax, y1 + ay, aw, ah)
                    else:
                        # Fallback to full frame if wood_detection ROI not configured
                        wood_detection = self.rgb_wood_detector.detect_wood_comprehensive(frame, camera=camera_name)

                    # Store wood detection results for overlay display
                    self.wood_detection_results[camera_name] = wood_detection

                    # Store dynamic ROI for defect detection
                    self.dynamic_roi[camera_name] = wood_detection.get('auto_roi')

                    # Check if wood is detected (any candidates found)
                    wood_detected = wood_detection['wood_detected']

                    if wood_detected:
                        # The wood width is already updated by ColorWoodDetector.detect_wood_comprehensive()
                        # via update_wood_width_dynamic(), so we don't need to recalculate it here
                        
                        # STEP 4: List wood detection details (only once per camera per detection session in auto mode)
                        if self.auto_detection_active and (not hasattr(self, '_wood_reported') or not self._wood_reported.get(camera_name, False)):
                            if not hasattr(self, '_wood_reported'):
                                self._wood_reported = {'top': False, 'bottom': False}

                            candidates = wood_detection.get('wood_candidates', [])
                            print(f"🪵 {camera_name.upper()} CAMERA: {len(candidates)} wood pieces detected (confidence: {wood_detection.get('confidence', 0):.2f})")
                            for i, candidate in enumerate(candidates, 1):
                                bbox = candidate['bbox']
                                confidence = candidate['confidence']
                                area = candidate['area']
                                print(f"   {i}. Wood piece: {bbox} (confidence: {confidence:.2f}, area: {area:.0f}px)")

                            self._wood_reported[camera_name] = True
                    else:
                        # No wood detected - clear ROI and skip defect detection
                        self.dynamic_roi[camera_name] = None
                        if not hasattr(self, '_wood_reported'):
                            self._wood_reported = {'top': False, 'bottom': False}
                        self._wood_reported[camera_name] = False
                except Exception as e:
                    print(f"❌ Error in wood detection for {camera_name}: {e}")
                    self.dynamic_roi[camera_name] = None
                    # Clear wood detection results on error
                    self.wood_detection_results[camera_name] = None
            else:
                # When not detecting (idle mode), clear wood detection results and ROI
                self.dynamic_roi[camera_name] = None
                self.wood_detection_results[camera_name] = None

            # Only proceed with defect detection if wood was detected (for hierarchy)
            has_wood_detected = (hasattr(self, 'wood_detection_results') and
                                self.wood_detection_results.get(camera_name) and
                                self.wood_detection_results[camera_name].get('wood_detected', False))

            if should_detect and has_wood_detected:
                # STEP 3: Detect Defects only in the Green ROI (static camera ROI)
                # Use the static Green ROI for defect detection to maintain hierarchy
                detection_frame, roi_info = self.apply_roi(frame, camera_name)
                print(f"🎯 Using Green ROI for {camera_name} defect detection")

                # Pre-resize frame for faster processing if it's very large
                height, width = detection_frame.shape[:2]
                if width > 1280 or height > 720:
                    # Resize for detection processing to improve speed
                    scale_factor = min(1280/width, 720/height)
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    resized_frame = cv2.resize(detection_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                    # Run detection on resized ROI frame
                    result = self.analyze_frame(resized_frame, camera_name, run_defect_model=True)

                    # Scale detection results back to ROI size
                    if len(result) == 3:
                        annotated_frame, defect_dict, detections_for_grading = result
                        # Scale annotated frame back to ROI size
                        annotated_frame = cv2.resize(annotated_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                    else:
                        annotated_frame, defect_dict = result
                        annotated_frame = cv2.resize(annotated_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                        detections_for_grading = []
                else:
                    # ROI frame is already optimal size, process normally
                    result = self.analyze_frame(detection_frame, camera_name, run_defect_model=True)

                    # Handle both old and new return formats for compatibility
                    if len(result) == 3:
                        annotated_frame, defect_dict, detections_for_grading = result
                    else:
                        annotated_frame, defect_dict = result
                        detections_for_grading = []

                # If ROI was applied, place the annotated ROI back into the full frame
                if roi_info is not None:
                    full_frame_annotated = frame.copy()
                    full_frame_annotated[roi_info["y1"]:roi_info["y2"], roi_info["x1"]:roi_info["x2"]] = annotated_frame
                    # Add ROI overlay to show the detection area
                    annotated_frame = self.draw_roi_overlay(full_frame_annotated, camera_name)
                else:
                    # No ROI applied, use the annotated frame as is
                    pass

                # Add wood detection overlay if available
                annotated_frame = self.draw_wood_detection_overlay(annotated_frame, camera_name)

                # Store the detection results for automatic detection session
                self.live_detections[camera_name] = defect_dict

                # Store measurements for sophisticated grading
                if not hasattr(self, 'live_measurements'):
                    self.live_measurements = {"top": [], "bottom": []}
                self.live_measurements[camera_name] = detections_for_grading

                # During automatic detection, store frame for potential PDF report
                if self.auto_detection_active:
                    # Keep the original session data structure for compatibility
                    detection_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "camera": camera_name,
                        "defects": defect_dict.copy(),
                        "measurements": detections_for_grading.copy() if detections_for_grading else [],
                        "frame_captured": True
                    }

                    # Add to session data
                    self.detection_session_data["total_detections"][camera_name].append(detection_entry)

                    # Save best frame (frame with most detections or first significant detection)
                    if (self.detection_session_data["best_frames"][camera_name] is None or
                        sum(defect_dict.values()) > 0):
                        # Convert frame to RGB for saving
                        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        self.detection_session_data["best_frames"][camera_name] = frame_rgb.copy()

                    # Store frame for potential PDF report
                    if len(self.detection_frames) < 50:  # Limit stored frames to prevent memory issues
                        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        self.detection_frames.append({
                            "camera": camera_name,
                            "timestamp": datetime.now().isoformat(),
                            "frame": frame_rgb.copy(),
                            "defects": defect_dict.copy()
                        })

                # Calculate grade for this camera using sophisticated grading
                if detections_for_grading:
                    surface_grade = self.determine_surface_grade(detections_for_grading)
                    grade_info = {
                        'grade': surface_grade,
                        'text': f'{surface_grade} - SS-EN 1611-1 ({camera_name.title()} Camera)',
                        'total_defects': len(detections_for_grading),
                        'color': self.get_grade_color(surface_grade)
                    }
                else:
                    grade_info = self.calculate_grade(defect_dict)  # Fallback to simple grading

                self.live_grades[camera_name] = grade_info

                # Update dashboard every 5th frame for smoother updates
                if self._detection_frame_skip[camera_name] % 5 == 0:
                    self.update_dashboard_display(camera_name, defect_dict, detections_for_grading)

                # Update the live grading display every 3rd frame
                if self._detection_frame_skip[camera_name] % 3 == 0:
                    self.update_live_grading_display()

                cv2image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            else:
                # No defect detection (either not should_detect or no wood detected)
                # Just show raw feed without detection processing
                # Add ROI overlay to show the detection area even when not detecting
                frame_with_roi = self.draw_roi_overlay(frame, camera_name)
                # Add wood detection overlay if available (skip for scan mode live feed)
                if self.current_mode != "SCAN_PHASE":
                    frame_with_overlays = self.draw_wood_detection_overlay(frame_with_roi, camera_name)
                else:
                    # For scan mode, only show yellow ROI on live feed (green overlay only on captured frames)
                    frame_with_overlays = self.draw_roi_overlay(frame, camera_name)  # Yellow ROI only
                cv2image = cv2.cvtColor(frame_with_overlays, cv2.COLOR_BGR2RGB)

                # Reset detections only when automatic detection is not active
                if not self.auto_detection_active:
                    self.live_detections[camera_name] = {}
                    self.live_grades[camera_name] = " "
                    if hasattr(self, 'live_measurements'):
                        self.live_measurements[camera_name] = []
                    # Update dashboard every 10th frame when no detection
                    if self._detection_frame_skip[camera_name] % 10 == 0:
                        self.update_dashboard_display(camera_name, {}, [])
                        self.update_live_grading_display()
            
            # Increment frame skip counter
            self._detection_frame_skip[camera_name] += 1
            
            # Convert to PIL Image and ensure consistent scaling
            img = Image.fromarray(cv2image)
            
            # Cache label dimensions to avoid repeated calculations
            if not hasattr(self, '_label_dimensions'):
                self._label_dimensions = {}
            
            cache_key = f"{camera_name}_dimensions"
            if cache_key not in self._label_dimensions:
                label.update_idletasks()
                self._label_dimensions[cache_key] = (label.winfo_width(), label.winfo_height())
            
            label_width, label_height = self._label_dimensions[cache_key]
            
            # Only resize if label has valid dimensions
            if label_width > 1 and label_height > 1:
                # Cache display dimensions calculation
                if f"{cache_key}_display" not in self._label_dimensions:
                    # Force consistent display size for both cameras (720p aspect ratio)
                    target_aspect_ratio = 16 / 9  # 1280x720 = 16:9
                    
                    # Use minimal margin
                    margin = 3
                    available_width = label_width - (2 * margin)
                    available_height = label_height - (2 * margin)
                    
                    # Calculate standardized size based on available space and 16:9 ratio
                    if available_width / available_height > target_aspect_ratio:
                        # Available space is wider than 16:9, constrain by height
                        display_height = available_height
                        display_width = int(display_height * target_aspect_ratio)
                    else:
                        # Available space is taller than 16:9, constrain by width
                        display_width = available_width
                        display_height = int(display_width / target_aspect_ratio)
                    
                    # Ensure dimensions don't exceed available space
                    display_width = min(display_width, available_width)
                    display_height = min(display_height, available_height)
                    
                    # Cache calculated dimensions and offsets
                    x_offset = (label_width - display_width) // 2
                    y_offset = (label_height - display_height) // 2
                    
                    self._label_dimensions[f"{cache_key}_display"] = {
                        'display_width': display_width,
                        'display_height': display_height,
                        'x_offset': x_offset,
                        'y_offset': y_offset
                    }
                
                # Use cached dimensions
                display_dims = self._label_dimensions[f"{cache_key}_display"]
                
                # Resize the camera image to exactly these dimensions (stretch if needed)
                # Use NEAREST for speed in real-time processing
                img = img.resize((display_dims['display_width'], display_dims['display_height']), Image.NEAREST)
                
                # Create a black background of the full label size
                final_img = Image.new('RGB', (label_width, label_height), 'black')
                
                # Paste the resized image
                final_img.paste(img, (display_dims['x_offset'], display_dims['y_offset']))
                img = final_img
            
            imgtk = ImageTk.PhotoImage(image=img)

            # Handle both Label and Canvas widgets
            if hasattr(label, 'configure') and 'image' in label.configure():
                # This is a Label widget
                label.imgtk = imgtk
                label.configure(image=imgtk)
            else:
                # This is a Canvas widget - use create_image method
                label.delete("all")  # Clear previous image
                label.create_image(0, 0, image=imgtk, anchor="nw")
                label.imgtk = imgtk  # Store reference to prevent garbage collection

    def calculate_grade(self, defect_dict):
        """Fallback grade calculation based on defect dictionary - simplified version"""
        total_defects = sum(defect_dict.values()) if defect_dict else 0

        if total_defects == 0:
            return {
                'grade': GRADE_G2_0,
                'text': 'Perfect (No Defects)',
                'total_defects': 0,
                'color': 'dark green'
            }
        elif total_defects <= 2:
            return {
                'grade': GRADE_G2_0,
                'text': f'Good (G2-0) - {total_defects} defects',
                'total_defects': total_defects,
                'color': 'dark green'
            }
        elif total_defects <= 6:
            return {
                'grade': GRADE_G2_2,
                'text': f'Fair (G2-2) - {total_defects} defects',
                'total_defects': total_defects,
                'color': 'orange'
            }
        else:
            return {
                'grade': GRADE_G2_4,
                'text': f'Poor (G2-4) - {total_defects} defects',
                'total_defects': total_defects,
                'color': 'red'
            }

    def update_live_grading_display(self):
        """Update the live grading display with current detection results using SS-EN 1611-1"""
        # Update mode status display
        if hasattr(self, 'mode_status_label'):
            if self.current_mode == "IDLE":
                self.mode_status_label.config(text=" ", foreground="gray")
            elif self.current_mode == "TRIGGER":
                self.mode_status_label.config(text="Waiting for Detection Process", foreground="blue")
            elif self.current_mode == "SCAN_PHASE":
                self.mode_status_label.config(text="Waiting for Detection Process", foreground="blue")
            elif self.current_mode == "CONTINUOUS":
                self.mode_status_label.config(text="Live Detection Active", foreground="green")
            else:
                self.mode_status_label.config(text="", foreground="black")

        # Update individual camera grades
        top_grade = self.live_grades["top"]
        bottom_grade = self.live_grades["bottom"]

        if isinstance(top_grade, dict):
            self.top_grade_label.config(text=top_grade['text'], foreground=top_grade['color'])
        else:
            self.top_grade_label.config(text=top_grade, foreground="gray")

        if isinstance(bottom_grade, dict):
            self.bottom_grade_label.config(text=bottom_grade['text'], foreground=bottom_grade['color'])
        else:
            self.bottom_grade_label.config(text=bottom_grade, foreground="gray")

        # Calculate combined grade using sophisticated method
        wood_detected = False
        top_surface_grade = None
        bottom_surface_grade = None

        # Get sophisticated grades from measurements if available
        if hasattr(self, 'live_measurements'):
            if self.live_measurements.get("top"):
                wood_detected = True
                top_surface_grade = self.determine_surface_grade(self.live_measurements["top"])

            if self.live_measurements.get("bottom"):
                wood_detected = True
                bottom_surface_grade = self.determine_surface_grade(self.live_measurements["bottom"])

        # Fallback to detection-based grading if measurements not available
        if not wood_detected:
            for camera_name in ["top", "bottom"]:
                if self.live_detections[camera_name]:
                    wood_detected = True
                    break

        # For scan mode, use the per-side grades from live_grades if available
        if self.current_mode == "SCAN_PHASE" and isinstance(top_grade, dict) and isinstance(bottom_grade, dict):
            top_surface_grade = top_grade['grade']
            bottom_surface_grade = bottom_grade['grade']
            wood_detected = True

        if wood_detected:
            # Use sophisticated grading if measurements available
            if top_surface_grade is not None or bottom_surface_grade is not None:
                final_grade = self.determine_final_grade(top_surface_grade, bottom_surface_grade)
                combined_text = f"Final Grade: {final_grade} (SS-EN 1611-1)"
                combined_color = self.get_grade_color(final_grade)
            else:
                # Fallback to simple combined defect counting
                combined_defects = {}
                for camera_name in ["top", "bottom"]:
                    if self.live_detections[camera_name]:
                        for defect, count in self.live_detections[camera_name].items():
                            combined_defects[defect] = combined_defects.get(defect, 0) + count

                combined_grade = self.calculate_grade(combined_defects)
                combined_text = combined_grade['text']
                combined_color = combined_grade['color']
                final_grade = None

            self.combined_grade_label.config(text=combined_text, foreground=combined_color)

            # Auto-grade functionality - COMPLETELY DISABLED
            # Grading only happens when beam clears in TRIGGER mode
            # This ensures accurate grading based on complete defect tracking
            pass

        else:
            self.combined_grade_label.config(text=" ", foreground="gray")

    def update_detection_details(self, camera_name, defect_dict, measurements=None):
        """Update the detection details display for a specific camera with SS-EN 1611-1 details"""
        # Determine which details text widget to update
        if camera_name == "top":
            details_widget = self.top_details
        elif camera_name == "bottom":
            details_widget = self.bottom_details
        else:
            return
        
        # Check if user is currently scrolling - if so, don't update
        if self._user_scrolling.get(camera_name, False):
            return
        
        # Format the detection details with sophisticated grading info
        if defect_dict and measurements:
            # Create a formatted string showing SS-EN 1611-1 grading details
            details_text = f"SS-EN 1611-1 Grading ({camera_name.title()} Camera):\n"
            
            # Show camera calibration info
            if camera_name == "top":
                details_text += f"Distance: {TOP_CAMERA_DISTANCE_CM}cm, Factor: {TOP_CAMERA_PIXEL_TO_MM:.3f}mm/px\n"
            else:
                details_text += f"Distance: {BOTTOM_CAMERA_DISTANCE_CM}cm, Factor: {BOTTOM_CAMERA_PIXEL_TO_MM:.3f}mm/px\n"
            
            total_defects = len(measurements)
            details_text += f"Wood Height: {WOOD_PALLET_WIDTH_MM}mm | Defects: {total_defects}\n"
            details_text += "═" * 50 + "\n"
            
            # Show individual defect analysis
            for i, (defect_type, size_mm, percentage) in enumerate(measurements, 1):
                individual_grade = self.grade_individual_defect(defect_type, size_mm, percentage)
                details_text += f"{i}. {defect_type.replace('_', ' ')}\n"
                details_text += f"   Size: {size_mm:.1f}mm ({percentage:.1f}% of width)\n"
                details_text += f"   Individual Grade: {individual_grade}\n"
                details_text += f"   Threshold Info: "
                
                # Show which threshold was applied using new grading system
                constants = GRADING_CONSTANTS.get(defect_type, {})
                limit = (0.10 * WOOD_PALLET_WIDTH_MM) + constants.get(individual_grade, 0)
                details_text += f"Limit: ≤{limit:.1f}mm (0.10×{WOOD_PALLET_WIDTH_MM}mm + {constants.get(individual_grade, 0)})\n"
                
                details_text += "\n"
            
            details_text += "═" * 50 + "\n"
            
            # Show surface grade determination
            surface_grade = self.determine_surface_grade(measurements)
            details_text += f"Final Surface Grade: {surface_grade}\n"
            
            # Show grade reasoning with detailed explanation
            if total_defects > 6:
                details_text += "Grade Reasoning: More than 6 defects detected\n"
                details_text += "SS-EN 1611-1 Rule: >6 defects = Automatic G2-4"
            elif total_defects > 4:
                details_text += "Grade Reasoning: More than 4 defects detected\n"
                details_text += "SS-EN 1611-1 Rule: >4 defects = Maximum G2-3"
            elif total_defects > 2:
                details_text += "Grade Reasoning: More than 2 defects detected\n"
                details_text += "SS-EN 1611-1 Rule: >2 defects = Maximum G2-2"
            else:
                details_text += "Grade Reasoning: Based on worst individual defect grade\n"
                details_text += "SS-EN 1611-1 Rule: ≤2 defects = Use individual grades"
                
        elif defect_dict:
            # Fallback to simple display if measurements not available
            details_text = f"Simple Detection ({camera_name.title()} Camera):\n"
            
            # Show camera info
            if camera_name == "top":
                details_text += f"Distance: {TOP_CAMERA_DISTANCE_CM}cm\n"
            else:
                details_text += f"Distance: {BOTTOM_CAMERA_DISTANCE_CM}cm\n"
            
            total_defects = sum(defect_dict.values())
            details_text += f"Total Defects: {total_defects}\n"
            details_text += "─" * 40 + "\n"
            
            # Sort defects by count (highest first)
            sorted_defects = sorted(defect_dict.items(), key=lambda x: x[1], reverse=True)
            
            for defect_type, count in sorted_defects:
                formatted_name = defect_type.replace('_', ' ').title()
                details_text += f"• {formatted_name}: {count} detected\n"
            
            details_text += "─" * 40 + "\n"
            details_text += f"Status: {len(defect_dict)} defect type(s) detected\n"
            details_text += "Note: Size measurements not available in simple mode"
        else:
            details_text = f"SS-EN 1611-1 Grading ({camera_name.title()}):\n"
            
            # Show camera calibration even when no detection
            if camera_name == "top":
                details_text += f"Distance: {TOP_CAMERA_DISTANCE_CM}cm, {TOP_CAMERA_PIXEL_TO_MM:.3f}mm/px\n"
            else:
                details_text += f"Distance: {BOTTOM_CAMERA_DISTANCE_CM}cm, {BOTTOM_CAMERA_PIXEL_TO_MM:.3f}mm/px\n"
            
            details_text += f"Wood Height: {WOOD_PALLET_WIDTH_MM}mm\n"
            details_text += "═" * 50 + "\n"
            details_text += "No wood or defects detected\n"
            details_text += "═" * 50 + "\n"
            details_text += "Status: Waiting for detection...\n"
            details_text += "\nReady to analyze:\n"
            details_text += "• Sound Knots (Live knots)\n"
            details_text += "• Unsound Knots (Dead/Missing/Crack knots)\n"
            details_text += "\nGrading according to SS-EN 1611-1 standard"
        
        # Only update if content has actually changed OR if this is the first update
        if (details_text != self._last_detection_content.get(camera_name, "") or 
            not self._last_detection_content.get(camera_name)):
            # Store current scroll position before update
            current_scroll_pos = details_widget.yview()[0]
            
            # Update the text widget
            details_widget.config(state=tk.NORMAL)
            details_widget.delete(1.0, tk.END)
            details_widget.insert(1.0, details_text)
            details_widget.config(state=tk.DISABLED)
            
            # Restore scroll position if user was scrolling, otherwise scroll to top for new content
            if current_scroll_pos > 0.1:  # If user had scrolled down
                try:
                    details_widget.yview_moveto(current_scroll_pos)
                except:
                    pass  # If restore fails, just continue
            else:
                details_widget.see(1.0)  # Scroll to top for new content
            
            # Cache the content
            self._last_detection_content[camera_name] = details_text

    def _identify_arduino_port(self):
        """Identify Arduino port by testing communication"""
        import glob

        # Get all potential serial ports
        ports_to_try = [
            # ACM ports (Arduino Uno R3, Leonardo, Micro with native USB)
            '/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2', '/dev/ttyACM3',
            # USB ports (Arduino Nano, Pro Mini with FTDI/CH340)
            '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3',
            # Reassigned ports (when disconnection occurs)
            '/dev/ttyUSB01', '/dev/ttyACM01',
            # Windows COM ports
            'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'COM10'
        ]

        # Also check for any ttyACM* or ttyUSB* devices that might exist
        acm_ports = glob.glob('/dev/ttyACM*')
        usb_ports = glob.glob('/dev/ttyUSB*')
        all_ports = list(set(ports_to_try + acm_ports + usb_ports))

        arduino_ports = []

        for port in all_ports:
            try:
                print(f"Testing Arduino port {port}...")
                # Try to open the port (same settings as setup_arduino)
                ser = serial.Serial(
                    port=port,
                    baudrate=9600,
                    timeout=2,
                    write_timeout=2,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    xonxoff=False,
                    rtscts=False,
                    dsrdtr=False
                )

                # Clear buffers
                ser.reset_input_buffer()
                ser.reset_output_buffer()

                # Test communication by sending stop command (same as setup_arduino)
                ser.write(b'X')
                ser.flush()
                time.sleep(0.5)  # Give Arduino time to process

                # If we get here without exception, port is accessible
                arduino_ports.append(port)
                print(f"✅ Found accessible serial port {port}")
                ser.close()

            except (serial.SerialException, OSError, UnicodeDecodeError) as e:
                print(f"❌ Port {port} not accessible: {e}")
                continue

        return arduino_ports

    def setup_arduino(self):
        # Don't attempt to setup Arduino if shutting down
        if hasattr(self, '_shutting_down') and self._shutting_down:
            return

        try:
            # Close existing connection if any
            if hasattr(self, 'ser') and self.ser:
                try:
                    self.ser.close()
                except:
                    pass

            # Try dynamic Arduino identification first
            arduino_ports = self._identify_arduino_port()
            if arduino_ports:
                port = arduino_ports[0]  # Use first found Arduino
                print(f"Using dynamically identified Arduino port: {port}")
            else:
                # Fallback to manual port list
                ports_to_try = [
                    '/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2', '/dev/ttyACM3',
                    '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3',
                    '/dev/ttyUSB01', '/dev/ttyACM01',
                    'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'COM10'
                ]
                port = None
                for p in ports_to_try:
                    try:
                        print(f"Trying to connect to Arduino on {p}...")
                        ser = serial.Serial(
                            port=p,
                            baudrate=9600,
                            timeout=2,
                            write_timeout=2,
                            bytesize=serial.EIGHTBITS,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE,
                            xonxoff=False,
                            rtscts=False,
                            dsrdtr=False
                        )
                        time.sleep(2)
                        ser.reset_input_buffer()
                        ser.reset_output_buffer()
                        ser.write(b'X')
                        ser.flush()
                        time.sleep(0.5)
                        port = p
                        ser.close()
                        print(f"✅ Arduino connected successfully on {p}")
                        break
                    except (serial.SerialException, OSError):
                        continue

                if not port:
                    raise serial.SerialException("No Arduino found on any port")

            # Connect to the identified port
            self.ser = serial.Serial(
                port=port,
                baudrate=9600,
                timeout=2,
                write_timeout=2,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False
            )
            time.sleep(3)  # Allow Arduino to stabilize

            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

            # Start Arduino listener thread if not already running and not shutting down
            if not (hasattr(self, '_shutting_down') and self._shutting_down):
                if not hasattr(self, 'arduino_thread') or not self.arduino_thread.is_alive():
                    self.arduino_thread = threading.Thread(target=self.listen_for_arduino, daemon=True)
                    self.arduino_thread.start()

            if hasattr(self, 'status_label'):
                self.update_status_text("Status: Arduino connected. Ready for automatic detection.")
                self.update_status_text("Status: Ready - Waiting for IR beam trigger", STATUS_READY_COLOR)

        except serial.SerialException as e:
            self.ser = None
            print(f"Arduino connection failed: {e}")
            if hasattr(self, 'status_label'):
                self.update_status_text("Status: Arduino not found. Running in manual mode.", STATUS_WARNING_COLOR)

    def process_message_queue(self):
        """Process messages from background threads safely in the main thread"""
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                
                if msg_type == "arduino_message":
                    message = data

                    # --- IR BEAM HANDLING (Arduino sends "B" for beam broken) ---
                    if message == "B":
                        print("🔥 ARDUINO SENT 'B' - IR BEAM BROKEN DETECTED! 🔥")
                        # Respond to IR triggers in TRIGGER or SCAN_PHASE mode
                        if self.current_mode == "TRIGGER":
                            if not self.auto_detection_active:
                                print("✅ TRIGGER MODE: Starting detection...")
                                print("🔧 Arduino should now set motorActiveForTrigger = true")
                                print("⚡ Stepper motor should start running NOW!")

                                if hasattr(self, 'status_label'):
                                    # status_label is a Text widget, not a Label widget
                                    self.status_label.config(state=tk.NORMAL)
                                    self.status_label.delete(1.0, tk.END)
                                    self.status_label.insert(1.0, "Status: IR TRIGGERED - Motor should be running!")
                                    self.status_label.config(foreground="orange", state=tk.DISABLED)
                                self.start_automatic_detection()
                                # Keep auto_grade_var False in TRIGGER mode - grading triggered by beam clear
                                print(f"After 'B': live_detection_var: {self.live_detection_var.get()}, auto_grade_var: {self.auto_grade_var.get()}")
                            else:
                                print("⚠️ IR beam broken but detection already active")
                        elif self.current_mode == "SCAN_PHASE":
                            if not self.scan_phase_active:
                                print("✅ SCAN_PHASE MODE: Starting segmented scanning...")
                                print("🔧 Arduino should now start segmented scanning")
                                print("⚡ Stepper motor should start running NOW!")

                                if hasattr(self, 'status_label'):
                                    self.status_label.config(state=tk.NORMAL)
                                    self.status_label.delete(1.0, tk.END)
                                    self.status_label.insert(1.0, "Status: SCAN_PHASE STARTED - Scanning segments...")
                                    self.status_label.config(foreground="orange", state=tk.DISABLED)
                                self.start_scan_phase()
                            else:
                                print("⚠️ IR beam broken but scan already active")
                        else:
                            # In IDLE or CONTINUOUS mode, just log the IR signal but don't act on it
                            print(f"❌ IR beam broken received but system is in {self.current_mode} mode - ignoring trigger")
                        continue  # skip other checks for this message

                    # --- SCAN PHASE SEGMENT HANDLING ---
                    if "complete. Pausing" in message and self.scan_phase_active:
                        # Extract segment number
                        try:
                            segment_num = int(message.split()[1])
                            self.capture_segment_frame(segment_num)
                        except (ValueError, IndexError):
                            print(f"Could not parse segment number from: {message}")
                        continue

                    # --- PAUSE COMPLETE HANDLING ---
                    if message == "PAUSE_COMPLETE":
                        print("Arduino signaled pause complete - resuming live feed")
                        self.resume_live_feed()
                        continue

                    # --- SCAN PHASE COMPLETION ---
                    if "Last scan phase complete" in message and self.scan_phase_active:
                        self.grade_all_woods()
                        continue

                    # --- CAPTURE HANDLING (Arduino sends "CAPTURE:X" for frame capture) ---
                    if message.startswith("CAPTURE:"):
                        try:
                            segment_num = int(message.split(':')[1])
                            print(f"Arduino signaled to capture segment {segment_num}")

                            # Set display to "No Wood Graded" when beam is blocked (moved here as cue)
                            self.live_grades = {"top": " ", "bottom": " "}
                            self.update_live_grading_display()

                            self.capture_segment_frame(segment_num)
                        except (ValueError, IndexError):
                            print(f"Could not parse CAPTURE message: {message}")
                        continue  # skip other checks for this message

                    # --- RESUME LIVE FEED HANDLING (Arduino sends "RESUME_LIVE_FEED" after pause) ---
                    if message == "RESUME_LIVE_FEED":
                        print("Arduino signaled to resume live feed")
                        self.resume_live_feed()
                        continue  # skip other checks for this message

                    # --- LENGTH HANDLING (Arduino sends "L:duration" when beam clears) ---
                    if message.startswith("L:"):
                        try:
                            duration_ms = int(message.split(':')[1])
                            # Length calculation removed - only measuring defect width

                            # In TRIGGER mode, process grading if detection was active
                            if self.current_mode == "TRIGGER":
                                # Stop detection when beam clears, regardless of whether detection was active
                                self.auto_grade_var.set(False)
                                print("IR beam cleared – stopping detection")
                                print(f"After 'L': live_detection_var: {self.live_detection_var.get()}, auto_grade_var: {self.auto_grade_var.get()}")

                                if self.auto_detection_active:
                                    if hasattr(self, 'status_label'):
                                        # status_label is a Text widget, not a Label widget
                                        self.status_label.config(state=tk.NORMAL)
                                        self.status_label.delete(1.0, tk.END)
                                        self.status_label.insert(1.0, "Status: Processing results...")
                                        self.status_label.config(foreground="red", state=tk.DISABLED)
                                    self.stop_automatic_detection_and_grade()
                                    if hasattr(self, 'status_label'):
                                        # status_label is a Text widget, not a Label widget
                                        self.status_label.config(state=tk.NORMAL)
                                        self.status_label.delete(1.0, tk.END)
                                        self.status_label.insert(1.0, "Status: Ready - Waiting for IR beam trigger")
                                        self.status_label.config(foreground="green", state=tk.DISABLED)
                                else:
                                    print(f"Length signal received (duration: {duration_ms}ms) but no detection was active")
                            else:
                                print(f"Length signal received (duration: {duration_ms}ms) but system is in {self.current_mode} mode")
                        except (ValueError, IndexError):
                            print(f"Could not parse length message: {message}")
                        continue  # skip other checks for this message

                    # --- SCAN COMPLETE HANDLING ---
                    elif "Last scan phase complete" in message:
                        print("Arduino signaled scan phase complete")
                        self.grade_all_woods()
                        continue

                    # --- OTHER ARDUINO MESSAGES ---
                    else:
                        print(f"Arduino message received: '{message}'")
                        if hasattr(self, 'status_label'):
                            # status_label is a Text widget, not a Label widget
                            self.status_label.config(state=tk.NORMAL)
                            self.status_label.delete(1.0, tk.END)
                            self.status_label.insert(1.0, f"Status: Arduino: {message}")
                            self.status_label.config(state=tk.DISABLED)

                elif msg_type == "status_update":
                    if hasattr(self, 'status_label'):
                        # status_label is a Text widget, not a Label widget
                        self.status_label.config(state=tk.NORMAL)
                        self.status_label.delete(1.0, tk.END)
                        self.status_label.insert(1.0, f"Status: {data}")
                        self.status_label.config(state=tk.DISABLED)
                    
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error in process_message_queue: {e}")
        
        # Schedule next check
        self.after(50, self.process_message_queue)

    def listen_for_arduino(self):
        """Robust Arduino listener with automatic reconnection"""
        reconnect_attempts = 0
        max_reconnect_attempts = 10  # Increased max attempts

        while True:
            try:
                # Check if we're shutting down
                if hasattr(self, '_shutting_down') and self._shutting_down:
                    print("Arduino listener: Shutdown detected, exiting thread")
                    break

                # Check if serial connection exists and is open
                if self.ser and hasattr(self.ser, 'is_open') and self.ser.is_open:
                    if self.ser.in_waiting > 0:
                        try:
                            message = self.ser.readline().decode('utf-8').strip()
                            if not message:
                                continue

                            self.reset_inactivity_timer()
                            print(f"📨 Arduino Message: '{message}' (Port: {self.ser.port})")

                            # Put message in queue for main thread to process
                            self.message_queue.put(("arduino_message", message))
                            reconnect_attempts = 0  # Reset counter on successful communication

                        except UnicodeDecodeError as e:
                            print(f"⚠️ Arduino message decode error: {e}")
                            # Clear the buffer and continue
                            try:
                                self.ser.reset_input_buffer()
                            except:
                                pass
                            continue

                elif not self.ser or (hasattr(self.ser, 'is_open') and not self.ser.is_open):
                    # Serial connection is closed or doesn't exist
                    if reconnect_attempts < max_reconnect_attempts:
                        reconnect_attempts += 1
                        print(f"🔄 Arduino disconnected, attempting reconnection {reconnect_attempts}/{max_reconnect_attempts}...")
                        if not self.arduino_disconnected_popup_shown:
                            messagebox.showwarning("Arduino Disconnection", "Arduino has been disconnected. Attempting reconnection...")
                            self.arduino_disconnected_popup_shown = True
                        time.sleep(3)  # Increased wait time for Arduino to stabilize

                        # Try to reconnect with multiple attempts per reconnection cycle
                        reconnected = False
                        for attempt in range(3):  # Try 3 times per reconnection attempt
                            try:
                                self.setup_arduino()
                                if self.ser and self.ser.is_open:
                                    print(f"✅ Arduino reconnected successfully on {self.ser.port}")
                                    messagebox.showinfo("Arduino Reconnection", f"Arduino has been reconnected on {self.ser.port}")
                                    self.arduino_disconnected_popup_shown = False
                                    reconnect_attempts = 0
                                    reconnected = True
                                    break
                                else:
                                    print(f"❌ Reconnection sub-attempt {attempt + 1} failed, retrying...")
                                    time.sleep(1)
                            except Exception as e:
                                print(f"❌ Reconnection sub-attempt {attempt + 1} failed: {e}")
                                time.sleep(1)

                        if not reconnected:
                            print(f"❌ All reconnection sub-attempts failed for attempt {reconnect_attempts}")
                    else:
                        print(f"❌ Max reconnection attempts ({max_reconnect_attempts}) reached, exiting listener thread")
                        break

                time.sleep(0.1)  # Small delay to prevent CPU spinning

            except (serial.SerialException, OSError, TypeError) as e:
                print(f"🔥 Arduino communication error: {e}")

                # Check if this is due to application shutdown
                if hasattr(self, '_shutting_down') and self._shutting_down:
                    print("Arduino listener: Application shutting down, exiting thread")
                    break

                # Attempt reconnection with enhanced logic
                if reconnect_attempts < max_reconnect_attempts:
                    reconnect_attempts += 1
                    print(f"🔄 Communication error, attempting reconnection {reconnect_attempts}/{max_reconnect_attempts}...")
                    time.sleep(3)  # Increased wait time

                    # Multiple reconnection attempts
                    reconnected = False
                    for attempt in range(3):
                        try:
                            self.setup_arduino()
                            if self.ser and self.ser.is_open:
                                print(f"✅ Arduino reconnected after error on {self.ser.port}")
                                reconnect_attempts = 0
                                reconnected = True
                                break
                            time.sleep(1)
                        except Exception as reconnect_error:
                            print(f"❌ Reconnection sub-attempt {attempt + 1} failed: {reconnect_error}")
                            time.sleep(1)

                    if not reconnected:
                        print(f"❌ All reconnection attempts failed")
                else:
                    print(f"❌ Max reconnection attempts reached after error, exiting thread")
                    break

            except Exception as e:
                print(f"❌ Unexpected error in Arduino listener: {e}")
                time.sleep(1)
                self.message_queue.put(("status_update", "Arduino connection lost"))
                break

    def send_arduino_command(self, command):
        # Don't send commands if shutting down
        if hasattr(self, '_shutting_down') and self._shutting_down:
            return
            
        self.reset_inactivity_timer()
        
        # Add rate limiting to prevent overwhelming Arduino
        current_time = time.time()
        if hasattr(self, '_last_command_time'):
            time_since_last = current_time - self._last_command_time
            if time_since_last < 0.1:  # Minimum 100ms between commands
                time.sleep(0.1 - time_since_last)
        
        try:
            if self.ser:
                # Check if serial connection is still valid
                if not hasattr(self.ser, 'is_open') or not self.ser.is_open:
                    print("Serial connection is closed, attempting to reconnect...")
                    self.setup_arduino()
                    if not self.ser:
                        return
                
                # Clear buffers before sending command to prevent overflow
                try:
                    self.ser.reset_input_buffer()
                    self.ser.reset_output_buffer()
                except:
                    pass
                
                # Send command with error handling
                command_bytes = command.encode('utf-8')
                self.ser.write(command_bytes)
                self.ser.flush()  # Ensure data is sent immediately
                
                # Record timestamp for rate limiting
                self._last_command_time = time.time()
                
                print(f"✅ Sent command to Arduino: '{command}' (Port: {self.ser.port})")
            else:
                print("❌ Cannot send command: Arduino not connected.")
                if hasattr(self, 'status_label'):
                    # status_label is a Text widget, not a Label widget
                    self.status_label.config(state=tk.NORMAL)
                    self.status_label.delete(1.0, tk.END)
                    self.status_label.insert(1.0, "Status: Arduino not connected.")
                    self.status_label.config(state=tk.DISABLED)
                    
        except (serial.SerialException, OSError, TypeError) as e:
            print(f"🔥 Error sending Arduino command '{command}': {e}")
            if hasattr(self, 'status_label'):
                # status_label is a Text widget, not a Label widget
                self.status_label.config(state=tk.NORMAL)
                self.status_label.delete(1.0, tk.END)
                self.status_label.insert(1.0, "Status: Arduino communication error - attempting reconnect...")
                self.status_label.config(state=tk.DISABLED)
            
            # Try to reconnect only if not shutting down
            if not (hasattr(self, '_shutting_down') and self._shutting_down):
                print("🔄 Attempting Arduino reconnection...")
                self.ser = None
                time.sleep(1)  # Brief pause before reconnection attempt
                self.setup_arduino()

    def set_continuous_mode(self):
        """Sets the system to fully automatic continuous mode."""
        print("Setting Continuous (Live + Auto Grade) Mode")
        self.current_mode = "CONTINUOUS"
        self.send_arduino_command('C')  # Send command to Arduino
        self.live_detection_var.set(True)
        self.auto_grade_var.set(True)
        self.update_status_text("Status: CONTINUOUS", STATUS_READY_COLOR)
        self.update_detection_status_display() # Update the status label

    def set_trigger_mode(self):
        """Sets the system to wait for an IR beam trigger."""
        print("Setting Trigger Mode")
        self.current_mode = "TRIGGER"
        print(f"Sending 'T' command to Arduino...")
        self.send_arduino_command('T')  # Send command to Arduino
        self.live_detection_var.set(True)  # Enable live detection for triggering
        self.auto_grade_var.set(False)
        self.update_status_text("Status: TRIGGER", STATUS_READY_COLOR)
        self.update_detection_status_display() # Update the status label
        print(f"Trigger mode set - Python mode: {self.current_mode}")

    def set_idle_mode(self):
        """Disables all operations and stops the conveyor."""
        print("Setting IDLE Mode")
        self.current_mode = "IDLE"
        self.send_arduino_command('X')  # Send stop command to Arduino
        self.live_detection_var.set(False)
        self.auto_grade_var.set(False)
        self.auto_detection_active = False  # Ensure automatic detection is disabled
        # Clear wood detection results when entering idle mode
        self.wood_detection_results = {"top": None, "bottom": None}
        self.dynamic_roi = {}
        # Reset grades to empty when entering idle mode
        self.live_grades = {"top": "", "bottom": ""}
        self.update_live_grading_display()
        # status_label is a Text widget, not a Label widget
        self.status_label.config(state=tk.NORMAL)
        self.status_label.delete(1.0, tk.END)
        self.status_label.insert(1.0, "Status: IDLE")
        self.status_label.config(foreground="gray", state=tk.DISABLED)

    def set_scan_mode(self):
        """Sets the system to SCAN_PHASE mode for segmented scanning."""
        print("Setting SCAN_PHASE Mode")
        self.current_mode = "SCAN_PHASE"
        self.send_arduino_command('S')  # Send scan phase command to Arduino
        self.live_detection_var.set(False)  # Disable live detection - only show live feed with ROI
        self.auto_grade_var.set(False)  # Grading happens after scan completion
        self.update_status_text("Status: SCAN_PHASE", STATUS_READY_COLOR)
        self.update_detection_status_display()
        print(f"Scan mode set - Python mode: {self.current_mode}")

    def start_scan_phase(self):
        """Initialize scan phase when Arduino detects beam break in SCAN_PHASE mode."""
        print("Starting SCAN_PHASE detection...")
        self.scan_phase_active = True
        self.current_wood_number += 1  # Increment for each new wood piece
        self.scan_session_start_time = time.time()
        if not hasattr(self, 'scan_session_folder') or not self.scan_session_folder:
            self.scan_session_folder = self.create_session_folder()
        self.captured_frames = {"top": [], "bottom": []}
        self.segment_defects = {"top": [], "bottom": []}
        self.scan_session_data = {}

        # Update live grading display to show scanning in progress
        self.live_grades = {
            "top": "",
            "bottom": "",
            "combined": ""
        }
        self.update_live_grading_display()

        self.update_status_text("Status: SCAN_PHASE active", STATUS_READY_COLOR)

    def create_segment_visualization(self, frame, wood_detection_result, camera_name):
        """Create visualization - currently just returns the frame as-is (no overlays)."""
        # For now, return the frame without any overlays to focus on basic capture
        return frame.copy()

    def create_session_folder(self):
        """Create session folder with timestamp."""
        timestamp = datetime.now().strftime("%m%d%Y:%H%M") + "H-Session"
        folder = os.path.join("testIR", "Detections", timestamp)
        os.makedirs(folder, exist_ok=True)
        print(f"Created session folder: {folder}")
        return folder

    def capture_segment_frame(self, segment_num):
        """Capture and save frames for a specific segment with ROI-based detection."""
        print(f"Capturing frames for segment {segment_num}...")

        # Capture frames from both cameras
        ret_top, frame_top = self.cap_top.read()
        ret_bottom, frame_bottom = self.cap_bottom.read()

        if not ret_top or not ret_bottom:
            print(f"Failed to capture frames for segment {segment_num}")
            return

        # Flip bottom camera frame horizontally (matching the other app)
        frame_bottom = cv2.flip(frame_bottom, 1)

        # Process each camera with wood detection first, then defect detection
        processed_frames = {}
        for camera_name, frame in [("top", frame_top), ("bottom", frame_bottom)]:
            print(f"Processing {camera_name} camera for segment {segment_num}")

            # Step 1: Run wood detection ONLY within Yellow ROI (camera ROI)
            wood_detection_result = None
            if self.roi_enabled.get(camera_name, True):
                roi_coords = self.roi_coordinates.get(camera_name)
                if roi_coords:
                    x1, y1, x2, y2 = roi_coords["x1"], roi_coords["y1"], roi_coords["x2"], roi_coords["y2"]
                    yellow_roi_frame = frame[y1:y2, x1:x2]
                    print(f"Running wood detection on Yellow ROI: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                    wood_detection_result = self.rgb_wood_detector.detect_wood_comprehensive(
                        yellow_roi_frame, camera=camera_name
                    )

                    # Adjust bounding boxes back to full frame coordinates
                    if wood_detection_result.get('wood_detected', False):
                        for candidate in wood_detection_result['wood_candidates']:
                            bbox_x, bbox_y, bbox_w, bbox_h = candidate['bbox']
                            candidate['bbox'] = (bbox_x + x1, bbox_y + y1, bbox_w, bbox_h)

                        if wood_detection_result.get('auto_roi'):
                            roi_x, roi_y, roi_w, roi_h = wood_detection_result['auto_roi']
                            wood_detection_result['auto_roi'] = (roi_x + x1, roi_y + y1, roi_w, roi_h)
                else:
                    print(f"No Yellow ROI defined, skipping wood detection")
            else:
                print(f"Wood detection ROI disabled")

            # Store wood detection results
            self.wood_detection_results[camera_name] = wood_detection_result

            # The wood width is already updated by ColorWoodDetector.detect_wood_comprehensive()
            # via update_wood_width_dynamic(), so we don't need to recalculate it here

            # Step 2: Apply defect detection ONLY within detected wood area (Green ROI based on wood detection)
            if wood_detection_result and wood_detection_result.get('wood_detected', False) and wood_detection_result.get('auto_roi'):
                wood_roi = wood_detection_result['auto_roi']
                x1, y1, w, h = wood_roi
                x2, y2 = x1 + w, y1 + h

                print(f"Wood detected on {camera_name}, running defect detection on wood ROI: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                # Crop frame to detected wood area
                wood_roi_frame = frame[y1:y2, x1:x2]

                # Step 3: Run defect detection on wood ROI-cropped frame
                result = self.analyze_frame(wood_roi_frame, camera_name, run_defect_model=True)

                if len(result) == 3:
                    annotated_roi, defect_dict, detections_for_grading = result
                else:
                    annotated_roi, defect_dict = result
                    detections_for_grading = []

                # Step 4: Place annotated wood ROI back onto full frame
                full_frame_annotated = frame.copy()
                full_frame_annotated[y1:y2, x1:x2] = annotated_roi

                # Step 5: Add wood detection overlay to show detected wood pieces
                final_frame = self.draw_wood_detection_overlay(full_frame_annotated, camera_name)

                # Step 6: Add ROI overlay to show detection area
                final_frame = self.draw_roi_overlay(final_frame, camera_name)

                processed_frames[camera_name] = final_frame

                # Store detection results for this segment
                if not hasattr(self, 'segment_defects'):
                    self.segment_defects = {"top": [], "bottom": []}
                self.segment_defects[camera_name].append({
                    "segment": segment_num,
                    "defects": defect_dict,
                    "measurements": detections_for_grading
                })

            else:
                # No wood detected, show wood detection attempt
                print(f"No wood detected on {camera_name} in Yellow ROI")
                final_frame = self.draw_wood_detection_overlay(frame, camera_name)
                final_frame = self.draw_roi_overlay(final_frame, camera_name)
                processed_frames[camera_name] = final_frame

        # Update UI with processed frames FIRST (so user can see them before saving)
        self.display_captured_frames(processed_frames["top"], processed_frames["bottom"])

        # Save processed frames with ROI-based detection results AFTER display
        self.save_segment_frames(self.current_wood_number, segment_num,
                                processed_frames["top"], processed_frames["bottom"])

        print(f"Frames captured, ROI-processed, and saved for segment {segment_num}")

    def save_segment_frames(self, wood_number, segment_num, top_frame, bottom_frame):
        """Save captured frames for a specific segment of a wood piece."""
        wood_folder = os.path.join(self.scan_session_folder, f"Wood ({wood_number})")
        top_folder = os.path.join(wood_folder, "Top Panel")
        bottom_folder = os.path.join(wood_folder, "Bottom Panel")

        os.makedirs(top_folder, exist_ok=True)
        os.makedirs(bottom_folder, exist_ok=True)

        # Save frames with segment-specific names
        top_filename = f"detection_segment_{segment_num}.jpg"
        bottom_filename = f"detection_segment_{segment_num}.jpg"

        top_path = os.path.join(top_folder, top_filename)
        bottom_path = os.path.join(bottom_folder, bottom_filename)

        # Save frames directly (already in BGR format from camera)
        success_top = cv2.imwrite(top_path, top_frame)
        success_bottom = cv2.imwrite(bottom_path, bottom_frame)

        if success_top and success_bottom:
            print(f"Saved segment {segment_num} frames for Wood {wood_number}")
        else:
            print(f"Failed to save segment {segment_num} frames: top={success_top}, bottom={success_bottom}")

    def save_final_wood_data(self, wood_number, top_grade=None, bottom_grade=None, final_grade=None):
        """Save accumulated defect data after all segments are captured for a wood piece."""
        if wood_number not in self.scan_session_data:
            return

        wood_folder = os.path.join(self.scan_session_folder, f"Wood ({wood_number})")
        data = self.scan_session_data[wood_number]

        # Use per-wood width if available, else fallback
        # CRITICAL: session data 'width_mm' contains the detected_width_mm value stored during grading
        wood_width = data.get('width_mm', WOOD_PALLET_WIDTH_MM if WOOD_PALLET_WIDTH_MM > 0 else 100)
        print(f"🔗 Save data synchronization: Using wood_width={wood_width:.1f}mm (from session data 'width_mm'={data.get('width_mm', 'N/A')}, fallback WOOD_PALLET_WIDTH_MM={WOOD_PALLET_WIDTH_MM:.1f}mm)")
        print(f"🔍 Authority chain: detected_width_mm → session_data['width_mm'] → wood_width = {wood_width:.1f}mm")

        # Use provided grades or get from live_grades
        if top_grade is None:
            top_grade = self.live_grades.get("top", {}).get("grade", "Unknown") if isinstance(self.live_grades.get("top"), dict) else "Unknown"
        if bottom_grade is None:
            bottom_grade = self.live_grades.get("bottom", {}).get("grade", "Unknown") if isinstance(self.live_grades.get("bottom"), dict) else "Unknown"
        if final_grade is None:
            grader = SSEN1611_1_PineGrader_Final(width_mm=wood_width)
            # Convert grade strings back to measurements for proper grading
            # For now, use placeholder - this should be improved to store actual measurements
            final_grade = grader.determine_final_grade([], [])  # Empty measurements for now

        # Defect display name mapping
        defect_display_names = {
            "Sound_Knot": "Live Knot",
            "Dead_Knot": "Dead Knot",
            "Missing_Knot": "Missing Knot",
            "Crack_Knot": "Knot with Crack",
            "Unsound_Knot": "Unsound Knot"  # Includes Crack Knots
        }

        # Save accumulated defect details
        with open(os.path.join(wood_folder, "defects.txt"), "w") as f:
            f.write(f"Wood No. ({wood_number}) - {wood_width}mm\n")
            f.write(f"Top Panel Grade: {top_grade}\n")
            f.write(f"Bottom Panel Grade: {bottom_grade}\n")
            f.write(f"Final Grade: {final_grade}\n\n")
            f.write("Top Panel Defects:\n")
            if data['top_defects']:
                for i, (defect_type, size_mm, percentage) in enumerate(data['top_defects'], 1):
                    display_name = defect_display_names.get(defect_type, defect_type.replace('_', ' '))
                    f.write(f"{i}. {display_name} - {size_mm:.1f}mm\n")
            else:
                f.write("No defects detected\n")
            f.write("\nBottom Panel Defects:\n")
            if data['bottom_defects']:
                for i, (defect_type, size_mm, percentage) in enumerate(data['bottom_defects'], 1):
                    display_name = defect_display_names.get(defect_type, defect_type.replace('_', ' '))
                    f.write(f"{i}. {display_name} - {size_mm:.1f}mm\n")
            else:
                f.write("No defects detected\n")

        print(f"Saved final defect data for Wood {wood_number}")

    def _display_frame_on_canvas(self, frame, canvas):
        """Convert frame to PhotoImage and display on canvas at 360p resolution, centered"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)

            # Always resize to 360p (640x360) resolution for display
            display_width = 640
            display_height = 360
            pil_image = pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)

            # Store reference to prevent garbage collection
            if canvas == self.top_canvas:
                self._top_photo = photo
            elif canvas == self.bottom_canvas:
                self._bottom_photo = photo

            # Clear canvas and display on canvas, centered
            canvas.delete("all")  # Clear any existing content

            # Calculate center position
            canvas_width = self.canvas_width
            canvas_height = self.canvas_height
            x = (canvas_width - display_width) // 2
            y = (canvas_height - display_height) // 2

            canvas.create_image(x, y, anchor=tk.NW, image=photo)

        except Exception as e:
            print(f"Error displaying frame on canvas: {e}")
            import traceback
            traceback.print_exc()

    def display_captured_frames(self, top_frame, bottom_frame):
        """Display captured frames with overlays in the UI canvases."""
        # Convert to PhotoImage and display both processed frames immediately
        self._display_frame_on_canvas(top_frame, self.top_canvas)
        self._display_frame_on_canvas(bottom_frame, self.bottom_canvas)

        # Set flag to indicate we're displaying processed frames
        self.displaying_processed_frame = True

        # Schedule live feed resumption after 3000ms
        if self.processed_frame_timer:
            self.after_cancel(self.processed_frame_timer)
        self.processed_frame_timer = self.after(3000, self.resume_live_feed)

    def resume_live_feed(self):
        """Resume live feed after processed frame display period"""
        self.displaying_processed_frame = False
        self.processed_frame_timer = None
        print("Resumed live feed after processed frame display")

    def grade_all_woods(self):
        """Grade all detected woods after scan phase completion using segment defect data."""
        print("Grading all detected woods from segment data...")

        # Only process if we have exactly one wood piece (current implementation processes all segments as one wood)
        if self.current_wood_number > 0:
            wood_num = self.current_wood_number  # Process only the current wood
            print(f"Processing Wood {wood_num}...")

            # Collect all defects from all segments for this wood piece
            wood_top_defects = []
            wood_bottom_defects = []

            # Aggregate defects from all segments
            for segment_data in self.segment_defects["top"]:
                if segment_data.get("measurements"):
                    wood_top_defects.extend(segment_data["measurements"])

            for segment_data in self.segment_defects["bottom"]:
                if segment_data.get("measurements"):
                    wood_bottom_defects.extend(segment_data["measurements"])

            # Deduplicate defects before grading to prevent overcounting
            # Convert defect tuples to detection dicts for deduplication
            top_detections_for_dedup = []
            bottom_detections_for_dedup = []

            # Create detection dicts with proper ISO timestamps based on segment order
            base_time = datetime.now()
            segment_timestamp = 0
            for segment_data in self.segment_defects["top"]:
                if segment_data.get("measurements"):
                    segment_time = base_time + timedelta(seconds=segment_timestamp)
                    for defect_type, size_mm, percentage in segment_data["measurements"]:
                        top_detections_for_dedup.append({
                            'timestamp': segment_time.isoformat(),
                            'defect_type': defect_type,
                            'size_mm': size_mm,
                            'percentage': percentage
                        })
                segment_timestamp += 0.1  # Increment timestamp for each segment

            segment_timestamp = 0
            for segment_data in self.segment_defects["bottom"]:
                if segment_data.get("measurements"):
                    segment_time = base_time + timedelta(seconds=segment_timestamp)
                    for defect_type, size_mm, percentage in segment_data["measurements"]:
                        bottom_detections_for_dedup.append({
                            'timestamp': segment_time.isoformat(),
                            'defect_type': defect_type,
                            'size_mm': size_mm,
                            'percentage': percentage
                        })
                segment_timestamp += 0.1  # Increment timestamp for each segment

            # Deduplicate detections
            deduplicated_top = self.deduplicator.deduplicate(top_detections_for_dedup)
            deduplicated_bottom = self.deduplicator.deduplicate(bottom_detections_for_dedup)

            # Convert back to defect tuples for grading
            deduplicated_top_defects = [(d['defect_type'], d['size_mm'], d['percentage']) for d in deduplicated_top]
            deduplicated_bottom_defects = [(d['defect_type'], d['size_mm'], d['percentage']) for d in deduplicated_bottom]

            print(f"  Top defects: {len(wood_top_defects)} raw -> {len(deduplicated_top_defects)} deduplicated")
            print(f"  Bottom defects: {len(wood_bottom_defects)} raw -> {len(deduplicated_bottom_defects)} deduplicated")

            # Store deduplicated defects in scan session data
            # CRITICAL: Use the authoritative WOOD_PALLET_WIDTH_MM (which follows detected_width_mm)
            wood_width = WOOD_PALLET_WIDTH_MM if WOOD_PALLET_WIDTH_MM > 0 else 100  # Use current detected width
            print(f"🔗 Grading synchronization: Using wood_width={wood_width:.1f}mm (from WOOD_PALLET_WIDTH_MM={WOOD_PALLET_WIDTH_MM:.1f}mm)")
            print(f"🔍 Confirming detected_width_mm authority: wood_width={wood_width:.1f}mm follows detected_width_mm")
            
            self.scan_session_data[wood_num] = {
                'top_defects': deduplicated_top_defects,
                'bottom_defects': deduplicated_bottom_defects,
                'width_mm': wood_width  # This stores the detected_width_mm value
            }

            # Grade the wood piece using PineGrader
            # Use wood width for this piece
            grader = SSEN1611_1_PineGrader_Final(width_mm=wood_width)

            # Get grades from PineGrader
            all_measurements = deduplicated_top_defects + deduplicated_bottom_defects
            top_grade = grader.determine_surface_grade(deduplicated_top_defects)
            bottom_grade = grader.determine_surface_grade(deduplicated_bottom_defects)
            final_grade = grader.determine_final_grade(deduplicated_top_defects, deduplicated_bottom_defects)

            # Save final defect data
            self.save_final_wood_data(wood_num, top_grade, bottom_grade, final_grade)

            print(f"Wood {wood_num}: Top={top_grade}, Bottom={bottom_grade}, Final={final_grade}")
            print(f"  Total defects: Top={len(wood_top_defects)}, Bottom={len(wood_bottom_defects)}")

            # Update live grades for UI display (per-side grading)
            self.live_grades["top"] = {
                'grade': top_grade,
                'text': f'{top_grade} - SS-EN 1611-1 (Top Camera)',
                'color': self.get_grade_color(top_grade)
            }
            self.live_grades["bottom"] = {
                'grade': bottom_grade,
                'text': f'{bottom_grade} - SS-EN 1611-1 (Bottom Camera)',
                'color': self.get_grade_color(bottom_grade)
            }

            # Update the live grading display to show per-side grades
            self.update_live_grading_display()

            # Only call finalize_grading once for this wood piece
            self.finalize_grading(final_grade, all_measurements)
            
            # Clear segment defects after processing to prevent reprocessing
            self.segment_defects = {"top": [], "bottom": []}
            print(f"Cleared segment defects after processing Wood {wood_num}")

        self.scan_phase_active = False
        self.update_status_text("Status: SCAN_PHASE completed - grading finished", STATUS_READY_COLOR)

    def finalize_grading(self, final_grade, all_measurements):
        """Central function to log piece details, update stats, and send Arduino command."""
        # 1. Convert grade to Arduino command for sorting and stats
        arduino_command = self.convert_grade_to_arduino_command(final_grade)

        # 2. Increment piece count and create log entry
        self.total_pieces_processed += 1
        piece_number = self.total_pieces_processed
        
        defects_for_log = []
        if all_measurements:
            # Summarize defects for cleaner logging
            defect_summary = {}
            for defect_type, size_mm, percentage in all_measurements:
                # Group defects by type
                if defect_type not in defect_summary:
                    defect_summary[defect_type] = {'count': 0, 'sizes_mm': []}
                defect_summary[defect_type]['count'] += 1
                defect_summary[defect_type]['sizes_mm'].append(f"{size_mm:.1f}")

            # Format the summary for the log
            for defect_type, data in defect_summary.items():
                defects_for_log.append({
                    'type': defect_type.replace('_', ' '),
                    'count': data['count'],
                    'sizes': ', '.join(data['sizes_mm'])
                })
        
        log_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "piece_number": piece_number,
            "final_grade": final_grade,
            "defects": defects_for_log
        }
        self.session_log.append(log_entry)

        # 3. Update UI statistics (count by grade, not by command)
        grade_to_stat_index = {
            GRADE_G2_0: 1,
            GRADE_G2_1: 2,
            GRADE_G2_2: 3,
            GRADE_G2_3: 4,
            GRADE_G2_4: 5
        }
        stat_index = grade_to_stat_index.get(final_grade, 5)
        self.grade_counts[stat_index] += 1
        self.live_stats[f"grade{stat_index}"] += 1
        self.update_live_stats_display()

        # 4. Send command to Arduino if it's connected
        if self.ser and self.ser.is_open:
            self.send_arduino_command(str(arduino_command))
        else:
            print("Arduino not connected. Command not sent.")

        # 5. Update status label and console
        status_text = f"Piece #{piece_number} Graded: {final_grade} (Cmd: {arduino_command})"
        print(f"✅ Grading Finalized - {status_text}")
        self.update_status_text(f"Status: {status_text}", STATUS_READY_COLOR)
        self.log_action(f"Graded Piece #{piece_number} as {final_grade} -> Arduino Cmd: {arduino_command}")


    def _execute_manual_grade(self):
        """Execute manual grading based on current detections."""
        wood_detected = False
        all_measurements = self.live_measurements.get("top", []) + self.live_measurements.get("bottom", [])

        if all_measurements:
            wood_detected = True

        if wood_detected:
            # Use current wood width (default to 100mm if not detected)
            # CRITICAL: WOOD_PALLET_WIDTH_MM follows detected_width_mm as the authoritative source
            wood_width = WOOD_PALLET_WIDTH_MM if WOOD_PALLET_WIDTH_MM > 0 else 100
            print(f"🔗 Manual grading synchronization: Using wood_width={wood_width:.1f}mm (from WOOD_PALLET_WIDTH_MM={WOOD_PALLET_WIDTH_MM:.1f}mm)")
            print(f"🔍 Authority chain: detected_width_mm → WOOD_PALLET_WIDTH_MM → wood_width = {wood_width:.1f}mm")

            # Create grader instance
            grader = SSEN1611_1_PineGrader_Final(width_mm=wood_width)

            # Get final grade from PineGrader
            final_grade = grader.determine_final_grade(
                self.live_measurements.get("top", []),
                self.live_measurements.get("bottom", [])
            )
            print(f"Manual grade trigger - SS-EN 1611-1 Final grade: {final_grade}")
            self.finalize_grading(final_grade, all_measurements)
        else:
            print("Manual grade trigger - No wood currently detected")
            # status_label is a Text widget, not a Label widget
            self.status_label.config(state=tk.NORMAL)
            self.status_label.delete(1.0, tk.END)
            self.status_label.insert(1.0, "Status: Manual grade - no wood detected")
            self.status_label.config(state=tk.DISABLED)

    def analyze_frame(self, frame, camera_name="top", run_defect_model=True):
        """Analyze frame using DeGirum model with object tracking"""
        if self.model is None:
            return frame, {}, [], []

        try:
            # Run inference using DeGirum
            inference_result = self.model(frame)

            # Get annotated frame
            annotated_frame = inference_result.image_overlay

            # Process detections for object tracking
            current_detections = []
            for det in inference_result.results:
                model_label = det['label']
                bbox = det['bbox']
                confidence = det.get('confidence', 0.7)

                # Extract bounding box for size calculation
                bbox_info = {'bbox': bbox}

                # Calculate defect size in mm and percentage using camera-specific calibration
                size_mm, percentage = self.calculate_defect_size(bbox_info, camera_name)

                # Map to standard defect type
                standard_defect_type = self.map_model_output_to_standard(model_label)

                # Prepare detection for tracker (bbox, defect_type, size_mm, confidence)
                current_detections.append((bbox, standard_defect_type, size_mm, confidence))

            # Process detections (no ROI filtering since frame is already cropped to relevant area)
            detections_for_grading = []
            final_defect_dict = {}

            for bbox, defect_type, size_mm, confidence in current_detections:
                # Use detection information directly for grading (already within cropped area)
                detections_for_grading.append((defect_type, size_mm, 0.0))  # percentage not needed for grading

                # Count by defect type for display
                if defect_type in final_defect_dict:
                    final_defect_dict[defect_type] += 1
                else:
                    final_defect_dict[defect_type] = 1

            # Use the model's default overlay annotations
            # The image_overlay from DeGirum already includes appropriate defect labels

            return annotated_frame, final_defect_dict, detections_for_grading

        except Exception as e:
            print(f"Error during DeGirum inference on {camera_name} camera: {e}")
            return frame, {}, []

    def log_action(self, message):
        """Log actions to file with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} - {message}\n"
            
            with open("wood_sorting_activity_log.txt", "a") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Error logging action: {e}")

    def update_live_stats_display(self):
        """Update the live statistics display with enhanced tabbed interface and thread safety"""
        # Skip update if currently in active inference to prevent UI conflicts
        if getattr(self, '_in_active_inference', False):
            return
            
        # Safety check to ensure all required attributes exist
        if not hasattr(self, 'live_stats'):
            self.live_stats = {"grade0": 0, "grade1": 0, "grade2": 0, "grade3": 0}
        if not hasattr(self, 'live_stats_labels'):
            return  # Skip update if labels aren't initialized yet
            
        # Update basic grade counts in the Grade Summary tab with error handling
        try:
            for grade_key, count in self.live_stats.items():
                if grade_key in self.live_stats_labels and self.live_stats_labels[grade_key].winfo_exists():
                    # Use after_idle to ensure UI updates happen on main thread
                    self.after_idle(lambda key=grade_key, cnt=count: 
                                  self._safe_update_label(key, cnt))
        except Exception as e:
            print(f"Error updating live stats display: {e}")
        
        # Update other tabs with thread safety
        try:
            self.update_grade_details_tab()
            self.update_performance_tab()
            self.update_recent_activity_tab()
        except Exception as e:
            print(f"Error updating statistics tabs: {e}")
    
    def _safe_update_label(self, grade_key, count):
        """Safely update a label with error handling"""
        try:
            if (grade_key in self.live_stats_labels and 
                self.live_stats_labels[grade_key].winfo_exists()):
                self.live_stats_labels[grade_key].config(text=str(count))
        except Exception as e:
            print(f"Error updating label {grade_key}: {e}")

    def update_grade_details_tab(self):
        """Update the Grade Details tab with current grade information"""
        if not hasattr(self, 'grading_details_frame'):
            return
            
        # Clear existing content
        for widget in self.grading_details_frame.winfo_children():
            widget.destroy()
        
        # Grading thresholds reference - Dynamic based on current wood width
        thresholds_frame = ttk.LabelFrame(self.grading_details_frame, text="SS-EN 1611-1 Grading Thresholds", padding="5")
        thresholds_frame.pack(fill="x", pady=2)

        # Calculate dynamic thresholds based on current wood width
        wood_width = WOOD_PALLET_WIDTH_MM if WOOD_PALLET_WIDTH_MM > 0 else 115  # Default to 115mm if not detected

        threshold_text = f"SS-EN 1611-1 Grading Thresholds (Wood Width: {wood_width}mm):\n"
        threshold_text += "Limit = (0.10 × width) + constant\n\n"

        # Sound Knots thresholds
        sound_limits = {}
        for grade, constant in GRADING_CONSTANTS["Sound_Knot"].items():
            limit = (0.10 * wood_width) + constant
            sound_limits[grade] = limit

        threshold_text += "Sound Knots:\n"
        threshold_text += f"  G2-0: ≤{sound_limits.get('G2-0', 21.5):.1f}mm  |  G2-1: ≤{sound_limits.get('G2-1', 31.5):.1f}mm  |  G2-2: ≤{sound_limits.get('G2-2', 46.5):.1f}mm  |  G2-3: ≤{sound_limits.get('G2-3', 61.5):.1f}mm\n\n"

        # Dead Knots thresholds
        dead_limits = {}
        for grade, constant in GRADING_CONSTANTS["Dead_Knot"].items():
            limit = (0.10 * wood_width) + constant
            dead_limits[grade] = limit

        threshold_text += "Dead Knots:\n"
        threshold_text += f"  G2-0: ≤{dead_limits.get('G2-0', 11.5):.1f}mm  |  G2-1: ≤{dead_limits.get('G2-1', 21.5):.1f}mm  |  G2-2: ≤{dead_limits.get('G2-2', 31.5):.1f}mm  |  G2-3: ≤{dead_limits.get('G2-3', 61.5):.1f}mm\n\n"

        # Unsound Knots thresholds (includes Crack Knots)
        unsound_limits = {}
        for grade, constant in GRADING_CONSTANTS["Unsound_Knot"].items():
            limit = (0.10 * wood_width) + constant
            unsound_limits[grade] = limit

        threshold_text += "Unsound Knots (incl. Crack Knots):\n"
        threshold_text += f"  G2-2: ≤{unsound_limits.get('G2-2', 25.5):.1f}mm  |  G2-3: ≤{unsound_limits.get('G2-3', 50.5):.1f}mm\n\n"

        threshold_text += "Count Limits (Dead/Unsound per meter):\n"
        threshold_text += "  G2-0: 0  |  G2-1: 1  |  G2-2: 2  |  G2-3: 5\n\n"
        threshold_text += "Note: 'Knot with Crack' is classified as Unsound Knot"

        ttk.Label(thresholds_frame, text=threshold_text, font=self.font_small, justify="left", wraplength=600).pack(anchor="w")

    def update_performance_tab(self):
        """Update the Performance Metrics tab"""
        if not hasattr(self, 'performance_frame'):
            return
            
        # Clear existing content
        for widget in self.performance_frame.winfo_children():
            widget.destroy()
        
        # System calibration info
        calibration_frame = ttk.LabelFrame(self.performance_frame, text="System Calibration", padding="5")
        calibration_frame.pack(fill="x", pady=2)
        
        calibration_text = f"Wood Pallet Width: {WOOD_PALLET_WIDTH_MM}mm\n"
        calibration_text += f"Top Camera: {TOP_CAMERA_DISTANCE_CM}cm distance, {TOP_CAMERA_PIXEL_TO_MM:.3f}mm/px\n"
        calibration_text += f"Bottom Camera: {BOTTOM_CAMERA_DISTANCE_CM}cm distance, {BOTTOM_CAMERA_PIXEL_TO_MM:.3f}mm/px\n"
        calibration_text += "Standard: SS-EN 1611-1 European Wood Grading"
        
        ttk.Label(calibration_frame, text=calibration_text, font=self.font_small, justify="left").pack(anchor="w")
        
        # Processing speed metrics
        speed_frame = ttk.LabelFrame(self.performance_frame, text="Processing Metrics", padding="5")
        speed_frame.pack(fill="x", pady=2)
        
        total_processed = getattr(self, 'total_pieces_processed', 0)
        session_start = getattr(self, 'session_start_time', time.time())
        elapsed = time.time() - session_start
        
        if elapsed > 0 and total_processed > 0:
            pieces_per_minute = (total_processed / elapsed) * 60
            avg_processing_time = elapsed / total_processed
            
            speed_text = f"Total Processing Time: {elapsed/60:.1f} minutes\n"
            speed_text += f"Processing Rate: {pieces_per_minute:.1f} pieces/minute\n"
            speed_text += f"Average Time per Piece: {avg_processing_time:.2f} seconds"
        else:
            speed_text = "No processing data available yet"
        
        ttk.Label(speed_frame, text=speed_text, font=self.font_small, justify="left").pack(anchor="w")
        
        # Grade distribution
        distribution_frame = ttk.LabelFrame(self.performance_frame, text="Grade Distribution", padding="5")
        distribution_frame.pack(fill="x", pady=2)

        if total_processed > 0:
            grade_counts = getattr(self, 'grade_counts', {1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
            distribution_text = ""
            grade_names = {1: "G2-0", 2: "G2-1", 3: "G2-2", 4: "G2-3", 5: "G2-4"}

            for grade, count in grade_counts.items():
                percentage = (count / total_processed) * 100 if total_processed > 0 else 0
                distribution_text += f"Grade {grade} ({grade_names.get(grade, 'Unknown')}): {count} pieces ({percentage:.1f}%)\n"
        else:
            distribution_text = "No processing data available yet"

        ttk.Label(distribution_frame, text=distribution_text, font=self.font_small, justify="left").pack(anchor="w")

    def update_recent_activity_tab(self):
        """Update the Recent Activity tab with widened summary and scrollable processing log from defects.txt files"""
        # Update Session Summary (wider display) - always update this, regardless of scrolling
        for widget in self.session_summary_frame.winfo_children():
            widget.destroy()

        total_processed = getattr(self, 'total_pieces_processed', 0)
        session_start = getattr(self, 'session_start_time', time.time())
        session_duration = time.time() - session_start
        hours = int(session_duration // 3600)
        minutes = int((session_duration % 3600) // 60)

        # Create a wider summary display with better formatting
        summary_main_frame = ttk.Frame(self.session_summary_frame)
        summary_main_frame.pack(fill="x", expand=True)

        # Left column - Basic stats
        left_frame = ttk.Frame(summary_main_frame)
        left_frame.pack(side="left", fill="both", expand=True)

        basic_stats = f"Total Pieces Processed: {total_processed}\n"
        basic_stats += f"Session Duration: {hours}h {minutes}m\n"
        if total_processed > 0:
            avg_per_hour = (total_processed / session_duration) * 3600 if session_duration > 0 else 0
            basic_stats += f"Average Rate: {avg_per_hour:.1f} pieces/hour"

        ttk.Label(left_frame, text=basic_stats, font=self.font_small, justify="left").pack(anchor="w")

        # Right column - Grade distribution
        if total_processed > 0:
            right_frame = ttk.Frame(summary_main_frame)
            right_frame.pack(side="right", fill="both", expand=True)

            grade_counts = getattr(self, 'grade_counts', {1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
            grade_stats = "Grade Distribution:\n"
            grade_names = {1: "G2-0", 2: "G2-1", 3: "G2-2", 4: "G2-3", 5: "G2-4"}

            for grade, count in grade_counts.items():
                percentage = (count / total_processed) * 100 if total_processed > 0 else 0
                grade_stats += f"{grade_names.get(grade, 'Unknown')}: {count} ({percentage:.1f}%)\n"

            ttk.Label(right_frame, text=grade_stats, font=self.font_small, justify="left").pack(anchor="w")

        # Don't update log if user is scrolling through it
        if getattr(self, '_user_scrolling_log', False):
            return

        # Check if number of defects files has changed (only update processing log when grading cycle completes)
        defects_entries = self._read_defects_files()
        current_file_count = len(defects_entries)

        # Only update processing log if the number of defects files has changed (new wood processed) OR if this is the first update
        if (current_file_count != self._last_defects_files_count or not hasattr(self, '_last_stats_content')):
            self._last_defects_files_count = current_file_count

            # Update Processing Log from defects.txt files
            for widget in self.processing_log_frame.winfo_children():
                widget.destroy()

            if defects_entries:
                # Sort by timestamp (newest first)
                defects_entries.sort(key=lambda x: x['timestamp'], reverse=True)

                for i, entry in enumerate(defects_entries):
                    wood_number = entry.get('wood_number', 'Unknown')
                    final_grade = entry.get('final_grade', 'Unknown')
                    top_grade = entry.get('top_grade', 'Unknown')
                    bottom_grade = entry.get('bottom_grade', 'Unknown')
                    wood_width = entry.get('wood_width', 'Unknown')
                    top_defects = entry.get('top_defects', [])
                    bottom_defects = entry.get('bottom_defects', [])
                    timestamp = entry.get('timestamp', 'Unknown')
                    session_name = entry.get('session_name', 'Unknown')

                    # Create a frame for each log entry for better formatting
                    entry_frame = ttk.Frame(self.processing_log_frame)
                    entry_frame.pack(fill="x", pady=2, padx=5)

                    # Display the clean defects.txt format
                    log_text = f"[{timestamp}] Wood No. ({wood_number}) - {wood_width}mm\n"
                    log_text += f"Top Panel Grade: {top_grade}\n"
                    log_text += f"Bottom Panel Grade: {bottom_grade}\n"
                    log_text += f"Final Grade: {final_grade}\n\n"

                    if top_defects:
                        log_text += "Top Panel Defects:\n"
                        for j, defect in enumerate(top_defects, 1):
                            log_text += f"{j}. {defect.get('type', 'Unknown')} - {defect.get('size', 'Unknown')}mm\n"
                    else:
                        log_text += "Top Panel Defects:\nNo defects detected\n"

                    log_text += "\n"

                    if bottom_defects:
                        log_text += "Bottom Panel Defects:\n"
                        for j, defect in enumerate(bottom_defects, 1):
                            log_text += f"{j}. {defect.get('type', 'Unknown')} - {defect.get('size', 'Unknown')}mm\n"
                    else:
                        log_text += "Bottom Panel Defects:\nNo defects detected\n"

                    # Color code by grade
                    grade_colors = {
                        "G2-0": "dark green",
                        "G2-1": "green",
                        "G2-2": "orange",
                        "G2-3": "red",
                        "G2-4": "dark red"
                    }
                    text_color = grade_colors.get(final_grade, "black")

                    log_label = ttk.Label(entry_frame, text=log_text, font=("Arial", 9),
                                        justify="left", foreground=text_color)
                    log_label.pack(anchor="w", fill="x")

                    # Add separator line (except for last entry)
                    if i < len(defects_entries) - 1:
                        separator = ttk.Separator(self.processing_log_frame, orient="horizontal")
                        separator.pack(fill="x", pady=2)
            else:
                # Show message when no defects files exist
                no_data_label = ttk.Label(self.processing_log_frame,
                                        text="No processed wood data yet...\nDefects files will appear here after processing.",
                                        font=self.font_small, foreground="gray")
                no_data_label.pack(pady=20)

            # Cache the content
            self._last_stats_content = self._generate_stats_content()

            # Update scroll region for log
            if hasattr(self, 'log_canvas'):
                self.log_canvas.configure(scrollregion=self.log_canvas.bbox("all"))

    def update_detailed_statistics(self):
        """Legacy method - now redirects to update_recent_activity_tab for compatibility"""
        self.update_recent_activity_tab()

    def _read_defects_files(self):
        """Read defects.txt files from Detections directory and parse wood processing data"""
        import os
        import glob
        from datetime import datetime

        defects_entries = []

        # Look for defects.txt files in Detections directory
        detections_dir = os.path.join("testIR", "Detections")
        if not os.path.exists(detections_dir):
            return defects_entries

        # Find all defects.txt files
        defects_files = glob.glob(os.path.join(detections_dir, "**", "defects.txt"), recursive=True)

        for defects_file in defects_files:
            try:
                with open(defects_file, 'r') as f:
                    content = f.read()

                # Parse the defects.txt content
                lines = content.strip().split('\n')
                if not lines:
                    continue

                # Extract session and wood info from file path
                # Path format: testIR/Detections/MMDDYY:TT-Session/Wood (No.)/defects.txt
                path_parts = defects_file.split(os.sep)
                session_name = ""
                wood_number = 0

                for part in path_parts:
                    if "H-Session" in part:
                        session_name = part
                    elif part.startswith("Wood (") and part.endswith(")"):
                        wood_number = int(part.replace("Wood (", "").replace(")", ""))

                # Parse content
                entry = {
                    'session_name': session_name,
                    'wood_number': wood_number,
                    'timestamp': 'Unknown',
                    'final_grade': 'Unknown',
                    'top_grade': 'Unknown',
                    'bottom_grade': 'Unknown',
                    'wood_width': 'Unknown',
                    'top_defects': [],
                    'bottom_defects': []
                }

                # Extract timestamp from session name (MMDDYY:TT format)
                if session_name:
                    try:
                        # Parse MMDDYY:TT format
                        date_part, time_part = session_name.split(':')
                        month = int(date_part[:2])
                        day = int(date_part[2:4])
                        year = 2000 + int(date_part[4:6])
                        hour = int(time_part[:2])

                        # Create datetime object (assume current minute/second for display)
                        dt = datetime(year, month, day, hour, 0, 0)
                        entry['timestamp'] = dt.strftime("%m/%d/%Y %H:00")
                    except:
                        entry['timestamp'] = session_name

                current_section = None
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Parse header line (Wood No. (X) - Ymm)
                    if line.startswith("Wood No."):
                        # Extract wood width if available
                        if " - " in line:
                            width_part = line.split(" - ")[1]
                            entry['wood_width'] = width_part.replace("mm", "").strip()
                        continue

                    # Parse grade lines
                    elif "Top Panel Grade:" in line:
                        entry['top_grade'] = line.split(": ")[1].strip()
                    elif "Bottom Panel Grade:" in line:
                        entry['bottom_grade'] = line.split(": ")[1].strip()
                    elif "Final Grade:" in line:
                        entry['final_grade'] = line.split(": ")[1].strip()

                    # Parse defect sections
                    elif line == "Top Panel Defects:":
                        current_section = 'top'
                    elif line == "Bottom Panel Defects:":
                        current_section = 'bottom'
                    elif current_section and line.startswith("No defects detected"):
                        current_section = None
                    elif current_section and line[0].isdigit():
                        # Parse defect line: "1. Defect Type - Size mm"
                        try:
                            parts = line.split('. ', 1)[1]  # Remove number prefix
                            if ' - ' in parts:
                                defect_type, size_part = parts.split(' - ', 1)
                                size_mm = float(size_part.replace('mm', '').strip())

                                defect_info = {
                                    'type': defect_type.strip(),
                                    'size': size_mm
                                }

                                if current_section == 'top':
                                    entry['top_defects'].append(defect_info)
                                elif current_section == 'bottom':
                                    entry['bottom_defects'].append(defect_info)
                        except:
                            continue

                # Only add entries that have valid data
                if entry['final_grade'] != 'Unknown':
                    defects_entries.append(entry)

            except Exception as e:
                print(f"Error reading defects file {defects_file}: {e}")
                continue

        return defects_entries

    def _generate_stats_content(self):
        """Generate a string representation of current stats for change detection"""
        content = f"processed:{getattr(self, 'total_pieces_processed', 0)}"

        grade_counts = getattr(self, 'grade_counts', {1: 0, 2: 0, 3: 0})
        for grade, count in grade_counts.items():
            content += f",g{grade}:{count}"

        # Include session log count for change detection
        if hasattr(self, 'session_log'):
            content += f",log_entries:{len(self.session_log)}"

        return content


    def on_closing(self):
        print("Releasing resources...")
        
        # Set a flag to indicate shutdown is in progress
        self._shutting_down = True
        
        # Close Arduino connection first to stop the listener thread
        if hasattr(self, 'ser') and self.ser:
            try:
                print("Closing Arduino connection...")
                self.ser.close()
                self.ser = None
            except Exception as e:
                print(f"Error closing Arduino connection: {e}")
        
        # Wait a moment for the Arduino thread to exit gracefully
        if hasattr(self, 'arduino_thread') and self.arduino_thread.is_alive():
            print("Waiting for Arduino thread to close...")
            self.arduino_thread.join(timeout=2.0)  # Wait up to 2 seconds
        
        # Release camera resources using CameraHandler
        try:
            print("Releasing camera resources...")
            if hasattr(self, 'camera_handler'):
                self.camera_handler.release_cameras()
        except Exception as e:
            print(f"Error releasing cameras: {e}")
        
        # Close the application
        try:
            self.destroy()
        except Exception as e:
            print(f"Error during application shutdown: {e}")

    def reset_inactivity_timer(self):
        self.last_activity_time = time.time()
        if hasattr(self, 'report_generated'):
            self.report_generated = False

    def check_inactivity(self):
        # Generate report after 30 seconds of inactivity (auto-log feature)
        if (not self.report_generated and
            (time.time() - self.last_activity_time > 30) and
            self.total_pieces_processed > 0): # Only generate if something was processed
            self.generate_report()
            self.report_generated = True
        self.after(1000, self.check_inactivity)

    def update_feeds(self):
        """Update camera feeds on canvases, but skip if displaying processed frame"""
        if self.displaying_processed_frame:
            # Skip updating live feed while showing processed frame
            self.after(100, self.update_feeds)
            return

        try:
            # Read frames from cameras independently
            ret_top, frame_top = self.cap_top.read() if self.cap_top else (False, None)
            ret_bottom, frame_bottom = self.cap_bottom.read() if self.cap_bottom else (False, None)

            # Process top camera frame if available
            if ret_top and frame_top is not None:
                # Apply ROI overlay to live feed if enabled
                if self.roi_enabled.get("top", True):
                    frame_top = self.draw_roi_overlay(frame_top, "top")
                # Display on top canvas
                self._display_frame_on_canvas(frame_top, self.top_canvas)

            # Process bottom camera frame if available
            if ret_bottom and frame_bottom is not None:
                # Flip bottom frame horizontally (matching the other app)
                frame_bottom = cv2.flip(frame_bottom, 1)
                # Apply ROI overlay to live feed if enabled
                if self.roi_enabled.get("bottom", True):
                    frame_bottom = self.draw_roi_overlay(frame_bottom, "bottom")
                # Display on bottom canvas
                self._display_frame_on_canvas(frame_bottom, self.bottom_canvas)

        except Exception as e:
            print(f"Error updating feeds: {e}")

        # Schedule next update
        self.after(100, self.update_feeds)

    def generate_report(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_filename = f"report_{timestamp}"
        txt_filename = f"{base_filename}.txt"
        pdf_filename = f"{base_filename}.pdf"
        log_filename = "wood_sorting_log.txt"
        
        # --- Build Report Content ---
        content = f"--- SS-EN 1611-1 Wood Sorting Report ---\n"
        content += f"Generated at: {timestamp}\n\n"
        content += "--- Session Summary ---\n"
        content += f"Total Pieces Processed: {self.total_pieces_processed}\n"
        content += f"Grade G2-0: {self.grade_counts.get(1, 0)}\n"
        content += f"Grade G2-1: {self.grade_counts.get(2, 0)}\n"
        content += f"Grade G2-2: {self.grade_counts.get(3, 0)}\n"
        content += f"Grade G2-3: {self.grade_counts.get(4, 0)}\n"
        content += f"Grade G2-4: {self.grade_counts.get(5, 0)}\n"
        
        content += "\n\n--- Individual Piece Log ---\n"
        if not self.session_log:
            content += "No pieces were processed in this session.\n"
        else:
            for entry in self.session_log:
                content += f"\nPiece #{entry['piece_number']}: Grade {entry['final_grade']}\n"
                if not entry['defects']:
                    content += "  - No defects detected.\n"
                else:
                    for defect in entry['defects']:
                        content += f"  - Defect: {defect['type']}, Count: {defect['count']}, Sizes (mm): {defect['sizes']}\n"

        # Save individual report files
        try:
            with open(txt_filename, 'w') as f:
                f.write(content)
            print(f"SS-EN 1611-1 report generated: {txt_filename}")
        except Exception as e:
            print(f"Error generating TXT report: {e}")
            messagebox.showerror("Report Error", f"Could not save TXT report: {e}")
            return

        # Append to main log file
        try:
            log_entry = f"{timestamp} | Pieces: {self.total_pieces_processed} | G2-0: {self.grade_counts.get(1, 0)} | G2-1: {self.grade_counts.get(2, 0)} | G2-2: {self.grade_counts.get(3, 0)} | G2-3: {self.grade_counts.get(4, 0)} | G2-4: {self.grade_counts.get(5, 0)}\n"
            with open(log_filename, 'a') as f:
                f.write(log_entry)
            print(f"Entry added to log file: {log_filename}")
            self.log_status_label.config(text="Log: Updated", foreground="blue")
        except Exception as e:
            print(f"Error updating log file: {e}")
            self.log_status_label.config(text="Log: Error", foreground="red")

        # Generate PDF Report
        try:
            c = canvas.Canvas(pdf_filename, pagesize=letter)
            width, height = letter
            
            c.setFont("Helvetica-Bold", 16)
            c.drawCentredString(width / 2.0, height - 1*inch, "SS-EN 1611-1 Wood Sorting System Report")
            c.setFont("Helvetica", 12)
            
            text = c.beginText(1*inch, height - 1.5*inch)
            text.textLine(f"Generated at: {timestamp}")
            text.textLine("")
            text.setFont("Helvetica-Bold", 12)
            text.textLine("Session Summary")
            text.setFont("Helvetica", 12)
            text.textLine(f"Total Pieces Processed: {self.total_pieces_processed}")
            text.textLine(f"Grade G2-0: {self.grade_counts.get(1, 0)}")
            text.textLine(f"Grade G2-1: {self.grade_counts.get(2, 0)}")
            text.textLine(f"Grade G2-2: {self.grade_counts.get(3, 0)}")
            text.textLine(f"Grade G2-3: {self.grade_counts.get(4, 0)}")
            text.textLine(f"Grade G2-4: {self.grade_counts.get(5, 0)}")
            text.textLine("")
            text.textLine("")
            text.setFont("Helvetica-Bold", 12)
            text.textLine("Individual Piece Log")
            text.setFont("Helvetica", 12)

            if not self.session_log:
                text.textLine("No pieces were processed in this session.")
            else:
                for entry in self.session_log:
                    # Check if we need a new page before drawing the next entry
                    if text.getY() < 2 * inch:
                        c.drawText(text)
                        c.showPage()
                        c.setFont("Helvetica", 12)
                        text = c.beginText(1*inch, height - 1*inch)

                    text.textLine("")
                    text.setFont("Helvetica-Bold", 10)
                    text.textLine(f"Piece #{entry['piece_number']}: Grade {entry['final_grade']}")
                    text.setFont("Helvetica", 10)
                    if not entry['defects']:
                        text.textLine("  - No defects detected.")
                    else:
                        for defect in entry['defects']:
                            text.textLine(f"  - Defect: {defect['type']}, Count: {defect['count']}, Sizes (mm): {defect['sizes']}")
            
            c.drawText(text)
            c.save()
            print(f"SS-EN 1611-1 PDF report generated: {pdf_filename}")
            
            self.last_report_path = pdf_filename
            self.last_report_label.config(text=f"Last Report: {os.path.basename(self.last_report_path)}")
            
            # Show notification only if toggle is enabled
            if self.show_report_notification.get():
                messagebox.showinfo("SS-EN 1611-1 Report", f"Reports saved as {txt_filename} and {pdf_filename}\nLog updated: {log_filename}")

        except Exception as e:
            print(f"Error generating PDF report: {e}")
            messagebox.showerror("Report Error", f"Could not save PDF report: {e}")
            
        # Reset the session log after generating the report
        self.session_log = []
        print("Session log has been cleared for the next report.")


    def manual_generate_report(self):
        """Manually generate a report"""
        self.generate_report()
        self.log_status_label.config(text="Log: Manual report generated", foreground="green")

    def view_session_folder(self):
        """Open the current session folder in file explorer"""
        import os
        import subprocess
        import platform

        try:
            # Check if we have a current session folder
            if hasattr(self, 'scan_session_folder') and self.scan_session_folder and os.path.exists(self.scan_session_folder):
                folder_path = self.scan_session_folder
            else:
                # Fallback to the Detections directory
                folder_path = os.path.join("testIR", "Detections")
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path, exist_ok=True)

            # Open folder based on platform
            if platform.system() == "Windows":
                os.startfile(folder_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", folder_path])
            else:  # Linux
                subprocess.run(["xdg-open", folder_path])

            print(f"Opened session folder: {folder_path}")
            self.log_status_label.config(text="Log: Session folder opened", foreground="blue")

        except Exception as e:
            print(f"Error opening session folder: {e}")
            self.log_status_label.config(text="Log: Error opening folder", foreground="red")

    def _reassign_cameras_ui(self):
        """UI wrapper for camera reassignment"""
        try:
            success = self.camera_handler.reassign_cameras_runtime()
            if success:
                # Update the cap references
                self.cap_top = self.camera_handler.top_camera
                self.cap_bottom = self.camera_handler.bottom_camera
                messagebox.showinfo("Success", "Cameras reassigned successfully!")
            else:
                messagebox.showerror("Error", "Camera reassignment failed. Check console for details.")
        except Exception as e:
            messagebox.showerror("Error", f"Camera reassignment error: {e}")

    def _reassign_arduino_ui(self):
        """UI wrapper for Arduino reassignment"""
        try:
            success = self.reassign_arduino_runtime()
            if success:
                messagebox.showinfo("Success", "Arduino reassigned successfully!")
            else:
                messagebox.showerror("Error", "Arduino reassignment failed. Check console for details.")
        except Exception as e:
            messagebox.showerror("Error", f"Arduino reassignment error: {e}")

    def simulate_ir_events(self):
        """Simulate IR beam events for testing TRIGGER mode"""
        print("Simulating IR beam events for testing...")

        # Simulate 'B' message (beam broken)
        print("Simulating 'B' message (beam broken)...")
        self.message_queue.put(("arduino_message", "B"))

        # Wait a moment then simulate 'L' message (beam clear with duration)
        self.after(2000, lambda: self.simulate_beam_clear())

    def simulate_scan_events(self):
        """Simulate SCAN_PHASE events for testing"""
        print("Simulating SCAN_PHASE events for testing...")

        # Simulate 'B' message (beam broken)
        print("Simulating 'B' message (beam broken)...")
        self.message_queue.put(("arduino_message", "B"))

        # Simulate segment captures
        self.after(1000, lambda: self.simulate_segment_capture(1))
        self.after(3000, lambda: self.simulate_segment_capture(2))
        self.after(5000, lambda: self.simulate_segment_capture(3))
        self.after(7000, lambda: self.simulate_scan_complete())

    def simulate_segment_capture(self, segment_num):
        """Simulate segment capture message"""
        print(f"Simulating 'CAPTURE:{segment_num}'")
        self.message_queue.put(("arduino_message", f"CAPTURE:{segment_num}"))

    def simulate_scan_complete(self):
        """Simulate scan completion"""
        print("Simulating 'Last scan phase complete. Clearing tail...'")
        self.message_queue.put(("arduino_message", "Last scan phase complete. Clearing tail..."))

    def simulate_beam_clear(self):
        """Simulate beam clear message"""
        print("Simulating 'L:1000' message (beam clear, 1000ms duration)...")
        self.message_queue.put(("arduino_message", "L:1000"))

    def toggle_fullscreen(self, event=None):
        """Toggle fullscreen mode (F11 key)"""
        self.is_fullscreen = not self.is_fullscreen
        self.attributes("-fullscreen", self.is_fullscreen)
        return "break"

    def exit_fullscreen(self, event=None):
        """Exit fullscreen mode (Escape key)"""
        self.is_fullscreen = False
        self.attributes("-fullscreen", False)
        return "break"

    def auto_fullscreen_rpi(self):
        """Automatically enable fullscreen for Raspberry Pi deployment"""
        try:
            # Check if running on Raspberry Pi
            with open('/proc/cpuinfo', 'r') as f:
                if 'Raspberry Pi' in f.read():
                    print("Raspberry Pi detected - enabling fullscreen mode")
                    self.is_fullscreen = True
                    self.attributes("-fullscreen", True)
        except FileNotFoundError:
            # Not running on Raspberry Pi, continue normally
            pass
        except Exception as e:
            print(f"Error checking system: {e}")

if __name__ == "__main__":
    import sys
    app = App()
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        app.set_trigger_mode()
        app.after(1000, app.simulate_ir_events)  # Start simulation after 1 second
    elif len(sys.argv) > 1 and sys.argv[1] == 'scan':
        app.set_scan_mode()
        app.after(1000, app.simulate_scan_events)  # Start scan simulation after 1 second
    app.mainloop()

