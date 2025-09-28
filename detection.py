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
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import numpy as np

# =============================================================================
# CONFIGURATION SECTION - Fine-tune these parameters for your setup
# =============================================================================

# SS-EN 1611-1 Grading Standards Implementation
# Grade constants (do not modify)
GRADE_G2_0 = "G2-0"
GRADE_G2_1 = "G2-1"
GRADE_G2_2 = "G2-2"
GRADE_G2_3 = "G2-3"
GRADE_G2_4 = "G2-4"

# -----------------------------------------------------------------------------
# CAMERA CALIBRATION SETTINGS
# Adjust these based on your camera distances and actual measurements
# -----------------------------------------------------------------------------
# Camera distances from wood surface (in cm)
TOP_CAMERA_DISTANCE_CM = 37      # Distance of top camera from wood surface
BOTTOM_CAMERA_DISTANCE_CM = 29    # Distance of bottom camera from wood surface

# Pixel-to-millimeter conversion factors (calibrate with known reference objects)
TOP_CAMERA_PIXEL_TO_MM = 0.4     # mm per pixel for top camera at 37cm distance
BOTTOM_CAMERA_PIXEL_TO_MM = 0.3   # mm per pixel for bottom camera at 29cm distance

# Wood pallet specifications
WOOD_PALLET_WIDTH_MM = 115        # Actual width of wood pallet in mm (11.5cm = 115mm)


# Detection confidence threshold (0.0 to 1.0)
DETECTION_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for defect detection

# -----------------------------------------------------------------------------
# CAMERA HARDWARE SETTINGS
# Adjust resolution and frame rate based on your camera capabilities
# -----------------------------------------------------------------------------
CAMERA_WIDTH = 1280              # Camera capture width in pixels
CAMERA_HEIGHT = 720              # Camera capture height in pixels
CAMERA_FPS = 30                  # Camera frame rate (frames per second)

# -----------------------------------------------------------------------------
# REGION OF INTEREST (ROI) SETTINGS
# Define areas to focus detection on (crop out irrelevant areas)
# Coordinates are in pixels: (x1, y1) to (x2, y2)
# -----------------------------------------------------------------------------
ROI_COORDINATES = {
    "top": {
        "x1": 400,  # Left boundary - exclude left equipment
        "y1": 0,   # Top boundary - exclude top area
        "x2": 850, # Right boundary - exclude right equipment
        "y2": 720   # Bottom boundary - focus on wood area
    },
    "bottom": {
        "x1": 150,    # Full frame for bottom camera (no ROI)
        "y1": 0,
        "x2": CAMERA_WIDTH,   # Use configured camera width
        "y2": CAMERA_HEIGHT   # Use configured camera height
    },
    "wood_detection": {
        "x1": 25,  # Left boundary for wood detection
        "y1": 0,    # Top boundary
        "x2": 100,  # Right boundary for wood detection
        "y2": 720   # Bottom boundary for wood detection
    },
    "exit_wood": {
        "x1": 1175,  # Left boundary for exit wood ROI
        "y1": 0,    # Top boundary
        "x2": 1250, # Right boundary for exit wood ROI
        "y2": 720   # Bottom boundary for exit wood ROI
    }
}

# -----------------------------------------------------------------------------
# SS-EN 1611-1 GRADING THRESHOLDS
# Thresholds for defect size limits: (max_mm, max_percentage_of_width)
# Adjust these if your grading standards differ
# -----------------------------------------------------------------------------
GRADING_THRESHOLDS = {
    "Sound_Knot": {  # Live knots
        GRADE_G2_0: (10, 5),      # (mm, percentage) - Perfect grade
        GRADE_G2_1: (30, 15),     # Good grade
        GRADE_G2_2: (50, 25),     # Fair grade
        GRADE_G2_3: (70, 35),     # Lower fair grade
        GRADE_G2_4: (float('inf'), float('inf'))  # Poor grade (no limit)
    },
    "Unsound_Knot": {  # Dead knots, missing knots, knots with cracks
        GRADE_G2_0: (7, 3.5),     # Stricter limits for unsound knots
        GRADE_G2_1: (20, 10),
        GRADE_G2_2: (35, 17.5),
        GRADE_G2_3: (50, 25),
        GRADE_G2_4: (float('inf'), float('inf'))
    }
}

# -----------------------------------------------------------------------------
# DEEP LEARNING MODEL SETTINGS
# Configure the AI model for defect detection
# -----------------------------------------------------------------------------
INFERENCE_HOST_ADDRESS = "@local"  # Use local inference
MODEL_ZOO_URL = "/home/inspectura/Desktop/InspecturaGUI/models/V2DefectCombined--640x640_quant_hailort_hailo8_1"
MODEL_NAME = "V2DefectCombined--640x640_quant_hailort_hailo8_1"


# -----------------------------------------------------------------------------
# PERFORMANCE AND TIMING SETTINGS
# Adjust for system performance and responsiveness
# -----------------------------------------------------------------------------
FRAME_SKIP_DETECTION = 5          # Process detection every Nth frame (reduce CPU load)
FRAME_SKIP_UI_UPDATE = 3          # Update UI every Nth frame
FRAME_SKIP_DASHBOARD_UPDATE = 15  # Update dashboard every Nth frame (reduce load)
FRAME_SKIP_NO_DETECTION_UPDATE = 10  # Update when no detection every Nth frame
MEMORY_CLEANUP_INTERVAL = 300     # Force garbage collection every N frames

# -----------------------------------------------------------------------------
# ARDUINO/CONVEYOR SETTINGS
# Serial ports to try for Arduino connection (in order of preference)
# -----------------------------------------------------------------------------
ARDUINO_PORTS = [
    '/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2', '/dev/ttyACM3',  # Native USB
    '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3',  # FTDI/CH340
    '/dev/ttyUSB01', '/dev/ttyACM01',  # Reassigned ports
    '/dev/ttyAMA0', '/dev/ttyAMA1', '/dev/ttyAMA10',  # Other Linux
    'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'COM10'  # Windows
]

# =============================================================================
# END OF CONFIGURATION SECTION
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

# =============================================================================
# END OF UI CONFIGURATION SECTION
# =============================================================================
# =============================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wood Sorting Application")
        
        # Get screen dimensions for dynamic sizing
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        # Set to fullscreen
        self.attributes("-fullscreen", True)
        
        # Set minimum size to prevent too small windows
        self.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        
        # For Raspberry Pi - detect if running in fullscreen environment
        self.is_fullscreen = False
        self.bind("<F11>", self.toggle_fullscreen)
        self.bind("<Escape>", self.exit_fullscreen)
        
        # Auto-fullscreen for Raspberry Pi (configurable)
        if ENABLE_FULLSCREEN_STARTUP:
            self.after(100, self.auto_fullscreen_rpi)
        
        # Calculate responsive font sizes based on screen size
        base_font_size = max(FONT_SIZE_BASE_MIN, min(FONT_SIZE_BASE_MAX, int(screen_height / FONT_SIZE_DIVISOR)))  # Responsive base size
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

        # Create message queue for thread communication
        self.message_queue = queue.Queue()

        # Helper method for updating status label (Text widget)
        def update_status_text(text, color=None):
            self.status_label.config(state=tk.NORMAL)
            self.status_label.delete(1.0, tk.END)
            self.status_label.insert(1.0, text)
            if color:
                self.status_label.config(foreground=color)
            self.status_label.config(state=tk.DISABLED)

        self.update_status_text = update_status_text

        # Initialize variables that might be accessed early by message processing
        self.total_pieces_processed = 0
        self.session_start_time = time.time()  # Track session start time for statistics
        self.grade_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Include grade 0 for perfect wood
        self.report_generated = False
        self.last_report_path = None
        self.last_activity_time = time.time()
        self.live_stats = {"grade0": 0, "grade1": 0, "grade2": 0, "grade3": 0}  # Include grade 0
        self._shutting_down = False  # Flag to indicate shutdown in progress
        self.session_log = [] # New: Log for individual piece details

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
        
        # Detection tracking variables
        self.detection_session_id = None
        self.piece_counter = 0
        self.current_piece_data = None
        
        # System mode tracking
        self.current_mode = "IDLE"  # Can be "IDLE", "TRIGGER", or "CONTINUOUS"
        
        # Automatic detection state (triggered by IR beam)
        self.live_detection_var = tk.BooleanVar(value=False) # For live inference mode
        self.auto_grade_var = tk.BooleanVar(value=False) # For auto grading in live mode
        self._last_auto_grade_time = 0
        self._in_active_inference = False  # Flag to prevent UI conflicts during inference

        self.auto_detection_active = False
        self.detection_frames = []  # Store frames during detection
        self.detection_session_data = {
            "start_time": None,
            "end_time": None,
            "total_detections": {"top": [], "bottom": []},
            "best_frames": {"top": None, "bottom": None},
            "final_grade": None
        }

        # --- DeGirum Model and Camera Initialization ---
        # DeGirum Configuration
        self.inference_host_address = INFERENCE_HOST_ADDRESS
        self.zoo_url = MODEL_ZOO_URL
        self.model_name = MODEL_NAME
        
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
        
        # Initialize cameras with specific resolution
        self.cap_top = cv2.VideoCapture(0)
        self.cap_bottom = cv2.VideoCapture(2)
        
        # Set camera resolution (configurable values)
        camera_width = CAMERA_WIDTH  # Desired width
        camera_height = CAMERA_HEIGHT  # Desired height

        # Configure top camera
        self.cap_top.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        self.cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        self.cap_top.set(cv2.CAP_PROP_FPS, CAMERA_FPS)  # Set FPS

        # Configure bottom camera
        self.cap_bottom.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        self.cap_bottom.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        self.cap_bottom.set(cv2.CAP_PROP_FPS, CAMERA_FPS)  # Set FPS

        # Store camera resolution for display scaling
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        # Verify camera settings
        actual_top_width = self.cap_top.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_top_height = self.cap_top.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_bottom_width = self.cap_bottom.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_bottom_height = self.cap_bottom.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        print(f"Top camera resolution: {actual_top_width}x{actual_top_height}")
        print(f"Bottom camera resolution: {actual_bottom_width}x{actual_bottom_height}")

        # Live detection tracking
        self.live_detections = {"top": {}, "bottom": {}}
        self.live_grades = {"top": "No wood detected", "bottom": "No wood detected"}

        # ROI (Region of Interest) settings
        self.roi_enabled = {"top": True, "bottom": False, "wood_detection": True, "exit_wood": True}  # Enable ROI for top camera, wood detection, and exit wood by default
        self.roi_coordinates = ROI_COORDINATES.copy()

        # Create canvases for camera feeds taking full width
        self.canvas_width = screen_width // 2 - 25
        self.canvas_height = 360
        self.top_canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.top_canvas.place(x=25, y=25, width=self.canvas_width, height=self.canvas_height)

        self.bottom_canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.bottom_canvas.place(x=self.canvas_width + 25, y=25, width=self.canvas_width, height=self.canvas_height)

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

        # Detection panel under bottom camera
        detection_frame = ttk.LabelFrame(self, text="Detection", padding=FRAME_PADDING)
        detection_frame.place(x=675, y=415, width=250, height=125)

        self.roi_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(detection_frame, text="Top ROI", variable=self.roi_var,
                       command=self.toggle_roi).pack(anchor="w")

        self.live_detection_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(detection_frame, text="Live Detect", variable=self.live_detection_var,
                       command=self.toggle_live_detection_mode).pack(anchor="w")

        self.auto_grade_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(detection_frame, text="Auto Grade", variable=self.auto_grade_var).pack(anchor="w")

        # Conveyor Control (place next to detection)
        control_frame = ttk.LabelFrame(self, text="Conveyor Control", padding=FRAME_PADDING)
        control_frame.place(x=935, y=415, width=655, height=125)

        control_inner_frame = ttk.Frame(control_frame)
        control_inner_frame.pack(fill="both", expand=True)
        control_inner_frame.grid_columnconfigure(0, weight=1)
        control_inner_frame.grid_columnconfigure(1, weight=1)
        control_inner_frame.grid_columnconfigure(2, weight=1)

        tk.Button(control_inner_frame, text="Continuous",
                 command=self.set_continuous_mode, bg=BUTTON_BACKGROUND_COLOR,
                 fg=BUTTON_TEXT_COLOR, activebackground=BUTTON_ACTIVE_COLOR,
                 font=self.font_button, relief="raised", borderwidth=2).grid(row=0, column=0, sticky="ew", padx=1, pady=1)
        tk.Button(control_inner_frame, text="Trigger",
                 command=self.set_trigger_mode, bg=BUTTON_BACKGROUND_COLOR,
                 fg=BUTTON_TEXT_COLOR, activebackground=BUTTON_ACTIVE_COLOR,
                 font=self.font_button, relief="raised", borderwidth=2).grid(row=0, column=1, sticky="ew", padx=1, pady=1)
        tk.Button(control_inner_frame, text="IDLE",
                 command=self.set_idle_mode, bg=BUTTON_BACKGROUND_COLOR,
                 fg=BUTTON_TEXT_COLOR, activebackground=BUTTON_ACTIVE_COLOR,
                 font=self.font_button, relief="raised", borderwidth=2).grid(row=0, column=2, sticky="ew", padx=1, pady=1)

        # Reports panel at fixed position (no overlap)
        REPORT_W, REPORT_H = 300, 125
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

        self.show_report_notification = tk.BooleanVar(value=True)
        ttk.Checkbutton(reports_frame, text="Notifications",
                       variable=self.show_report_notification).pack()

        self.last_report_label = ttk.Label(reports_frame, text="Last: None",
                                          font=self.font_small, wraplength=100)
        self.last_report_label.pack()

        # Statistics section full width at bottom
        stats_frame = ttk.LabelFrame(self, text="Statistics", padding=FRAME_PADDING)
        stats_frame.place(x=0, y=screen_height - 500, width=screen_width, height=500)
        stats_frame.grid_columnconfigure(0, weight=1)
        stats_frame.grid_rowconfigure(0, weight=1)

        # Create notebook for tabbed statistics
        self.stats_notebook = ttk.Notebook(stats_frame, height=STATS_TAB_HEIGHT - 40)  # Account for padding and tab headers
        self.stats_notebook.grid(row=0, column=0, sticky="nsew")

        # Tab 1: Grade Summary (Overview with Live Grading)
        grade_summary_tab = ttk.Frame(self.stats_notebook, height=STATS_TAB_HEIGHT - 60)
        self.stats_notebook.add(grade_summary_tab, text="Grade Summary")
        

        # Live Grading Section (includes both individual grades and grade counts)
        live_grading_frame = ttk.LabelFrame(grade_summary_tab, text="Live Grading Results", padding="10")
        live_grading_frame.pack(fill="both", expand=True, pady=(10, 5), padx=10)

        # Configure grid for the live grading frame
        live_grading_frame.grid_columnconfigure(0, weight=1)
        live_grading_frame.grid_columnconfigure(1, weight=1)
        live_grading_frame.grid_columnconfigure(2, weight=1)
        live_grading_frame.grid_rowconfigure(0, weight=0)  # Individual grades row
        live_grading_frame.grid_rowconfigure(1, weight=1)  # Grade counts row

        # Row 0: Individual camera grades (horizontal layout)
        individual_grades_frame = ttk.Frame(live_grading_frame)
        individual_grades_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        individual_grades_frame.grid_columnconfigure(0, weight=1)
        individual_grades_frame.grid_columnconfigure(1, weight=1)
        individual_grades_frame.grid_columnconfigure(2, weight=2)

        # Individual camera grades (horizontal layout)
        top_grade_container = ttk.Frame(individual_grades_frame)
        top_grade_container.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        ttk.Label(top_grade_container, text="Top Camera:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.top_grade_label = ttk.Label(top_grade_container, text="No wood detected",
                                        foreground="gray", font=self.font_small)
        self.top_grade_label.pack(anchor="w")

        bottom_grade_container = ttk.Frame(individual_grades_frame)
        bottom_grade_container.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Label(bottom_grade_container, text="Bottom Camera:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.bottom_grade_label = ttk.Label(bottom_grade_container, text="No wood detected",
                                           foreground="gray", font=self.font_small)
        self.bottom_grade_label.pack(anchor="w")

        # Combined grade (prominent display, takes more space)
        combined_container = ttk.Frame(individual_grades_frame)
        combined_container.grid(row=0, column=2, sticky="ew", padx=5, pady=2)
        ttk.Label(combined_container, text="Final Grade:", font=("Arial", 12, "bold")).pack(anchor="w")
        self.combined_grade_label = ttk.Label(combined_container, text="No wood detected",
                                             font=("Arial", 11, "bold"), foreground="gray")
        self.combined_grade_label.pack(anchor="w")

        # Row 1: Grade counts (moved inside live grading results container)
        grade_counts_container = ttk.Frame(live_grading_frame)
        grade_counts_container.grid(row=1, column=0, columnspan=3, sticky="nsew")
        grade_counts_container.grid_columnconfigure(0, weight=1, uniform="grade_counts")
        grade_counts_container.grid_columnconfigure(1, weight=1, uniform="grade_counts")
        grade_counts_container.grid_columnconfigure(2, weight=1, uniform="grade_counts")
        grade_counts_container.grid_columnconfigure(3, weight=1, uniform="grade_counts")
        grade_counts_container.grid_rowconfigure(0, weight=1, minsize=80)

        # Initialize stats labels with consistent spacing and fonts
        self.live_stats_labels = {}
        grade_info = [
            ("grade0", "Perfect\n(No Defects)", GRADE_PERFECT_COLOR),
            ("grade1", "Good\n(G2-0)", GRADE_GOOD_COLOR),
            ("grade2", "Fair\n(G2-1, G2-2, G2-3)", GRADE_FAIR_COLOR),
            ("grade3", "Poor\n(G2-4)", GRADE_POOR_COLOR)
        ]

        for i, (grade_key, label_text, color) in enumerate(grade_info):
            grade_container = ttk.Frame(grade_counts_container, relief="solid", borderwidth=2)
            grade_container.grid(row=0, column=i, sticky="nsew", padx=6, pady=5, ipadx=8, ipady=8)
            grade_container.grid_columnconfigure(0, weight=1)
            grade_container.grid_rowconfigure(0, weight=1)
            grade_container.grid_rowconfigure(1, weight=1)

            # Create a consistent inner frame for better control
            inner_frame = ttk.Frame(grade_container)
            inner_frame.grid(row=0, column=0, sticky="nsew", rowspan=2)

            # Title label with fixed font
            title_label = ttk.Label(inner_frame, text=label_text, font=("Arial", 9, "bold"),
                                   justify="center")
            title_label.pack(expand=True, pady=(8, 2))

            # Count label with fixed font and consistent positioning
            self.live_stats_labels[grade_key] = ttk.Label(inner_frame, text="0",
                                                         foreground=color, font=("Arial", 16, "bold"))
            self.live_stats_labels[grade_key].pack(expand=True, pady=(2, 8))

        # Tab 2: Defect Details
        defect_details_tab = ttk.Frame(self.stats_notebook, height=STATS_TAB_HEIGHT - 60)
        self.stats_notebook.add(defect_details_tab, text="Defect Details")
        
        # Defect details content
        self.defect_details_frame = ttk.Frame(defect_details_tab)
        self.defect_details_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Tab 3: Performance Metrics
        performance_tab = ttk.Frame(self.stats_notebook, height=STATS_TAB_HEIGHT - 60)
        self.stats_notebook.add(performance_tab, text="Performance")
        
        # Performance metrics content
        self.performance_frame = ttk.Frame(performance_tab)
        self.performance_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Tab 4: Recent Activity
        activity_tab = ttk.Frame(self.stats_notebook, height=STATS_TAB_HEIGHT - 60)
        self.stats_notebook.add(activity_tab, text="Recent Activity")
        
        # Main container with two sections
        activity_main_container = ttk.Frame(activity_tab)
        activity_main_container.pack(fill="both", expand=True, padx=5, pady=5)
        activity_main_container.grid_columnconfigure(0, weight=1)
        activity_main_container.grid_rowconfigure(0, weight=0)  # Summary section (fixed height)
        activity_main_container.grid_rowconfigure(1, weight=1)  # Log section (expandable)
        
        # Session Summary Section (wider, fixed height)
        self.session_summary_frame = ttk.LabelFrame(activity_main_container, text="Current Session Summary", padding="10")
        self.session_summary_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        # Processing Log Section (scrollable)
        log_container = ttk.LabelFrame(activity_main_container, text="Recent Processing Log", padding="5")
        log_container.grid(row=1, column=0, sticky="nsew")
        log_container.grid_columnconfigure(0, weight=1)
        log_container.grid_rowconfigure(0, weight=1)

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
        
        log_canvas.bind("<Button-4>", on_log_scroll)
        log_canvas.bind("<Button-5>", on_log_scroll)  
        log_canvas.bind("<MouseWheel>", on_log_scroll)
        log_scrollbar.bind("<ButtonPress-1>", lambda e: setattr(self, '_user_scrolling_log', True))
        
        log_canvas.create_window((0, 0), window=self.processing_log_frame, anchor="nw")
        log_canvas.configure(yscrollcommand=log_scrollbar.set)
        
        log_canvas.grid(row=0, column=0, sticky="nsew")
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Store canvas references for updates
        self.log_canvas = log_canvas
        self.activity_canvas = log_canvas  # Keep for compatibility
        
        # Initialize scroll state variables
        self._user_scrolling_log = False

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
        # Mapping from your model's labels to standard categories
        label_mapping = {
            # Your model outputs
            "sound_knots": "Sound_Knot",
            "unsound_knots": "Unsound_Knot",
            # Add variations of your model's actual output labels here
            "sound knots": "Sound_Knot",
            "unsound knots": "Unsound_Knot",
            "live_knot": "Sound_Knot",
            "dead_knot": "Unsound_Knot",
            "missing_knot": "Unsound_Knot",
            "crack_knot": "Unsound_Knot",
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
            width_px = abs(x2 - x1)
            height_px = abs(y2 - y1)
            
            # Use the larger dimension (worst case for grading)
            max_dimension_px = max(width_px, height_px)
            
            # Use camera-specific conversion factor
            if camera_name == "top":
                pixel_to_mm = TOP_CAMERA_PIXEL_TO_MM
            else:  # bottom camera
                pixel_to_mm = BOTTOM_CAMERA_PIXEL_TO_MM
            
            # Convert to millimeters
            size_mm = max_dimension_px * pixel_to_mm
            
            # Calculate percentage of actual wood pallet width
            percentage = (size_mm / WOOD_PALLET_WIDTH_MM) * 100
            
            return size_mm, percentage
            
        except Exception as e:
            print(f"Error calculating defect size: {e}")
            # Return conservative values if calculation fails
            return 50.0, 35.0  # Assumes large defect for safety

    def grade_individual_defect(self, defect_type, size_mm, percentage):
        """Grade an individual defect based on SS-EN 1611-1 standards"""
        if defect_type not in GRADING_THRESHOLDS:
            print(f"Unknown defect type: {defect_type}, defaulting to Unsound_Knot")
            defect_type = "Unsound_Knot"
        
        thresholds = GRADING_THRESHOLDS[defect_type]
        
        # Check each grade threshold (size OR percentage can trigger the grade)
        for grade in [GRADE_G2_0, GRADE_G2_1, GRADE_G2_2, GRADE_G2_3]:
            mm_threshold, pct_threshold = thresholds[grade]
            if size_mm <= mm_threshold or percentage <= pct_threshold:
                return grade
        
        # If no threshold met, it's the worst grade
        return GRADE_G2_4

    def determine_surface_grade(self, defect_measurements):
        """Determine overall grade for a surface based on individual defect measurements"""
        if not defect_measurements:
            return GRADE_G2_0
        
        # Get individual grades for each defect
        defect_grades = []
        defect_counts = {}
        
        for defect_type, size_mm, percentage in defect_measurements:
            # Get grade for this individual defect
            grade = self.grade_individual_defect(defect_type, size_mm, percentage)
            defect_grades.append(grade)
            
            # Count defects by type
            if defect_type not in defect_counts:
                defect_counts[defect_type] = 0
            defect_counts[defect_type] += 1
        
        # Count total defects
        total_defects = sum(defect_counts.values())
        
        # Grade hierarchy for finding worst grade
        grade_hierarchy = [GRADE_G2_0, GRADE_G2_1, GRADE_G2_2, GRADE_G2_3, GRADE_G2_4]
        
        # Find the worst individual defect grade
        worst_grade_index = 0
        for grade in defect_grades:
            if grade in grade_hierarchy:
                grade_index = grade_hierarchy.index(grade)
                worst_grade_index = max(worst_grade_index, grade_index)
        
        worst_individual_grade = grade_hierarchy[worst_grade_index]
        
        # Apply defect count limitations per SS-EN 1611-1
        if total_defects > 6:
            return GRADE_G2_4
        elif total_defects > 4:
            # Maximum G2-3 regardless of individual grades
            return grade_hierarchy[min(3, worst_grade_index)]  # G2-3 is index 3
        elif total_defects > 2:
            # Maximum G2-2 regardless of individual grades  
            return grade_hierarchy[min(2, worst_grade_index)]  # G2-2 is index 2
        
        # Return the worst individual grade if defect count allows
        return worst_individual_grade

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
        # Map the 5 standard grades to sorting gates following strict classification:
        # Good: G2-0 | Fair: G2-1, G2-2, G2-3 | Poor: G2-4
        grade_to_command = {
            GRADE_G2_0: 1,    # Good (G2-0) - Gate 1
            GRADE_G2_1: 2,    # Fair (G2-1) - Gate 2  
            GRADE_G2_2: 2,    # Fair (G2-2) - Gate 2
            GRADE_G2_3: 2,    # Fair (G2-3) - Gate 2
            GRADE_G2_4: 3     # Poor (G2-4) - Gate 3
        }
        
        return grade_to_command.get(standard_grade, 3)  # Default to worst gate if unknown

    def get_grade_color(self, grade):
        """Get color coding for grades"""
        color_map = {
            GRADE_G2_0: 'dark green',
            GRADE_G2_1: 'green', 
            GRADE_G2_2: 'orange',
            GRADE_G2_3: 'dark orange',
            GRADE_G2_4: 'red'
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
        canvas = tk.Canvas(frame, height=DETECTION_DETAILS_HEIGHT)
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
        
        widgets['wood_width_label'] = ttk.Label(calib_frame, 
                                              text=f"Wood Width: {WOOD_PALLET_WIDTH_MM}mm",
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
        
        wood_label = ttk.Label(calib_frame, text=f"Wood Width: {WOOD_PALLET_WIDTH_MM}mm", 
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
            "wood_width_mm": WOOD_PALLET_WIDTH_MM
        }
        
        # Add individual defect details
        for i, (defect_type, size_mm, percentage) in enumerate(measurements, 1):
            individual_grade = self.grade_individual_defect(defect_type, size_mm, percentage)
            
            defect_detail = {
                "defect_id": i,
                "type": defect_type,
                "size_mm": round(size_mm, 2),
                "percentage_of_width": round(percentage, 2),
                "individual_grade": individual_grade,
                "grading_standard": "SS-EN 1611-1"
            }
            
            # Add threshold information
            thresholds = GRADING_THRESHOLDS.get(defect_type, GRADING_THRESHOLDS["Unsound_Knot"])
            for grade in [GRADE_G2_0, GRADE_G2_1, GRADE_G2_2, GRADE_G2_3]:
                mm_threshold, pct_threshold = thresholds[grade]
                if size_mm <= mm_threshold or percentage <= pct_threshold:
                    defect_detail["applied_threshold"] = f"≤{mm_threshold}mm or ≤{pct_threshold}%"
                    defect_detail["threshold_grade"] = grade
                    break
            else:
                defect_detail["applied_threshold"] = f">{thresholds[GRADE_G2_3][0]}mm and >{thresholds[GRADE_G2_3][1]}%"
                defect_detail["threshold_grade"] = GRADE_G2_3
            
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
        from datetime import datetime
        
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
        from datetime import datetime
        
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
                individual_grade = self.grade_individual_defect(defect_type, size_mm, percentage)
                
                size_label = ttk.Label(defect_frame, 
                                     text=f"Size: {size_mm:.1f}mm ({percentage:.1f}% of width)",
                                     font=self.font_small)
                size_label.pack(anchor="w")
                
                grade_color = self.get_grade_color(individual_grade)
                grade_label = ttk.Label(defect_frame, 
                                      text=f"Individual Grade: {individual_grade}",
                                      font=self.font_small, foreground=grade_color)
                grade_label.pack(anchor="w")
                
                # Show threshold info
                thresholds = GRADING_THRESHOLDS.get(defect_type, GRADING_THRESHOLDS["Unsound_Knot"])
                for grade in [GRADE_G2_0, GRADE_G2_1, GRADE_G2_2, GRADE_G2_3]:
                    mm_threshold, pct_threshold = thresholds[grade]
                    if size_mm <= mm_threshold or percentage <= pct_threshold:
                        threshold_text = f"Threshold: ≤{mm_threshold}mm or ≤{pct_threshold}%"
                        break
                else:
                    threshold_text = f"Threshold: >{thresholds[GRADE_G2_3][0]}mm and >{thresholds[GRADE_G2_3][1]}%"
                
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
        # Update dashboard at configured interval to reduce load
        if not hasattr(self, '_frame_counter'):
            self._frame_counter = 0

        self._frame_counter += 1
        if self._frame_counter % STATS_UPDATE_SKIP == 0:
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
            if self.auto_detection_active:
                total_detections = (len(self.detection_session_data["total_detections"]["top"]) +
                                   len(self.detection_session_data["total_detections"]["bottom"]))
                self.update_status_text(
                    f"Status: AUTO (IR) DETECTION ACTIVE 🔍 ({total_detections} detections)",
                    STATUS_WARNING_COLOR
                )
            elif self.live_detection_var.get():
                self.update_status_text(
                    f"Status: {self.current_mode} MODE - Live Detection ACTIVE",
                    STATUS_READY_COLOR
                )
            elif self.current_mode == "IDLE":
                self.update_status_text(
                    "Status: IDLE MODE - System disabled, no operations",
                    SECONDARY_TEXT_COLOR
                )
            elif self.current_mode == "TRIGGER":
                self.update_status_text(
                    "Status: TRIGGER MODE - Waiting for IR beam trigger",
                    STATUS_READY_COLOR
                )
            elif self.current_mode == "CONTINUOUS":
                self.update_status_text(
                    "Status: CONTINUOUS MODE - Live detection enabled",
                    STATUS_READY_COLOR
                )
            else:
                # Fallback for unknown states
                self.update_status_text(
                    f"Status: {self.current_mode} MODE - Ready",
                    STATUS_READY_COLOR
                )

    def toggle_live_detection_mode(self):
        """Handle toggling between IR trigger and live detection modes."""
        self.update_detection_status_display()

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
        
        # Clear previous live detections
        self.live_detections = {"top": {}, "bottom": {}}
        self.live_grades = {"top": "Detecting...", "bottom": "Detecting..."}
        
        print("🔍 Automatic detection STARTED - Object detected by IR beam")
        self.log_action("Automatic detection started - IR beam triggered")

    def stop_automatic_detection_and_grade(self):
        """Stop automatic detection and send grade when object clears IR beam"""
        if not self.auto_detection_active:
            return
            
        self.auto_detection_active = False
        self._in_active_inference = False  # Clear inference flag to resume normal UI updates
        self.detection_session_data["end_time"] = datetime.now()
        
        # Process all collected detection data
        top_detections = self.detection_session_data["total_detections"]["top"]
        bottom_detections = self.detection_session_data["total_detections"]["bottom"]
        
        # Determine final grades from accumulated detections
        final_top_grade = self.determine_final_grade_from_session("top", top_detections)
        final_bottom_grade = self.determine_final_grade_from_session("bottom", bottom_detections)
        
        # Combine grades for final decision
        combined_grade = self.determine_final_grade(final_top_grade, final_bottom_grade)
        self.detection_session_data["final_grade"] = combined_grade
        
        # --- New Centralized Logging ---
        all_top_measurements = [m for d in top_detections for m in d.get("measurements", [])]
        all_bottom_measurements = [m for d in bottom_detections for m in d.get("measurements", [])]
        self.finalize_grading(combined_grade, all_top_measurements + all_bottom_measurements)
        
        # Update live grading display
        self.live_grades["top"] = final_top_grade
        self.live_grades["bottom"] = final_bottom_grade
        self.update_live_grading_display()
        
        # Clear detection data for next piece
        self.detection_frames = []

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
            grade_map = {0: GRADE_G2_0, 1: GRADE_G2_1, 2: GRADE_G2_2, 3: GRADE_G2_4}
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

    def toggle_roi(self):
        """Toggle ROI for top camera"""
        self.roi_enabled["top"] = self.roi_var.get()
        status = "enabled" if self.roi_enabled["top"] else "disabled"

        print(f"ROI for top camera {status}")

    def apply_roi(self, frame, camera_name):
        """Apply Region of Interest (ROI) to frame for focused detection"""
        if not self.roi_enabled.get(camera_name, False):
            return frame, None
        
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

    def draw_bounding_boxes_on_canvas(self, canvas, detections, camera_name):
        """Draw bounding boxes for detections on canvas with ROI-based coloring and labels"""
        for det in detections:
            bbox = det.get('bbox', [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]

                # Scale from model coordinates (640x640) to canvas coordinates
                x1_canvas = x1 * (self.canvas_width / 640.0)
                y1_canvas = y1 * (self.canvas_height / 640.0)
                x2_canvas = x2 * (self.canvas_width / 640.0)
                y2_canvas = y2 * (self.canvas_height / 640.0)

                # Check if bounding box intersects with ROI
                intersects_roi = self.bbox_intersects_roi(bbox, camera_name)

                # Color based on ROI intersection: green = counted, red = not counted
                box_color = "green" if intersects_roi else "red"
                label_color = "white" if intersects_roi else "yellow"

                # Draw rectangle with appropriate color
                canvas.create_rectangle(x1_canvas, y1_canvas, x2_canvas, y2_canvas,
                                      outline=box_color, width=3, tags="bbox")

                # Add defect type label above the bounding box on the left side
                defect_label = det.get('label', 'Unknown')
                # Position label slightly above the top-left corner of the box
                label_x = x1_canvas
                label_y = y1_canvas - 15  # 15 pixels above the box

                # Ensure label doesn't go off the top of canvas
                if label_y < 5:
                    label_y = y1_canvas + 5  # Put it inside the box if too high

                canvas.create_text(label_x, label_y, text=defect_label,
                                 fill=label_color, font=("Arial", 10, "bold"),
                                 anchor="sw", tags="bbox_label")  # Left-aligned above box

        # Keep boxes and labels on top
        canvas.tag_raise("bbox")
        canvas.tag_raise("bbox_label")

    def draw_roi_on_canvas(self, canvas):
        """Draw ROI overlay on top camera canvas"""
        if not self.roi_enabled.get("top", False) and not self.roi_enabled.get("wood_detection", False):
            return

        # Clear previous ROI
        canvas.delete("roi")
        canvas.delete("roi_label")
        canvas.delete("wood_roi")
        canvas.delete("wood_roi_label")
        canvas.delete("exit_roi")
        canvas.delete("exit_roi_label")

        # Scale factors for canvas
        scale_x = self.canvas_width / 1280.0
        scale_y = self.canvas_height / 720.0

        # Draw top ROI if enabled
        if self.roi_enabled.get("top", False):
            # Get ROI coordinates
            roi_coords = self.roi_coordinates.get("top", {})
            roi_x1 = roi_coords.get("x1", 150)
            roi_y1 = roi_coords.get("y1", 80)
            roi_x2 = roi_coords.get("x2", 1280)
            roi_y2 = roi_coords.get("y2", 720)

            x1 = roi_x1 * scale_x
            y1 = roi_y1 * scale_y
            x2 = roi_x2 * scale_x
            y2 = roi_y2 * scale_y

            # Draw ROI rectangle (yellow border)
            canvas.create_rectangle(x1, y1, x2, y2, outline="yellow", width=3, tags="roi")

            # Add ROI label
            canvas.create_text(x1 + 10, y1 + 30, text="ROI - TOP",
                              fill="yellow", font=("Helvetica", 12, "bold"), anchor="nw", tags="roi_label")

        # Draw wood detection ROI if enabled
        if self.roi_enabled.get("wood_detection", False):
            # Get wood detection ROI coordinates
            wood_roi_coords = self.roi_coordinates.get("wood_detection", {})
            wood_roi_x1 = wood_roi_coords.get("x1", 100)
            wood_roi_y1 = wood_roi_coords.get("y1", 0)
            wood_roi_x2 = wood_roi_coords.get("x2", 500)
            wood_roi_y2 = wood_roi_coords.get("y2", 300)

            wood_x1 = wood_roi_x1 * scale_x
            wood_y1 = wood_roi_y1 * scale_y
            wood_x2 = wood_roi_x2 * scale_x
            wood_y2 = wood_roi_y2 * scale_y

            # Draw wood detection ROI rectangle (green border)
            canvas.create_rectangle(wood_x1, wood_y1, wood_x2, wood_y2, outline="green", width=3, tags="wood_roi")

            # Add wood detection ROI label
            canvas.create_text(wood_x1 + 10, wood_y1 + 30, text="WOOD DETECTION ROI",
                              fill="green", font=("Helvetica", 12, "bold"), anchor="nw", tags="wood_roi_label")

        # Draw exit wood ROI if enabled
        if self.roi_enabled.get("exit_wood", False):
            # Get exit wood ROI coordinates
            exit_roi_coords = self.roi_coordinates.get("exit_wood", {})
            exit_roi_x1 = exit_roi_coords.get("x1", 700)
            exit_roi_y1 = exit_roi_coords.get("y1", 0)
            exit_roi_x2 = exit_roi_coords.get("x2", 1100)
            exit_roi_y2 = exit_roi_coords.get("y2", 250)

            exit_x1 = exit_roi_x1 * scale_x
            exit_y1 = exit_roi_y1 * scale_y
            exit_x2 = exit_roi_x2 * scale_x
            exit_y2 = exit_roi_y2 * scale_y

            # Draw exit wood ROI rectangle (blue border)
            canvas.create_rectangle(exit_x1, exit_y1, exit_x2, exit_y2, outline="blue", width=3, tags="exit_roi")

            # Add exit wood ROI label
            canvas.create_text(exit_x1 + 10, exit_y1 + 30, text="EXIT WOOD ROI",
                              fill="blue", font=("Helvetica", 12, "bold"), anchor="nw", tags="exit_roi_label")

        # Keep ROIs on top
        canvas.tag_raise("roi")
        canvas.tag_raise("roi_label")
        canvas.tag_raise("wood_roi")
        canvas.tag_raise("wood_roi_label")
        canvas.tag_raise("exit_roi")
        canvas.tag_raise("exit_roi_label")

    def update_single_feed(self, cap, canvas, camera_name):
        ret, frame = cap.read()
        if ret:
            # Resize frame to fit the canvas size
            frame = cv2.resize(frame, (self.canvas_width, self.canvas_height))

            # Skip detection processing if frame rate is too high
            if not hasattr(self, '_detection_frame_skip'):
                self._detection_frame_skip = {"top": 0, "bottom": 0}

            # Process detection based on automatic IR beam OR live detection toggle
            should_detect = self.auto_detection_active or self.live_detection_var.get()

            if should_detect:
                # Run detection on the full frame (ROI-independent)
                result = self.analyze_frame(frame, camera_name, run_defect_model=True)

                # Handle return formats
                if len(result) >= 3:
                    annotated_frame, defect_dict, measurements = result[:3]
                    detections = result[3] if len(result) > 3 else []
                else:
                    annotated_frame, defect_dict = result
                    measurements = []
                    detections = []

                # Resize annotated frame to canvas size for display
                if annotated_frame.shape[:2] != (self.canvas_height, self.canvas_width):
                    annotated_frame = cv2.resize(annotated_frame, (self.canvas_width, self.canvas_height))

                # Filter detections and measurements based on ROI intersection for grading
                roi_filtered_defect_dict = {}
                roi_filtered_measurements = []

                if self.roi_enabled.get(camera_name, False) and detections:
                    # Filter based on bounding box intersection with ROI
                    roi_intersecting_detections = []

                    for det in detections:
                        bbox = det.get('bbox', [])
                        if len(bbox) >= 4 and self.bbox_intersects_roi(bbox, camera_name):
                            roi_intersecting_detections.append(det)

                    # Re-calculate defect_dict and measurements from ROI-intersecting detections only
                    for det in roi_intersecting_detections:
                        model_label = det.get('label', '')
                        standard_defect_type = self.map_model_output_to_standard(model_label)

                        # Count defects by type
                        if standard_defect_type in roi_filtered_defect_dict:
                            roi_filtered_defect_dict[standard_defect_type] += 1
                        else:
                            roi_filtered_defect_dict[standard_defect_type] = 1

                        # Calculate size for measurements
                        bbox_info = {'bbox': det['bbox']}
                        size_mm, percentage = self.calculate_defect_size(bbox_info, camera_name)
                        roi_filtered_measurements.append((standard_defect_type, size_mm, percentage))
                else:
                    # No ROI filtering - use all detections
                    roi_filtered_defect_dict = defect_dict.copy()
                    roi_filtered_measurements = measurements.copy()

                # Store filtered detection results (only ROI-intersecting for grading)
                self.live_detections[camera_name] = roi_filtered_defect_dict
                if not hasattr(self, 'live_measurements'):
                    self.live_measurements = {"top": [], "bottom": []}
                self.live_measurements[camera_name] = roi_filtered_measurements

                # During automatic detection, collect data
                if self.auto_detection_active:
                    detection_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "camera": camera_name,
                        "defects": defect_dict.copy(),
                        "measurements": measurements.copy() if measurements else [],
                        "frame_captured": True
                    }
                    self.detection_session_data["total_detections"][camera_name].append(detection_entry)

                    # Save best frame
                    if (self.detection_session_data["best_frames"][camera_name] is None or
                        sum(defect_dict.values()) > 0):
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.detection_session_data["best_frames"][camera_name] = frame_rgb.copy()

                    # Store frame for report
                    if len(self.detection_frames) < 50:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.detection_frames.append({
                            "camera": camera_name,
                            "timestamp": datetime.now().isoformat(),
                            "frame": frame_rgb.copy(),
                            "defects": defect_dict.copy()
                        })

                # Calculate grade
                if measurements:
                    surface_grade = self.determine_surface_grade(measurements)
                    grade_info = {
                        'grade': surface_grade,
                        'text': f'{surface_grade} - SS-EN 1611-1 ({camera_name.title()} Camera)',
                        'total_defects': len(measurements),
                        'color': self.get_grade_color(surface_grade)
                    }
                else:
                    grade_info = self.calculate_grade(defect_dict)

                self.live_grades[camera_name] = grade_info

                # Update displays at intervals
                if self._detection_frame_skip[camera_name] % FRAME_SKIP_DETECTION == 0:
                    self.update_dashboard_display(camera_name, defect_dict, measurements)
                if self._detection_frame_skip[camera_name] % UI_UPDATE_SKIP == 0:
                    self.update_live_grading_display()

            else:
                # Reset detections when not active
                if not self.auto_detection_active:
                    self.live_detections[camera_name] = {}
                    self.live_grades[camera_name] = "No wood detected"
                    if hasattr(self, 'live_measurements'):
                        self.live_measurements[camera_name] = []
                    if self._detection_frame_skip[camera_name] % LOG_UPDATE_SKIP == 0:
                        self.update_dashboard_display(camera_name, {}, [])
                        self.update_live_grading_display()

            # Increment frame skip counter
            self._detection_frame_skip[camera_name] += 1

            # Convert to PhotoImage
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            photo = ImageTk.PhotoImage(image=img)

            # Update canvas safely
            canvas.delete('cam_img')  # Remove previous camera image
            canvas.delete('bbox')     # Remove previous bounding boxes
            canvas.delete('bbox_label')  # Remove previous bounding box labels
            if camera_name == "top":
                self._top_photo = photo
                canvas.create_image(0, 0, anchor='nw', image=self._top_photo, tags='cam_img')
            else:
                self._bottom_photo = photo
                canvas.create_image(0, 0, anchor='nw', image=self._bottom_photo, tags='cam_img')

            # Draw bounding boxes if detections exist (only when detecting)
            if should_detect and 'detections' in locals() and detections:
                self.draw_bounding_boxes_on_canvas(canvas, detections, camera_name)

            # Draw ROI on top camera if enabled (ensures it's on top)
            if camera_name == "top":
                self.draw_roi_on_canvas(canvas)

    def calculate_grade(self, defect_dict):
        """Calculate grade based on defect dictionary and return grade info"""
        total_defects = sum(defect_dict.values()) if defect_dict else 0
        
        if total_defects == 0:
            return {
                'grade': 0,  # Grade 0 for perfect wood
                'text': 'Perfect (No Defects)',
                'total_defects': 0,
                'color': 'dark green'
            }
        elif total_defects <= 2:
            return {
                'grade': 1,
                'text': f'Good (G2-0) - {total_defects} defects',
                'total_defects': total_defects,
                'color': 'green'
            }
        elif total_defects <= 6:
            return {
                'grade': 2,
                'text': f'Fair (G2-1, G2-2, G2-3) - {total_defects} defects',
                'total_defects': total_defects,
                'color': 'orange'
            }
        else:
            return {
                'grade': 3,
                'text': f'Poor (G2-4) - {total_defects} defects',
                'total_defects': total_defects,
                'color': 'red'
            }

    def update_live_grading_display(self):
        """Update the live grading display with current detection results using SS-EN 1611-1"""
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
            
            # Auto-grade functionality - send grade automatically if enabled
            if self.auto_grade_var.get():
                
                current_time = time.time()
                if current_time - self._last_auto_grade_time > 2.0:  # Only send grade every 2 seconds
                    if final_grade:
                        print(f"Auto-grade triggered - Final SS-EN 1611-1 grade: {final_grade}")
                        # To log, we need the measurements that led to this grade
                        all_measurements = self.live_measurements.get("top", []) + self.live_measurements.get("bottom", [])
                        self.finalize_grading(final_grade, all_measurements)
                    else:
                        # Fallback for simple grading
                        combined_defects = {}
                        for camera_name in ["top", "bottom"]:
                            if self.live_detections[camera_name]:
                                for defect, count in self.live_detections[camera_name].items():
                                    combined_defects[defect] = combined_defects.get(defect, 0) + count
                        print(f"Auto-grade triggered - Combined defects: {combined_defects}")
                        # Simple grading doesn't have measurement details, so we pass an empty list
                        simple_grade_info = self.calculate_grade(combined_defects)
                        grade_map = {0: GRADE_G2_0, 1: GRADE_G2_1, 2: GRADE_G2_2, 3: GRADE_G2_4}
                        final_grade_text = grade_map.get(simple_grade_info.get('grade'), GRADE_G2_4)
                        self.finalize_grading(final_grade_text, [])

                    self._last_auto_grade_time = current_time

        else:
            self.combined_grade_label.config(text="No wood detected", foreground="gray")

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
            details_text += f"Wood Width: {WOOD_PALLET_WIDTH_MM}mm | Defects: {total_defects}\n"
            details_text += "═" * 50 + "\n"
            
            # Show individual defect analysis
            for i, (defect_type, size_mm, percentage) in enumerate(measurements, 1):
                individual_grade = self.grade_individual_defect(defect_type, size_mm, percentage)
                details_text += f"{i}. {defect_type.replace('_', ' ')}\n"
                details_text += f"   Size: {size_mm:.1f}mm ({percentage:.1f}% of width)\n"
                details_text += f"   Individual Grade: {individual_grade}\n"
                details_text += f"   Threshold Info: "
                
                # Show which threshold was applied
                thresholds = GRADING_THRESHOLDS.get(defect_type, GRADING_THRESHOLDS["Unsound_Knot"])
                for grade in [GRADE_G2_0, GRADE_G2_1, GRADE_G2_2, GRADE_G2_3]:
                    mm_threshold, pct_threshold = thresholds[grade]
                    if size_mm <= mm_threshold or percentage <= pct_threshold:
                        details_text += f"≤{mm_threshold}mm or ≤{pct_threshold}%\n"
                        break
                else:
                    details_text += f">{thresholds[GRADE_G2_3][0]}mm and >{thresholds[GRADE_G2_3][1]}%\n"
                
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
            
            details_text += f"Wood Width: {WOOD_PALLET_WIDTH_MM}mm\n"
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
            
            # Try multiple common serial ports for Arduino
            # Prioritize ACM ports for Arduino Uno R3/Leonardo with native USB
            # Include USB ports for Arduino Nano/Pro Mini with FTDI/CH340 chips
            # Include potential reassigned ports (ACM1, USB01, etc.)
            ports_to_try = ARDUINO_PORTS
            
            for port in ports_to_try:
                try:
                    print(f"Trying to connect to Arduino on {port}...")
                    # More robust serial connection settings
                    self.ser = serial.Serial(
                        port=port, 
                        baudrate=9600, 
                        timeout=2,           # Increased timeout
                        write_timeout=2,     # Prevent hanging on write
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        xonxoff=False,       # Disable software flow control
                        rtscts=False,        # Disable hardware flow control
                        dsrdtr=False         # Disable DSR/DTR flow control
                    )
                    time.sleep(3)  # Extended time for Arduino to reset and stabilize
                    
                    # Clear any existing data in buffers
                    self.ser.reset_input_buffer()
                    self.ser.reset_output_buffer()
                    
                    # Test the connection with a gentle ping
                    self.ser.write(b'X')  # Send stop command as test
                    self.ser.flush()
                    time.sleep(0.5)
                    
                    print(f"✅ Arduino connected successfully on {port}")
                    break
                except (serial.SerialException, OSError) as e:
                    print(f"❌ Failed to connect on {port}: {e}")
                    continue
            else:
                # No port worked
                raise serial.SerialException("No Arduino found on any port")
            
            # Start Arduino listener thread if not already running and not shutting down
            if not (hasattr(self, '_shutting_down') and self._shutting_down):
                if not hasattr(self, 'arduino_thread') or not self.arduino_thread.is_alive():
                    self.arduino_thread = threading.Thread(target=self.listen_for_arduino, daemon=True)
                    self.arduino_thread.start()
            
            if hasattr(self, 'status_label'):
                self.update_status_text("Status: Arduino connected. Ready for automatic detection.")
                # Start with auto detection ready status
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
                        # ONLY respond to IR triggers in TRIGGER mode
                        if self.current_mode == "TRIGGER":
                            if not self.auto_detection_active:
                                print("✅ TRIGGER MODE: Starting detection...")
                                print("🔧 Arduino should now set motorActiveForTrigger = true")
                                print("⚡ Stepper motor should start running NOW!")
                                if hasattr(self, 'status_label'):
                                    self.update_status_text(
                                        "Status: IR TRIGGERED - Motor should be running!", STATUS_WARNING_COLOR
                                    )
                                self.start_automatic_detection()
                            else:
                                print("⚠️ IR beam broken but detection already active")
                        else:
                            # In IDLE or CONTINUOUS mode, just log the IR signal but don't act on it
                            print(f"❌ IR beam broken received but system is in {self.current_mode} mode - ignoring trigger")
                        continue  # skip other checks for this message

                    # --- LENGTH HANDLING (Arduino sends "L:duration" when beam clears) ---
                    if message.startswith("L:"):
                        try:
                            duration_ms = int(message.split(':')[1])
                            self.calculate_and_display_length(duration_ms)
                            
                            # In TRIGGER mode, stop detection when beam clears (length message received)
                            if self.current_mode == "TRIGGER" and self.auto_detection_active:
                                print("IR beam cleared – stopping detection (TRIGGER MODE)")
                                if hasattr(self, 'status_label'):
                                    self.update_status_text(
                                        "Status: Processing results...", STATUS_ERROR_COLOR
                                    )
                                self.stop_automatic_detection_and_grade()
                                if hasattr(self, 'status_label'):
                                    self.update_status_text(
                                        "Status: Ready - Waiting for IR beam trigger", STATUS_READY_COLOR
                                    )
                            else:
                                print(f"Length signal received (duration: {duration_ms}ms) but system is in {self.current_mode} mode or no detection active")
                        except (ValueError, IndexError):
                            print(f"Could not parse length message: {message}")
                        continue  # skip other checks for this message

                    # --- OTHER ARDUINO MESSAGES ---
                    else:
                        print(f"Arduino message received: '{message}'")
                        if hasattr(self, 'status_label'):
                            self.update_status_text(f"Status: Arduino: {message}")

                elif msg_type == "status_update":
                    if hasattr(self, 'status_label'):
                        self.update_status_text(f"Status: {data}")
                    
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error in process_message_queue: {e}")
        
        # Schedule next check
        self.after(50, self.process_message_queue)

    def listen_for_arduino(self):
        """Robust Arduino listener with automatic reconnection"""
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        
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
                        time.sleep(2)  # Wait before reconnect attempt
                        
                        # Try to reconnect
                        try:
                            self.setup_arduino()
                            if self.ser and self.ser.is_open:
                                print(f"✅ Arduino reconnected successfully on {self.ser.port}")
                                reconnect_attempts = 0
                            else:
                                print(f"❌ Reconnection attempt {reconnect_attempts} failed")
                        except Exception as e:
                            print(f"❌ Reconnection attempt {reconnect_attempts} failed: {e}")
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
                    
                # Attempt reconnection
                if reconnect_attempts < max_reconnect_attempts:
                    reconnect_attempts += 1
                    print(f"🔄 Communication error, attempting reconnection {reconnect_attempts}/{max_reconnect_attempts}...")
                    time.sleep(2)
                    try:
                        self.setup_arduino()
                        if self.ser and self.ser.is_open:
                            print(f"✅ Arduino reconnected after error on {self.ser.port}")
                            reconnect_attempts = 0
                    except Exception as reconnect_error:
                        print(f"❌ Reconnection failed: {reconnect_error}")
                else:
                    print(f"❌ Max reconnection attempts reached after error, exiting thread")
                    break
                    
            except Exception as e:
                print(f"❌ Unexpected error in Arduino listener: {e}")
                time.sleep(1)
                self.message_queue.put(("status_update", "Arduino connection lost"))
                break
            except Exception as e:
                print(f"Unexpected error in Arduino listener: {e}")
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
                    self.update_status_text("Status: Arduino not connected.", STATUS_ERROR_COLOR)
                    
        except (serial.SerialException, OSError, TypeError) as e:
            print(f"🔥 Error sending Arduino command '{command}': {e}")
            if hasattr(self, 'status_label'):
                self.update_status_text("Status: Arduino communication error - attempting reconnect...", STATUS_ERROR_COLOR)
            
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
        self.update_detection_status_display() # Update the status label

    def set_trigger_mode(self):
        """Sets the system to wait for an IR beam trigger."""
        print("Setting Trigger Mode")
        self.current_mode = "TRIGGER"
        print(f"Sending 'T' command to Arduino...")
        self.send_arduino_command('T')  # Send command to Arduino
        self.live_detection_var.set(False)
        self.auto_grade_var.set(False)
        self.update_detection_status_display() # Update the status label
        print(f"Trigger mode set - Python mode: {self.current_mode}")

    def set_idle_mode(self):
        """Disables all operations and stops the conveyor."""
        print("Setting IDLE Mode")
        self.current_mode = "IDLE"
        self.send_arduino_command('X')  # Send stop command to Arduino
        self.live_detection_var.set(False)
        self.auto_grade_var.set(False)
        self.update_status_text("Status: IDLE - Conveyor Stopped", SECONDARY_TEXT_COLOR)

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

        # 3. Update UI statistics
        self.grade_counts[arduino_command] += 1
        self.live_stats[f"grade{arduino_command}"] += 1
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
        top_surface_grade = None
        bottom_surface_grade = None
        
        all_measurements = self.live_measurements.get("top", []) + self.live_measurements.get("bottom", [])
        
        if all_measurements:
            wood_detected = True
            top_surface_grade = self.determine_surface_grade(self.live_measurements.get("top", []))
            bottom_surface_grade = self.determine_surface_grade(self.live_measurements.get("bottom", []))

        if wood_detected:
            final_grade = self.determine_final_grade(top_surface_grade, bottom_surface_grade)
            print(f"Manual grade trigger - SS-EN 1611-1 Final grade: {final_grade}")
            self.finalize_grading(final_grade, all_measurements)
        else:
            print("Manual grade trigger - No wood currently detected")
            self.update_status_text("Status: Manual grade - no wood detected")

    def analyze_frame(self, frame, camera_name="top", run_defect_model=True):
        """Analyze frame using DeGirum model for defect detection with size measurement"""
        if self.model is None:
            return frame, {}, []

        try:
            # Resize frame to model's expected input size (640x640) and convert to RGB
            model_input = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB)
            print(f"Debug: Running inference on {camera_name} camera, input shape: {model_input.shape}")

            # Run inference using DeGirum
            inference_result = self.model(model_input)
            print(f"Debug: Inference successful on {camera_name} camera")

            # Get annotated frame
            annotated_frame = inference_result.image_overlay

            # Process detections to count defects and measure sizes
            final_defect_dict = {}
            defect_measurements = []  # Store detailed measurements for grading
            detections = inference_result.results

            for det in detections:
                model_label = det['label']

                # Map model output to standard defect types
                standard_defect_type = self.map_model_output_to_standard(model_label)

                # Extract bounding box and scale to original frame coordinates for size calculation
                x1, y1, x2, y2 = det['bbox']
                scale_x = 1280.0 / 640.0  # Original width / model width
                scale_y = 720.0 / 640.0   # Original height / model height

                # Scale bbox to original frame coordinates
                bbox_original = {
                    'bbox': [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                }

                # Calculate defect size in mm and percentage using camera-specific calibration
                size_mm, percentage = self.calculate_defect_size(bbox_original, camera_name)

                # Store detailed measurement for sophisticated grading
                defect_measurements.append((standard_defect_type, size_mm, percentage))

                # Count defects by standardized label (for simple display)
                if standard_defect_type in final_defect_dict:
                    final_defect_dict[standard_defect_type] += 1
                else:
                    final_defect_dict[standard_defect_type] = 1

            return annotated_frame, final_defect_dict, defect_measurements, detections

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
            self.update_defect_details_tab()
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

    def update_defect_details_tab(self):
        """Update the Defect Details tab with current defect information"""
        if not hasattr(self, 'defect_details_frame'):
            return
            
        # Clear existing content
        for widget in self.defect_details_frame.winfo_children():
            widget.destroy()
        
        # Current detection status
        current_frame = ttk.LabelFrame(self.defect_details_frame, text="Current Detection", padding="5")
        current_frame.pack(fill="x", pady=2)
        
        if hasattr(self, 'live_measurements') and any(self.live_measurements.values()):
            for camera_name in ["top", "bottom"]:
                measurements = self.live_measurements.get(camera_name, [])
                if measurements:
                    camera_text = f"{camera_name.title()} Camera: {len(measurements)} defects detected\n"
                    
                    # Group by defect type
                    defect_summary = {}
                    for defect_type, size_mm, percentage in measurements:
                        if defect_type not in defect_summary:
                            defect_summary[defect_type] = []
                        defect_summary[defect_type].append((size_mm, percentage))
                    
                    for defect_type, sizes in defect_summary.items():
                        avg_size = sum(s[0] for s in sizes) / len(sizes)
                        avg_percentage = sum(s[1] for s in sizes) / len(sizes)
                        camera_text += f"  • {defect_type.replace('_', ' ')}: {len(sizes)} defects, avg {avg_size:.1f}mm ({avg_percentage:.1f}%)\n"
                    
                    ttk.Label(current_frame, text=camera_text, font=self.font_small, justify="left").pack(anchor="w")
        else:
            ttk.Label(current_frame, text="No defects currently detected", font=self.font_small).pack(anchor="w")
        
        # Grading thresholds reference
        thresholds_frame = ttk.LabelFrame(self.defect_details_frame, text="SS-EN 1611-1 Grading Thresholds", padding="5")
        thresholds_frame.pack(fill="x", pady=2)
        
        threshold_text = "Sound Knots (mm, %):\n"
        threshold_text += "  G2-0: ≤10mm, ≤5%  |  G2-1: ≤30mm, ≤15%  |  G2-2: ≤50mm, ≤25%  |  G2-3: ≤70mm, ≤35%\n\n"
        threshold_text += "Unsound Knots (mm, %):\n"
        threshold_text += "  G2-0: ≤7mm, ≤3.5%  |  G2-1: ≤20mm, ≤10%  |  G2-2: ≤35mm, ≤17.5%  |  G2-3: ≤50mm, ≤25%"
        
        ttk.Label(thresholds_frame, text=threshold_text, font=self.font_small, justify="left").pack(anchor="w")

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
            grade_counts = getattr(self, 'grade_counts', {0: 0, 1: 0, 2: 0, 3: 0})
            distribution_text = ""
            grade_names = {0: "Perfect", 1: "Good (G2-0)", 2: "Fair (G2-1,G2-2,G2-3)", 3: "Poor (G2-4)"}
            
            for grade, count in grade_counts.items():
                percentage = (count / total_processed) * 100 if total_processed > 0 else 0
                distribution_text += f"Grade {grade} ({grade_names.get(grade, 'Unknown')}): {count} pieces ({percentage:.1f}%)\n"
        else:
            distribution_text = "No processing data available yet"
        
        ttk.Label(distribution_frame, text=distribution_text, font=self.font_small, justify="left").pack(anchor="w")

    def update_recent_activity_tab(self):
        """Update the Recent Activity tab with widened summary and scrollable processing log"""
        # Don't update log if user is scrolling through it
        if getattr(self, '_user_scrolling_log', False):
            return
            
        # Generate content first to check if it changed
        new_stats_content = self._generate_stats_content()
        
        # Only update if content actually changed OR if this is the first update
        if (new_stats_content != self._last_stats_content or not self._last_stats_content):
            
            # Update Session Summary (wider display)
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
                
                grade_counts = getattr(self, 'grade_counts', {0: 0, 1: 0, 2: 0, 3: 0})
                grade_stats = "Grade Distribution:\n"
                grade_names = {0: "Perfect", 1: "Good (G2-0)", 2: "Fair (G2-1,G2-2,G2-3)", 3: "Poor (G2-4)"}
                
                for grade, count in grade_counts.items():
                    percentage = (count / total_processed) * 100 if total_processed > 0 else 0
                    grade_stats += f"{grade_names.get(grade, 'Unknown')}: {count} ({percentage:.1f}%)\n"
                
                ttk.Label(right_frame, text=grade_stats, font=self.font_small, justify="left").pack(anchor="w")
            
            # Update Processing Log (scrollable with many entries and detailed defect info)
            for widget in self.processing_log_frame.winfo_children():
                widget.destroy()
            
            if hasattr(self, 'session_log') and self.session_log:
                # Show all entries, not just last 10 (since it's now scrollable)
                recent_entries_copy = self.session_log.copy()
                recent_entries_copy.reverse()  # Show newest first
                
                for i, entry in enumerate(recent_entries_copy):
                    timestamp = entry.get('timestamp', 'Unknown')
                    piece_number = entry.get('piece_number', 'Unknown')
                    grade = entry.get('final_grade', 'Unknown')
                    defects = entry.get('defects', [])
                    
                    # Create a frame for each log entry for better formatting
                    entry_frame = ttk.Frame(self.processing_log_frame)
                    entry_frame.pack(fill="x", pady=2, padx=5)
                    
                    # Format defects info with specific details including sizes (single line)
                    if defects:
                        defects_details = []
                        for d in defects:
                            defect_type = d.get('type', 'Unknown')
                            count = d.get('count', 0)
                            sizes = d.get('sizes', '')
                            if sizes:
                                defects_details.append(f"{defect_type} (x{count}): {sizes}mm")
                            else:
                                defects_details.append(f"{defect_type} (x{count})")
                        
                        defects_info = " | ".join(defects_details)
                        log_text = f"[{timestamp}] Piece #{piece_number}: Grade {grade} - Defects: {defects_info}"
                    else:
                        log_text = f"[{timestamp}] Piece #{piece_number}: Grade {grade} - No defects detected"
                    
                    # Color code by grade
                    grade_colors = {
                        "G2-0": "green",
                        "G2-1": "blue", 
                        "G2-2": "orange",
                        "G2-3": "red",
                        "G2-4": "red"
                    }
                    text_color = grade_colors.get(grade, "black")
                    
                    log_label = ttk.Label(entry_frame, text=log_text, font=("Arial", 9), 
                                        justify="left", foreground=text_color)
                    log_label.pack(anchor="w", fill="x")
                    
                    # Add separator line (except for last entry)
                    if i < len(recent_entries_copy) - 1:
                        separator = ttk.Separator(self.processing_log_frame, orient="horizontal")
                        separator.pack(fill="x", pady=2)
            else:
                # Show message when no log entries exist
                no_data_label = ttk.Label(self.processing_log_frame, 
                                        text="No processing data yet...", 
                                        font=self.font_small, foreground="gray")
                no_data_label.pack(pady=20)
            
            # Cache the content
            self._last_stats_content = new_stats_content
            
            # Update scroll region for log
            if hasattr(self, 'log_canvas'):
                self.log_canvas.configure(scrollregion=self.log_canvas.bbox("all"))

    def update_detailed_statistics(self):
        """Legacy method - now redirects to update_recent_activity_tab for compatibility"""
        self.update_recent_activity_tab()

    def _generate_stats_content(self):
        """Generate a string representation of current stats for change detection"""
        content = f"processed:{getattr(self, 'total_pieces_processed', 0)}"
        
        grade_counts = getattr(self, 'grade_counts', {0: 0, 1: 0, 2: 0, 3: 0})
        for grade, count in grade_counts.items():
            content += f",g{grade}:{count}"
        
        # Include session log count for change detection
        if hasattr(self, 'session_log'):
            content += f",log_entries:{len(self.session_log)}"
                
        return content

    def calculate_and_display_length(self, duration_ms):
        try:
            speed_cm_s = float(self.speed_var.get())
            length_cm = (duration_ms / 1000.0) * speed_cm_s
            length_text = f"\nEstimated Length: {length_cm:.2f} cm"
            print(f"Wood piece length calculated: {length_cm:.2f} cm")

        except ValueError:
            self.update_status_text("Status: Invalid speed value!")

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
        
        # Release camera resources
        try:
            print("Releasing camera resources...")
            if hasattr(self, 'cap_top'):
                self.cap_top.release()
            if hasattr(self, 'cap_bottom'):
                self.cap_bottom.release()
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
        content += f"Grade Perfect (No Defects): {self.grade_counts.get(0, 0)}\n"
        content += f"Grade G2-0/G2-1 (Good Quality): {self.grade_counts.get(1, 0)}\n"  
        content += f"Grade G2-2/G2-3 (Fair Quality): {self.grade_counts.get(2, 0)}\n"
        content += f"Grade G2-4 (Poor Quality): {self.grade_counts.get(3, 0)}\n"
        
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
            log_entry = f"{timestamp} | Pieces: {self.total_pieces_processed} | Perfect: {self.grade_counts.get(0, 0)} | G1: {self.grade_counts.get(1, 0)} | G2: {self.grade_counts.get(2, 0)} | G3: {self.grade_counts.get(3, 0)}\n"
            with open(log_filename, 'a') as f:
                f.write(log_entry)
            print(f"Entry added to log file: {log_filename}")
            self.log_status_label.config(text="Log: Updated", foreground=STATUS_READY_COLOR)
        except Exception as e:
            print(f"Error updating log file: {e}")
            self.log_status_label.config(text="Log: Error", foreground=STATUS_ERROR_COLOR)

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
            text.textLine(f"Grade Perfect (No Defects): {self.grade_counts.get(0, 0)}")
            text.textLine(f"Grade G2-0/G2-1 (Good Quality): {self.grade_counts.get(1, 0)}")
            text.textLine(f"Grade G2-2/G2-3 (Fair Quality): {self.grade_counts.get(2, 0)}")
            text.textLine(f"Grade G2-4 (Poor Quality): {self.grade_counts.get(3, 0)}")
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
        self.log_status_label.config(text="Log: Manual report generated", foreground=STATUS_READY_COLOR)

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
    app = App()
    app.mainloop()
