#!/usr/bin/env python3
"""
Wood Sorting Application - Main GUI Module

This module contains the main GUI application for the wood sorting system.
Includes enhanced error handling, performance monitoring, and live detection capabilities.
"""

import sys
import cv2
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame, QGridLayout, QCheckBox, QTabWidget, QGroupBox, QTextEdit, QProgressBar, QScrollArea, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QFont, QColor
from PyQt5.QtCore import Qt, QTimer, QDateTime, QThread, pyqtSignal
import queue
import time

from modules.camera_module import CameraModule
from modules.detection_module import DetectionModule
from modules.arduino_module import ArduinoModule
from modules.reporting_module import ReportingModule
from modules.grading_module import calculate_grade, determine_final_grade, get_grade_color
from modules.utils_module import TOP_CAMERA_PIXEL_TO_MM, BOTTOM_CAMERA_PIXEL_TO_MM, WOOD_PALLET_WIDTH_MM
from modules.error_handler import (
    log_info, log_warning, log_error, SystemComponent, 
    get_error_summary, error_handler
)
from modules.performance_monitor import get_performance_monitor, start_performance_monitoring
from config.settings import get_config

class WoodSortingApp(QMainWindow):
    def __init__(self, dev_mode=False, config=None):
        super().__init__()
        self.dev_mode = dev_mode
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize configuration
        self.config = config or get_config(environment="development" if dev_mode else "production")
        
        # Fixed 1080p dimensions from config
        self.WINDOW_WIDTH = self.config.gui.window_width
        self.WINDOW_HEIGHT = self.config.gui.window_height
        
        self.setWindowTitle("Wood Sorting Application - Enhanced (1080p Maximized)")
        
        # Set minimum window size for 1080p display but allow maximizing
        self.setMinimumSize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        
        # Maximize the window to fill the screen if configured
        if self.config.gui.maximize_on_startup:
            self.showMaximized()

        # Initialize message queue for thread communication
        self.message_queue = queue.Queue()

        # Initialize performance monitoring
        self.performance_monitor = get_performance_monitor()
        if self.config.performance.enable_monitoring:
            start_performance_monitoring()
            self.performance_monitor.add_update_callback(self.update_performance_display)

        # Initialize modules
        self.camera_module = CameraModule(dev_mode=self.dev_mode)
        self.camera_module.initialize_cameras()  # Initialize cameras
        self.detection_module = DetectionModule(dev_mode=self.dev_mode)
        self.arduino_module = ArduinoModule(message_queue=self.message_queue)
        self.reporting_module = ReportingModule()

        # System state tracking
        self.current_mode = "IDLE" # Can be "IDLE", "TRIGGER", or "CONTINUOUS"
        self.auto_detection_active = False # Triggered by IR beam
        self.live_detection_var = False # For live inference mode (continuous)
        self.auto_grade_var = False # For auto grading in live mode
        
        # Wood detection state (IR-triggered workflow)
        self.ir_triggered = False
        self.wood_confirmed = False
        self.wood_detection_active = False
        self.current_detection_session = None

        # Initialize variables for statistics and logging
        self.total_pieces_processed = 0
        self.session_start_time = QDateTime.currentMSecsSinceEpoch() / 1000 # Unix timestamp
        self.grade_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.live_stats = {"grade0": 0, "grade1": 0, "grade2": 0, "grade3": 0}
        self.session_log = []

        # Store original frame sizes and defect information
        self.top_frame_original = None
        self.bottom_frame_original = None
        self.current_defects = {}
        self.current_grade_info = None
        self.wood_classification = "Unknown"  # Wood type classification
        self.detection_state = "Waiting"  # Detection state for UI

        # UI initialization
        self.setup_connections()
        self.setup_ui()
        self.setup_dev_mode()
        
        # Start the UI update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_feeds)
        self.timer.start(50)  # Update every 50ms (20 FPS)

    def setup_connections(self):
        pass

    def setup_ui(self):
        # Main layout with responsive margins for maximized mode
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(15, 15, 15, 15)  # Responsive margins
        self.main_layout.setSpacing(12)  # Responsive spacing

        # Top Section: Camera Feeds + Defect Analysis Panel (Responsive height)
        top_section_container = QWidget()
        top_section_layout = QHBoxLayout(top_section_container)
        top_section_layout.setContentsMargins(5, 5, 5, 5)
        top_section_layout.setSpacing(15)

        # Left Side: Camera Feeds (Takes 70% of available width)
        cameras_container = QWidget()
        cameras_layout = QHBoxLayout(cameras_container)
        cameras_layout.setContentsMargins(0, 0, 0, 0)
        cameras_layout.setSpacing(15)

        # Top Camera Frame (Responsive sizing)
        top_camera_group = QGroupBox("Top Camera View")
        top_camera_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        top_camera_layout = QVBoxLayout(top_camera_group)
        top_camera_layout.setContentsMargins(5, 15, 5, 5)
        self.top_camera_label = QLabel("Initializing Camera...")
        self.top_camera_label.setAlignment(Qt.AlignCenter)
        self.top_camera_label.setMinimumSize(400, 225)  # Minimum 16:9 ratio
        self.top_camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.top_camera_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        top_camera_layout.addWidget(self.top_camera_label)
        cameras_layout.addWidget(top_camera_group)

        # Bottom Camera Frame (Responsive sizing)
        bottom_camera_group = QGroupBox("Bottom Camera View")
        bottom_camera_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        bottom_camera_layout = QVBoxLayout(bottom_camera_group)
        bottom_camera_layout.setContentsMargins(5, 15, 5, 5)
        self.bottom_camera_label = QLabel("Initializing Camera...")
        self.bottom_camera_label.setAlignment(Qt.AlignCenter)
        self.bottom_camera_label.setMinimumSize(400, 225)  # Minimum 16:9 ratio
        self.bottom_camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.bottom_camera_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        bottom_camera_layout.addWidget(self.bottom_camera_label)
        cameras_layout.addWidget(bottom_camera_group)

        # Right Side: Live Defect Analysis Panel (Takes 30% of available width)
        self.defect_analysis_panel = QGroupBox("Live Defect Analysis")
        self.defect_analysis_panel.setMinimumWidth(350)  # Minimum width
        self.defect_analysis_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.defect_analysis_panel.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        defect_analysis_layout = QVBoxLayout(self.defect_analysis_panel)
        defect_analysis_layout.setContentsMargins(10, 15, 10, 10)

        # Defect details display
        self.defect_details_text = QTextEdit()
        self.defect_details_text.setReadOnly(True)
        self.defect_details_text.setMinimumHeight(200)  # Minimum height
        self.defect_details_text.setMaximumHeight(300)  # Maximum height for responsive design
        self.defect_details_text.setStyleSheet("font-size: 12px; background-color: #f9f9f9;")
        defect_analysis_layout.addWidget(self.defect_details_text)

        # Current grade display
        self.current_grade_label = QLabel("Final Grade: Waiting for wood...")
        self.current_grade_label.setAlignment(Qt.AlignCenter)
        self.current_grade_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px; border: 2px solid #ccc; border-radius: 5px;")
        defect_analysis_layout.addWidget(self.current_grade_label)

        # Wood classification display
        self.wood_classification_label = QLabel("Wood Classification: Unknown")
        self.wood_classification_label.setAlignment(Qt.AlignCenter)
        self.wood_classification_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #666;")
        defect_analysis_layout.addWidget(self.wood_classification_label)

        # Add defect panel to cameras container
        top_section_layout.addWidget(cameras_container, 7)  # 70% width
        top_section_layout.addWidget(self.defect_analysis_panel, 3)  # 30% width

        self.main_layout.addWidget(top_section_container)

        # Middle Section: Controls (Responsive layout)
        controls_container = QWidget()
        controls_layout = QHBoxLayout(controls_container)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        controls_layout.setSpacing(15)

        # Conveyor Control Group (Responsive width)
        conveyor_group = QGroupBox("Conveyor Control")
        conveyor_group.setMinimumWidth(250)  # Minimum width
        conveyor_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        conveyor_layout = QGridLayout(conveyor_group)
        conveyor_layout.setContentsMargins(10, 15, 10, 10)
        conveyor_layout.setSpacing(8)
        
        btn_continuous = QPushButton("Continuous")
        btn_continuous.setMinimumHeight(30)  # Minimum button height
        btn_continuous.setStyleSheet("font-size: 14px; font-weight: bold;")
        btn_continuous.clicked.connect(self.set_continuous_mode)
        conveyor_layout.addWidget(btn_continuous, 0, 0)
        
        btn_trigger = QPushButton("Trigger")
        btn_trigger.setMinimumHeight(30)  # Minimum button height
        btn_trigger.setStyleSheet("font-size: 14px; font-weight: bold;")
        btn_trigger.clicked.connect(self.set_trigger_mode)
        conveyor_layout.addWidget(btn_trigger, 0, 1)
        
        btn_idle = QPushButton("IDLE")
        btn_idle.setMinimumHeight(30)  # Minimum button height
        btn_idle.setStyleSheet("font-size: 14px; font-weight: bold;")
        btn_idle.clicked.connect(self.set_idle_mode)
        conveyor_layout.addWidget(btn_idle, 0, 2)
        controls_layout.addWidget(conveyor_group)

        # Detection Settings Group (Responsive width)
        detection_group = QGroupBox("Detection")
        detection_group.setMinimumWidth(250)  # Minimum width
        detection_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        detection_layout = QVBoxLayout(detection_group)
        detection_layout.setContentsMargins(10, 15, 10, 10)
        detection_layout.setSpacing(5)

        # Detection State Label
        self.detection_state_label = QLabel("State: Waiting")
        self.detection_state_label.setStyleSheet("font-size: 12px; color: #666; font-weight: bold;")
        detection_layout.addWidget(self.detection_state_label)

        self.roi_checkbox = QCheckBox("Top ROI Active")
        self.roi_checkbox.setChecked(True)
        self.roi_checkbox.setStyleSheet("font-size: 14px;")
        self.roi_checkbox.toggled.connect(self.toggle_roi)
        detection_layout.addWidget(self.roi_checkbox)

        self.wood_detection_checkbox = QCheckBox("Show Wood Detection")
        self.wood_detection_checkbox.setChecked(True)
        self.wood_detection_checkbox.setStyleSheet("font-size: 14px;")
        self.wood_detection_checkbox.toggled.connect(self.toggle_wood_detection)
        detection_layout.addWidget(self.wood_detection_checkbox)

        self.live_detection_checkbox = QCheckBox("Live Detection")
        self.live_detection_checkbox.setStyleSheet("font-size: 14px;")
        self.live_detection_checkbox.toggled.connect(self.toggle_live_detection)
        detection_layout.addWidget(self.live_detection_checkbox)

        self.auto_grade_checkbox = QCheckBox("Auto Grade")
        self.auto_grade_checkbox.setStyleSheet("font-size: 14px;")
        self.auto_grade_checkbox.toggled.connect(self.toggle_auto_grade)
        detection_layout.addWidget(self.auto_grade_checkbox)
        controls_layout.addWidget(detection_group)

        # Status Information Group (Responsive width)
        status_group = QGroupBox("System Status")
        status_group.setMinimumWidth(200)  # Minimum width
        status_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        status_layout = QVBoxLayout(status_group)
        status_layout.setContentsMargins(10, 15, 10, 10)
        status_layout.setSpacing(5)
        
        self.top_camera_status = QLabel("Top Camera: Initializing")
        self.top_camera_status.setStyleSheet("font-size: 12px;")
        status_layout.addWidget(self.top_camera_status)
        
        self.bottom_camera_status = QLabel("Bottom Camera: Initializing")
        self.bottom_camera_status.setStyleSheet("font-size: 12px;")
        status_layout.addWidget(self.bottom_camera_status)
        
        self.arduino_status = QLabel("Arduino: Disconnected")
        self.arduino_status.setStyleSheet("font-size: 12px; color: red;")
        status_layout.addWidget(self.arduino_status)

        self.system_status_label = QLabel("System: Initializing")
        self.system_status_label.setStyleSheet("font-size: 12px; color: #666;")
        status_layout.addWidget(self.system_status_label)

        controls_layout.addWidget(status_group)

        # Reports Group (Responsive width)
        reports_group = QGroupBox("Reports")
        reports_group.setMinimumWidth(200)  # Minimum width
        reports_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        reports_layout = QVBoxLayout(reports_group)
        reports_layout.setContentsMargins(10, 15, 10, 10)
        reports_layout.setSpacing(5)
        
        self.log_status_label = QLabel("Log: Ready")
        self.log_status_label.setStyleSheet("color: green; font-size: 14px;")
        reports_layout.addWidget(self.log_status_label)
        
        btn_generate_report = QPushButton("Generate Report")
        btn_generate_report.setMinimumHeight(25)  # Minimum button height
        btn_generate_report.setStyleSheet("font-size: 14px; font-weight: bold;")
        btn_generate_report.clicked.connect(self.manual_generate_report)
        reports_layout.addWidget(btn_generate_report)
        
        self.show_report_notification_checkbox = QCheckBox("Notifications")
        self.show_report_notification_checkbox.setChecked(True) # Default from original
        self.show_report_notification_checkbox.setStyleSheet("font-size: 14px;")
        reports_layout.addWidget(self.show_report_notification_checkbox)
        
        self.last_report_label = QLabel("Last: None")
        self.last_report_label.setStyleSheet("font-size: 12px; color: #666;")
        self.last_report_label.setWordWrap(True)
        reports_layout.addWidget(self.last_report_label)
        
        controls_layout.addWidget(reports_group)

        self.main_layout.addWidget(controls_container)

        # Bottom Section: Enhanced Tabbed Statistics (Responsive for maximized)
        self.stats_notebook = QTabWidget()
        self.stats_notebook.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stats_notebook.setMinimumHeight(250)  # Minimum height for content
        self.stats_notebook.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                padding: 12px 24px;
                margin-right: 2px;
                font-size: 14px;
                min-width: 120px;
                text-align: center;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #007acc;
            }
        """)
        self.main_layout.addWidget(self.stats_notebook)

        # Tab 1: Grade Summary (Enhanced with fixed dimensions)
        grade_summary_widget = QWidget()
        grade_summary_layout = QVBoxLayout(grade_summary_widget)
        grade_summary_layout.setContentsMargins(15, 15, 15, 15)
        grade_summary_layout.setSpacing(10)

        # Grade statistics frame
        grade_stats_frame = QFrame()
        grade_stats_frame.setStyleSheet("border: 1px solid #ccc; border-radius: 5px; padding: 10px; background-color: #f9f9f9;")
        grade_stats_layout = QGridLayout(grade_stats_frame)
        grade_stats_layout.setContentsMargins(10, 10, 10, 10)
        grade_stats_layout.setSpacing(10)

        # Grade labels with enhanced styling
        for i in range(4):
            grade_label = QLabel(f"Grade {i}")
            grade_label.setStyleSheet("font-size: 16px; font-weight: bold;")
            grade_stats_layout.addWidget(grade_label, i, 0)
            
            count_label = QLabel("0")
            count_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #333;")
            count_label.setAlignment(Qt.AlignCenter)
            setattr(self, f"grade_{i}_count", count_label)
            grade_stats_layout.addWidget(count_label, i, 1)
            
            percentage_label = QLabel("0%")
            percentage_label.setStyleSheet("font-size: 14px; color: #666;")
            percentage_label.setAlignment(Qt.AlignCenter)
            setattr(self, f"grade_{i}_percentage", percentage_label)
            grade_stats_layout.addWidget(percentage_label, i, 2)

        grade_summary_layout.addWidget(grade_stats_frame)

        # Session information
        session_info_frame = QFrame()
        session_info_frame.setStyleSheet("border: 1px solid #ccc; border-radius: 5px; padding: 10px; background-color: #f0f8ff;")
        session_info_layout = QGridLayout(session_info_frame)
        session_info_layout.setContentsMargins(10, 10, 10, 10)

        session_info_layout.addWidget(QLabel("Total Processed:"), 0, 0)
        self.total_processed_label = QLabel("0")
        self.total_processed_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        session_info_layout.addWidget(self.total_processed_label, 0, 1)

        session_info_layout.addWidget(QLabel("Session Duration:"), 1, 0)
        self.session_duration_label = QLabel("00:00:00")
        self.session_duration_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        session_info_layout.addWidget(self.session_duration_label, 1, 1)

        grade_summary_layout.addWidget(session_info_frame)
        self.stats_notebook.addTab(grade_summary_widget, "Grade Summary")

        # Tab 2: Performance Metrics
        performance_widget = QWidget()
        performance_layout = QVBoxLayout(performance_widget)
        performance_layout.setContentsMargins(15, 15, 15, 15)

        # Performance display area
        self.performance_display = QTextEdit()
        self.performance_display.setReadOnly(True)
        self.performance_display.setStyleSheet("font-family: monospace; background-color: #f5f5f5;")
        performance_layout.addWidget(self.performance_display)

        self.stats_notebook.addTab(performance_widget, "Performance")

        # Tab 3: System Log
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(15, 15, 15, 15)

        # Log display area
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("font-family: monospace; font-size: 11px; background-color: #f8f8f8;")
        log_layout.addWidget(self.log_display)

        # Log controls
        log_controls_layout = QHBoxLayout()
        btn_clear_log = QPushButton("Clear Log")
        btn_clear_log.clicked.connect(lambda: self.log_display.clear())
        log_controls_layout.addWidget(btn_clear_log)
        
        btn_export_log = QPushButton("Export Log")
        btn_export_log.clicked.connect(self.export_log)
        log_controls_layout.addWidget(btn_export_log)
        log_controls_layout.addStretch()
        
        log_layout.addLayout(log_controls_layout)
        self.stats_notebook.addTab(log_widget, "System Log")

    def setup_dev_mode(self):
        if self.dev_mode:
            print("Running in Development Mode: Simulating camera feeds and data.")
            self.simulate_camera_feed()
            # Mock Arduino module
            self.arduino_module.send_arduino_command = lambda cmd: print(f"DEV MODE: Mock Arduino command '{cmd}' sent.")
            self.arduino_module.setup_arduino = lambda: print("DEV MODE: Mock Arduino setup.")
            self.arduino_module.close_connection = lambda: print("DEV MODE: Mock Arduino closed.")
            # Mock detection module if needed for specific tests
            self.detection_module.analyze_frame = self._mock_analyze_frame
            self.detection_module.load_model = lambda: print("DEV MODE: Mock DeGirum model loaded.")
            self.detection_module.detect_wood = self._mock_detect_wood

    def _mock_analyze_frame(self, frame, camera_name="top"):
        # Simple mock for analyze_frame
        h, w, _ = frame.shape
        annotated_frame = frame.copy()
        # Draw a dummy detection
        cv2.rectangle(annotated_frame, (w//4, h//4), (w*3//4, h*3//4), (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Mock Defect", (w//4 + 10, h//4 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return annotated_frame, {"mock_defect": 1}, [("mock_defect", 10.0, 5.0)]

    def _mock_detect_wood(self, frame):
        # Simple mock for detect_wood
        # Simulate wood detection based on a simple condition or toggle
        return True # Always detect wood in dev mode for now

    def simulate_camera_feed(self):
        """Simulate camera feeds in development mode"""
        import numpy as np
        
        # Create mock images for top and bottom cameras
        height, width = 480, 640
        
        # Top camera - create a mock wood piece image
        top_image = np.random.randint(100, 200, (height, width, 3), dtype=np.uint8)
        # Add some wood-like texture
        cv2.rectangle(top_image, (50, 100), (590, 380), (139, 69, 19), -1)  # Brown wood color
        cv2.putText(top_image, "TOP CAMERA - MOCK FEED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Bottom camera - create another mock wood piece image
        bottom_image = np.random.randint(80, 180, (height, width, 3), dtype=np.uint8)
        cv2.rectangle(bottom_image, (60, 120), (580, 360), (101, 67, 33), -1)  # Different brown
        cv2.putText(bottom_image, "BOTTOM CAMERA - MOCK FEED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Store mock frames
        self.top_frame_original = top_image
        self.bottom_frame_original = bottom_image

    def display_message(self, message, msg_type="info"):
        """Display message in the log with timestamp"""
        timestamp = QDateTime.currentDateTime().toString('hh:mm:ss')
        log_entry = f"[{timestamp}] {message}"
        
        # Add to log display
        if hasattr(self, 'log_display'):
            self.log_display.append(log_entry)
        
        # Print to console
        print(log_entry)

    def update_feeds(self):
        """Update camera feeds and process detection"""
        try:
            print(f"DEBUG: update_feeds called - live_detection_var: {self.live_detection_var}")
            # Get frames from cameras or use mock frames in dev mode
            if self.dev_mode:
                top_frame = self.top_frame_original.copy() if self.top_frame_original is not None else None
                bottom_frame = self.bottom_frame_original.copy() if self.bottom_frame_original is not None else None
            else:
                top_frame = self.camera_module.get_top_frame()
                bottom_frame = self.camera_module.get_bottom_frame()

            # Update camera status
            self.update_camera_status("top", top_frame is not None)
            self.update_camera_status("bottom", bottom_frame is not None)

            # Process frames if available
            if top_frame is not None:
                # Validate frame before processing
                if not self._validate_frame(top_frame):
                    self.display_message("Invalid frame data received from top camera", "warning")
                    top_frame = None
                else:
                    # Run detection if live detection is enabled
                    if self.live_detection_var:
                        try:
                            print(f"DEBUG: GUI calling detection_module.analyze_frame for top camera")
                            annotated_frame, defects, defect_list, alignment_result = self.detection_module.analyze_frame(top_frame, "top")
                            print(f"DEBUG: GUI received result from detection_module.analyze_frame for top camera")
                            self.current_defects["top"] = {"defects": defects, "defect_list": defect_list}

                            # Auto grade if enabled
                            if self.auto_grade_var:
                                self.calculate_and_display_grade()
                        except Exception as e:
                            self.display_message(f"Detection error on top camera: {str(e)}", "error")
                            print(f"DEBUG: Exception in GUI top camera detection: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            annotated_frame = top_frame
                    else:
                        annotated_frame = top_frame

                    # Always apply alignment overlay for ROI visualization
                    try:
                        alignment_result = self.detection_module.alignment_module.check_wood_alignment(annotated_frame, None)
                        annotated_frame = self.detection_module.alignment_module.draw_alignment_overlay(annotated_frame, alignment_result)
                    except Exception as e:
                        self.display_message(f"Error applying alignment overlay: {str(e)}", "warning")

                    # Check for ROI intersection and add alerts regardless of live detection status
                    try:
                        # Always run wood detection for ROI alerts and bounding boxes
                        wood_detected, wood_confidence, wood_bbox = self.detection_module.detect_wood_presence(annotated_frame)

                        if wood_detected and wood_bbox:
                            # Check if wood bbox intersects with any ROI
                            wood_intersects_roi = self._check_wood_roi_intersection(wood_bbox, "top")

                            # Add red border and text if wood intersects ROI
                            if wood_intersects_roi:
                                self._add_misalignment_indicators(annotated_frame)
                                print(f"DEBUG: Added misalignment indicators for top camera - wood bbox: {wood_bbox}")

                            # Draw bounding boxes regardless of live detection status
                            if self.wood_detection_checkbox.isChecked():
                                # Draw wood bounding box
                                x1, y1, x2, y2 = wood_bbox
                                if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= annotated_frame.shape[1] and y2 <= annotated_frame.shape[0]:
                                    # Choose color based on ROI intersection
                                    box_color = (0, 0, 255) if wood_intersects_roi else (0, 255, 0)  # Red if intersects, Green if not

                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 5)

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

                                    # Add label
                                    status_text = "NOT ALIGNED" if wood_intersects_roi else "ALIGNED"
                                    cv2.putText(annotated_frame, f"WOOD DETECTED ({status_text}, Conf: {wood_confidence:.2f})",
                                               (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, 3)

                                    print(f"DEBUG: Drew bounding box for top camera - wood bbox: {wood_bbox}, intersects ROI: {wood_intersects_roi}")
                    except Exception as e:
                        self.display_message(f"Error in ROI alert system: {str(e)}", "warning")
                        print(f"DEBUG: Exception in top camera ROI system: {str(e)}")

                    # Convert and display
                    self.display_frame(annotated_frame, self.top_camera_label)

            if bottom_frame is not None:
                # Validate frame before processing
                if not self._validate_frame(bottom_frame):
                    self.display_message("Invalid frame data received from bottom camera", "warning")
                    bottom_frame = None
                else:
                    # Run detection if live detection is enabled
                    if self.live_detection_var:
                        try:
                            print(f"DEBUG: GUI calling detection_module.analyze_frame for bottom camera")
                            annotated_frame, defects, defect_list, alignment_result = self.detection_module.analyze_frame(bottom_frame, "bottom")
                            print(f"DEBUG: GUI received result from detection_module.analyze_frame for bottom camera")
                            self.current_defects["bottom"] = {"defects": defects, "defect_list": defect_list}

                            # Auto grade if enabled (combined with top camera results)
                            if self.auto_grade_var:
                                self.calculate_and_display_grade()
                        except Exception as e:
                            self.display_message(f"Detection error on bottom camera: {str(e)}", "error")
                            print(f"DEBUG: Exception in GUI bottom camera detection: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            annotated_frame = bottom_frame
                    else:
                        annotated_frame = bottom_frame

                    # Always apply alignment overlay for ROI visualization
                    try:
                        alignment_result = self.detection_module.alignment_module.check_wood_alignment(annotated_frame, None)
                        annotated_frame = self.detection_module.alignment_module.draw_alignment_overlay(annotated_frame, alignment_result)
                    except Exception as e:
                        self.display_message(f"Error applying alignment overlay: {str(e)}", "warning")

                    # Check for ROI intersection and add alerts regardless of live detection status
                    try:
                        # Always run wood detection for ROI alerts and bounding boxes
                        wood_detected, wood_confidence, wood_bbox = self.detection_module.detect_wood_presence(annotated_frame)

                        if wood_detected and wood_bbox:
                            # Check if wood bbox intersects with any ROI
                            wood_intersects_roi = self._check_wood_roi_intersection(wood_bbox, "bottom")

                            # Add red border and text if wood intersects ROI
                            if wood_intersects_roi:
                                self._add_misalignment_indicators(annotated_frame)
                                print(f"DEBUG: Added misalignment indicators for bottom camera - wood bbox: {wood_bbox}")

                            # Draw bounding boxes regardless of live detection status
                            if self.wood_detection_checkbox.isChecked():
                                # Draw wood bounding box
                                x1, y1, x2, y2 = wood_bbox
                                if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= annotated_frame.shape[1] and y2 <= annotated_frame.shape[0]:
                                    # Choose color based on ROI intersection
                                    box_color = (0, 0, 255) if wood_intersects_roi else (0, 255, 0)  # Red if intersects, Green if not

                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 5)

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

                                    # Add label
                                    status_text = "NOT ALIGNED" if wood_intersects_roi else "ALIGNED"
                                    cv2.putText(annotated_frame, f"WOOD DETECTED ({status_text}, Conf: {wood_confidence:.2f})",
                                               (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, 3)

                                    print(f"DEBUG: Drew bounding box for bottom camera - wood bbox: {wood_bbox}, intersects ROI: {wood_intersects_roi}")
                    except Exception as e:
                        self.display_message(f"Error in ROI alert system: {str(e)}", "warning")
                        print(f"DEBUG: Exception in bottom camera ROI system: {str(e)}")

                    # Convert and display
                    self.display_frame(annotated_frame, self.bottom_camera_label)

            # Update session duration
            self.update_session_duration()
            
            # Process message queue
            self.process_message_queue()

        except Exception as e:
            self.display_message(f"Error in update_feeds: {str(e)}", "error")

    def _validate_frame(self, frame):
        """Validate frame data to prevent processing corrupted frames."""
        try:
            # Check if frame is None
            if frame is None:
                return False

            # Check if frame has valid shape
            if not hasattr(frame, 'shape') or len(frame.shape) != 3:
                return False

            # Check dimensions
            height, width, channels = frame.shape
            if height <= 0 or width <= 0 or channels not in [1, 3, 4]:
                return False

            # Check reasonable size limits (prevent extremely large frames)
            if width > 4096 or height > 4096 or width < 16 or height < 16:
                return False

            # Check data type
            if hasattr(frame, 'dtype') and frame.dtype != 'uint8':
                return False

            # Check if frame data is accessible and contiguous
            try:
                # Try to access a small portion of the frame
                _ = frame[0:1, 0:1]
                # Check if frame is contiguous in memory
                if not frame.flags['C_CONTIGUOUS']:
                    # Try to make it contiguous
                    frame = frame.copy()
                    _ = frame[0:1, 0:1]
            except:
                return False

            # Additional validation for OpenCV matrix operations
            try:
                # Test basic OpenCV operations that might fail
                import cv2
                # Test color conversion (this often fails with corrupted frames)
                if channels == 3:
                    test_frame = cv2.cvtColor(frame[0:10, 0:10], cv2.COLOR_BGR2RGB)
                    if test_frame.shape != (10, 10, 3):
                        return False
            except:
                return False

            return True

        except Exception as e:
            self.display_message(f"Frame validation failed: {str(e)}", "warning")
            return False

    def display_frame(self, frame, label_widget):
        """Convert OpenCV frame to QPixmap and display in label"""
        if frame is not None:
            try:
                print(f"DEBUG: Displaying frame with shape: {frame.shape}, dtype: {frame.dtype}")

                # Check if frame has valid data
                if frame.size == 0:
                    print("DEBUG: Frame has zero size")
                    return

                # Ensure frame is in correct format
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    print(f"DEBUG: Frame has unexpected shape: {frame.shape}")
                    rgb_image = frame

                h, w = rgb_image.shape[:2]
                ch = rgb_image.shape[2] if len(rgb_image.shape) > 2 else 1

                print(f"DEBUG: RGB image shape: {rgb_image.shape}, channels: {ch}")

                if ch == 3:
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                elif ch == 1:
                    bytes_per_line = w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
                else:
                    print(f"DEBUG: Unsupported number of channels: {ch}")
                    return

                if qt_image.isNull():
                    print("DEBUG: QImage is null")
                    return

                # Scale image to fit label while maintaining aspect ratio
                pixmap = QPixmap.fromImage(qt_image)
                if pixmap.isNull():
                    print("DEBUG: QPixmap is null")
                    return

                scaled_pixmap = pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                if scaled_pixmap.isNull():
                    print("DEBUG: Scaled QPixmap is null")
                    return

                label_widget.setPixmap(scaled_pixmap)
                print(f"DEBUG: Successfully set pixmap on label")
            except Exception as e:
                self.display_message(f"Error displaying frame: {str(e)}", "error")
                print(f"DEBUG: Exception in display_frame: {str(e)}")
                import traceback
                traceback.print_exc()
                # Clear the label on error
                label_widget.clear()
        else:
            print("DEBUG: Frame is None")

    def update_camera_status(self, camera_name, is_available):
        """Update camera status display"""
        status_text = "Connected" if is_available else "Disconnected"
        color = "green" if is_available else "red"
        
        if camera_name == "top":
            self.top_camera_status.setText(f"Top Camera: {status_text}")
            self.top_camera_status.setStyleSheet(f"font-size: 12px; color: {color};")
        else:
            self.bottom_camera_status.setText(f"Bottom Camera: {status_text}")
            self.bottom_camera_status.setStyleSheet(f"font-size: 12px; color: {color};")

    def calculate_and_display_grade(self):
        """Calculate grade from current defects and display results"""
        try:
            # Combine defects from both cameras
            all_defects = {}
            all_defect_lists = []
            
            for camera in ["top", "bottom"]:
                if camera in self.current_defects:
                    camera_defects = self.current_defects[camera]["defects"]
                    camera_defect_list = self.current_defects[camera]["defect_list"]
                    
                    # Merge defect counts
                    for defect_type, count in camera_defects.items():
                        all_defects[defect_type] = all_defects.get(defect_type, 0) + count
                    
                    # Add defect details
                    all_defect_lists.extend(camera_defect_list)

            # Calculate grade
            grade_info = calculate_grade(all_defects)
            final_grade = determine_final_grade(grade_info)
            
            # Store current grade info
            self.current_grade_info = {
                'grade': final_grade,
                'defects': all_defects,
                'defect_list': all_defect_lists,
                'grade_info': grade_info
            }
            
            # Update grade display
            grade_color = get_grade_color(final_grade)
            self.current_grade_label.setText(f"Final Grade: {final_grade}")
            self.current_grade_label.setStyleSheet(f"""
                font-size: 18px; font-weight: bold; padding: 10px;
                border: 2px solid {grade_color}; border-radius: 5px;
                background-color: {grade_color}20; color: {grade_color};
            """)

            # Update wood classification
            self.update_wood_classification()

            # Update defect details
            self.update_defect_details(all_defects, all_defect_lists, grade_info)

            # Update detection state
            self.update_detection_state("Grading")
            
        except Exception as e:
            self.display_message(f"Error calculating grade: {str(e)}", "error")

    def update_defect_details(self, defects, defect_list, grade_info):
        """Update the defect details display"""
        details_text = "=== DEFECT ANALYSIS ===\n\n"
        
        # Defect summary
        if defects:
            details_text += "Defect Summary:\n"
            for defect_type, count in defects.items():
                details_text += f"• {defect_type}: {count}\n"
        else:
            details_text += "No defects detected.\n"
        
        details_text += "\n=== GRADE CALCULATION ===\n\n"
        
        # Grade breakdown
        if grade_info:
            for grade, criteria in grade_info.items():
                if criteria['meets_criteria']:
                    details_text += f"✓ Grade {grade}: MEETS CRITERIA\n"
                    details_text += f"  Max defects allowed: {criteria['max_defects']}\n"
                    details_text += f"  Current defects: {criteria['current_defects']}\n\n"
                else:
                    details_text += f"✗ Grade {grade}: EXCEEDS LIMITS\n"
                    details_text += f"  Max defects allowed: {criteria['max_defects']}\n"
                    details_text += f"  Current defects: {criteria['current_defects']}\n\n"
        
        # Detailed defect list
        if defect_list:
            details_text += "=== DETAILED DEFECTS ===\n\n"
            for i, (defect_type, x, y) in enumerate(defect_list, 1):
                details_text += f"{i}. {defect_type} at ({x:.1f}, {y:.1f})\n"
        
        self.defect_details_text.setPlainText(details_text)

    def update_session_duration(self):
        """Update session duration display"""
        if hasattr(self, 'session_start_time'):
            current_time = QDateTime.currentMSecsSinceEpoch() / 1000
            duration_seconds = int(current_time - self.session_start_time)
            
            hours = duration_seconds // 3600
            minutes = (duration_seconds % 3600) // 60
            seconds = duration_seconds % 60
            
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            self.session_duration_label.setText(duration_str)

    def process_message_queue(self):
        """Process messages from Arduino module"""
        try:
            while not self.message_queue.empty():
                message = self.message_queue.get_nowait()
                self.handle_arduino_message(message)
        except:
            pass

    def handle_arduino_message(self, message):
        """Handle messages received from Arduino"""
        try:
            log_info(SystemComponent.GUI, f"Arduino message received: {message}")

            if message == "B":
                # IR beam broken - start wood detection workflow
                self.handle_ir_beam_broken()
            elif message.startswith("L:"):
                # Length measurement received
                try:
                    duration_ms = int(message.split(":")[1])
                    self.handle_length_measurement(duration_ms)
                except (ValueError, IndexError) as e:
                    log_error(SystemComponent.GUI, f"Invalid length message format: {message}", e)
            else:
                # Other Arduino messages
                self.update_system_status(f"Arduino: {message}")

        except Exception as e:
            log_error(SystemComponent.GUI, f"Error handling Arduino message '{message}': {str(e)}", e)

    def handle_ir_beam_broken(self):
        """Handle IR beam broken event - start wood detection workflow"""
        try:
            log_info(SystemComponent.GUI, "IR beam broken - starting wood detection workflow")

            # Only respond to IR triggers in TRIGGER mode
            if self.current_mode == "TRIGGER":
                if not self.auto_detection_active:
                    log_info(SystemComponent.GUI, "✅ TRIGGER MODE: Starting detection...")
                    self.live_detection_checkbox.setChecked(True)
                    self.auto_grade_checkbox.setChecked(True)

                    # Update internal state variables
                    self.live_detection_var = True
                    self.auto_grade_var = True

                    # Update states
                    self.ir_triggered = True
                    self.wood_confirmed = False
                    self.auto_detection_active = True

                    self.update_system_status("Status: IR TRIGGERED - Motor should be running!")
                    self.update_detection_state("Detecting")

                else:
                    log_warning(SystemComponent.GUI, "⚠️ IR beam broken but detection already active")
            else:
                # In IDLE or CONTINUOUS mode, just log the IR signal but don't act on it
                log_info(SystemComponent.GUI, f"❌ IR beam broken received but system is in {self.current_mode} mode - ignoring trigger")
                self.update_system_status(f"Status: IR signal ignored ({self.current_mode} mode)")
                self.update_detection_state("Waiting")

        except Exception as e:
            log_error(SystemComponent.GUI, f"Error handling IR beam broken: {str(e)}", e)

    def handle_length_measurement(self, duration_ms):
        """Handle length measurement from Arduino - IR beam cleared, stop detection"""
        try:
            # Calculate estimated length based on conveyor speed
            estimated_speed_mm_per_ms = 0.1  # Adjust this value based on testing
            estimated_length_mm = duration_ms * estimated_speed_mm_per_ms

            log_info(SystemComponent.GUI, f"Length measurement: {duration_ms}ms → ~{estimated_length_mm:.1f}mm")

            # In TRIGGER mode, stop detection when beam clears (length message received)
            if self.current_mode == "TRIGGER" and self.auto_detection_active:
                log_info(SystemComponent.GUI, "IR beam cleared – stopping detection (TRIGGER MODE)")

                # Deactivate the checkboxes
                self.live_detection_checkbox.setChecked(False)
                self.auto_grade_checkbox.setChecked(False)

                # Update internal state variables
                self.live_detection_var = False
                self.auto_grade_var = False

                # Stop detection session
                self.auto_detection_active = False
                self.ir_triggered = False

                if self.wood_confirmed:
                    # Wood was detected, process grading
                    self.wood_confirmed = False
                    self.update_system_status("Status: Processing results... → Ready for next trigger")
                    self.update_detection_state("Processing")

                    # Process any final grading if needed
                    self.finalize_detection_session()

                    # Return to ready state
                    self.update_system_status("Status: TRIGGER MODE - Waiting for IR beam trigger")
                    self.update_detection_state("Waiting")
                else:
                    # No wood detected, start 3-second buffer
                    self.update_system_status("Status: Object passed - No wood detected, waiting 3 seconds...")
                    self.update_detection_state("Waiting")
                    if self.no_wood_timer is None:
                        self.no_wood_timer = QTimer(self)
                        self.no_wood_timer.setSingleShot(True)
                        self.no_wood_timer.timeout.connect(self.mark_object_cleared)
                        self.no_wood_timer.start(3000)  # 3 seconds

            else:
                log_info(SystemComponent.GUI, f"Length signal received (duration: {duration_ms}ms) but system is in {self.current_mode} mode or no detection active")
                self.update_system_status(f"Status: Object length: ~{estimated_length_mm:.1f}mm")

        except Exception as e:
            log_error(SystemComponent.GUI, f"Error handling length measurement: {str(e)}", e)

    def finalize_detection_session(self):
        """Finalize the detection session and perform grading if needed"""
        try:
            # This method would contain the logic to finalize grading
            # Based on accumulated detection data during the session
            log_info(SystemComponent.GUI, "Finalizing detection session...")

            # Increment counters and update statistics
            if self.current_grade_info:
                grade = self.current_grade_info['grade']
                if grade in self.grade_counts:
                    self.grade_counts[grade] += 1
                    self.live_stats[f"grade{grade}"] += 1
                    self.total_pieces_processed += 1

                    # Update UI counters
                    self.update_grade_counters()

            # Log completion
            log_info(SystemComponent.GUI, "Detection session finalized")

        except Exception as e:
            log_error(SystemComponent.GUI, f"Error finalizing detection session: {str(e)}", e)

    def mark_object_cleared(self):
        """Mark object as cleared after 3-second buffer for no wood detection"""
        try:
            self.update_system_status("Status: Object cleared - No wood detected")
            self.wood_confirmed = False
            self.auto_detection_active = False
            self.ir_triggered = False
            if self.no_wood_timer:
                self.no_wood_timer = None
            log_info(SystemComponent.GUI, "Object marked as cleared - no wood detected")
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error marking object cleared: {str(e)}", e)

    def update_performance_display(self, metrics):
        """Update performance metrics display"""
        if hasattr(self, 'performance_display'):
            performance_text = "=== PERFORMANCE METRICS ===\n\n"
            
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    performance_text += f"{metric_name}: {metric_value:.2f}\n"
                else:
                    performance_text += f"{metric_name}: {metric_value}\n"
            
            self.performance_display.setPlainText(performance_text)

    # Control Methods
    def set_continuous_mode(self):
        """Set system to continuous mode"""
        self.current_mode = "CONTINUOUS"
        self.display_message("System set to CONTINUOUS mode")
        self.update_system_status("Status: CONTINUOUS MODE - Live detection & auto-grading enabled")
        self.update_detection_state("Waiting")

        # Reset IR-triggered state
        self.ir_triggered = False
        self.wood_confirmed = False
        self.auto_detection_active = False

        # Send command to Arduino
        success = self.arduino_module.send_arduino_command('C')
        if not success:
            log_warning(SystemComponent.GUI, "Failed to send continuous mode command to Arduino")

    def set_trigger_mode(self):
        """Set system to trigger mode"""
        self.current_mode = "TRIGGER"
        self.display_message("System set to TRIGGER mode")
        self.update_system_status("Status: TRIGGER MODE - Waiting for IR beam trigger")
        self.update_detection_state("Waiting")

        # Reset state
        self.ir_triggered = False
        self.wood_confirmed = False
        self.auto_detection_active = False

        # Send command to Arduino
        success = self.arduino_module.send_arduino_command('T')
        if not success:
            log_warning(SystemComponent.GUI, "Failed to send trigger mode command to Arduino")

    def set_idle_mode(self):
        """Set system to idle mode"""
        self.current_mode = "IDLE"
        self.display_message("System set to IDLE mode")
        self.update_system_status("Status: IDLE MODE - System disabled, conveyor stopped")
        self.update_detection_state("Waiting")

        # Reset state
        self.ir_triggered = False
        self.wood_confirmed = False
        self.auto_detection_active = False

        # Send command to Arduino
        success = self.arduino_module.send_arduino_command('X')
        if not success:
            log_warning(SystemComponent.GUI, "Failed to send idle mode command to Arduino")

    def toggle_live_detection(self, checked):
        """Toggle live detection mode"""
        self.live_detection_var = checked
        status = "enabled" if checked else "disabled"
        self.display_message(f"Live detection {status}")

        # Update status display
        if checked:
            self.update_system_status(f"Status: {self.current_mode} MODE - Live detection ACTIVE")
        else:
            self.update_system_status(f"Status: {self.current_mode} MODE - Live detection DISABLED")

    def toggle_auto_grade(self, checked):
        """Toggle auto grading mode"""
        self.auto_grade_var = checked
        status = "enabled" if checked else "disabled"
        self.display_message(f"Auto grading {status}")

    def toggle_roi(self, checked):
        """Toggle ROI selection"""
        roi_status = "Active" if checked else "Disabled"
        self.display_message(f"Top ROI {roi_status}")

    def toggle_wood_detection(self, checked):
        """Toggle wood detection visualization"""
        detection_status = "enabled" if checked else "disabled"
        self.display_message(f"Wood detection visualization {detection_status}")

    def _check_wood_roi_intersection(self, wood_bbox, camera_name):
        """Check if wood bounding box intersects with any ROI (both top and bottom ROIs)"""
        try:
            if not wood_bbox:
                return False

            wx1, wy1, wx2, wy2 = wood_bbox
            print(f"DEBUG: Checking ROI intersection for {camera_name} camera")
            print(f"DEBUG: Wood bbox: [{wx1}, {wy1}, {wx2}, {wy2}]")

            # Use hardcoded ROI coordinates based on the log output
            # From log: "Top: (64,0) to (1216,108), Bottom: (64,612) to (1216,720)"
            frame_height = 720
            frame_width = 1280

            # Check both ROIs regardless of camera
            top_roi = (64, 0, 1216, 108)  # Top ROI: (64,0) to (1216,108)
            bottom_roi = (64, 612, 1216, 720)  # Bottom ROI: (64,612) to (1216,720)

            # Check intersection with top ROI
            top_x_overlap = (wx1 < top_roi[2]) and (wx2 > top_roi[0])
            top_y_overlap = (wy1 < top_roi[3]) and (wy2 > top_roi[1])
            top_intersection = top_x_overlap and top_y_overlap

            # Check intersection with bottom ROI
            bottom_x_overlap = (wx1 < bottom_roi[2]) and (wx2 > bottom_roi[0])
            bottom_y_overlap = (wy1 < bottom_roi[3]) and (wy2 > bottom_roi[1])
            bottom_intersection = bottom_x_overlap and bottom_y_overlap

            # Wood intersects ROI if it touches either top OR bottom ROI
            intersection = top_intersection or bottom_intersection

            print(f"DEBUG: Top ROI: {top_roi} - X overlap: {top_x_overlap}, Y overlap: {top_y_overlap}, Intersection: {top_intersection}")
            print(f"DEBUG: Bottom ROI: {bottom_roi} - X overlap: {bottom_x_overlap}, Y overlap: {bottom_y_overlap}, Intersection: {bottom_intersection}")
            print(f"DEBUG: Overall ROI intersection result: {intersection}")

            return intersection

        except Exception as e:
            self.display_message(f"Error checking ROI intersection: {str(e)}", "warning")
            print(f"DEBUG: Exception in ROI intersection check: {str(e)}")
            return False

    def _add_misalignment_indicators(self, frame):
        """Add red border and 'Wood not aligned' text to frame"""
        try:
            height, width = frame.shape[:2]

            # Add red border around the entire frame
            border_thickness = 8
            cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), border_thickness)

            # Add "Wood not aligned" text in the center
            text = "WOOD NOT ALIGNED"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            font_thickness = 4

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

            # Calculate text position (center of frame)
            text_x = (width - text_width) // 2
            text_y = (height + text_height) // 2

            # Add black outline for better visibility
            cv2.putText(frame, text, (text_x-2, text_y-2), font, font_scale, (0, 0, 0), font_thickness + 2)
            cv2.putText(frame, text, (text_x+2, text_y-2), font, font_scale, (0, 0, 0), font_thickness + 2)
            cv2.putText(frame, text, (text_x-2, text_y+2), font, font_scale, (0, 0, 0), font_thickness + 2)
            cv2.putText(frame, text, (text_x+2, text_y+2), font, font_scale, (0, 0, 0), font_thickness + 2)

            # Add red text
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

        except Exception as e:
            self.display_message(f"Error adding misalignment indicators: {str(e)}", "warning")

    def update_detection_state(self, state):
        """Update the detection state label"""
        try:
            state_colors = {
                "Waiting": "#666",
                "Detecting": "#f39c12",
                "Grading": "#27ae60",
                "Processing": "#3498db"
            }
            color = state_colors.get(state, "#666")
            self.detection_state_label.setText(f"State: {state}")
            self.detection_state_label.setStyleSheet(f"font-size: 12px; color: {color}; font-weight: bold;")
            self.detection_state = state
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error updating detection state: {str(e)}", e)

    def update_wood_classification(self):
        """Update wood classification display"""
        # Mock wood classification - in real implementation this would come from detection module
        wood_types = ["Pine", "Oak", "Maple", "Cedar", "Spruce"]
        import random
        self.wood_classification = random.choice(wood_types)
        self.wood_classification_label.setText(f"Wood Classification: {self.wood_classification}")
        self.wood_classification_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")

    def update_system_status(self, status_text):
        """Update system status label"""
        self.system_status_label.setText(status_text)
        self.system_status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")

    def update_grade_counters(self):
        """Update grade counters in the UI"""
        try:
            # Update individual grade counts
            for grade in range(4):
                count_label = getattr(self, f"grade_{grade}_count", None)
                if count_label:
                    count = self.live_stats.get(f"grade{grade}", 0)
                    count_label.setText(str(count))

            # Update total processed
            if hasattr(self, 'total_processed_label'):
                self.total_processed_label.setText(str(self.total_pieces_processed))

            # Update percentages
            total = sum(self.live_stats.values())
            if total > 0:
                for grade in range(4):
                    percentage_label = getattr(self, f"grade_{grade}_percentage", None)
                    if percentage_label:
                        count = self.live_stats.get(f"grade{grade}", 0)
                        percentage = (count / total) * 100
                        percentage_label.setText(f"{percentage:.1f}%")

        except Exception as e:
            log_error(SystemComponent.GUI, f"Error updating grade counters: {str(e)}", e)

    def manual_generate_report(self):
        """Generate manual report with enhanced data"""
        try:
            report_data = {
                'timestamp': QDateTime.currentDateTime().toString(),
                'total_processed': self.total_pieces_processed,
                'grade_counts': self.grade_counts,
                'session_duration': self.session_duration_label.text() if hasattr(self, 'session_duration_label') else "00:00:00",
                'wood_classification': self.wood_classification,
                'detection_state': self.detection_state,
                'current_mode': self.current_mode,
                'live_stats': self.live_stats
            }

            filename = self.reporting_module.generate_report(report_data)
            self.display_message(f"Report generated: {filename}")
            self.last_report_label.setText(f"Last: {filename}")

        except Exception as e:
            self.display_message(f"Error generating report: {str(e)}", "error")

    def export_log(self):
        """Export system log to file"""
        try:
            if hasattr(self, 'log_display'):
                log_content = self.log_display.toPlainText()
                timestamp = QDateTime.currentDateTime().toString('yyyy-MM-dd_hh-mm-ss')
                filename = f"logs/system_log_{timestamp}.txt"
                
                with open(filename, 'w') as f:
                    f.write(log_content)
                
                self.display_message(f"Log exported to: {filename}")
            
        except Exception as e:
            self.display_message(f"Error exporting log: {str(e)}", "error")

def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = WoodSortingApp(dev_mode=False)  # Set to False to use webcam in production mode
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
