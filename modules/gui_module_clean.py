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
        self.current_grade_label = QLabel("Grade: Not Detected")
        self.current_grade_label.setAlignment(Qt.AlignCenter)
        self.current_grade_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px; border: 2px solid #ccc; border-radius: 5px;")
        defect_analysis_layout.addWidget(self.current_grade_label)

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
                # Run detection if live detection is enabled
                if self.live_detection_var:
                    annotated_frame, defects, defect_list = self.detection_module.analyze_frame(top_frame, "top")
                    self.current_defects["top"] = {"defects": defects, "defect_list": defect_list}
                    
                    # Auto grade if enabled
                    if self.auto_grade_var:
                        self.calculate_and_display_grade()
                else:
                    annotated_frame = top_frame
                
                # Convert and display
                self.display_frame(annotated_frame, self.top_camera_label)

            if bottom_frame is not None:
                # Run detection if live detection is enabled
                if self.live_detection_var:
                    annotated_frame, defects, defect_list = self.detection_module.analyze_frame(bottom_frame, "bottom")
                    self.current_defects["bottom"] = {"defects": defects, "defect_list": defect_list}
                    
                    # Auto grade if enabled (combined with top camera results)
                    if self.auto_grade_var:
                        self.calculate_and_display_grade()
                else:
                    annotated_frame = bottom_frame
                
                # Convert and display
                self.display_frame(annotated_frame, self.bottom_camera_label)

            # Update session duration
            self.update_session_duration()
            
            # Process message queue
            self.process_message_queue()

        except Exception as e:
            self.display_message(f"Error in update_feeds: {str(e)}", "error")

    def display_frame(self, frame, label_widget):
        """Convert OpenCV frame to QPixmap and display in label"""
        if frame is not None:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale image to fit label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label_widget.setPixmap(scaled_pixmap)

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
            self.current_grade_label.setText(f"Grade: {final_grade}")
            self.current_grade_label.setStyleSheet(f"""
                font-size: 18px; font-weight: bold; padding: 10px; 
                border: 2px solid {grade_color}; border-radius: 5px;
                background-color: {grade_color}20; color: {grade_color};
            """)
            
            # Update defect details
            self.update_defect_details(all_defects, all_defect_lists, grade_info)
            
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
                self.display_message(f"Arduino: {message}")
        except:
            pass

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
        
    def set_trigger_mode(self):
        """Set system to trigger mode"""
        self.current_mode = "TRIGGER"
        self.display_message("System set to TRIGGER mode")
        
    def set_idle_mode(self):
        """Set system to idle mode"""
        self.current_mode = "IDLE"
        self.display_message("System set to IDLE mode")

    def toggle_live_detection(self, checked):
        """Toggle live detection mode"""
        self.live_detection_var = checked
        status = "enabled" if checked else "disabled"
        self.display_message(f"Live detection {status}")

    def toggle_auto_grade(self, checked):
        """Toggle auto grading mode"""
        self.auto_grade_var = checked
        status = "enabled" if checked else "disabled"
        self.display_message(f"Auto grading {status}")

    def manual_generate_report(self):
        """Generate manual report"""
        try:
            report_data = {
                'timestamp': QDateTime.currentDateTime().toString(),
                'total_processed': self.total_pieces_processed,
                'grade_counts': self.grade_counts,
                'session_duration': self.session_duration_label.text() if hasattr(self, 'session_duration_label') else "00:00:00"
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
    window = WoodSortingApp(dev_mode=True)  # Set to True for development mode
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
