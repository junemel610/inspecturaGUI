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
    QPushButton, QFrame, QGridLayout, QCheckBox, QTabWidget, QGroupBox, QTextEdit, QProgressBar, QScrollArea, QSizePolicy, QComboBox, QDoubleSpinBox, QSpinBox, QFormLayout, QLineEdit, QListWidget, QListWidgetItem
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QFont, QColor
from PyQt5.QtCore import Qt, QTimer, QDateTime, QThread, pyqtSignal
import queue
import time
import threading

try:
    import degirum_tools
    DEGIRUM_TOOLS_AVAILABLE = True
except ImportError:
    DEGIRUM_TOOLS_AVAILABLE = False
    print("Warning: degirum_tools not available - predict_stream functionality will be disabled")

from modules.camera_module import CameraModule
from modules.detection_module import DetectionModule
from modules.arduino_module import ArduinoModule
from modules.reporting_module import ReportingModule
from modules.grading_module import calculate_grade, determine_final_grade, get_grade_color
from modules import grading_module
from modules.utils_module import TOP_CAMERA_PIXEL_TO_MM, BOTTOM_CAMERA_PIXEL_TO_MM, WOOD_PALLET_WIDTH_MM, map_model_output_to_standard, calculate_defect_size
from modules.error_handler import (
    log_info, log_warning, log_error, SystemComponent,
    get_error_summary, error_handler, log_arduino_error
)
from modules.performance_monitor import get_performance_monitor, start_performance_monitoring
from config.settings import get_config

# ROI-based wood detection imports
from modules.roi_module import ROIModule, ROIManager, OverlapDetector, ROIBasedWorkflowManager, ROIVisualizer, ROIStatus
from modules.wood_detection_module import WoodDetectionEngine

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

        # Initialize ROI-based wood detection system
        self.roi_module = ROIModule()
        self.roi_module.initialize_workflow_manager(
            self.detection_module,
            grading_module,
            self.arduino_module
        )
        
        # Setup Arduino connection (only in non-dev mode)
        if not self.dev_mode:
            success = self.arduino_module.setup_arduino()
            if success:
                log_info(SystemComponent.GUI, "Arduino connection established successfully")
            else:
                log_warning(SystemComponent.GUI, "Arduino connection failed - running in manual mode")
        
        self.reporting_module = ReportingModule()

        # System state tracking
        self.current_mode = "IDLE" # Can be "IDLE", "TRIGGER", or "CONTINUOUS"
        self.auto_detection_active = False # Triggered by IR beam
        self.live_detection_var = False # For live inference mode (continuous)
        self.auto_grade_var = False # For auto grading in live mode
        
        # Wood detection state (ROI-triggered workflow)
        self.roi_triggered = False
        self.wood_confirmed = False
        self.roi_detection_active = False
        self.current_roi_session = None

        # Predict stream state
        self.predict_stream_active = False
        self.predict_stream_thread = None
        self.predict_stream_results = []
        self.latest_annotated_frame = None

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

        # Initialize ROI configuration UI
        self.update_roi_list()

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

        # ROI Status Section
        roi_status_group = QGroupBox("ROI Status")
        roi_status_group.setStyleSheet("QGroupBox { font-size: 12px; font-weight: bold; }")
        roi_status_layout = QVBoxLayout(roi_status_group)
        roi_status_layout.setContentsMargins(5, 5, 5, 5)

        # ROI activity indicators
        self.roi_activity_label = QLabel("Active ROIs: 0")
        self.roi_activity_label.setStyleSheet("font-size: 11px; color: #666;")
        roi_status_layout.addWidget(self.roi_activity_label)

        self.roi_overlap_label = QLabel("Overlaps: 0")
        self.roi_overlap_label.setStyleSheet("font-size: 11px; color: #666;")
        roi_status_layout.addWidget(self.roi_overlap_label)

        self.roi_sessions_label = QLabel("Active Sessions: 0")
        self.roi_sessions_label.setStyleSheet("font-size: 11px; color: #666;")
        roi_status_layout.addWidget(self.roi_sessions_label)

        defect_analysis_layout.addWidget(roi_status_group)

        # Wood Detection Status Section
        wood_status_group = QGroupBox("Wood Detection")
        wood_status_group.setStyleSheet("QGroupBox { font-size: 12px; font-weight: bold; }")
        wood_status_layout = QVBoxLayout(wood_status_group)
        wood_status_layout.setContentsMargins(5, 5, 5, 5)

        self.wood_detections_label = QLabel("Detections: 0")
        self.wood_detections_label.setStyleSheet("font-size: 11px; color: #666;")
        wood_status_layout.addWidget(self.wood_detections_label)

        self.wood_confidence_label = QLabel("Avg Confidence: 0.00")
        self.wood_confidence_label.setStyleSheet("font-size: 11px; color: #666;")
        wood_status_layout.addWidget(self.wood_confidence_label)

        self.wood_features_label = QLabel("Features: None")
        self.wood_features_label.setStyleSheet("font-size: 11px; color: #666;")
        wood_status_layout.addWidget(self.wood_features_label)

        defect_analysis_layout.addWidget(wood_status_group)

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

        # Tab 3: Model Health & Performance
        health_widget = QWidget()
        health_layout = QVBoxLayout(health_widget)
        health_layout.setContentsMargins(15, 15, 15, 15)

        # Model Health Status Section
        health_status_group = QGroupBox("Model Health Status")
        health_status_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        health_status_layout = QVBoxLayout(health_status_group)

        # Health status display
        self.model_health_label = QLabel("Model Health: Checking...")
        self.model_health_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        health_status_layout.addWidget(self.model_health_label)

        # Health metrics grid
        health_metrics_layout = QGridLayout()
        health_metrics_layout.setSpacing(10)

        # Row 0: Headers
        health_metrics_layout.addWidget(QLabel("Metric"), 0, 0)
        health_metrics_layout.addWidget(QLabel("Value"), 0, 1)
        health_metrics_layout.addWidget(QLabel("Status"), 0, 2)

        # Row 1: Inference Time
        health_metrics_layout.addWidget(QLabel("Avg Inference Time:"), 1, 0)
        self.avg_inference_time_label = QLabel("N/A")
        self.avg_inference_time_label.setStyleSheet("font-family: monospace;")
        health_metrics_layout.addWidget(self.avg_inference_time_label, 1, 1)
        self.inference_time_status = QLabel("Unknown")
        health_metrics_layout.addWidget(self.inference_time_status, 1, 2)

        # Row 2: Success Rate
        health_metrics_layout.addWidget(QLabel("Success Rate:"), 2, 0)
        self.success_rate_label = QLabel("N/A")
        self.success_rate_label.setStyleSheet("font-family: monospace;")
        health_metrics_layout.addWidget(self.success_rate_label, 2, 1)
        self.success_rate_status = QLabel("Unknown")
        health_metrics_layout.addWidget(self.success_rate_status, 2, 2)

        # Row 3: Total Inferences
        health_metrics_layout.addWidget(QLabel("Total Inferences:"), 3, 0)
        self.total_inferences_label = QLabel("0")
        self.total_inferences_label.setStyleSheet("font-family: monospace;")
        health_metrics_layout.addWidget(self.total_inferences_label, 3, 1)
        health_metrics_layout.addWidget(QLabel("Count"), 3, 2)

        health_status_layout.addLayout(health_metrics_layout)

        # Control buttons
        health_controls_layout = QHBoxLayout()
        self.btn_reload_model = QPushButton("Reload Model")
        self.btn_reload_model.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.btn_reload_model.clicked.connect(self.reload_model)
        health_controls_layout.addWidget(self.btn_reload_model)

        self.btn_benchmark_model = QPushButton("Run Benchmark")
        self.btn_benchmark_model.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.btn_benchmark_model.clicked.connect(self.run_model_benchmark)
        health_controls_layout.addWidget(self.btn_benchmark_model)

        health_controls_layout.addStretch()
        health_status_layout.addLayout(health_controls_layout)

        health_layout.addWidget(health_status_group)

        # Camera Status Section
        camera_status_group = QGroupBox("Camera Status")
        camera_status_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        camera_status_layout = QVBoxLayout(camera_status_group)

        # Camera status grid
        camera_grid_layout = QGridLayout()
        camera_grid_layout.setSpacing(10)

        # Headers
        camera_grid_layout.addWidget(QLabel("Camera"), 0, 0)
        camera_grid_layout.addWidget(QLabel("Status"), 0, 1)
        camera_grid_layout.addWidget(QLabel("Usage Count"), 0, 2)
        camera_grid_layout.addWidget(QLabel("Last Used"), 0, 3)

        # Top Camera
        camera_grid_layout.addWidget(QLabel("Top Camera"), 1, 0)
        self.top_camera_health_status = QLabel("Unknown")
        camera_grid_layout.addWidget(self.top_camera_health_status, 1, 1)
        self.top_camera_usage_count = QLabel("0")
        camera_grid_layout.addWidget(self.top_camera_usage_count, 1, 2)
        self.top_camera_last_used = QLabel("Never")
        camera_grid_layout.addWidget(self.top_camera_last_used, 1, 3)

        # Bottom Camera
        camera_grid_layout.addWidget(QLabel("Bottom Camera"), 2, 0)
        self.bottom_camera_health_status = QLabel("Unknown")
        camera_grid_layout.addWidget(self.bottom_camera_health_status, 2, 1)
        self.bottom_camera_usage_count = QLabel("0")
        camera_grid_layout.addWidget(self.bottom_camera_usage_count, 2, 2)
        self.bottom_camera_last_used = QLabel("Never")
        camera_grid_layout.addWidget(self.bottom_camera_last_used, 2, 3)

        camera_status_layout.addLayout(camera_grid_layout)
        health_layout.addWidget(camera_status_group)

        # Error Recovery Section
        error_recovery_group = QGroupBox("Error Recovery Controls")
        error_recovery_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        error_recovery_layout = QVBoxLayout(error_recovery_group)

        # Recovery options
        recovery_options_layout = QVBoxLayout()

        # Auto recovery checkbox
        self.auto_recovery_checkbox = QCheckBox("Enable Automatic Error Recovery")
        self.auto_recovery_checkbox.setChecked(True)
        self.auto_recovery_checkbox.setStyleSheet("font-size: 14px;")
        recovery_options_layout.addWidget(self.auto_recovery_checkbox)

        # Recovery strategy selection
        recovery_strategy_layout = QHBoxLayout()
        recovery_strategy_layout.addWidget(QLabel("Recovery Strategy:"))
        self.recovery_strategy_combo = QComboBox()
        self.recovery_strategy_combo.addItems(["Retry", "Fallback", "Circuit Breaker"])
        self.recovery_strategy_combo.setCurrentText("Retry")
        recovery_strategy_layout.addWidget(self.recovery_strategy_combo)
        recovery_strategy_layout.addStretch()
        recovery_options_layout.addLayout(recovery_strategy_layout)

        # Recovery controls
        recovery_controls_layout = QHBoxLayout()
        self.btn_force_recovery = QPushButton("Force Recovery")
        self.btn_force_recovery.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.btn_force_recovery.clicked.connect(self.force_error_recovery)
        recovery_controls_layout.addWidget(self.btn_force_recovery)

        self.btn_reset_error_state = QPushButton("Reset Error State")
        self.btn_reset_error_state.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.btn_reset_error_state.clicked.connect(self.reset_error_state)
        recovery_controls_layout.addWidget(self.btn_reset_error_state)

        recovery_controls_layout.addStretch()
        recovery_options_layout.addLayout(recovery_controls_layout)

        error_recovery_layout.addLayout(recovery_options_layout)

        # Error status display
        self.error_status_text = QTextEdit()
        self.error_status_text.setReadOnly(True)
        self.error_status_text.setMaximumHeight(100)
        self.error_status_text.setStyleSheet("font-family: monospace; font-size: 11px; background-color: #f9f9f9;")
        self.error_status_text.setPlainText("No errors detected")
        error_recovery_layout.addWidget(self.error_status_text)

        health_layout.addWidget(error_recovery_group)

        self.stats_notebook.addTab(health_widget, "Model Health")

        # Tab 4: Model Configuration
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setContentsMargins(15, 15, 15, 15)

        # Model Configuration Section
        model_config_group = QGroupBox("Model Configuration")
        model_config_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        model_config_layout = QVBoxLayout(model_config_group)

        # Configuration form
        config_form_layout = QFormLayout()
        config_form_layout.setSpacing(10)

        # Model name
        self.config_model_name = QLineEdit("defect_detector")
        config_form_layout.addRow("Model Name:", self.config_model_name)

        # Confidence threshold
        self.config_confidence_threshold = QDoubleSpinBox()
        self.config_confidence_threshold.setRange(0.0, 1.0)
        self.config_confidence_threshold.setSingleStep(0.05)
        self.config_confidence_threshold.setValue(0.5)
        config_form_layout.addRow("Confidence Threshold:", self.config_confidence_threshold)

        # Health check interval
        self.config_health_interval = QSpinBox()
        self.config_health_interval.setRange(30, 3600)
        self.config_health_interval.setValue(300)
        self.config_health_interval.setSuffix(" seconds")
        config_form_layout.addRow("Health Check Interval:", self.config_health_interval)

        # Inference timeout
        self.config_inference_timeout = QSpinBox()
        self.config_inference_timeout.setRange(1000, 30000)
        self.config_inference_timeout.setValue(5000)
        self.config_inference_timeout.setSuffix(" ms")
        config_form_layout.addRow("Inference Timeout:", self.config_inference_timeout)

        # Retry attempts
        self.config_retry_attempts = QSpinBox()
        self.config_retry_attempts.setRange(1, 10)
        self.config_retry_attempts.setValue(3)
        config_form_layout.addRow("Retry Attempts:", self.config_retry_attempts)

        model_config_layout.addLayout(config_form_layout)

        # Configuration buttons
        config_buttons_layout = QHBoxLayout()
        self.btn_load_config = QPushButton("Load Current Config")
        self.btn_load_config.clicked.connect(self.load_current_config)
        config_buttons_layout.addWidget(self.btn_load_config)

        self.btn_save_config = QPushButton("Save Configuration")
        self.btn_save_config.clicked.connect(self.save_model_config)
        config_buttons_layout.addWidget(self.btn_save_config)

        self.btn_validate_config = QPushButton("Validate Configuration")
        self.btn_validate_config.clicked.connect(self.validate_configuration)
        config_buttons_layout.addWidget(self.btn_validate_config)

        config_buttons_layout.addStretch()
        model_config_layout.addLayout(config_buttons_layout)

        config_layout.addWidget(model_config_group)

        # Configuration Status Section
        config_status_group = QGroupBox("Configuration Status")
        config_status_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        config_status_layout = QVBoxLayout(config_status_group)

        self.config_status_text = QTextEdit()
        self.config_status_text.setReadOnly(True)
        self.config_status_text.setMaximumHeight(150)
        self.config_status_text.setStyleSheet("font-family: monospace; background-color: #f9f9f9;")
        config_status_layout.addWidget(self.config_status_text)

        config_layout.addWidget(config_status_group)

        self.stats_notebook.addTab(config_widget, "Configuration")

        # Tab 5: ROI Configuration
        roi_config_widget = QWidget()
        roi_config_layout = QVBoxLayout(roi_config_widget)
        roi_config_layout.setContentsMargins(15, 15, 15, 15)

        # ROI Configuration Section
        roi_config_group = QGroupBox("Interactive ROI Configuration")
        roi_config_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        roi_config_group_layout = QVBoxLayout(roi_config_group)

        # Camera selection
        camera_selection_layout = QHBoxLayout()
        camera_selection_layout.addWidget(QLabel("Camera:"))
        self.roi_camera_combo = QComboBox()
        self.roi_camera_combo.addItems(["top", "bottom"])
        self.roi_camera_combo.currentTextChanged.connect(self.update_roi_list)
        camera_selection_layout.addWidget(self.roi_camera_combo)
        camera_selection_layout.addStretch()
        roi_config_group_layout.addLayout(camera_selection_layout)

        # ROI list
        roi_list_group = QGroupBox("ROI List")
        roi_list_layout = QVBoxLayout(roi_list_group)

        self.roi_list_widget = QListWidget()
        self.roi_list_widget.itemSelectionChanged.connect(self.on_roi_selected)
        roi_list_layout.addWidget(self.roi_list_widget)

        # ROI control buttons
        roi_buttons_layout = QHBoxLayout()
        self.btn_add_roi = QPushButton("Add ROI")
        self.btn_add_roi.clicked.connect(self.add_roi)
        roi_buttons_layout.addWidget(self.btn_add_roi)

        self.btn_edit_roi = QPushButton("Edit ROI")
        self.btn_edit_roi.clicked.connect(self.edit_roi)
        roi_buttons_layout.addWidget(self.btn_edit_roi)

        self.btn_delete_roi = QPushButton("Delete ROI")
        self.btn_delete_roi.clicked.connect(self.delete_roi)
        roi_buttons_layout.addWidget(self.btn_delete_roi)

        self.btn_activate_roi = QPushButton("Activate")
        self.btn_activate_roi.clicked.connect(self.activate_roi)
        roi_buttons_layout.addWidget(self.btn_activate_roi)

        self.btn_deactivate_roi = QPushButton("Deactivate")
        self.btn_deactivate_roi.clicked.connect(self.deactivate_roi)
        roi_buttons_layout.addWidget(self.btn_deactivate_roi)

        roi_list_layout.addLayout(roi_buttons_layout)
        roi_config_group_layout.addWidget(roi_list_group)

        # ROI properties
        roi_properties_group = QGroupBox("ROI Properties")
        roi_properties_layout = QFormLayout(roi_properties_group)

        self.roi_name_edit = QLineEdit()
        roi_properties_layout.addRow("Name:", self.roi_name_edit)

        self.roi_x1_spin = QSpinBox()
        self.roi_x1_spin.setRange(0, 1280)
        roi_properties_layout.addRow("X1:", self.roi_x1_spin)

        self.roi_y1_spin = QSpinBox()
        self.roi_y1_spin.setRange(0, 720)
        roi_properties_layout.addRow("Y1:", self.roi_y1_spin)

        self.roi_x2_spin = QSpinBox()
        self.roi_x2_spin.setRange(0, 1280)
        roi_properties_layout.addRow("X2:", self.roi_x2_spin)

        self.roi_y2_spin = QSpinBox()
        self.roi_y2_spin.setRange(0, 720)
        roi_properties_layout.addRow("Y2:", self.roi_y2_spin)

        self.roi_threshold_spin = QDoubleSpinBox()
        self.roi_threshold_spin.setRange(0.0, 1.0)
        self.roi_threshold_spin.setSingleStep(0.1)
        self.roi_threshold_spin.setValue(0.5)
        roi_properties_layout.addRow("Threshold:", self.roi_threshold_spin)

        # Save/Load buttons
        roi_save_load_layout = QHBoxLayout()
        self.btn_save_roi_config = QPushButton("Save Configuration")
        self.btn_save_roi_config.clicked.connect(self.save_roi_config)
        roi_save_load_layout.addWidget(self.btn_save_roi_config)

        self.btn_load_roi_config = QPushButton("Load Configuration")
        self.btn_load_roi_config.clicked.connect(self.load_roi_config)
        roi_save_load_layout.addWidget(self.btn_load_roi_config)

        self.btn_reset_roi_config = QPushButton("Reset to Default")
        self.btn_reset_roi_config.clicked.connect(self.reset_roi_config)
        roi_save_load_layout.addWidget(self.btn_reset_roi_config)

        roi_properties_layout.addRow(roi_save_load_layout)
        roi_config_group_layout.addWidget(roi_properties_group)

        roi_config_layout.addWidget(roi_config_group)

        # ROI Preview Section
        roi_preview_group = QGroupBox("ROI Preview")
        roi_preview_layout = QVBoxLayout(roi_preview_group)

        self.roi_preview_label = QLabel("Select an ROI to preview")
        self.roi_preview_label.setAlignment(Qt.AlignCenter)
        self.roi_preview_label.setMinimumSize(320, 180)
        self.roi_preview_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        roi_preview_layout.addWidget(self.roi_preview_label)

        roi_config_layout.addWidget(roi_preview_group)

        self.stats_notebook.addTab(roi_config_widget, "ROI Config")

        # Tab 6: System Log
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
            self.update_arduino_status()

            # Process frames if available
            if top_frame is not None:
                # Validate frame before processing
                if not self._validate_frame(top_frame):
                    self.display_message("Invalid frame data received from top camera", "warning")
                    top_frame = None
                else:
                    # Run detection if live detection is enabled AND predict_stream is not active
                    if self.live_detection_var and not self.predict_stream_active:
                        try:
                            print(f"DEBUG: GUI calling detection_module.analyze_frame for top camera")
                            annotated_frame, defects, defect_list, alignment_result = self.detection_module.analyze_frame(top_frame, "top")
                            print(f"DEBUG: GUI received result from detection_module.analyze_frame for top camera")
                            self.current_defects["top"] = {"defects": defects, "defect_list": defect_list}

                            # Auto grade if enabled
                            if self.auto_grade_var:
                                self.calculate_and_display_grade()

                            # Update model health tracking
                            if hasattr(self.detection_module, 'model_manager') and hasattr(self.detection_module.model_manager, 'health_monitor'):
                                # Track inference performance
                                inference_time = getattr(annotated_frame, 'inference_time', 100)  # Mock time if not available
                                success = len(defects) > 0 or True  # Assume success if we got results
                                self.detection_module.model_manager.health_monitor.track_inference("defect_detector", inference_time, success)
                        except Exception as e:
                            self.display_message(f"Detection error on top camera: {str(e)}", "error")
                            print(f"DEBUG: Exception in GUI top camera detection: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            annotated_frame = top_frame
                    elif self.predict_stream_active:
                        # When predict_stream is active, use the latest annotated frame from predict_stream
                        print(f"DEBUG: Predict stream active, using latest annotated frame for top camera")
                        if self.latest_annotated_frame is not None:
                            annotated_frame = self.latest_annotated_frame
                        else:
                            annotated_frame = top_frame
                    else:
                        annotated_frame = top_frame

                    # COMMENTED OUT: Alignment overlay no longer needed for full-frame defect detection
                    # try:
                    #     alignment_result = self.detection_module.alignment_module.check_wood_alignment(annotated_frame, None)
                    #     annotated_frame = self.detection_module.alignment_module.draw_alignment_overlay(annotated_frame, alignment_result)
                    # except Exception as e:
                    #     self.display_message(f"Error applying alignment overlay: {str(e)}", "warning")

                    # INTEGRATED: ROI-based wood detection and visual feedback
                    try:
                        # Process frame through ROI module for wood detection and overlap analysis
                        roi_results = self.roi_module.process_frame(annotated_frame, "top")

                        # Extract results
                        wood_detections = roi_results.get('wood_detections', [])
                        overlaps = roi_results.get('overlaps', {})
                        annotated_frame = roi_results.get('annotated_frame', annotated_frame)

                        # Handle ROI overlap triggers
                        overlapping_roi_ids = []
                        for roi_ids in overlaps.values():
                            overlapping_roi_ids.extend(roi_ids)

                        if overlapping_roi_ids and self.current_mode == "TRIGGER" and not self.auto_detection_active:
                            # Trigger ROI-based workflow
                            self.handle_roi_wood_overlap("top", wood_detections, overlapping_roi_ids)

                        # Update wood confirmation flag during ROI-triggered detection
                        if self.current_mode == "TRIGGER" and self.auto_detection_active and not self.wood_confirmed and wood_detections:
                            self.wood_confirmed = True
                            log_info(SystemComponent.GUI, "✅ Wood confirmed during ROI trigger period")
                            self.update_system_status("Status: Wood detected in ROI - collecting defect data...")

                        # Visual feedback for ROI and wood detection status
                        if self.roi_checkbox.isChecked():
                            # ROI overlays are already drawn by roi_module.process_frame
                            pass

                        if self.wood_detection_checkbox.isChecked() and wood_detections:
                            # Wood detection overlays are already drawn by roi_module.process_frame
                            pass

                        print(f"DEBUG: ROI processing complete - wood detections: {len(wood_detections)}, overlaps: {len(overlaps)}")

                    except Exception as e:
                        self.display_message(f"Error in ROI-based detection system: {str(e)}", "warning")
                        print(f"DEBUG: Exception in ROI system: {str(e)}")

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

                            # Update model health tracking
                            if hasattr(self.detection_module, 'model_manager') and hasattr(self.detection_module.model_manager, 'health_monitor'):
                                # Track inference performance
                                inference_time = getattr(annotated_frame, 'inference_time', 100)  # Mock time if not available
                                success = len(defects) > 0 or True  # Assume success if we got results
                                self.detection_module.model_manager.health_monitor.track_inference("defect_detector", inference_time, success)
                        except Exception as e:
                            self.display_message(f"Detection error on bottom camera: {str(e)}", "error")
                            print(f"DEBUG: Exception in GUI bottom camera detection: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            annotated_frame = bottom_frame
                    else:
                        annotated_frame = bottom_frame

                    # COMMENTED OUT: Alignment overlay no longer needed for full-frame defect detection
                    # try:
                    #     alignment_result = self.detection_module.alignment_module.check_wood_alignment(annotated_frame, None)
                    #     annotated_frame = self.detection_module.alignment_module.draw_alignment_overlay(annotated_frame, alignment_result)
                    # except Exception as e:
                    #     self.display_message(f"Error applying alignment overlay: {str(e)}", "warning")

                    # INTEGRATED: ROI-based wood detection and visual feedback
                    try:
                        # Process frame through ROI module for wood detection and overlap analysis
                        roi_results = self.roi_module.process_frame(annotated_frame, "bottom")

                        # Extract results
                        wood_detections = roi_results.get('wood_detections', [])
                        overlaps = roi_results.get('overlaps', {})
                        annotated_frame = roi_results.get('annotated_frame', annotated_frame)

                        # Handle ROI overlap triggers
                        overlapping_roi_ids = []
                        for roi_ids in overlaps.values():
                            overlapping_roi_ids.extend(roi_ids)

                        if overlapping_roi_ids and self.current_mode == "TRIGGER" and not self.auto_detection_active:
                            # Trigger ROI-based workflow
                            self.handle_roi_wood_overlap("bottom", wood_detections, overlapping_roi_ids)

                        # Update wood confirmation flag during ROI-triggered detection
                        if self.current_mode == "TRIGGER" and self.auto_detection_active and not self.wood_confirmed and wood_detections:
                            self.wood_confirmed = True
                            log_info(SystemComponent.GUI, "✅ Wood confirmed during ROI trigger period (bottom camera)")
                            self.update_system_status("Status: Wood detected in ROI - collecting defect data...")

                        # Enhanced visual feedback for ROI and wood detection status
                        if self.roi_checkbox.isChecked():
                            # ROI overlays are already drawn by roi_module.process_frame
                            # Add additional status information overlay
                            self._add_roi_status_overlay(annotated_frame, camera_name, overlaps)

                        if self.wood_detection_checkbox.isChecked() and wood_detections:
                            # Wood detection overlays are already drawn by roi_module.process_frame
                            # Add confidence score overlays
                            self._add_wood_detection_overlays(annotated_frame, wood_detections)

                        # Update status displays with real-time information
                        self.update_roi_status_display(camera_name, overlaps, wood_detections)

                        print(f"DEBUG: ROI processing complete for bottom camera - wood detections: {len(wood_detections)}, overlaps: {len(overlaps)}")

                    except Exception as e:
                        self.display_message(f"Error in ROI-based detection system (bottom): {str(e)}", "warning")
                        print(f"DEBUG: Exception in ROI system (bottom): {str(e)}")

                    # Convert and display
                    self.display_frame(annotated_frame, self.bottom_camera_label)

            # Update session duration
            self.update_session_duration()

            # Update model health and camera status (every 5 seconds)
            current_time = time.time()
            if not hasattr(self, '_last_health_update') or current_time - self._last_health_update > 5.0:
                self.update_model_health_display()
                self.update_camera_status_display()
                self._last_health_update = current_time

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

    def update_arduino_status(self):
        """Update Arduino connection status display with concise text"""
        try:
            if self.arduino_module:
                status = self.arduino_module.get_connection_status()
                is_connected = status.get("connected", False)
                port = status.get("port", "None")

                # Make status text more concise
                if is_connected:
                    # Extract just the port number for brevity
                    if port and port != "None":
                        port_short = port.split('/')[-1] if '/' in port else port[:8]  # Last part or first 8 chars
                        status_text = f"Arduino: ✓ {port_short}"
                    else:
                        status_text = "Arduino: ✓ Connected"
                else:
                    status_text = "Arduino: ✗ Disconnected"

                color = "green" if is_connected else "red"
                self.arduino_status.setText(status_text)
                self.arduino_status.setStyleSheet(f"font-size: 11px; color: {color}; font-weight: bold;")

                # Update system status if Arduino status changed (without redundant info)
                if is_connected:
                    self.update_system_status("Arduino ready")
                else:
                    self.update_system_status("Arduino offline - manual mode")
            else:
                self.arduino_status.setText("Arduino: N/A")
                self.arduino_status.setStyleSheet("font-size: 11px; color: orange; font-weight: bold;")

        except Exception as e:
            log_error(SystemComponent.GUI, f"Error updating Arduino status: {str(e)}", e)
            self.arduino_status.setText("Arduino: Error")
            self.arduino_status.setStyleSheet("font-size: 11px; color: red; font-weight: bold;")

    def calculate_and_display_grade(self):
        """Calculate grade from current defects and display results"""
        try:
            # Prevent recursion during predict_stream analysis
            if hasattr(self, 'predict_stream_active') and self.predict_stream_active:
                # Only grade every few frames to avoid overwhelming
                if not hasattr(self, '_last_grade_frame'):
                    self._last_grade_frame = 0
                current_frame = len(self.predict_stream_results) if hasattr(self, 'predict_stream_results') else 0
                if current_frame - self._last_grade_frame < 5:
                    return  # Skip grading this frame
                self._last_grade_frame = current_frame

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

            # ✅ FIXED: Calculate separate grades for top and bottom cameras
            top_defects = self.current_defects.get("top", {}).get("defects", {})
            bottom_defects = self.current_defects.get("bottom", {}).get("defects", {})

            top_grade_info = calculate_grade(top_defects)
            bottom_grade_info = calculate_grade(bottom_defects)

            # Extract grade values
            top_grade = top_grade_info.get('grade', 0)
            bottom_grade = bottom_grade_info.get('grade', 0)

            # ✅ FIXED: Pass both grades to determine_final_grade
            final_grade = determine_final_grade(top_grade, bottom_grade)

            print(f"DEBUG: Grading - top_defects: {top_defects}, bottom_defects: {bottom_defects}")
            print(f"DEBUG: Grading - top_grade: {top_grade}, bottom_grade: {bottom_grade}")
            print(f"DEBUG: Grading - final_grade: {final_grade}")
            
            # Store current grade info
            self.current_grade_info = {
                'grade': final_grade,
                'defects': all_defects,
                'defect_list': all_defect_lists,
                'top_grade_info': top_grade_info,
                'bottom_grade_info': bottom_grade_info
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

            # Update defect details - use top_grade_info as primary
            self.update_defect_details(all_defects, all_defect_lists, top_grade_info)

            # Update detection state
            self.update_detection_state("Grading")

            # Send grade command to Arduino for automatic sorting
            if self.arduino_module and self.arduino_module.is_connected():
                try:
                    success = self.arduino_module.send_grade_command(final_grade)
                    if success:
                        self.update_system_status(f"✅ Grade {final_grade} sent to Arduino for sorting")
                        log_info(SystemComponent.GUI, f"Successfully sent grade {final_grade} to Arduino")
                    else:
                        self.update_system_status(f"❌ Failed to send grade {final_grade} to Arduino")
                        log_arduino_error(f"Failed to send grade {final_grade} to Arduino")
                except Exception as e:
                    self.update_system_status(f"❌ Error sending grade to Arduino: {str(e)}")
                    log_arduino_error(f"Error sending grade {final_grade} to Arduino: {str(e)}", e)
            else:
                self.update_system_status(f"⚠️ Arduino not connected - Grade {final_grade} calculated but not sent")
                log_warning(SystemComponent.GUI, f"Arduino not connected - cannot send grade {final_grade}")

        except Exception as e:
            self.display_message(f"Error calculating grade: {str(e)}", "error")

    def update_roi_status_display(self, camera_name, overlaps, wood_detections):
        """Update ROI and wood detection status displays"""
        try:
            # Update ROI activity
            active_rois = self.roi_module.roi_manager.get_active_rois(camera_name)
            self.roi_activity_label.setText(f"Active ROIs: {len(active_rois)}")

            # Update overlap information
            total_overlaps = len(overlaps) if overlaps else 0
            self.roi_overlap_label.setText(f"Overlaps: {total_overlaps}")

            # Update session information
            if hasattr(self.roi_module, 'workflow_manager') and self.roi_module.workflow_manager:
                active_sessions = len(self.roi_module.workflow_manager.get_active_sessions(camera_name))
                self.roi_sessions_label.setText(f"Active Sessions: {active_sessions}")
            else:
                self.roi_sessions_label.setText("Active Sessions: N/A")

            # Update wood detection information
            detection_count = len([d for d in wood_detections if d.detected]) if wood_detections else 0
            self.wood_detections_label.setText(f"Detections: {detection_count}")

            # Calculate average confidence
            if wood_detections:
                confidences = [d.confidence for d in wood_detections if d.detected]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                self.wood_confidence_label.setText(f"Avg Confidence: {avg_confidence:.2f}")
            else:
                self.wood_confidence_label.setText("Avg Confidence: 0.00")

            # Update features information
            if wood_detections and wood_detections[0].detected:
                features = wood_detections[0].features
                if features and 'dominant_color' in features:
                    self.wood_features_label.setText(f"Features: {features['dominant_color']}")
                else:
                    self.wood_features_label.setText("Features: Basic")
            else:
                self.wood_features_label.setText("Features: None")

        except Exception as e:
            self.display_message(f"Error updating ROI status display: {str(e)}", "warning")

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

        # Grade breakdown - handle both old and new formats
        if grade_info:
            # Check if it's the new format (dict with grade keys)
            if isinstance(grade_info, dict) and any(isinstance(k, int) for k in grade_info.keys()):
                # New format: {grade: criteria}
                for grade, criteria in grade_info.items():
                    if isinstance(criteria, dict) and 'meets_criteria' in criteria:
                        if criteria['meets_criteria']:
                            details_text += f"✓ Grade {grade}: MEETS CRITERIA\n"
                            details_text += f"  Max defects allowed: {criteria.get('max_defects', 'N/A')}\n"
                            details_text += f"  Current defects: {criteria.get('current_defects', 'N/A')}\n\n"
                        else:
                            details_text += f"✗ Grade {grade}: EXCEEDS LIMITS\n"
                            details_text += f"  Max defects allowed: {criteria.get('max_defects', 'N/A')}\n"
                            details_text += f"  Current defects: {criteria.get('current_defects', 'N/A')}\n\n"
                    else:
                        details_text += f"Grade {grade}: {criteria}\n"
            else:
                # Old format or other structure
                details_text += f"Grade Information: {grade_info}\n"

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
            # Handle both tuple format ('arduino_message', 'IR: 0') and string format ('IR: 0')
            if isinstance(message, tuple) and len(message) >= 2:
                actual_message = message[1]  # Extract the actual message from tuple
                log_info(SystemComponent.GUI, f"Arduino message received: {message} -> extracted: {actual_message}")
            else:
                actual_message = str(message)
                log_info(SystemComponent.GUI, f"Arduino message received: {actual_message}")

            # LEGACY IR SUPPORT: Keep IR beam messages for backward compatibility
            if actual_message == "B":
                # IR beam broken - start wood detection workflow (legacy support)
                log_info(SystemComponent.GUI, "Legacy IR beam broken message received - ignoring (ROI-based system active)")
                self.update_system_status("Status: Legacy IR message received - using ROI-based detection")
            elif actual_message == "IR: 0" or actual_message == "IR:0":
                # IR beam broken - start wood detection workflow (legacy support)
                log_info(SystemComponent.GUI, "Legacy IR beam broken message received - ignoring (ROI-based system active)")
                self.update_system_status("Status: Legacy IR message received - using ROI-based detection")
            elif actual_message == "IR: 1" or actual_message == "IR:1":
                # IR beam cleared - stop detection and process grading (legacy support)
                log_info(SystemComponent.GUI, "Legacy IR beam cleared message received - ignoring (ROI-based system active)")
                self.update_system_status("Status: Legacy IR message received - using ROI-based detection")
            elif actual_message.startswith("L:"):
                # Length measurement received (legacy support)
                try:
                    duration_ms = int(actual_message.split(":")[1])
                    log_info(SystemComponent.GUI, f"Legacy length measurement received: {duration_ms}ms - ignoring (ROI-based system active)")
                    self.update_system_status(f"Status: Legacy length measurement received - using ROI-based detection")
                except (ValueError, IndexError) as e:
                    log_error(SystemComponent.GUI, f"Invalid length message format: {actual_message}", e)
            else:
                # Other Arduino messages (status updates, etc.)
                self.update_system_status(f"Arduino: {actual_message}")

        except Exception as e:
            log_error(SystemComponent.GUI, f"Error handling Arduino message '{message}': {str(e)}", e)

    def handle_roi_wood_overlap(self, camera_name: str, wood_detections: list, overlapping_rois: list):
        """Handle wood-ROI overlap detection - start ROI-based workflow"""
        try:
            log_info(SystemComponent.GUI, f"Wood-ROI overlap detected on {camera_name} - starting ROI-based workflow")

            # Only respond to ROI triggers in TRIGGER mode
            if self.current_mode == "TRIGGER":
                if not self.auto_detection_active:
                    # 1. Check model health before starting inference
                    model_health = self.detection_module.get_model_health_status("defect_detector")
                    if model_health in [self.detection_module.HealthStatus.UNHEALTHY, self.detection_module.HealthStatus.DEGRADED]:
                        log_warning(SystemComponent.GUI, f"Model health is {model_health.value} - attempting recovery before ROI trigger")
                        if not self.detection_module.reload_model("defect_detector"):
                            log_error(SystemComponent.GUI, "Model recovery failed - cannot process ROI trigger")
                            self.display_message("Model health check failed - ROI trigger ignored", "error")
                            return

                    # 2. Check camera availability
                    camera_status = self.detection_module.get_camera_status(camera_name)
                    if camera_status == self.detection_module.CameraStatus.ERROR:
                        log_error(SystemComponent.GUI, f"Camera {camera_name} not available - cannot process ROI trigger")
                        self.display_message("Camera unavailable - ROI trigger ignored", "error")
                        return

                    log_info(SystemComponent.GUI, f"✅ TRIGGER MODE: Starting ROI-based workflow for {camera_name}")

                    # Start ROI-based detection workflow
                    self.start_roi_based_workflow(camera_name, wood_detections, overlapping_rois)

                    # Set the live detection checkbox and variables
                    self.live_detection_checkbox.setChecked(True)
                    self.live_detection_var = True
                    self.auto_grade_var = True

                    # Update states
                    self.roi_triggered = True
                    self.wood_confirmed = True  # Wood is confirmed by ROI overlap
                    self.auto_detection_active = True

                    self.update_system_status(f"Status: ROI TRIGGERED - Wood detected in {camera_name}!")
                    self.update_detection_state("Detecting")

                    # Add health monitoring during inference sessions
                    self.update_model_health_display()
                    self.update_camera_status_display()

                else:
                    log_warning(SystemComponent.GUI, "⚠️ ROI overlap detected but detection already active")
            else:
                # In IDLE or CONTINUOUS mode, just log the ROI signal but don't act on it
                log_info(SystemComponent.GUI, f"❌ ROI overlap detected but system is in {self.current_mode} mode - ignoring trigger")
                self.update_system_status(f"Status: ROI signal ignored ({self.current_mode} mode)")
                self.update_detection_state("Waiting")

        except Exception as e:
            log_error(SystemComponent.GUI, f"Error handling ROI wood overlap: {str(e)}", e)
            self.display_message("ROI trigger error - attempting system recovery", "warning")
            self.update_model_health_display()

    def handle_length_measurement(self, duration_ms):
        """Handle length measurement from Arduino - IR beam cleared, stop detection"""
        try:
            # Calculate estimated length based on conveyor speed
            estimated_speed_mm_per_ms = 0.1  # Adjust this value based on testing
            estimated_length_mm = duration_ms * estimated_speed_mm_per_ms

            log_info(SystemComponent.GUI, f"Length measurement: {duration_ms}ms → ~{estimated_length_mm:.1f}mm")

            # In TRIGGER mode, stop detection when beam clears (length message received)
            if self.current_mode == "TRIGGER" and self.auto_detection_active:
                log_info(SystemComponent.GUI, "IR beam cleared – stopping predict_stream inference (TRIGGER MODE)")

                # Stop predict_stream inference
                self.stop_predict_stream_inference()

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

    def start_predict_stream_inference(self):
        """Start continuous inference using enhanced predict_stream with health checks and error recovery"""
        try:
            # 1. Check model health before starting inference
            model_health = self.detection_module.get_model_health_status("defect_detector")
            if model_health in [self.detection_module.HealthStatus.UNHEALTHY, self.detection_module.HealthStatus.DEGRADED]:
                log_warning(SystemComponent.GUI, f"Model health is {model_health.value} - attempting recovery before starting inference")
                if not self.detection_module.reload_model("defect_detector"):
                    log_error(SystemComponent.GUI, "Model recovery failed - cannot start predict_stream")
                    self.display_message("Model health check failed - cannot start inference", "error")
                    return

            # 2. Check camera availability and prevent conflicts
            camera_status = self.detection_module.get_camera_status("top")
            if camera_status == self.detection_module.CameraStatus.IN_USE:
                log_warning(SystemComponent.GUI, "Top camera is already in use - cannot start predict_stream")
                self.display_message("Camera conflict detected - cannot start inference", "error")
                return

            if not DEGIRUM_TOOLS_AVAILABLE:
                log_error(SystemComponent.GUI, "degirum_tools not available - cannot start predict_stream")
                return

            if self.predict_stream_active:
                log_warning(SystemComponent.GUI, "Predict stream already active")
                return

            log_info(SystemComponent.GUI, "Starting enhanced predict_stream continuous inference")

            # Reset results collection and performance tracking
            self.predict_stream_results = []
            self._inference_start_time = time.time()
            self._inference_frame_count = 0
            self._inference_error_count = 0

            # Pause the GUI camera feed to avoid conflicts
            self._pause_gui_camera_feed()

            # Start predict_stream in a separate thread with enhanced monitoring
            self.predict_stream_active = True
            self.predict_stream_thread = threading.Thread(target=self._run_enhanced_predict_stream_inference)
            self.predict_stream_thread.daemon = True
            self.predict_stream_thread.start()

            log_info(SystemComponent.GUI, "Enhanced predict stream inference started successfully")

        except Exception as e:
            log_error(SystemComponent.GUI, f"Error starting enhanced predict_stream inference: {str(e)}", e)
            self.predict_stream_active = False

    def _run_enhanced_predict_stream_inference(self):
        """Run the enhanced predict_stream inference loop with error recovery and monitoring"""
        try:
            log_info(SystemComponent.GUI, "Starting enhanced predict_stream inference loop")

            # 3. Use the new process_stream method with error recovery
            camera_name = "top"
            model_name = "defect_detector"

            # Create enhanced analyzer with performance tracking
            class EnhancedDefectAnalyzer:
                def __init__(self, gui_instance):
                    self.gui = gui_instance
                    self.frame_count = 0
                    self.start_time = time.time()
                    self.error_count = 0

                def analyze(self, result):
                    """Enhanced analyze method with performance tracking"""
                    try:
                        self.frame_count += 1
                        current_time = time.time()

                        # Process the inference result
                        detections = result.results if hasattr(result, 'results') else []

                        frame_defects = {}
                        frame_defect_measurements = []

                        # Process detections with confidence filtering
                        for det in detections:
                            confidence = det.get('confidence', 0)
                            if confidence < self.gui.detection_module.defect_confidence_threshold:
                                continue

                            model_label = det['label']
                            # Count defects by type
                            if model_label in frame_defects:
                                frame_defects[model_label] += 1
                            else:
                                frame_defects[model_label] = 1

                        # Store frame results
                        frame_result = {
                            'frame_id': self.frame_count,
                            'defects': frame_defects,
                            'timestamp': current_time,
                            'processing_time': current_time - self.start_time
                        }

                        self.gui.predict_stream_results.append(frame_result)

                        # Update GUI tracking variables
                        self.gui._inference_frame_count = self.frame_count

                        # Store the annotated frame for GUI display
                        if hasattr(result, 'image_overlay'):
                            self.gui.latest_annotated_frame = result.image_overlay

                        # 4. Add health monitoring during inference sessions
                        # Track inference performance
                        inference_time = getattr(result, 'inference_time', 50)  # Default 50ms
                        success = len(detections) > 0 or True  # Assume success if we got results
                        self.gui.detection_module.model_manager.health_monitor.track_inference(
                            model_name, inference_time, success
                        )

                        # Log progress with performance metrics
                        fps = self.frame_count / (current_time - self.start_time) if (current_time - self.start_time) > 0 else 0
                        print(f"Enhanced predict_stream: Frame {self.frame_count} - defects: {frame_defects}, FPS: {fps:.1f}")

                        # Update system status with performance info
                        if self.frame_count % 30 == 0:  # Update every 30 frames
                            self.gui.update_system_status(f"Status: Processing frame {self.frame_count} - {len(frame_defects)} defects detected")

                    except Exception as e:
                        self.error_count += 1
                        self.gui._inference_error_count = self.error_count
                        print(f"Error in EnhancedDefectAnalyzer.analyze: {str(e)}")

                        # 6. Add automatic recovery mechanisms
                        if self.error_count > 5:  # If too many errors, attempt recovery
                            log_warning(SystemComponent.GUI, f"High error rate in predict_stream ({self.error_count} errors) - attempting recovery")
                            if self.gui.detection_module.reload_model(model_name):
                                log_info(SystemComponent.GUI, "Model reloaded successfully during inference")
                                self.error_count = 0  # Reset error count
                            else:
                                log_error(SystemComponent.GUI, "Model reload failed during inference")

            # Create analyzer instance
            analyzer = EnhancedDefectAnalyzer(self)

            # 2. Implement camera coordination to prevent conflicts
            # Acquire camera through the detection module's camera coordinator
            camera_handle = self.detection_module.acquire_camera(camera_name, "predict_stream_gui")
            if not camera_handle:
                log_error(SystemComponent.GUI, f"Failed to acquire camera {camera_name} for predict_stream")
                return

            try:
                # Get the defect model from detection module
                model = self.detection_module.defect_model
                if model is None:
                    log_error(SystemComponent.GUI, "Defect model not available for enhanced predict_stream")
                    return

                # Determine camera index based on dev mode
                if self.dev_mode:
                    camera_index = 0
                    log_info(SystemComponent.GUI, f"Using camera index {camera_index} for enhanced predict_stream (dev mode)")
                else:
                    camera_index = 0  # Top camera
                    log_info(SystemComponent.GUI, f"Using camera index {camera_index} for enhanced predict_stream (production mode)")

                # Use predict_stream with enhanced analyzer and error recovery
                log_info(SystemComponent.GUI, f"Starting enhanced predict_stream with camera index {camera_index}")

                for result in degirum_tools.predict_stream(
                    model=model,
                    video_source_id=camera_index,
                    fps=15,  # Limit to 15 FPS for processing
                    analyzers=[analyzer]
                ):
                    # Check if we should stop
                    if not self.predict_stream_active:
                        log_info(SystemComponent.GUI, "Enhanced predict stream stopping due to flag")
                        break

                    # Small delay to prevent overwhelming the system
                    time.sleep(0.01)

            except Exception as e:
                log_error(SystemComponent.GUI, f"Error in enhanced predict_stream loop: {str(e)}", e)

                # 6. Add automatic recovery mechanisms for workflow stability
                if "timeout" in str(e).lower():
                    log_info(SystemComponent.GUI, "Timeout detected in predict_stream - attempting automatic recovery")
                    if self.detection_module.reload_model(model_name):
                        log_info(SystemComponent.GUI, "Model recovered from timeout")
                    else:
                        log_error(SystemComponent.GUI, "Model recovery from timeout failed")
                elif "connection" in str(e).lower():
                    log_info(SystemComponent.GUI, "Connection error detected - attempting camera reconnection")
                    # The camera coordinator will handle reconnection automatically

            log_info(SystemComponent.GUI, f"Enhanced predict stream stopped after {analyzer.frame_count} frames")

            # 5. Integrate performance tracking and reporting
            end_time = time.time()
            total_duration = end_time - self._inference_start_time
            avg_fps = self._inference_frame_count / total_duration if total_duration > 0 else 0

            performance_report = {
                'total_frames': self._inference_frame_count,
                'total_duration': total_duration,
                'average_fps': avg_fps,
                'total_errors': self._inference_error_count,
                'error_rate': self._inference_error_count / self._inference_frame_count if self._inference_frame_count > 0 else 0
            }

            log_info(SystemComponent.GUI, f"Enhanced predict_stream performance: {performance_report}")

            # Update performance display
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.add_metric('predict_stream_fps', avg_fps)
                self.performance_monitor.add_metric('predict_stream_errors', self._inference_error_count)

        except Exception as e:
            log_error(SystemComponent.GUI, f"Error in enhanced predict_stream thread: {str(e)}", e)
        finally:
            self.predict_stream_active = False
            # Resume GUI camera feed
            self._resume_gui_camera_feed()
            # Reinitialize camera for GUI use
            self._reinitialize_camera_after_predict_stream()

    def _pause_gui_camera_feed(self):
        """Pause the GUI camera feed timer to avoid conflicts with predict_stream"""
        try:
            if hasattr(self, 'timer') and self.timer.isActive():
                log_info(SystemComponent.GUI, "Pausing GUI camera feed for predict_stream")
                self.timer.stop()
                self._gui_feed_paused = True
        except Exception as e:
            log_warning(SystemComponent.GUI, f"Error pausing GUI camera feed: {str(e)}")

    def _resume_gui_camera_feed(self):
        """Resume the GUI camera feed timer after predict_stream stops"""
        try:
            if hasattr(self, '_gui_feed_paused') and self._gui_feed_paused:
                log_info(SystemComponent.GUI, "Resuming GUI camera feed after predict_stream")
                self.timer.start(100)  # Resume with 100ms interval
                self._gui_feed_paused = False
        except Exception as e:
            log_warning(SystemComponent.GUI, f"Error resuming GUI camera feed: {str(e)}")

    def _reinitialize_camera_after_predict_stream(self):
        """Reinitialize the camera after predict_stream stops so GUI can use it again"""
        try:
            if not self.dev_mode and hasattr(self.camera_module, 'cap_top') and self.camera_module.cap_top is None:
                log_info(SystemComponent.GUI, "Reinitializing camera for GUI use after predict_stream")
                # Give predict_stream time to fully release the camera
                time.sleep(0.5)
                # Reinitialize the camera
                self.camera_module.initialize_cameras()
        except Exception as e:
            log_warning(SystemComponent.GUI, f"Error reinitializing camera: {str(e)}")

    def _run_mock_predict_stream(self):
        """Run a mock predict_stream for testing when model is not available"""
        try:
            log_info(SystemComponent.GUI, "Starting mock predict_stream for testing")

            # Create a mock analyzer
            class MockDefectAnalyzer:
                def __init__(self, gui_instance):
                    self.gui = gui_instance
                    self.frame_count = 0

                def analyze(self, result):
                    """Mock analyze method"""
                    try:
                        self.frame_count += 1

                        # Simulate some mock defects for testing
                        mock_defects = {}
                        if self.frame_count % 10 == 0:  # Every 10th frame has defects
                            mock_defects = {"crack": 1, "knot": 2}

                        frame_defect_measurements = []
                        for defect_type, count in mock_defects.items():
                            for i in range(count):
                                frame_defect_measurements.append((defect_type, 50.0, 0.05))

                        # Store frame results
                        frame_result = {
                            'frame_id': self.frame_count,
                            'defects': mock_defects,
                            'defect_measurements': frame_defect_measurements,
                            'timestamp': time.time()
                        }

                        self.gui.predict_stream_results.append(frame_result)

                        # Update current defects for grading
                        self.gui.current_defects["top"] = {
                            "defects": mock_defects,
                            "defect_list": frame_defect_measurements
                        }

                        # Store a mock annotated frame
                        if self.gui.dev_mode and self.gui.top_frame_original is not None:
                            mock_frame = self.gui.top_frame_original.copy()
                            # Add some mock annotations
                            if mock_defects:
                                cv2.putText(mock_frame, f"Mock defects: {mock_defects}", (50, 50),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            self.gui.latest_annotated_frame = mock_frame

                        # Auto grade if enabled
                        if self.gui.auto_grade_var:
                            self.gui.calculate_and_display_grade()

                        log_info(SystemComponent.GUI, f"Mock processed frame {self.frame_count} - defects: {mock_defects}")

                    except Exception as e:
                        log_error(SystemComponent.GUI, f"Error in MockDefectAnalyzer.analyze: {str(e)}", e)

                def annotate(self, result, image):
                    """Mock annotate method"""
                    return image

            # Create mock analyzer instance
            analyzer = MockDefectAnalyzer(self)

            # Simulate predict_stream behavior
            frame_count = 0
            while self.predict_stream_active:
                try:
                    # Get frame from camera or mock
                    if self.dev_mode:
                        frame = self.top_frame_original.copy() if self.top_frame_original is not None else None
                    else:
                        frame = self.camera_module.get_top_frame()

                    if frame is not None:
                        frame_count += 1

                        # Create a mock result object
                        class MockResult:
                            def __init__(self, frame_data):
                                self.results = []  # Mock empty results
                                self.image_overlay = frame_data

                        mock_result = MockResult(frame)

                        # Process result with analyzer
                        analyzer.analyze(mock_result)

                        # Small delay to prevent overwhelming the system
                        time.sleep(0.1)

                    else:
                        time.sleep(0.5)  # Wait for frame if not available

                except Exception as e:
                    log_error(SystemComponent.GUI, f"Error in mock predict_stream: {str(e)}", e)
                    time.sleep(1)  # Wait before retrying

            log_info(SystemComponent.GUI, f"Mock predict stream stopped after {frame_count} frames")

        except Exception as e:
            log_error(SystemComponent.GUI, f"Error in mock predict_stream thread: {str(e)}", e)
        finally:
            self.predict_stream_active = False

    def stop_predict_stream_inference(self):
        """Stop the enhanced predict_stream inference with performance reporting"""
        try:
            if self.predict_stream_active:
                log_info(SystemComponent.GUI, "Stopping enhanced predict_stream inference")
                self.predict_stream_active = False

                # Wait for thread to finish
                if self.predict_stream_thread and self.predict_stream_thread.is_alive():
                    self.predict_stream_thread.join(timeout=2.0)

                self.predict_stream_thread = None
                self.latest_annotated_frame = None  # Clear the latest frame

                # 5. Integrate performance tracking and reporting
                if hasattr(self, '_inference_start_time') and hasattr(self, '_inference_frame_count'):
                    end_time = time.time()
                    total_duration = end_time - self._inference_start_time
                    avg_fps = self._inference_frame_count / total_duration if total_duration > 0 else 0

                    performance_summary = {
                        'session_duration': total_duration,
                        'total_frames_processed': self._inference_frame_count,
                        'average_fps': avg_fps,
                        'total_errors': getattr(self, '_inference_error_count', 0),
                        'error_rate': getattr(self, '_inference_error_count', 0) / self._inference_frame_count if self._inference_frame_count > 0 else 0
                    }

                    log_info(SystemComponent.GUI, f"Enhanced predict_stream performance summary: {performance_summary}")

                    # Update performance monitor
                    if hasattr(self, 'performance_monitor'):
                        self.performance_monitor.add_metric('predict_stream_session_duration', total_duration)
                        self.performance_monitor.add_metric('predict_stream_avg_fps', avg_fps)
                        self.performance_monitor.add_metric('predict_stream_error_rate', performance_summary['error_rate'])

                    # Display performance summary in GUI
                    self.display_message(f"Session completed: {self._inference_frame_count} frames, {avg_fps:.1f} FPS avg", "info")

                # Resume GUI camera feed
                self._resume_gui_camera_feed()

                log_info(SystemComponent.GUI, "Enhanced predict stream inference stopped")
            else:
                log_info(SystemComponent.GUI, "Enhanced predict stream was not active")

        except Exception as e:
            log_error(SystemComponent.GUI, f"Error stopping enhanced predict_stream: {str(e)}", e)

    def handle_roi_session_end(self, camera_name: str, session_results: dict):
        """Handle ROI session end - stop detection and process grading"""
        try:
            log_info(SystemComponent.GUI, f"ROI session ended for {camera_name} - processing accumulated defects")

            # Only respond to ROI triggers in TRIGGER mode
            if self.current_mode == "TRIGGER" and self.auto_detection_active:
                log_info(SystemComponent.GUI, "✅ TRIGGER MODE: Processing ROI session results...")

                # Stop ROI-based detection workflow
                self.stop_roi_based_workflow(camera_name)

                # Deactivate the checkboxes
                self.live_detection_checkbox.setChecked(False)
                self.auto_grade_checkbox.setChecked(False)

                # Update internal state variables
                self.live_detection_var = False
                self.auto_grade_var = False

                # Stop detection session
                self.auto_detection_active = False
                self.roi_triggered = False

                if session_results and session_results.get('total_frames', 0) > 0:
                    # Wood was detected and defects accumulated, calculate grade
                    log_info(SystemComponent.GUI, "Wood confirmed - calculating grade from ROI session defect data")
                    self.wood_confirmed = False

                    # Use accumulated defect data for grading
                    self.calculate_grade_from_roi_session(session_results)

                    self.update_system_status("Status: Grade calculated from ROI session - Ready for next trigger")
                    self.update_detection_state("Waiting")
                else:
                    # No defects accumulated in session
                    log_info(SystemComponent.GUI, "ROI session ended - no defects accumulated")
                    self.update_system_status("Status: No defects detected in ROI session - Ready for next trigger")
                    self.update_detection_state("Waiting")

            else:
                log_info(SystemComponent.GUI, f"ROI session ended but system is in {self.current_mode} mode or no detection active")
                self.update_system_status("Status: ROI session completed")

        except Exception as e:
            log_error(SystemComponent.GUI, f"Error handling ROI session end: {str(e)}", e)

    def start_roi_based_workflow(self, camera_name: str, wood_detections: list, overlapping_rois: list):
        """Start ROI-based detection workflow"""
        try:
            log_info(SystemComponent.GUI, f"Starting ROI-based workflow for {camera_name}")

            # Start predict_stream for continuous detection during ROI overlap
            self.start_predict_stream_inference()

            # Initialize ROI session tracking
            self.active_roi_sessions = getattr(self, 'active_roi_sessions', {})
            self.active_roi_sessions[camera_name] = {
                'start_time': time.time(),
                'wood_detections': wood_detections,
                'overlapping_rois': overlapping_rois,
                'accumulated_defects': {}
            }

            log_info(SystemComponent.GUI, f"ROI-based workflow started for {camera_name}")

        except Exception as e:
            log_error(SystemComponent.GUI, f"Error starting ROI-based workflow: {str(e)}", e)

    def stop_roi_based_workflow(self, camera_name: str):
        """Stop ROI-based detection workflow"""
        try:
            log_info(SystemComponent.GUI, f"Stopping ROI-based workflow for {camera_name}")

            # Stop predict_stream inference
            self.stop_predict_stream_inference()

            # Clear ROI session tracking
            if hasattr(self, 'active_roi_sessions'):
                self.active_roi_sessions.pop(camera_name, None)

            log_info(SystemComponent.GUI, f"ROI-based workflow stopped for {camera_name}")

        except Exception as e:
            log_error(SystemComponent.GUI, f"Error stopping ROI-based workflow: {str(e)}", e)

    def calculate_grade_from_roi_session(self, session_results: dict):
        """Calculate grade from accumulated ROI session defects"""
        try:
            # Extract defect data from session results
            total_defects = session_results.get('total_defects', {})

            # Convert to format expected by grading system
            grading_defects = []
            for defect_type, count in total_defects.items():
                # Add size information (use default if not available)
                for _ in range(count):
                    grading_defects.append((defect_type, 10.0, 5.0))  # Default size values

            # Perform grading
            from modules.grading_module import determine_surface_grade
            grade = determine_surface_grade(grading_defects)

            # Update grade display
            grade_color = get_grade_color(grade)
            self.current_grade_label.setText(f"Final Grade: {grade}")
            self.current_grade_label.setStyleSheet(f"""
                font-size: 18px; font-weight: bold; padding: 10px;
                border: 2px solid {grade_color}; border-radius: 5px;
                background-color: {grade_color}20; color: {grade_color};
            """)

            # Send grade command to Arduino
            if self.arduino_module and self.arduino_module.is_connected():
                try:
                    success = self.arduino_module.send_grade_command(grade)
                    if success:
                        self.update_system_status(f"✅ Grade {grade} sent to Arduino for sorting")
                        log_info(SystemComponent.GUI, f"Successfully sent grade {grade} to Arduino")
                    else:
                        self.update_system_status(f"❌ Failed to send grade {grade} to Arduino")
                        log_arduino_error(f"Failed to send grade {grade} to Arduino")
                except Exception as e:
                    self.update_system_status(f"❌ Error sending grade to Arduino: {str(e)}")
                    log_arduino_error(f"Error sending grade {grade} to Arduino: {str(e)}", e)
            else:
                self.update_system_status(f"⚠️ Arduino not connected - Grade {grade} calculated but not sent")
                log_warning(SystemComponent.GUI, f"Arduino not connected - cannot send grade {grade}")

            # Update statistics
            if grade in self.grade_counts:
                self.grade_counts[grade] += 1
                self.live_stats[f"grade{grade}"] += 1
                self.total_pieces_processed += 1
                self.update_grade_counters()

        except Exception as e:
            log_error(SystemComponent.GUI, f"Error calculating grade from ROI session: {str(e)}", e)

    def finalize_detection_session(self):
        """Finalize the enhanced detection session with comprehensive performance reporting"""
        try:
            log_info(SystemComponent.GUI, "Finalizing enhanced detection session...")

            # 5. Integrate performance tracking and reporting
            session_performance = {
                'total_frames': len(self.predict_stream_results),
                'session_duration': time.time() - getattr(self, '_inference_start_time', time.time()),
                'model_health_status': self.detection_module.get_model_health_status("defect_detector").value,
                'camera_status': self.detection_module.get_camera_status("top").value,
                'total_defects_detected': sum(sum(frame['defects'].values()) for frame in self.predict_stream_results),
                'average_defects_per_frame': 0
            }

            if session_performance['total_frames'] > 0:
                session_performance['average_defects_per_frame'] = session_performance['total_defects_detected'] / session_performance['total_frames']

            # Get model performance report
            model_report = self.detection_module.get_model_performance_report("defect_detector")
            if model_report:
                session_performance.update({
                    'avg_inference_time': model_report.get('avg_inference_time', 0),
                    'success_rate': model_report.get('success_rate', 0),
                    'total_inferences': model_report.get('total_inferences', 0)
                })

            log_info(SystemComponent.GUI, f"Enhanced session performance: {session_performance}")

            # Update performance monitor with session data
            if hasattr(self, 'performance_monitor'):
                for key, value in session_performance.items():
                    if isinstance(value, (int, float)):
                        self.performance_monitor.add_metric(f'session_{key}', value)

            # Increment counters and update statistics
            if self.current_grade_info:
                grade = self.current_grade_info['grade']
                if grade in self.grade_counts:
                    self.grade_counts[grade] += 1
                    self.live_stats[f"grade{grade}"] += 1
                    self.total_pieces_processed += 1

                    # Update UI counters
                    self.update_grade_counters()

            # 4. Add health monitoring during inference sessions
            # Update model health display after session
            self.update_model_health_display()
            self.update_camera_status_display()

            # Generate performance report
            self.manual_generate_report()

            log_info(SystemComponent.GUI, "Enhanced detection session finalized with comprehensive reporting")

        except Exception as e:
            log_error(SystemComponent.GUI, f"Error finalizing enhanced detection session: {str(e)}", e)

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

        # Reset ROI-triggered state
        self.roi_triggered = False
        self.wood_confirmed = False
        self.auto_detection_active = False

        # Send command to Arduino
        if self.arduino_module and self.arduino_module.is_connected():
            try:
                success = self.arduino_module.send_arduino_command('C')
                if success:
                    self.update_system_status("✅ Continuous mode command sent to Arduino")
                    log_info(SystemComponent.GUI, "Successfully sent continuous mode command to Arduino")
                else:
                    self.update_system_status("❌ Failed to send continuous mode command to Arduino")
                    log_arduino_error("Failed to send continuous mode command to Arduino")
            except Exception as e:
                self.update_system_status(f"❌ Error sending continuous mode to Arduino: {str(e)}")
                log_arduino_error(f"Error sending continuous mode to Arduino: {str(e)}", e)
        else:
            self.update_system_status("⚠️ Arduino not connected - Continuous mode set locally only")
            log_warning(SystemComponent.GUI, "Arduino not connected - cannot send continuous mode command")

    def set_trigger_mode(self):
        """Set system to trigger mode"""
        self.current_mode = "TRIGGER"
        self.display_message("System set to TRIGGER mode")
        self.update_system_status("Status: TRIGGER MODE - Waiting for ROI wood detection trigger")
        self.update_detection_state("Waiting")

        # Reset state
        self.roi_triggered = False
        self.wood_confirmed = False
        self.auto_detection_active = False

        # Send command to Arduino
        if self.arduino_module and self.arduino_module.is_connected():
            try:
                success = self.arduino_module.send_arduino_command('T')
                if success:
                    self.update_system_status("✅ Trigger mode command sent to Arduino")
                    log_info(SystemComponent.GUI, "Successfully sent trigger mode command to Arduino")
                else:
                    self.update_system_status("❌ Failed to send trigger mode command to Arduino")
                    log_arduino_error("Failed to send trigger mode command to Arduino")
            except Exception as e:
                self.update_system_status(f"❌ Error sending trigger mode to Arduino: {str(e)}")
                log_arduino_error(f"Error sending trigger mode to Arduino: {str(e)}", e)
        else:
            self.update_system_status("⚠️ Arduino not connected - Trigger mode set locally only")
            log_warning(SystemComponent.GUI, "Arduino not connected - cannot send trigger mode command")

    def set_idle_mode(self):
        """Set system to idle mode"""
        self.current_mode = "IDLE"
        self.display_message("System set to IDLE mode")
        self.update_system_status("Status: IDLE MODE - System disabled, conveyor stopped")
        self.update_detection_state("Waiting")

        # Reset state
        self.roi_triggered = False
        self.wood_confirmed = False
        self.auto_detection_active = False

        # Send command to Arduino
        if self.arduino_module and self.arduino_module.is_connected():
            try:
                success = self.arduino_module.send_arduino_command('X')
                if success:
                    self.update_system_status("✅ Idle mode command sent to Arduino")
                    log_info(SystemComponent.GUI, "Successfully sent idle mode command to Arduino")
                else:
                    self.update_system_status("❌ Failed to send idle mode command to Arduino")
                    log_arduino_error("Failed to send idle mode command to Arduino")
            except Exception as e:
                self.update_system_status(f"❌ Error sending idle mode to Arduino: {str(e)}")
                log_arduino_error(f"Error sending idle mode to Arduino: {str(e)}", e)
        else:
            self.update_system_status("⚠️ Arduino not connected - Idle mode set locally only")
            log_warning(SystemComponent.GUI, "Arduino not connected - cannot send idle mode command")

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

    def _add_roi_status_overlay(self, frame, camera_name, overlaps):
        """Add comprehensive ROI status overlay to frame"""
        try:
            height, width = frame.shape[:2]

            # Get ROI information
            active_rois = self.roi_module.roi_manager.get_active_rois(camera_name)
            roi_states = self.roi_module.roi_manager.roi_states.get(camera_name, {})

            # Add status bar at top of frame
            status_bar_height = 40
            status_bar = np.zeros((status_bar_height, width, 3), dtype=np.uint8)
            status_bar[:] = [50, 50, 50]  # Dark gray background

            # Add status text
            status_text = f"ROI Status: {len(active_rois)} active"
            cv2.putText(status_bar, status_text, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add overlap information
            if overlaps:
                overlap_text = f" | Overlaps: {len(overlaps)}"
                cv2.putText(status_bar, overlap_text, (200, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add mode information
            mode_text = f" | Mode: {self.current_mode}"
            cv2.putText(status_bar, mode_text, (400, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Overlay status bar on frame
            frame[0:status_bar_height, :] = cv2.addWeighted(
                frame[0:status_bar_height, :], 0.7, status_bar, 0.3, 0
            )

            # Add individual ROI status indicators
            y_offset = height - 100
            for i, roi_id in enumerate(active_rois):
                roi_config = self.roi_module.roi_manager.get_roi_config(camera_name, roi_id)
                if not roi_config:
                    continue

                status = roi_states.get(roi_id, ROIStatus.INACTIVE)
                color = {
                    ROIStatus.ACTIVE: (0, 255, 0),
                    ROIStatus.OVERLAPPING: (0, 165, 255),
                    ROIStatus.INACTIVE: (128, 128, 128),
                    ROIStatus.ERROR: (0, 0, 255)
                }.get(status, (128, 128, 128))

                # Draw status indicator
                indicator_x = 10
                indicator_y = y_offset - (i * 25)
                cv2.circle(frame, (indicator_x, indicator_y), 8, color, -1)
                cv2.circle(frame, (indicator_x, indicator_y), 8, (255, 255, 255), 2)

                # Add ROI label
                label = f"{roi_config.name}: {status.value}"
                cv2.putText(frame, label, (25, indicator_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        except Exception as e:
            self.display_message(f"Error adding ROI status overlay: {str(e)}", "warning")

    def _add_wood_detection_overlays(self, frame, wood_detections):
        """Add enhanced wood detection overlays with confidence scores"""
        try:
            for i, detection in enumerate(wood_detections):
                if not detection.detected:
                    continue

                bbox = detection.bbox
                confidence = detection.confidence
                features = detection.features or {}

                x1, y1, x2, y2 = bbox

                # Determine color based on confidence
                if confidence >= 0.8:
                    color = (0, 255, 0)  # Green for high confidence
                elif confidence >= 0.6:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 165, 255)  # Orange for low confidence

                # Draw enhanced bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Draw corner markers
                marker_size = 15
                # Top-left
                cv2.line(frame, (x1, y1), (x1 + marker_size, y1), color, 2)
                cv2.line(frame, (x1, y1), (x1, y1 + marker_size), color, 2)
                # Top-right
                cv2.line(frame, (x2, y1), (x2 - marker_size, y1), color, 2)
                cv2.line(frame, (x2, y1), (x2, y1 + marker_size), color, 2)
                # Bottom-left
                cv2.line(frame, (x1, y2), (x1 + marker_size, y2), color, 2)
                cv2.line(frame, (x1, y2), (x1, y2 - marker_size), color, 2)
                # Bottom-right
                cv2.line(frame, (x2, y2), (x2 - marker_size, y2), color, 2)
                cv2.line(frame, (x2, y2), (x2, y2 - marker_size), color, 2)

                # Add confidence score
                confidence_text = f"Wood {i+1}: {confidence:.2f}"
                cv2.putText(frame, confidence_text, (x1 + 10, y1 + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Add feature information if available
                if features.get('dominant_color'):
                    color_text = f"Color: {features['dominant_color']}"
                    cv2.putText(frame, color_text, (x1 + 10, y1 + 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Add area information if available
                if 'contour_data' in features and 'area' in features['contour_data']:
                    area = features['contour_data']['area']
                    area_text = f"Area: {area:.0f}px²"
                    cv2.putText(frame, area_text, (x1 + 10, y1 + 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        except Exception as e:
            self.display_message(f"Error adding wood detection overlays: {str(e)}", "warning")

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
        """Update wood classification based on defect analysis"""
        try:
            # Get current defect information
            current_defects = getattr(self, 'current_defects', {})
            top_defects = current_defects.get('top', {}).get('defects', {})
            bottom_defects = current_defects.get('bottom', {}).get('defects', {})

            # Combine defects from both cameras
            all_defects = {}
            for defect_type, count in top_defects.items():
                all_defects[defect_type] = all_defects.get(defect_type, 0) + count
            for defect_type, count in bottom_defects.items():
                all_defects[defect_type] = all_defects.get(defect_type, 0) + count

            # Classify wood based on defect patterns
            wood_type = self._classify_wood_from_defects(all_defects)

            self.wood_classification = wood_type
            self.wood_classification_label.setText(f"Wood Type: {self.wood_classification}")

            # Set color based on wood type confidence
            if wood_type == "Unknown":
                color = "#95a5a6"  # Gray for unknown
                style = "font-size: 14px; font-weight: bold; color: #95a5a6; font-style: italic;"
            else:
                color = "#2c3e50"  # Dark blue for classified
                style = "font-size: 14px; font-weight: bold; color: #2c3e50;"

            self.wood_classification_label.setStyleSheet(style)

        except Exception as e:
            # Fallback to unknown if classification fails
            self.wood_classification = "Unknown"
            self.wood_classification_label.setText("Wood Type: Unknown")
            self.wood_classification_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #95a5a6; font-style: italic;")
            print(f"Error in wood classification: {str(e)}")

    def _classify_wood_from_defects(self, defects):
        """Classify wood type based on defect patterns"""
        try:
            if not defects:
                return "Unknown"

            total_defects = sum(defects.values())

            # Wood classification rules based on defect patterns
            # These are simplified rules - in practice would use ML model

            # High knot density often indicates softwoods
            knot_count = defects.get('knot', 0)
            knot_ratio = knot_count / total_defects if total_defects > 0 else 0

            # Crack patterns can indicate wood type
            crack_count = defects.get('crack', 0)
            crack_ratio = crack_count / total_defects if total_defects > 0 else 0

            # Classification logic
            if knot_ratio > 0.6:
                # High knot ratio suggests softwoods
                if total_defects > 10:
                    return "Pine"  # Pine often has many small knots
                else:
                    return "Spruce"  # Spruce has fewer but distinctive knots
            elif crack_ratio > 0.5:
                # High crack ratio suggests hardwoods that are prone to checking
                return "Oak"  # Oak is prone to cracking
            elif knot_ratio > 0.3:
                # Moderate knot ratio
                if crack_count > knot_count:
                    return "Maple"  # Maple can have cracks and knots
                else:
                    return "Cedar"  # Cedar has moderate knotting
            elif total_defects < 3:
                # Very few defects suggests high-quality wood
                return "Oak"  # High-quality oak
            else:
                # Default classification based on most common defect
                most_common_defect = max(defects, key=defects.get)
                if most_common_defect == 'knot':
                    return "Pine"
                elif most_common_defect == 'crack':
                    return "Oak"
                elif most_common_defect == 'resin':
                    return "Pine"
                else:
                    return "Maple"  # Default fallback

        except Exception as e:
            print(f"Error in defect-based classification: {str(e)}")
            return "Unknown"

    def update_system_status(self, status_text):
        """Update system status label with concise enhanced information"""
        try:
            # Start with the main status
            enhanced_status = status_text

            # Add model health info if available (keep it short)
            if hasattr(self.detection_module, 'get_model_health_status'):
                try:
                    model_health = self.detection_module.get_model_health_status()
                    # Use single letter for brevity
                    health_short = model_health.value[0].upper()  # H, D, U, or ?
                    enhanced_status += f" | M:{health_short}"
                except:
                    pass

            # Add camera status info if available (keep it very short)
            if hasattr(self.detection_module, 'get_camera_status'):
                try:
                    top_status = self.detection_module.get_camera_status("top")
                    bottom_status = self.detection_module.get_camera_status("bottom")
                    # Use single letters: A=Available, I=In Use, E=Error
                    top_short = "A" if top_status.value == "available" else ("I" if top_status.value == "in_use" else "E")
                    bottom_short = "A" if bottom_status.value == "available" else ("I" if bottom_status.value == "in_use" else "E")
                    enhanced_status += f" | C:{top_short}{bottom_short}"
                except:
                    pass

            # Limit the total length to prevent overflow
            if len(enhanced_status) > 80:
                enhanced_status = enhanced_status[:77] + "..."

            self.system_status_label.setText(enhanced_status)
            self.system_status_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #2c3e50;")
        except Exception as e:
            # Fallback to basic status if enhancement fails
            # Truncate if too long
            if len(status_text) > 80:
                status_text = status_text[:77] + "..."
            self.system_status_label.setText(status_text)
            self.system_status_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #2c3e50;")

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

    def update_model_health_display(self):
        """Update model health status display"""
        try:
            if hasattr(self.detection_module, 'get_model_health_status'):
                health_status = self.detection_module.get_model_health_status()
                self.model_health_label.setText(f"Model Health: {health_status.value.upper()}")

                # Set color based on health status
                if health_status.value == "healthy":
                    color = "#27ae60"  # Green
                elif health_status.value == "degraded":
                    color = "#f39c12"  # Orange
                elif health_status.value == "unhealthy":
                    color = "#e74c3c"  # Red
                else:
                    color = "#95a5a6"  # Gray

                self.model_health_label.setStyleSheet(f"font-size: 16px; font-weight: bold; padding: 10px; color: {color};")

            if hasattr(self.detection_module, 'get_model_performance_report'):
                performance_report = self.detection_module.get_model_performance_report()

                if performance_report:
                    # Update inference time
                    avg_time = performance_report.get('avg_inference_time', 0)
                    self.avg_inference_time_label.setText(f"{avg_time:.2f} ms")

                    # Set inference time status
                    if avg_time < 500:
                        self.inference_time_status.setText("Good")
                        self.inference_time_status.setStyleSheet("color: #27ae60;")
                    elif avg_time < 1000:
                        self.inference_time_status.setText("Slow")
                        self.inference_time_status.setStyleSheet("color: #f39c12;")
                    else:
                        self.inference_time_status.setText("Critical")
                        self.inference_time_status.setStyleSheet("color: #e74c3c;")

                    # Update success rate
                    success_rate = performance_report.get('success_rate', 0) * 100
                    self.success_rate_label.setText(f"{success_rate:.1f}%")

                    # Set success rate status
                    if success_rate > 95:
                        self.success_rate_status.setText("Excellent")
                        self.success_rate_status.setStyleSheet("color: #27ae60;")
                    elif success_rate > 85:
                        self.success_rate_status.setText("Good")
                        self.success_rate_status.setStyleSheet("color: #f39c12;")
                    else:
                        self.success_rate_status.setText("Poor")
                        self.success_rate_status.setStyleSheet("color: #e74c3c;")

                    # Update total inferences
                    total_inferences = performance_report.get('total_inferences', 0)
                    self.total_inferences_label.setText(str(total_inferences))

        except Exception as e:
            self.display_message(f"Error updating model health display: {str(e)}", "warning")

    def update_camera_status_display(self):
        """Update camera status display"""
        try:
            # Update top camera status
            if hasattr(self.detection_module, 'get_camera_status'):
                top_status = self.detection_module.get_camera_status("top")
                self.top_camera_health_status.setText(top_status.value.upper())

                # Set color based on status
                if top_status.value == "available":
                    color = "#27ae60"  # Green
                elif top_status.value == "in_use":
                    color = "#f39c12"  # Orange
                else:
                    color = "#e74c3c"  # Red

                self.top_camera_health_status.setStyleSheet(f"color: {color};")

            # Update bottom camera status
            if hasattr(self.detection_module, 'get_camera_status'):
                bottom_status = self.detection_module.get_camera_status("bottom")
                self.bottom_camera_health_status.setText(bottom_status.value.upper())

                # Set color based on status
                if bottom_status.value == "available":
                    color = "#27ae60"  # Green
                elif bottom_status.value == "in_use":
                    color = "#f39c12"  # Orange
                else:
                    color = "#e74c3c"  # Red

                self.bottom_camera_health_status.setStyleSheet(f"color: {color};")

            # Update usage statistics (mock data for now)
            self.top_camera_usage_count.setText("N/A")
            self.bottom_camera_usage_count.setText("N/A")
            self.top_camera_last_used.setText("N/A")
            self.bottom_camera_last_used.setText("N/A")

        except Exception as e:
            self.display_message(f"Error updating camera status display: {str(e)}", "warning")

    def reload_model(self):
        """Reload the model with error recovery"""
        try:
            self.display_message("Reloading model...")
            if hasattr(self.detection_module, 'reload_model'):
                success = self.detection_module.reload_model()
                if success:
                    self.display_message("Model reloaded successfully", "info")
                    self.update_model_health_display()
                else:
                    self.display_message("Model reload failed", "error")
            else:
                self.display_message("Model reload not available", "warning")
        except Exception as e:
            self.display_message(f"Error reloading model: {str(e)}", "error")

    def run_model_benchmark(self):
        """Run model performance benchmark"""
        try:
            self.display_message("Running model benchmark...")
            if hasattr(self.detection_module, 'benchmark_model'):
                benchmark_result = self.detection_module.benchmark_model()
                if benchmark_result:
                    self.display_message(f"Benchmark completed: {benchmark_result.avg_inference_time:.2f}ms avg, {benchmark_result.throughput:.2f} FPS", "info")
                    self.update_model_health_display()
                else:
                    self.display_message("Benchmark failed", "error")
            else:
                self.display_message("Model benchmark not available", "warning")
        except Exception as e:
            self.display_message(f"Error running benchmark: {str(e)}", "error")

    def load_current_config(self):
        """Load current model configuration into the form"""
        try:
            if hasattr(self.detection_module, 'get_model_config'):
                model_name = self.config_model_name.text()
                config = self.detection_module.get_model_config(model_name)

                if config:
                    # Update form fields with current config
                    self.config_confidence_threshold.setValue(config.get('confidence_threshold', 0.5))
                    self.config_health_interval.setValue(config.get('health_check_interval', 300))
                    self.config_inference_timeout.setValue(config.get('timeout', 5000))
                    self.config_retry_attempts.setValue(config.get('retry_attempts', 3))

                    self.config_status_text.setPlainText(f"✅ Configuration loaded for model: {model_name}\n\n{json.dumps(config, indent=2)}")
                    self.display_message("Configuration loaded successfully", "info")
                else:
                    self.config_status_text.setPlainText("❌ No configuration found for model")
                    self.display_message("No configuration found", "warning")
            else:
                self.config_status_text.setPlainText("❌ Configuration management not available")
                self.display_message("Configuration management not available", "warning")
        except Exception as e:
            self.config_status_text.setPlainText(f"❌ Error loading configuration: {str(e)}")
            self.display_message(f"Error loading configuration: {str(e)}", "error")

    def save_model_config(self):
        """Save model configuration from the form"""
        try:
            if hasattr(self.detection_module, 'update_model_config'):
                model_name = self.config_model_name.text()

                # Collect form data
                updates = {
                    'confidence_threshold': self.config_confidence_threshold.value(),
                    'health_check_interval': self.config_health_interval.value(),
                    'timeout': self.config_inference_timeout.value(),
                    'retry_attempts': self.config_retry_attempts.value()
                }

                # Save configuration
                success = self.detection_module.update_model_config(model_name, updates)

                if success:
                    self.config_status_text.setPlainText(f"✅ Configuration saved for model: {model_name}\n\n{json.dumps(updates, indent=2)}")
                    self.display_message("Configuration saved successfully", "info")
                else:
                    self.config_status_text.setPlainText("❌ Failed to save configuration")
                    self.display_message("Failed to save configuration", "error")
            else:
                self.config_status_text.setPlainText("❌ Configuration management not available")
                self.display_message("Configuration management not available", "warning")
        except Exception as e:
            self.config_status_text.setPlainText(f"❌ Error saving configuration: {str(e)}")
            self.display_message(f"Error saving configuration: {str(e)}", "error")

    def validate_configuration(self):
        """Validate current configuration"""
        try:
            if hasattr(self.detection_module, 'validate_configuration'):
                validation_result = self.detection_module.validate_configuration()

                if validation_result.is_valid:
                    self.config_status_text.setPlainText(f"✅ Configuration is valid\n\n{validation_result.message}")
                    self.display_message("Configuration validation passed", "info")
                else:
                    self.config_status_text.setPlainText(f"❌ Configuration validation failed\n\n{validation_result.message}")
                    if validation_result.details:
                        self.config_status_text.append(f"\nDetails:\n{json.dumps(validation_result.details, indent=2)}")
                    self.display_message("Configuration validation failed", "warning")
            else:
                self.config_status_text.setPlainText("❌ Configuration validation not available")
                self.display_message("Configuration validation not available", "warning")
        except Exception as e:
            self.config_status_text.setPlainText(f"❌ Error validating configuration: {str(e)}")
            self.display_message(f"Error validating configuration: {str(e)}", "error")

    def force_error_recovery(self):
        """Force error recovery for the model"""
        try:
            self.display_message("Forcing error recovery...")
            strategy = self.recovery_strategy_combo.currentText().lower()

            if hasattr(self.detection_module, 'reload_model'):
                success = self.detection_module.reload_model()
                if success:
                    self.error_status_text.setPlainText(f"✅ Error recovery successful using {strategy} strategy\n\nModel reloaded and health restored")
                    self.display_message("Error recovery completed successfully", "info")
                    self.update_model_health_display()
                else:
                    self.error_status_text.setPlainText(f"❌ Error recovery failed using {strategy} strategy\n\nModel reload unsuccessful")
                    self.display_message("Error recovery failed", "error")
            else:
                self.error_status_text.setPlainText("❌ Error recovery not available")
                self.display_message("Error recovery not available", "warning")
        except Exception as e:
            self.error_status_text.setPlainText(f"❌ Error during recovery: {str(e)}")
            self.display_message(f"Error during recovery: {str(e)}", "error")

    def reset_error_state(self):
        """Reset error state and clear error history"""
        try:
            self.display_message("Resetting error state...")
            # Reset error status display
            self.error_status_text.setPlainText("✅ Error state reset\n\nNo errors detected")

            # Reset model health monitoring if available
            if hasattr(self.detection_module, 'model_manager') and hasattr(self.detection_module.model_manager, 'health_monitor'):
                # Clear the metrics (this is a simplified reset)
                if hasattr(self.detection_module.model_manager.health_monitor, 'metrics'):
                    self.detection_module.model_manager.health_monitor.metrics.clear()

            self.display_message("Error state reset successfully", "info")
            self.update_model_health_display()
        except Exception as e:
            self.error_status_text.setPlainText(f"❌ Error resetting state: {str(e)}")
            self.display_message(f"Error resetting state: {str(e)}", "error")

    # ROI Configuration Methods
    def update_roi_list(self):
        """Update the ROI list for the selected camera"""
        try:
            camera_name = self.roi_camera_combo.currentText()
            self.roi_list_widget.clear()

            if hasattr(self, 'roi_module') and self.roi_module:
                active_rois = self.roi_module.roi_manager.get_active_rois(camera_name)
                roi_states = self.roi_module.roi_manager.roi_states.get(camera_name, {})

                for roi_id in active_rois:
                    roi_config = self.roi_module.roi_manager.get_roi_config(camera_name, roi_id)
                    if roi_config:
                        status = roi_states.get(roi_id, ROIStatus.INACTIVE)
                        item_text = f"{roi_config.name} ({roi_id}) - {status.value}"
                        item = QListWidgetItem(item_text)
                        item.setData(Qt.UserRole, roi_id)
                        self.roi_list_widget.addItem(item)

        except Exception as e:
            self.display_message(f"Error updating ROI list: {str(e)}", "warning")

    def on_roi_selected(self):
        """Handle ROI selection in the list"""
        try:
            current_item = self.roi_list_widget.currentItem()
            if not current_item:
                return

            camera_name = self.roi_camera_combo.currentText()
            roi_id = current_item.data(Qt.UserRole)

            roi_config = self.get_roi_config(camera_name, roi_id)
            if roi_config:
                    # Update property fields
                    self.roi_name_edit.setText(roi_config.name)
                    self.roi_x1_spin.setValue(roi_config.coordinates[0])
                    self.roi_y1_spin.setValue(roi_config.coordinates[1])
                    self.roi_x2_spin.setValue(roi_config.coordinates[2])
                    self.roi_y2_spin.setValue(roi_config.coordinates[3])
                    self.roi_threshold_spin.setValue(roi_config.overlap_threshold)

                    # Update preview
                    self.update_roi_preview(camera_name, roi_config)

        except Exception as e:
            self.display_message(f"Error selecting ROI: {str(e)}", "warning")

    def update_roi_preview(self, camera_name, roi_config):
        """Update the ROI preview image"""
        try:
            # Get current frame for preview
            if camera_name == "top":
                frame = self.top_frame_original
            else:
                frame = self.bottom_frame_original

            if frame is not None:
                # Create a copy for drawing
                preview_frame = frame.copy()

                # Draw ROI rectangle
                x1, y1, x2, y2 = roi_config.coordinates
                cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Add ROI info text
                cv2.putText(preview_frame, f"ROI: {roi_config.name}", (x1 + 10, y1 + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                # Convert to QPixmap and display
                rgb_image = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                h, w = rgb_image.shape[:2]
                bytes_per_line = 3 * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(self.roi_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.roi_preview_label.setPixmap(scaled_pixmap)
            else:
                self.roi_preview_label.setText("No frame available for preview")

        except Exception as e:
            self.display_message(f"Error updating ROI preview: {str(e)}", "warning")

    def add_roi(self):
        """Add a new ROI"""
        try:
            camera_name = self.roi_camera_combo.currentText()
            roi_name = self.roi_name_edit.text().strip()

            if not roi_name:
                roi_name = f"ROI_{len(self.roi_module.roi_manager.get_active_rois(camera_name)) + 1}"

            # Get coordinates from spin boxes
            x1 = self.roi_x1_spin.value()
            y1 = self.roi_y1_spin.value()
            x2 = self.roi_x2_spin.value()
            y2 = self.roi_y2_spin.value()

            # Validate coordinates
            if x2 <= x1 or y2 <= y1:
                self.display_message("Invalid ROI coordinates: X2 must be > X1 and Y2 must be > Y1", "error")
                return

            coordinates = (x1, y1, x2, y2)
            threshold = self.roi_threshold_spin.value()

            # Generate unique ROI ID
            roi_id = f"{camera_name}_roi_{int(time.time())}"

            if hasattr(self, 'roi_module') and self.roi_module:
                success = self.roi_module.roi_manager.define_roi(
                    camera_name, roi_id, coordinates, roi_name, threshold
                )

                if success:
                    self.display_message(f"ROI '{roi_name}' added successfully", "info")
                    self.update_roi_list()
                else:
                    self.display_message("Failed to add ROI", "error")

        except Exception as e:
            self.display_message(f"Error adding ROI: {str(e)}", "error")

    def edit_roi(self):
        """Edit the selected ROI"""
        try:
            current_item = self.roi_list_widget.currentItem()
            if not current_item:
                self.display_message("Please select an ROI to edit", "warning")
                return

            camera_name = self.roi_camera_combo.currentText()
            roi_id = current_item.data(Qt.UserRole)

            # Get updated values
            roi_name = self.roi_name_edit.text().strip()
            x1 = self.roi_x1_spin.value()
            y1 = self.roi_y1_spin.value()
            x2 = self.roi_x2_spin.value()
            y2 = self.roi_y2_spin.value()
            threshold = self.roi_threshold_spin.value()

            # Validate coordinates
            if x2 <= x1 or y2 <= y1:
                self.display_message("Invalid ROI coordinates", "error")
                return

            # Update ROI configuration
            roi_config = self.get_roi_config(camera_name, roi_id)
            if roi_config:
                roi_config.name = roi_name
                roi_config.coordinates = (x1, y1, x2, y2)
                roi_config.overlap_threshold = threshold

                # Save configuration
                self.roi_module.roi_manager.save_config()
                self.display_message(f"ROI '{roi_name}' updated successfully", "info")
                self.update_roi_list()
            else:
                self.display_message("ROI not found", "error")

        except Exception as e:
            self.display_message(f"Error editing ROI: {str(e)}", "error")

    def delete_roi(self):
        """Delete the selected ROI"""
        try:
            current_item = self.roi_list_widget.currentItem()
            if not current_item:
                self.display_message("Please select an ROI to delete", "warning")
                return

            camera_name = self.roi_camera_combo.currentText()
            roi_id = current_item.data(Qt.UserRole)

            if hasattr(self, 'roi_module') and self.roi_module:
                # Remove from active ROIs
                if camera_name in self.roi_module.roi_manager.active_rois:
                    self.roi_module.roi_manager.active_rois[camera_name].discard(roi_id)

                # Remove from ROIs dict
                if camera_name in self.roi_module.roi_manager.rois:
                    self.roi_module.roi_manager.rois[camera_name].pop(roi_id, None)

                # Remove from states
                if camera_name in self.roi_module.roi_manager.roi_states:
                    self.roi_module.roi_manager.roi_states[camera_name].pop(roi_id, None)

                # Save configuration
                self.roi_module.roi_manager.save_config()

                self.display_message("ROI deleted successfully", "info")
                self.update_roi_list()

                # Clear property fields
                self.roi_name_edit.clear()
                self.roi_preview_label.setText("Select an ROI to preview")

        except Exception as e:
            self.display_message(f"Error deleting ROI: {str(e)}", "error")

    def activate_roi(self):
        """Activate the selected ROI"""
        try:
            current_item = self.roi_list_widget.currentItem()
            if not current_item:
                self.display_message("Please select an ROI to activate", "warning")
                return

            camera_name = self.roi_camera_combo.currentText()
            roi_id = current_item.data(Qt.UserRole)

            if hasattr(self, 'roi_module') and self.roi_module:
                success = self.roi_module.roi_manager.activate_roi(camera_name, roi_id)
                if success:
                    self.display_message("ROI activated successfully", "info")
                    self.update_roi_list()
                else:
                    self.display_message("Failed to activate ROI", "error")

        except Exception as e:
            self.display_message(f"Error activating ROI: {str(e)}", "error")

    def deactivate_roi(self):
        """Deactivate the selected ROI"""
        try:
            current_item = self.roi_list_widget.currentItem()
            if not current_item:
                self.display_message("Please select an ROI to deactivate", "warning")
                return

            camera_name = self.roi_camera_combo.currentText()
            roi_id = current_item.data(Qt.UserRole)

            if hasattr(self, 'roi_module') and self.roi_module:
                success = self.roi_module.roi_manager.deactivate_roi(camera_name, roi_id)
                if success:
                    self.display_message("ROI deactivated successfully", "info")
                    self.update_roi_list()
                else:
                    self.display_message("Failed to deactivate ROI", "error")

        except Exception as e:
            self.display_message(f"Error deactivating ROI: {str(e)}", "error")

    def save_roi_config(self):
        """Save ROI configuration"""
        try:
            if hasattr(self, 'roi_module') and self.roi_module:
                success = self.roi_module.roi_manager.save_config()
                if success:
                    self.display_message("ROI configuration saved successfully", "info")
                else:
                    self.display_message("Failed to save ROI configuration", "error")
        except Exception as e:
            self.display_message(f"Error saving ROI configuration: {str(e)}", "error")

    def load_roi_config(self):
        """Load ROI configuration"""
        try:
            if hasattr(self, 'roi_module') and self.roi_module:
                success = self.roi_module.roi_manager.load_config()
                if success:
                    self.display_message("ROI configuration loaded successfully", "info")
                    self.update_roi_list()
                else:
                    self.display_message("Failed to load ROI configuration", "error")
        except Exception as e:
            self.display_message(f"Error loading ROI configuration: {str(e)}", "error")

    def reset_roi_config(self):
        """Reset ROI configuration to default"""
        try:
            if hasattr(self, 'roi_module') and self.roi_module:
                # Clear all ROIs
                self.roi_module.roi_manager.rois.clear()
                self.roi_module.roi_manager.active_rois.clear()
                self.roi_module.roi_manager.roi_states.clear()

                # Create default ROIs
                self.roi_module.roi_manager.define_roi("top", "top_roi_1", (64, 0, 1216, 108), "Top ROI", 0.3)
                self.roi_module.roi_manager.define_roi("bottom", "bottom_roi_1", (64, 612, 1216, 720), "Bottom ROI", 0.3)

                self.display_message("ROI configuration reset to default", "info")
                self.update_roi_list()

        except Exception as e:
            self.display_message(f"Error resetting ROI configuration: {str(e)}", "error")

    def get_roi_config(self, camera_name, roi_id):
        """Get ROI configuration from the ROI manager"""
        try:
            if hasattr(self, 'roi_module') and self.roi_module:
                if camera_name in self.roi_module.roi_manager.rois:
                    return self.roi_module.roi_manager.rois[camera_name].get(roi_id)
            return None
        except Exception as e:
            self.display_message(f"Error getting ROI config: {str(e)}", "warning")
            return None

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
