import sys
import cv2
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
            # Use thread-safe update mechanism
            self.performance_monitor.add_update_callback(self.queue_performance_update)

        # Initialize modules with enhanced error handling
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

        # Error monitoring
        self.error_monitoring_active = True
        self.last_error_check = time.time()

        log_info(SystemComponent.GUI, f"WoodSortingApp initialized (dev_mode={dev_mode})")

        self.setup_ui()
        self.setup_connections()
        self.setup_dev_mode()

        # Start camera feeds and message processing
        if not self.dev_mode:
            self.camera_module.initialize_cameras()
            self.update_feeds() # Start the main update loop for real cameras
        else:
            self.update_feeds() # Start the main update loop for dev_mode logic
            
        # Start message queue processing
        self.process_message_queue()
        
        # Start error monitoring
        self.setup_error_monitoring()

    def center_window(self):
        """Center the window on the screen"""
        screen = QApplication.desktop().screenGeometry()
        window = self.geometry()
        x = (screen.width() - window.width()) // 2
        y = (screen.height() - window.height()) // 2
        self.move(x, y)

    def setup_connections(self):
        """Setup connections and signals between modules"""
        # This method sets up connections between different components
        # For now, we're using the message queue for communication
        log_info(SystemComponent.GUI, "Setting up module connections")

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
        defect_panel_layout = QVBoxLayout(self.defect_analysis_panel)
        defect_panel_layout.setContentsMargins(10, 15, 10, 10)
        defect_panel_layout.setSpacing(8)
        
        # Current Grade Display (Responsive height)
        self.current_grade_frame = QFrame()
        self.current_grade_frame.setMinimumHeight(50)  # Minimum height
        self.current_grade_frame.setMaximumHeight(80)  # Maximum height
        self.current_grade_frame.setFrameStyle(QFrame.StyledPanel)
        self.current_grade_frame.setStyleSheet("background-color: #f0f0f0; border: 2px solid #ccc; border-radius: 5px;")
        current_grade_layout = QVBoxLayout(self.current_grade_frame)
        current_grade_layout.setContentsMargins(5, 5, 5, 5)
        
        self.current_grade_label = QLabel("Final Grade: Waiting for wood...")
        self.current_grade_label.setAlignment(Qt.AlignCenter)
        self.current_grade_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #666; padding: 8px;")
        current_grade_layout.addWidget(self.current_grade_label)
        
        defect_panel_layout.addWidget(self.current_grade_frame)
        
        # Defect Details Area (Responsive sizing)
        self.defect_details_scroll = QScrollArea()
        self.defect_details_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.defect_details_widget = QWidget()
        self.defect_details_layout = QVBoxLayout(self.defect_details_widget)
        self.defect_details_layout.setAlignment(Qt.AlignTop)
        self.defect_details_layout.setContentsMargins(5, 5, 5, 5)
        self.defect_details_layout.setSpacing(5)
        
        # Initial placeholder
        placeholder_label = QLabel("No defects detected\n\nWaiting for wood detection...")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setStyleSheet("color: #888; font-style: italic; padding: 20px; font-size: 18px;")
        self.defect_details_layout.addWidget(placeholder_label)
        
        self.defect_details_scroll.setWidget(self.defect_details_widget)
        self.defect_details_scroll.setWidgetResizable(True)
        self.defect_details_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.defect_details_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        defect_panel_layout.addWidget(self.defect_details_scroll)
        
        # Camera Status Summary (Responsive height)
        camera_status_frame = QFrame()
        camera_status_frame.setMinimumHeight(40)  # Minimum height
        camera_status_frame.setMaximumHeight(60)  # Maximum height
        camera_status_frame.setFrameStyle(QFrame.StyledPanel)
        camera_status_layout = QVBoxLayout(camera_status_frame)
        camera_status_layout.setContentsMargins(5, 5, 5, 5)
        
        self.top_camera_status = QLabel("Top Camera: Ready")
        self.top_camera_status.setStyleSheet("font-size: 14px; color: #666;")
        self.bottom_camera_status = QLabel("Bottom Camera: Ready")
        self.bottom_camera_status.setStyleSheet("font-size: 14px; color: #666;")
        
        camera_status_layout.addWidget(self.top_camera_status)
        camera_status_layout.addWidget(self.bottom_camera_status)
        defect_panel_layout.addWidget(camera_status_frame)
        
        # Add camera container and defect panel to top section with proportions
        top_section_layout.addWidget(cameras_container, 70)  # 70% width for cameras
        top_section_layout.addWidget(self.defect_analysis_panel, 30)  # 30% width for defect panel

        self.main_layout.addWidget(top_section_container)

        # Middle Section: Controls (Responsive height)
        controls_container = QWidget()
        controls_container.setMinimumHeight(100)  # Minimum height
        controls_container.setMaximumHeight(140)  # Maximum height
        controls_layout = QHBoxLayout(controls_container)
        controls_layout.setContentsMargins(10, 5, 10, 5)
        controls_layout.setSpacing(15)

        # System Status Group (Responsive width)
        status_group = QGroupBox("System Status")
        status_group.setMinimumWidth(250)  # Minimum width
        status_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        status_layout = QVBoxLayout(status_group)
        status_layout.setContentsMargins(10, 15, 10, 10)
        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-weight: bold; font-size: 16px; padding: 10px;")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        controls_layout.addWidget(status_group)

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
        
        self.roi_checkbox = QCheckBox("Top ROI")
        self.roi_checkbox.setChecked(True) # Default from original
        self.roi_checkbox.stateChanged.connect(self.toggle_roi)
        self.roi_checkbox.setStyleSheet("font-size: 14px;")
        detection_layout.addWidget(self.roi_checkbox)
        
        self.live_detect_checkbox = QCheckBox("Live Detect")
        self.live_detect_checkbox.setChecked(False) # Default from original
        self.live_detect_checkbox.stateChanged.connect(self.toggle_live_detection_mode)
        self.live_detect_checkbox.setStyleSheet("font-size: 14px;")
        detection_layout.addWidget(self.live_detect_checkbox)
        
        self.auto_grade_checkbox = QCheckBox("Auto Grade")
        self.auto_grade_checkbox.setChecked(False) # Default from original
        self.auto_grade_checkbox.stateChanged.connect(self.toggle_auto_grade)
        self.auto_grade_checkbox.setStyleSheet("font-size: 14px;")
        detection_layout.addWidget(self.auto_grade_checkbox)
        controls_layout.addWidget(detection_group)

        # Reports Group (Responsive width)
        reports_group = QGroupBox("Reports")
        reports_group.setMinimumWidth(250)  # Minimum width
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

        # Bottom Section: Enhanced Tabbed Statistics with Error Monitoring (Responsive for maximized)
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
        grade_summary_tab = QWidget()
        grade_summary_layout = QVBoxLayout(grade_summary_tab)
        grade_summary_layout.setContentsMargins(15, 10, 15, 10)
        grade_summary_layout.setSpacing(8)
        self.stats_notebook.addTab(grade_summary_tab, "Grade Summary")

        # Grade counts in a clean grid (Fixed dimensions)
        grade_counts_frame = QWidget()
        grade_counts_frame.setFixedHeight(120)  # Increased height for bigger text
        grade_counts_layout = QHBoxLayout(grade_counts_frame)
        grade_counts_layout.setSpacing(10)
        grade_counts_layout.setContentsMargins(10, 5, 10, 5)
        self.live_stats_labels = {}
        grade_info = [
            ("grade0", "Perfect\n(No Defects)", "dark green"),
            ("grade1", "Good\n(G2-0)", "green"), 
            ("grade2", "Fair\n(G2-1, G2-2, G2-3)", "orange"),
            ("grade3", "Poor\n(G2-4)", "red")
        ]
        for grade_key, label_text, color in grade_info:
            grade_container = QFrame()
            grade_container.setFrameShape(QFrame.StyledPanel)
            grade_container.setFrameShadow(QFrame.Sunken)
            grade_container.setMinimumSize(220, 90)  # Increased minimum container size
            grade_container.setMaximumSize(350, 120)  # Increased maximum container size
            grade_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            grade_container_layout = QVBoxLayout(grade_container)
            grade_container_layout.setContentsMargins(5, 5, 5, 5)
            grade_container_layout.setSpacing(3)
            
            title_label = QLabel(label_text)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setFont(QFont("Arial", 10, QFont.Bold))  # Reduced title font slightly
            title_label.setFixedHeight(35)  # Increased height for title
            grade_container_layout.addWidget(title_label)
            
            count_label = QLabel("0")
            count_label.setAlignment(Qt.AlignCenter)
            count_label.setStyleSheet(f"color: {color}; font-size: 22pt; font-weight: bold;")  # Slightly reduced
            count_label.setFixedHeight(50)  # Increased height for count
            self.live_stats_labels[grade_key] = count_label
            grade_container_layout.addWidget(count_label)
            
            grade_counts_layout.addWidget(grade_container)
        grade_summary_layout.addWidget(grade_counts_frame)

        # Live Grading Results (Fixed dimensions)
        live_grading_group = QGroupBox("Live Grading Results")
        live_grading_group.setFixedHeight(140)  # Increased height for bigger text
        live_grading_layout = QGridLayout(live_grading_group)
        live_grading_layout.setContentsMargins(15, 15, 15, 10)
        live_grading_layout.setSpacing(8)
        
        # Top Camera Result
        top_label = QLabel("Top Camera:")
        top_label.setFixedWidth(120)  # Increased width
        top_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        live_grading_layout.addWidget(top_label, 0, 0)
        self.top_grade_label = QLabel("No wood detected")
        self.top_grade_label.setStyleSheet("color: gray; font-size: 14px;")
        self.top_grade_label.setFixedWidth(220)  # Increased width
        live_grading_layout.addWidget(self.top_grade_label, 0, 1)

        # Bottom Camera Result
        bottom_label = QLabel("Bottom Camera:")
        bottom_label.setFixedWidth(120)  # Increased width
        bottom_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        live_grading_layout.addWidget(bottom_label, 1, 0)
        self.bottom_grade_label = QLabel("No wood detected")
        self.bottom_grade_label.setStyleSheet("color: gray; font-size: 14px;")
        self.bottom_grade_label.setFixedWidth(220)  # Increased width
        live_grading_layout.addWidget(self.bottom_grade_label, 1, 1)

        # Final Grade Result (spanning 2 rows)
        final_label = QLabel("Final Grade:")
        final_label.setFixedWidth(120)  # Increased width
        final_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        live_grading_layout.addWidget(final_label, 0, 2, 2, 1) # Span 2 rows
        self.combined_grade_label = QLabel("No wood detected")
        self.combined_grade_label.setStyleSheet("font-weight: bold; color: gray; font-size: 16px;")
        self.combined_grade_label.setFixedWidth(280)  # Increased width
        live_grading_layout.addWidget(self.combined_grade_label, 0, 3, 2, 1) # Span 2 rows
        
        grade_summary_layout.addWidget(live_grading_group)

        # Tab 2: System Health & Error Monitoring (Fixed dimensions)
        error_monitoring_tab = QWidget()
        error_monitoring_layout = QVBoxLayout(error_monitoring_tab)
        error_monitoring_layout.setContentsMargins(15, 10, 15, 10)
        error_monitoring_layout.setSpacing(8)
        self.stats_notebook.addTab(error_monitoring_tab, "System Health")
        
        # System status overview (Fixed dimensions)
        health_overview_group = QGroupBox("System Status Overview")
        health_overview_group.setFixedHeight(140)  # Fixed height
        health_overview_layout = QGridLayout(health_overview_group)
        health_overview_layout.setContentsMargins(15, 15, 15, 10)
        health_overview_layout.setSpacing(8)
        
        # Camera System Status
        cam_label = QLabel("Camera System:")
        cam_label.setFixedWidth(120)
        cam_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        health_overview_layout.addWidget(cam_label, 0, 0)
        self.camera_health_label = QLabel("Checking...")
        self.camera_health_label.setStyleSheet("color: gray; font-size: 14px;")
        self.camera_health_label.setFixedWidth(200)
        health_overview_layout.addWidget(self.camera_health_label, 0, 1)
        
        # Arduino Status
        ard_label = QLabel("Arduino:")
        ard_label.setFixedWidth(120)
        ard_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        health_overview_layout.addWidget(ard_label, 1, 0)
        self.arduino_health_label = QLabel("Checking...")
        self.arduino_health_label.setStyleSheet("color: gray; font-size: 14px;")
        self.arduino_health_label.setFixedWidth(200)
        health_overview_layout.addWidget(self.arduino_health_label, 1, 1)
        
        # Detection System Status
        det_label = QLabel("Detection System:")
        det_label.setFixedWidth(120)
        det_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        health_overview_layout.addWidget(det_label, 2, 0)
        self.detection_health_label = QLabel("OK")
        self.detection_health_label.setStyleSheet("color: green; font-size: 14px;")
        self.detection_health_label.setFixedWidth(200)
        health_overview_layout.addWidget(self.detection_health_label, 2, 1)
        
        # Wood Detection Status
        wood_label = QLabel("Wood Detection:")
        wood_label.setFixedWidth(120)
        wood_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        health_overview_layout.addWidget(wood_label, 3, 0)
        self.wood_detection_status_label = QLabel("Waiting for IR trigger")
        self.wood_detection_status_label.setStyleSheet("color: gray; font-size: 14px;")
        self.wood_detection_status_label.setFixedWidth(200)
        health_overview_layout.addWidget(self.wood_detection_status_label, 3, 1)
        
        error_monitoring_layout.addWidget(health_overview_group)
        
        # Error summary (Fixed dimensions)
        error_summary_group = QGroupBox("Recent Errors")
        error_summary_group.setFixedHeight(150)  # Fixed height
        error_summary_layout = QVBoxLayout(error_summary_group)
        error_summary_layout.setContentsMargins(15, 15, 15, 10)
        self.error_summary_text = QTextEdit()
        self.error_summary_text.setFixedHeight(115)  # Fixed content height
        self.error_summary_text.setReadOnly(True)
        self.error_summary_text.setStyleSheet("font-size: 12px; font-family: Consolas, monospace;")
        error_summary_layout.addWidget(self.error_summary_text)
        error_monitoring_layout.addWidget(error_summary_group)

        # Tab 3: Defect Details (Enhanced with fixed dimensions)
        defect_details_tab = QWidget()
        defect_details_layout = QVBoxLayout(defect_details_tab)
        defect_details_layout.setContentsMargins(15, 10, 15, 10)
        self.stats_notebook.addTab(defect_details_tab, "Defect Details")
        
        self.defect_details_text = QTextEdit()
        self.defect_details_text.setFixedHeight(290)  # Fixed height for content
        self.defect_details_text.setReadOnly(True)
        self.defect_details_text.setText("Waiting for detection data...")
        self.defect_details_text.setStyleSheet("font-size: 14px; font-family: Arial;")
        defect_details_layout.addWidget(self.defect_details_text)

        # Tab 4: Performance Metrics (Enhanced with fixed dimensions)
        performance_tab = QWidget()
        performance_layout = QVBoxLayout(performance_tab)
        performance_layout.setContentsMargins(15, 10, 15, 10)
        self.stats_notebook.addTab(performance_tab, "Performance")
        
        self.performance_text = QTextEdit()
        self.performance_text.setFixedHeight(290)  # Fixed height for content
        self.performance_text.setReadOnly(True)
        self.performance_text.setStyleSheet("font-size: 14px; font-family: Arial;")
        performance_layout.addWidget(self.performance_text)

        # Tab 5: Recent Activity (Enhanced with fixed dimensions)
        activity_tab = QWidget()
        activity_layout = QVBoxLayout(activity_tab)
        activity_layout.setContentsMargins(15, 10, 15, 10)
        self.stats_notebook.addTab(activity_tab, "Recent Activity")
        
        self.activity_text = QTextEdit()
        self.activity_text.setFixedHeight(290)  # Fixed height for content
        self.activity_text.setReadOnly(True)
        self.activity_text.setStyleSheet("font-size: 14px; font-family: Arial;")
        activity_layout.addWidget(self.activity_text)

    def setup_error_monitoring(self):
        """Setup periodic error monitoring"""
        self.error_timer = QTimer(self)
        self.error_timer.timeout.connect(self.update_error_monitoring)
        self.error_timer.start(5000)  # Update every 5 seconds
        log_info(SystemComponent.GUI, "Error monitoring started")

    def queue_performance_update(self, metrics):
        """Queue performance update to be processed in main thread"""
        try:
            # Use QTimer.singleShot to ensure GUI updates happen in main thread
            QTimer.singleShot(0, lambda: self.update_performance_display(metrics))
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error queuing performance update: {str(e)}", e)

    def update_performance_display(self, metrics):
        """Update performance display with real-time metrics"""
        try:
            if hasattr(self, 'performance_text'):
                performance_text = f"""Real-Time Performance Metrics
=================================

Frame Rate: {metrics.fps:.1f} FPS
Memory Usage: {metrics.memory_usage_mb:.1f} MB
CPU Usage: {metrics.cpu_usage_percent:.1f}%
Processing Time: {metrics.processing_time_ms:.1f} ms

Component Timing:
• Detection: {metrics.detection_time_ms:.1f} ms
• Arduino: {metrics.arduino_time_ms:.1f} ms  
• GUI Updates: {metrics.gui_update_time_ms:.1f} ms

System Status: {'OPTIMAL' if metrics.fps > 25 and metrics.cpu_usage_percent < 80 else 'DEGRADED' if metrics.fps > 15 else 'CRITICAL'}
"""
                self.performance_text.setText(performance_text)
                
                # Update frame rate for performance monitoring
                self.performance_monitor.update_frame_rate()
                
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error updating performance display: {str(e)}", e)

    def update_error_monitoring(self):
        """Update system health indicators and error summary"""
        if not self.error_monitoring_active:
            return
            
        try:
            # Update camera health
            camera_status = self.camera_module.get_camera_status()
            available_cameras = self.camera_module.get_available_cameras()
            
            if len(available_cameras) == 2:
                self.camera_health_label.setText("HEALTHY")
                self.camera_health_label.setStyleSheet("color: green; font-weight: bold;")
            elif len(available_cameras) == 1:
                self.camera_health_label.setText("DEGRADED (1 camera)")
                self.camera_health_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.camera_health_label.setText("CRITICAL (no cameras)")
                self.camera_health_label.setStyleSheet("color: red; font-weight: bold;")
            
            # Update Arduino health
            arduino_status = self.arduino_module.get_connection_status()
            if arduino_status.get("connected", False):
                self.arduino_health_label.setText("CONNECTED")
                self.arduino_health_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.arduino_health_label.setText("DISCONNECTED")
                self.arduino_health_label.setStyleSheet("color: red; font-weight: bold;")
            
            # Update wood detection status based on current mode and IR state
            if self.current_mode == "TRIGGER":
                if self.ir_triggered:
                    if self.wood_confirmed:
                        self.wood_detection_status_label.setText("Wood detected - Processing")
                        self.wood_detection_status_label.setStyleSheet("color: green; font-weight: bold;")
                    else:
                        self.wood_detection_status_label.setText("Object detected - Checking for wood")
                        self.wood_detection_status_label.setStyleSheet("color: orange; font-weight: bold;")
                else:
                    self.wood_detection_status_label.setText("Waiting for IR trigger")
                    self.wood_detection_status_label.setStyleSheet("color: gray;")
            elif self.current_mode == "CONTINUOUS":
                self.wood_detection_status_label.setText("Continuous mode - Always active")
                self.wood_detection_status_label.setStyleSheet("color: blue;")
            else:
                self.wood_detection_status_label.setText("IDLE - Detection inactive")
                self.wood_detection_status_label.setStyleSheet("color: gray;")
            
            # Update error summary
            error_summary = get_error_summary()
            error_text = f"Total Errors: {error_summary.get('total_errors', 0)}\n"
            
            if error_summary.get('last_errors'):
                error_text += "Recent Issues:\n"
                for component, error_info in error_summary['last_errors'].items():
                    error_text += f"• {component}: {error_info.get('message', 'Unknown')} ({error_info.get('timestamp', 'Unknown time')})\n"
            else:
                error_text += "No recent errors detected."
            
            self.error_summary_text.setText(error_text)
            
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error in error monitoring update: {str(e)}", e)

    def process_message_queue(self):
        """Process messages from Arduino and other background threads"""
        try:
            while not self.message_queue.empty():
                try:
                    message_type, message_data = self.message_queue.get_nowait()
                    
                    if message_type == "arduino_message":
                        self.handle_arduino_message(message_data)
                    elif message_type == "status_update":
                        self.status_label.setText(f"Status: {message_data}")
                        log_info(SystemComponent.GUI, f"Status update: {message_data}")
                    elif message_type == "error":
                        log_error(SystemComponent.GUI, f"Received error message: {message_data}")
                        
                except queue.Empty:
                    break
                except Exception as e:
                    log_error(SystemComponent.GUI, f"Error processing message: {str(e)}", e)
                    
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error in message queue processing: {str(e)}", e)
        
        # Schedule next check
        QTimer.singleShot(100, self.process_message_queue)

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
                self.status_label.setText(f"Status: Arduino: {message}")
                
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error handling Arduino message '{message}': {str(e)}", e)

    def handle_ir_beam_broken(self):
        """Handle IR beam broken event - start wood detection workflow with checkbox activation"""
        try:
            log_info(SystemComponent.GUI, "IR beam broken - starting wood detection workflow")
            
            # Only respond to IR triggers in TRIGGER mode (like original application)
            if self.current_mode == "TRIGGER":
                if not self.auto_detection_active:
                    log_info(SystemComponent.GUI, "✅ TRIGGER MODE: Starting detection...")
                    log_info(SystemComponent.GUI, "🔧 Arduino should now set motorActiveForTrigger = true")
                    log_info(SystemComponent.GUI, "⚡ Stepper motor should start running NOW!")
                    
                    # Activate the Live Detection and Auto Grade checkboxes (like original behavior)
                    self.live_detect_checkbox.setChecked(True)
                    self.auto_grade_checkbox.setChecked(True)
                    
                    # Update internal state variables
                    self.live_detection_var = True
                    self.auto_grade_var = True
                    
                    # Update states
                    self.ir_triggered = True
                    self.wood_confirmed = False
                    self.auto_detection_active = True
                    
                    self.status_label.setText("Status: IR TRIGGERED - Motor should be running!")
                    
                else:
                    log_warning(SystemComponent.GUI, "⚠️ IR beam broken but detection already active")
            else:
                # In IDLE or CONTINUOUS mode, just log the IR signal but don't act on it
                log_info(SystemComponent.GUI, f"❌ IR beam broken received but system is in {self.current_mode} mode - ignoring trigger")
                self.status_label.setText(f"Status: IR signal ignored ({self.current_mode} mode)")
            
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
                
                # Deactivate the checkboxes (like original behavior)
                self.live_detect_checkbox.setChecked(False)
                self.auto_grade_checkbox.setChecked(False)
                
                # Update internal state variables
                self.live_detection_var = False
                self.auto_grade_var = False
                
                # Stop detection session
                self.auto_detection_active = False
                self.ir_triggered = False
                self.wood_confirmed = False
                
                self.status_label.setText("Status: Processing results... → Ready for next trigger")
                
                # Process any final grading if needed
                self.finalize_detection_session()
                
                # Return to ready state
                self.status_label.setText("Status: TRIGGER MODE - Waiting for IR beam trigger")
                
            else:
                log_info(SystemComponent.GUI, f"Length signal received (duration: {duration_ms}ms) but system is in {self.current_mode} mode or no detection active")
                self.status_label.setText(f"Status: Object length: ~{estimated_length_mm:.1f}mm")
            
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error handling length measurement: {str(e)}", e)

    def finalize_detection_session(self):
        """Finalize the detection session and perform grading if needed"""
        try:
            # This method would contain the logic to finalize grading
            # Based on accumulated detection data during the session
            log_info(SystemComponent.GUI, "Finalizing detection session...")
            
            # Here you would typically:
            # 1. Collect all detection results from the session
            # 2. Determine final grade
            # 3. Send Arduino command
            # 4. Update statistics
            
            # For now, just log completion
            log_info(SystemComponent.GUI, "Detection session finalized")
            
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error finalizing detection session: {str(e)}", e)

    def update_defect_analysis_panel(self, camera_name, defect_dict, measurements=None):
        """Update the live defect analysis panel with detailed defect information"""
        try:
            # Clear existing defect widgets
            while self.defect_details_layout.count():
                child = self.defect_details_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            
            if measurements and defect_dict:
                # Update current grade display
                surface_grade = self.determine_surface_grade(measurements)
                grade_color = self.get_grade_color(surface_grade)
                self.current_grade_label.setText(f"Final Grade: {surface_grade}")
                self.current_grade_label.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {grade_color}; padding: 10px;")
                
                # Camera info header
                camera_header = QLabel(f"📹 {camera_name.title()} Camera Analysis")
                camera_header.setStyleSheet("font-size: 13px; font-weight: bold; color: #2c3e50; padding: 8px; background-color: #ecf0f1; border-radius: 3px;")
                self.defect_details_layout.addWidget(camera_header)
                
                # Defect count summary
                defect_count_label = QLabel(f"🔍 Total Defects Found: {len(measurements)}")
                defect_count_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #e74c3c; padding: 5px;")
                self.defect_details_layout.addWidget(defect_count_label)
                
                # Individual defect details
                for i, (defect_type, size_mm, percentage) in enumerate(measurements, 1):
                    defect_frame = QFrame()
                    defect_frame.setFrameStyle(QFrame.StyledPanel)
                    defect_frame.setStyleSheet("background-color: #ffffff; border: 1px solid #bdc3c7; border-radius: 5px; margin: 2px;")
                    defect_layout = QVBoxLayout(defect_frame)
                    defect_layout.setContentsMargins(8, 8, 8, 8)
                    
                    # Defect type and number
                    defect_title = QLabel(f"Defect #{i}: {defect_type.replace('_', ' ').title()}")
                    defect_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #2c3e50;")
                    defect_layout.addWidget(defect_title)
                    
                    # Size information with visual representation
                    size_text = f"📏 Size: {size_mm:.1f}mm ({percentage:.1f}% of wood width)"
                    size_label = QLabel(size_text)
                    size_label.setStyleSheet("font-size: 11px; color: #34495e; margin-left: 10px;")
                    defect_layout.addWidget(size_label)
                    
                    # Size bar visualization
                    size_bar = QProgressBar()
                    size_bar.setMaximum(100)
                    size_bar.setValue(min(int(percentage), 100))
                    size_bar.setTextVisible(False)
                    size_bar.setFixedHeight(8)
                    
                    # Color code the progress bar based on severity
                    if percentage < 10:
                        bar_color = "#27ae60"  # Green for small defects
                    elif percentage < 25:
                        bar_color = "#f39c12"  # Orange for medium defects
                    else:
                        bar_color = "#e74c3c"  # Red for large defects
                        
                    size_bar.setStyleSheet(f"""
                        QProgressBar {{
                            border: 1px solid #bdc3c7;
                            border-radius: 4px;
                            background-color: #ecf0f1;
                        }}
                        QProgressBar::chunk {{
                            background-color: {bar_color};
                            border-radius: 3px;
                        }}
                    """)
                    defect_layout.addWidget(size_bar)
                    
                    # Individual grade for this defect
                    individual_grade = self.grade_individual_defect(defect_type, size_mm, percentage)
                    grade_color = self.get_grade_color(individual_grade)
                    grade_text = f"⭐ Individual Grade: {individual_grade}"
                    grade_label = QLabel(grade_text)
                    grade_label.setStyleSheet(f"font-size: 11px; font-weight: bold; color: {grade_color}; margin-left: 10px;")
                    defect_layout.addWidget(grade_label)
                    
                    # SS-EN 1611-1 threshold information
                    threshold_info = self.get_threshold_info(defect_type, size_mm, percentage)
                    threshold_label = QLabel(f"📋 Threshold: {threshold_info}")
                    threshold_label.setStyleSheet("font-size: 10px; color: #7f8c8d; margin-left: 10px; font-style: italic;")
                    defect_layout.addWidget(threshold_label)
                    
                    self.defect_details_layout.addWidget(defect_frame)
                
                # Grading summary
                self.add_grading_summary(measurements)
                
                # Update camera status
                self.update_camera_status(camera_name, f"✅ {len(measurements)} defects detected")
                
            elif defect_dict:
                # Simple detection mode (no measurements)
                self.current_grade_label.setText("Final Grade: Simple Detection Mode")
                self.current_grade_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #f39c12; padding: 10px;")
                
                simple_info = QLabel(f"Simple detection for {camera_name.title()} Camera\nTotal defects: {sum(defect_dict.values())}")
                simple_info.setStyleSheet("color: #7f8c8d; padding: 10px;")
                self.defect_details_layout.addWidget(simple_info)
                
                self.update_camera_status(camera_name, f"⚠️ Simple mode: {sum(defect_dict.values())} defects")
                
            else:
                # No defects detected
                self.current_grade_label.setText("Final Grade: Waiting for wood...")
                self.current_grade_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #666; padding: 10px;")
                
                no_defects_label = QLabel("No defects detected\n\nWaiting for wood detection...")
                no_defects_label.setAlignment(Qt.AlignCenter)
                no_defects_label.setStyleSheet("color: #888; font-style: italic; padding: 20px;")
                self.defect_details_layout.addWidget(no_defects_label)
                
                self.update_camera_status(camera_name, "🔄 Ready for detection")
                
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error updating defect analysis panel: {str(e)}", e)

    def determine_surface_grade(self, measurements):
        """Determine surface grade from measurements - simplified version"""
        if not measurements:
            return "G2-0"  # Perfect grade if no defects
        
        # This is a simplified grading - in production you'd use the full SS-EN 1611-1 logic
        total_defects = len(measurements)
        if total_defects > 6:
            return "G2-4"
        elif total_defects > 4:
            return "G2-3"
        elif total_defects > 2:
            return "G2-2"
        else:
            return "G2-1"

    def grade_individual_defect(self, defect_type, size_mm, percentage):
        """Grade individual defect - simplified version"""
        if percentage > 35:
            return "G2-4"
        elif percentage > 25:
            return "G2-3"
        elif percentage > 15:
            return "G2-2"
        elif percentage > 5:
            return "G2-1"
        else:
            return "G2-0"

    def get_grade_color(self, grade):
        """Get color for grade display"""
        color_map = {
            "G2-0": "#27ae60",  # Green
            "G2-1": "#2ecc71",  # Light green
            "G2-2": "#f39c12",  # Orange
            "G2-3": "#e67e22",  # Dark orange
            "G2-4": "#e74c3c"   # Red
        }
        return color_map.get(grade, "#666")

    def get_threshold_info(self, defect_type, size_mm, percentage):
        """Get threshold information for display"""
        # Simplified threshold display
        if defect_type == "Sound_Knot":
            return f"Sound knot limits: G2-0≤10mm, G2-1≤30mm, G2-2≤50mm"
        else:
            return f"Unsound knot limits: G2-0≤7mm, G2-1≤20mm, G2-2≤35mm"

    def add_grading_summary(self, measurements):
        """Add grading summary to defect panel"""
        summary_frame = QFrame()
        summary_frame.setFrameStyle(QFrame.StyledPanel)
        summary_frame.setStyleSheet("background-color: #f8f9fa; border: 2px solid #3498db; border-radius: 5px; margin: 5px;")
        summary_layout = QVBoxLayout(summary_frame)
        
        summary_title = QLabel("📊 SS-EN 1611-1 Grading Summary")
        summary_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #2c3e50; padding: 5px;")
        summary_layout.addWidget(summary_title)
        
        total_defects = len(measurements)
        if total_defects > 6:
            reasoning = "More than 6 defects → Automatic G2-4"
        elif total_defects > 4:
            reasoning = "More than 4 defects → Maximum G2-3"
        elif total_defects > 2:
            reasoning = "More than 2 defects → Maximum G2-2"
        else:
            reasoning = "≤2 defects → Based on individual grades"
        
        reasoning_label = QLabel(f"Reasoning: {reasoning}")
        reasoning_label.setStyleSheet("font-size: 11px; color: #34495e; padding: 3px; margin-left: 10px;")
        summary_layout.addWidget(reasoning_label)
        
        self.defect_details_layout.addWidget(summary_frame)

    def update_camera_status(self, camera_name, status_text):
        """Update camera status in the defect panel"""
        if camera_name == "top":
            self.top_camera_status.setText(f"Top Camera: {status_text}")
        else:
            self.bottom_camera_status.setText(f"Bottom Camera: {status_text}")

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
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_simulated_feed)
        self.timer.start(100) # Update every 100 ms (10 FPS)

    def _update_simulated_feed(self):
        current_time = QDateTime.currentDateTime().toString(Qt.DefaultLocaleLongDate)
        dummy_image = QImage(self.camera_module.camera_width, self.camera_module.camera_height, QImage.Format_RGB32)
        dummy_image.fill(Qt.darkGray)
        
        painter = QPainter(dummy_image)
        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 24))
        painter.drawText(dummy_image.rect(), Qt.AlignCenter, f"SIMULATED FEED\n{current_time}\nDEV MODE")
        painter.end()

        pixmap = QPixmap.fromImage(dummy_image)
        self.top_camera_label.setPixmap(pixmap.scaled(self.top_camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.bottom_camera_label.setPixmap(pixmap.scaled(self.bottom_camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_feeds(self):
        """Enhanced camera feed update with integrated wood detection and error handling"""
        try:
            # In dev_mode, the simulated feed is handled by _update_simulated_feed's QTimer.
            if self.dev_mode:
                QTimer.singleShot(10, self.update_feeds)
                return

            # Read frames from cameras
            ret_top, frame_top = self.camera_module.read_frame("top")
            ret_bottom, frame_bottom = self.camera_module.read_frame("bottom")

            # Process top camera
            if ret_top and frame_top is not None:
                try:
                    # Determine if we should run defect detection
                    run_defect_detection = self._should_run_defect_detection()
                    
                    # Apply ROI if enabled
                    roi_frame_top, roi_info = self.camera_module.apply_roi(
                        frame_top, "top", {"top": True}, 
                        {"top": {"x1": 150, "y1": 80, "x2": 1130, "y2": 640}}
                    )
                    
                    if run_defect_detection:
                        # Run defect detection on ROI frame
                        processed_frame_top, defect_dict_top, measurements_top = self.detection_module.analyze_frame(
                            roi_frame_top if roi_info else frame_top, "top"
                        )
                        
                        # Update defect details and grading if we have detections
                        if defect_dict_top or measurements_top:
                            self.update_defect_details("top", defect_dict_top, measurements_top)
                            
                    else:
                        processed_frame_top = roi_frame_top if roi_info else frame_top
                        
                    # Draw ROI overlay for visualization
                    display_frame_top = self.camera_module.draw_roi_overlay(
                        processed_frame_top, "top", {"top": True},
                        {"top": {"x1": 150, "y1": 80, "x2": 1130, "y2": 640}}
                    )
                    
                    self._display_frame(display_frame_top, self.top_camera_label)
                    
                except Exception as e:
                    log_error(SystemComponent.GUI, f"Error processing top camera frame: {str(e)}", e)
                    self.top_camera_label.setText("Top Camera: Processing Error")
            else:
                self.top_camera_label.setText("Top Camera: Not Available")

            # Process bottom camera
            if ret_bottom and frame_bottom is not None:
                try:
                    run_defect_detection = self._should_run_defect_detection()
                    
                    if run_defect_detection:
                        processed_frame_bottom, defect_dict_bottom, measurements_bottom = self.detection_module.analyze_frame(
                            frame_bottom, "bottom"
                        )
                        
                        if defect_dict_bottom or measurements_bottom:
                            self.update_defect_details("bottom", defect_dict_bottom, measurements_bottom)
                    else:
                        processed_frame_bottom = frame_bottom
                        
                    self._display_frame(processed_frame_bottom, self.bottom_camera_label)
                    
                except Exception as e:
                    log_error(SystemComponent.GUI, f"Error processing bottom camera frame: {str(e)}", e)
                    self.bottom_camera_label.setText("Bottom Camera: Processing Error")
            else:
                self.bottom_camera_label.setText("Bottom Camera: Not Available")

            # Update other UI elements periodically
            current_time = time.time()
            if current_time - getattr(self, '_last_ui_update', 0) > 1.0:  # Update every second
                self._last_ui_update = current_time
                self.update_performance_metrics()
                self.update_recent_activity()
                
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error in update_feeds: {str(e)}", e)
        
        # Schedule next update
        QTimer.singleShot(10, self.update_feeds)

    def _should_run_defect_detection(self):
        """Determine if defect detection should run based on current mode and state"""
        if self.current_mode == "CONTINUOUS":
            return self.live_detection_var
        elif self.current_mode == "TRIGGER":
            return self.auto_detection_active and self.wood_confirmed
        else:  # IDLE
            return False

    def update_defect_details(self, camera_name, defect_dict, measurements):
        """Update defect details display using the new live defect analysis panel"""
        try:
            if not defect_dict and not measurements:
                # Update panel to show no defects
                self.update_defect_analysis_panel(camera_name, {}, [])
                return
            
            # Update the new live defect analysis panel (main display)
            self.update_defect_analysis_panel(camera_name, defect_dict, measurements)
            
            # Also update the defect details tab for historical tracking
            current_details = self.defect_details_text.toPlainText()
            
            timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
            new_details = f"\n[{timestamp}] {camera_name.upper()} Camera:\n"
            
            if measurements:
                for defect_type, size_mm, percentage in measurements:
                    new_details += f"  • {defect_type}: {size_mm:.1f}mm ({percentage:.1f}%)\n"
            elif defect_dict:
                for defect_type, count in defect_dict.items():
                    new_details += f"  • {defect_type}: {count} detected\n"
            
            # Limit text length
            if len(current_details) > 2000:
                lines = current_details.split('\n')
                current_details = '\n'.join(lines[-20:])  # Keep last 20 lines
                
            self.defect_details_text.setText(current_details + new_details)
            
            # Auto-scroll to bottom
            cursor = self.defect_details_text.textCursor()
            cursor.movePosition(cursor.End)
            self.defect_details_text.setTextCursor(cursor)
            
            # Handle grading if in appropriate mode
            if (self.current_mode == "TRIGGER" and self.auto_detection_active and 
                self.wood_confirmed and self.auto_grade_var):
                
                # This is a simplified grading - in a full implementation,
                # you'd collect measurements from both cameras before grading
                if measurements:
                    grade_info = calculate_grade(defect_dict)
                    log_info(SystemComponent.GUI, 
                            f"Calculated grade for {camera_name}: {grade_info}")
            
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error updating defect details: {str(e)}", e)

    def update_performance_metrics(self):
        """Update performance metrics tab"""
        try:
            uptime = time.time() - self.session_start_time
            
            metrics_text = f"""Performance Metrics (Session Uptime: {uptime/3600:.1f} hours)

Processing Statistics:
• Total Pieces Processed: {self.total_pieces_processed}
• Average Processing Rate: {self.total_pieces_processed / max(uptime/60, 1):.1f} pieces/minute

Camera System:
• Available Cameras: {len(self.camera_module.get_available_cameras())}/2
• Camera Health: {self.camera_module.get_system_health().get('status', 'Unknown')}

Arduino System:
• Connection Status: {'Connected' if self.arduino_module.is_connected() else 'Disconnected'}

Error Summary:
{get_error_summary().get('total_errors', 0)} total errors this session
"""
            
            self.performance_text.setText(metrics_text)
            
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error updating performance metrics: {str(e)}", e)

    def update_live_stats_display(self):
        """Update the live statistics display safely"""
        try:
            for grade_key, label in self.live_stats_labels.items():
                count = self.live_stats.get(grade_key, 0)
                label.setText(str(count))
                
            log_info(SystemComponent.GUI, f"Updated live stats: {self.live_stats}")
            
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error updating live stats display: {str(e)}", e)

    def update_recent_activity(self):
        """Update recent activity tab"""
        try:
            activity_lines = []
            
            # Add recent session log entries
            for log_entry in self.session_log[-10:]:  # Last 10 entries
                activity_lines.append(
                    f"[{log_entry.get('timestamp', 'Unknown')}] "
                    f"Piece #{log_entry.get('piece_number', '?')}: "
                    f"Grade {log_entry.get('final_grade', 'Unknown')}"
                )
                
                defects = log_entry.get('defects', [])
                if defects:
                    for defect in defects:
                        activity_lines.append(
                            f"    → {defect.get('type', 'Unknown')}: "
                            f"{defect.get('count', 0)} defects, "
                            f"sizes: {defect.get('sizes', 'N/A')}"
                        )
                else:
                    activity_lines.append("    → No defects detected")
                    
            if not activity_lines:
                activity_lines = ["No pieces processed in this session."]
                
            self.activity_text.setText("\n".join(activity_lines))
            
            # Auto-scroll to bottom
            cursor = self.activity_text.textCursor()
            cursor.movePosition(cursor.End)
            self.activity_text.setTextCursor(cursor)
            
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error updating recent activity: {str(e)}", e)

    def _display_frame(self, frame, label):
        # Convert OpenCV image to QPixmap
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(QPixmap.fromImage(p))

    def set_continuous_mode(self):
        """Set system to continuous mode with enhanced logging and checkbox activation"""
        try:
            log_info(SystemComponent.GUI, "Setting Continuous Mode")
            self.current_mode = "CONTINUOUS"
            
            # Block signals to prevent toggle functions from interfering
            self.live_detect_checkbox.blockSignals(True)
            self.auto_grade_checkbox.blockSignals(True)
            
            # Update checkbox states for Continuous mode (both enabled and checked)
            self.live_detect_checkbox.setEnabled(True)
            self.auto_grade_checkbox.setEnabled(True)
            self.live_detect_checkbox.setChecked(True)
            self.auto_grade_checkbox.setChecked(True)
            
            # Unblock signals
            self.live_detect_checkbox.blockSignals(False)
            self.auto_grade_checkbox.blockSignals(False)
            
            # Update internal state variables
            self.live_detection_var = True
            self.auto_grade_var = True
            
            self.status_label.setText("Status: CONTINUOUS MODE - Live detection & auto-grading enabled")
            
            # Reset IR-triggered state
            self.ir_triggered = False
            self.wood_confirmed = False
            self.auto_detection_active = False
            
            # Send command to Arduino
            success = self.arduino_module.send_arduino_command('C')
            if not success:
                log_warning(SystemComponent.GUI, "Failed to send continuous mode command to Arduino")
            else:
                log_info(SystemComponent.GUI, "Continuous mode command sent to Arduino successfully")
                
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error setting continuous mode: {str(e)}", e)

    def set_trigger_mode(self):
        """Set system to trigger mode with enhanced logging and checkbox deactivation"""
        try:
            log_info(SystemComponent.GUI, "Setting Trigger Mode")
            self.current_mode = "TRIGGER"
            
            # Block signals to prevent toggle functions from interfering
            self.live_detect_checkbox.blockSignals(True)
            self.auto_grade_checkbox.blockSignals(True)
            
            # Update checkbox states for Trigger mode (both enabled but unchecked - will be activated by IR trigger)
            self.live_detect_checkbox.setEnabled(True)
            self.auto_grade_checkbox.setEnabled(True)
            self.live_detect_checkbox.setChecked(False)
            self.auto_grade_checkbox.setChecked(False)
            
            # Unblock signals
            self.live_detect_checkbox.blockSignals(False)
            self.auto_grade_checkbox.blockSignals(False)
            
            # Update internal state variables
            self.live_detection_var = False
            self.auto_grade_var = False
            
            # Reset state
            self.ir_triggered = False
            self.wood_confirmed = False
            self.auto_detection_active = False
            
            self.status_label.setText("Status: TRIGGER MODE - Waiting for IR beam trigger")
            
            # Send command to Arduino
            success = self.arduino_module.send_arduino_command('T')
            if not success:
                log_warning(SystemComponent.GUI, "Failed to send trigger mode command to Arduino")
            else:
                log_info(SystemComponent.GUI, "Trigger mode command sent to Arduino successfully")
                
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error setting trigger mode: {str(e)}", e)

    def set_idle_mode(self):
        """Set system to idle mode with enhanced logging and checkbox deactivation"""
        try:
            log_info(SystemComponent.GUI, "Setting IDLE Mode")
            self.current_mode = "IDLE"
            
            # Block signals to prevent toggle functions from interfering
            self.live_detect_checkbox.blockSignals(True)
            self.auto_grade_checkbox.blockSignals(True)
            
            # Update checkbox states for IDLE mode (both unchecked but still enabled)
            self.live_detect_checkbox.setChecked(False)
            self.auto_grade_checkbox.setChecked(False)
            self.live_detect_checkbox.setEnabled(True)
            self.auto_grade_checkbox.setEnabled(True)
            
            # Unblock signals
            self.live_detect_checkbox.blockSignals(False)
            self.auto_grade_checkbox.blockSignals(False)
            
            # Update internal state variables
            self.live_detection_var = False
            self.auto_grade_var = False
            
            # Reset state
            self.ir_triggered = False
            self.wood_confirmed = False
            self.auto_detection_active = False
            
            self.status_label.setText("Status: IDLE MODE - System disabled, conveyor stopped")
            
            # Send command to Arduino
            success = self.arduino_module.send_arduino_command('X')
            if not success:
                log_warning(SystemComponent.GUI, "Failed to send idle mode command to Arduino")
            else:
                log_info(SystemComponent.GUI, "IDLE mode command sent to Arduino successfully")
                
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error setting idle mode: {str(e)}", e)
                
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error setting trigger mode: {str(e)}", e)

    def finalize_grading(self, final_grade, all_measurements):
        """Enhanced grading finalization with comprehensive logging"""
        try:
            # Convert grade to Arduino command for sorting
            arduino_command = self.arduino_module.convert_grade_to_arduino_command(final_grade)

            # Increment piece count and create log entry
            self.total_pieces_processed += 1
            piece_number = self.total_pieces_processed
            
            # Log to reporting module
            self.reporting_module.finalize_grading_log(final_grade, all_measurements, piece_number)

            # Update UI statistics
            self.grade_counts[arduino_command] += 1
            self.live_stats[f"grade{arduino_command}"] += 1
            self.update_live_stats_display()

            # Send command to Arduino if it's connected
            if not self.dev_mode and self.arduino_module.is_connected():
                success = self.arduino_module.send_grade_command(final_grade)
                if not success:
                    log_error(SystemComponent.GUI, f"Failed to send grade command for piece #{piece_number}")
            else:
                log_info(SystemComponent.GUI, f"DEV MODE: Mock Arduino command '{arduino_command}' for grade '{final_grade}'")

            # Update status and log
            status_text = f"Piece #{piece_number} Graded: {final_grade} (Cmd: {arduino_command})"
            self.status_label.setText(f"Status: {status_text}")
            
            log_info(SystemComponent.GUI, f"Grading finalized - {status_text}")
            self.reporting_module.log_action(f"Graded Piece #{piece_number} as {final_grade} -> Arduino Cmd: {arduino_command}")
            
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error finalizing grading: {str(e)}", e)

    def closeEvent(self, event):
        """Enhanced cleanup on application close"""
        try:
            log_info(SystemComponent.GUI, "Application closing - starting cleanup")
            
            # Stop error monitoring
            self.error_monitoring_active = False
            
            # Release camera resources
            self.camera_module.release_cameras()
            
            # Close Arduino connection
            self.arduino_module.close_connection()
            
            # Generate final report if pieces were processed
            if self.total_pieces_processed > 0:
                log_info(SystemComponent.GUI, "Generating final report before shutdown")
                self.reporting_module.generate_report()
            
            log_info(SystemComponent.GUI, "Application cleanup completed")
            event.accept()
            
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error during application cleanup: {str(e)}", e)
            event.accept()  # Accept anyway to ensure app closes

    def toggle_roi(self):
        """Toggle ROI for top camera with enhanced functionality"""
        try:
            roi_enabled = self.roi_checkbox.isChecked()
            # Update detection module ROI settings
            if hasattr(self.detection_module, 'set_roi_enabled'):
                self.detection_module.set_roi_enabled("top", roi_enabled)
            
            log_info(SystemComponent.GUI, f"ROI for top camera: {'enabled' if roi_enabled else 'disabled'}")
            
            # Update status
            roi_status = "Active" if roi_enabled else "Disabled"
            self.status_label.setText(f"Status: {self.current_mode} MODE - ROI {roi_status}")
            
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error toggling ROI: {str(e)}", e)

    def toggle_live_detection_mode(self):
        """Toggle live detection mode with checkbox synchronization"""
        try:
            # Get state from checkbox
            self.live_detection_var = self.live_detect_checkbox.isChecked()
            log_info(SystemComponent.GUI, f"Live detection mode toggled: {self.live_detection_var}")
            
            # Update status display
            if self.live_detection_var:
                self.status_label.setText(f"Status: {self.current_mode} MODE - Live detection ACTIVE")
            else:
                self.status_label.setText(f"Status: {self.current_mode} MODE - Live detection DISABLED")
                
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error toggling live detection mode: {str(e)}", e)

    def toggle_auto_grade(self):
        """Toggle auto grade mode with checkbox synchronization"""
        try:
            # Get state from checkbox
            self.auto_grade_var = self.auto_grade_checkbox.isChecked()
            log_info(SystemComponent.GUI, f"Auto grade mode toggled: {self.auto_grade_var}")
            
            # Update status display
            auto_grade_status = "ENABLED" if self.auto_grade_var else "DISABLED"
            current_status = self.status_label.text()
            
            # Update status to show auto-grade state
            if "Auto-grading" not in current_status:
                self.status_label.setText(f"{current_status} - Auto-grading {auto_grade_status}")
                
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error toggling auto grade mode: {str(e)}", e)
                
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error toggling live detection: {str(e)}", e)

    def manual_generate_report(self):
        """Generate report manually with enhanced error handling"""
        try:
            log_info(SystemComponent.GUI, "Manual report generation requested")
            self.reporting_module.generate_report()
            self.status_label.setText("Status: Report generated successfully")
            
            # Update last report label if possible
            if hasattr(self.reporting_module, 'last_report_path') and self.reporting_module.last_report_path:
                import os
                report_name = os.path.basename(self.reporting_module.last_report_path)
                self.last_report_label.setText(f"Last: {report_name}")
                
        except Exception as e:
            log_error(SystemComponent.GUI, f"Error generating manual report: {str(e)}", e)
            self.status_label.setText("Status: Report generation failed")

    def update_live_stats_display(self):
        """Update the live statistics display with enhanced tabbed interface and thread safety."""
        # Skip update if currently in active inference to prevent UI conflicts
        # (self._in_active_inference will be managed by the main processing loop)
        # if getattr(self, '_in_active_inference', False):
        #     return
            
        # Safety check to ensure all required attributes exist
        if not hasattr(self, 'live_stats'):
            self.live_stats = {"grade0": 0, "grade1": 0, "grade2": 0, "grade3": 0}
        if not hasattr(self, 'live_stats_labels'):
            return  # Skip update if labels aren't initialized yet
            
        # Update basic grade counts in the Grade Summary tab with error handling
        try:
            for grade_key, count in self.live_stats.items():
                if grade_key in self.live_stats_labels: # No winfo_exists() in PyQt
                    # Use after_idle to ensure UI updates happen on main thread
                    self.live_stats_labels[grade_key].setText(str(count))
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
        """Safely update a label with error handling (PyQt version)."""
        try:
            if grade_key in self.live_stats_labels:
                self.live_stats_labels[grade_key].setText(str(count))
        except Exception as e:
            print(f"Error updating label {grade_key}: {e}")

    def update_defect_details_tab(self):
        """Update the Defect Details tab with current defect information."""
        # This will require access to live_measurements, which will be populated by detection_module
        # For now, a placeholder.
        print("Update Defect Details Tab (implemented later)")

    def update_performance_tab(self):
        """Update the Performance Metrics tab."""
        # This will require access to total_pieces_processed, session_start_time, grade_counts
        # For now, a placeholder.
        print("Update Performance Tab (implemented later)")

    def update_recent_activity_tab(self):
        """Update the Recent Activity tab with widened summary and scrollable processing log."""
        # This will require access to session_log, total_pieces_processed, session_start_time, grade_counts
        # For now, a placeholder.
        print("Update Recent Activity Tab (implemented later)")

    def _generate_stats_content(self):
        """Generate a string representation of current stats for change detection."""
        content = f"processed:{getattr(self, 'total_pieces_processed', 0)}"
        
        grade_counts = getattr(self, 'grade_counts', {0: 0, 1: 0, 2: 0, 3: 0})
        for grade, count in grade_counts.items():
            content += f",g{grade}:{count}"
        
        # Include session log count for change detection
        if hasattr(self, 'session_log'):
            content += f",log_entries:{len(self.session_log)}"
                
        return content

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = WoodSortingApp(dev_mode=True)
    main_app.show()
    sys.exit(app.exec_())