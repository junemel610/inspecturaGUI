#!/usr/bin/env python3
"""
Enhanced Error Handler with Modal Notifications for Wood Sorting System

This module provides comprehensive error handling specifically designed for 
CASE 3.1 System Robustness Under Abnormal Conditions, including:
- Detection failures and misclassifications  
- Communication errors
- Wood plank misalignment
- Modal notifications for operator alerts
- Automatic system pause and recovery mechanisms
"""

import logging
import os
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, Callable
from PyQt5.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QIcon, QPixmap

class AbnormalCondition(Enum):
    """Specific abnormal conditions for CASE 3.1 testing"""
    NO_WOOD_PRESENT = "NO_WOOD_PRESENT"
    CAMERA_FEED_BLOCKED = "CAMERA_FEED_BLOCKED"
    LOW_CONFIDENCE_DETECTION = "LOW_CONFIDENCE_DETECTION"
    WOOD_MISALIGNMENT = "WOOD_MISALIGNMENT"
    COMMUNICATION_FAILURE = "COMMUNICATION_FAILURE"
    DETECTION_FAILURE = "DETECTION_FAILURE"
    CLASSIFICATION_ERROR = "CLASSIFICATION_ERROR"
    SORTING_MECHANISM_FAILURE = "SORTING_MECHANISM_FAILURE"

class ErrorSeverity(Enum):
    """Error severity levels for the wood sorting system"""
    INFO = "INFO"
    WARNING = "WARNING" 
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class SystemComponent(Enum):
    """System components for error tracking"""
    CAMERA = "CAMERA"
    DETECTION = "DETECTION"
    ARDUINO = "ARDUINO"
    GRADING = "GRADING"
    GUI = "GUI"
    REPORTING = "REPORTING"
    SORTING = "SORTING"
    GENERAL = "GENERAL"

class SystemResponse(Enum):
    """System response actions for abnormal conditions"""
    PAUSE_SYSTEM = "PAUSE_SYSTEM"
    ALERT_OPERATOR = "ALERT_OPERATOR"
    REROUTE_FOR_MANUAL = "REROUTE_FOR_MANUAL"
    RETRY_OPERATION = "RETRY_OPERATION"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    CONTINUE_WITH_WARNING = "CONTINUE_WITH_WARNING"

class AbnormalConditionDialog(QDialog):
    """Modal dialog for abnormal condition notifications"""
    
    action_selected = pyqtSignal(str)
    
    def __init__(self, condition: AbnormalCondition, details: str, parent=None):
        super().__init__(parent)
        self.condition = condition
        self.details = details
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the modal dialog UI"""
        self.setWindowTitle("System Abnormal Condition Detected")
        self.setModal(True)
        self.setFixedSize(600, 400)
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title with icon
        title_layout = QHBoxLayout()
        
        # Warning icon (you could add an actual icon file here)
        icon_label = QLabel("⚠️")
        icon_label.setStyleSheet("font-size: 48px;")
        title_layout.addWidget(icon_label)
        
        title_label = QLabel("ABNORMAL CONDITION DETECTED")
        title_label.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold; 
            color: #d32f2f;
            margin-left: 10px;
        """)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        layout.addLayout(title_layout)
        
        # Condition type
        condition_label = QLabel(f"Condition: {self.get_condition_description()}")
        condition_label.setStyleSheet("""
            font-size: 14px; 
            font-weight: bold; 
            padding: 10px; 
            background-color: #ffebee; 
            border-left: 4px solid #d32f2f;
        """)
        layout.addWidget(condition_label)
        
        # Details
        details_label = QLabel("Details:")
        details_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(details_label)
        
        details_text = QTextEdit()
        details_text.setPlainText(self.details)
        details_text.setReadOnly(True)
        details_text.setMaximumHeight(100)
        details_text.setStyleSheet("""
            font-size: 11px; 
            background-color: #f5f5f5; 
            border: 1px solid #ccc;
        """)
        layout.addWidget(details_text)
        
        # Recommended action
        action_label = QLabel("Recommended System Response:")
        action_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(action_label)
        
        response_text = QLabel(self.get_recommended_response())
        response_text.setStyleSheet("""
            font-size: 11px; 
            padding: 8px; 
            background-color: #e3f2fd; 
            border-left: 4px solid #1976d2;
        """)
        response_text.setWordWrap(True)
        layout.addWidget(response_text)
        
        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Pause system button
        pause_btn = QPushButton("Pause System")
        pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800; 
                color: white; 
                font-weight: bold; 
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
        """)
        pause_btn.clicked.connect(lambda: self.select_action("PAUSE_SYSTEM"))
        button_layout.addWidget(pause_btn)
        
        # Manual inspection button
        manual_btn = QPushButton("Manual Inspection")
        manual_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196f3; 
                color: white; 
                font-weight: bold; 
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
        """)
        manual_btn.clicked.connect(lambda: self.select_action("REROUTE_FOR_MANUAL"))
        button_layout.addWidget(manual_btn)
        
        # Retry button
        retry_btn = QPushButton("Retry Operation")
        retry_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50; 
                color: white; 
                font-weight: bold; 
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
        """)
        retry_btn.clicked.connect(lambda: self.select_action("RETRY_OPERATION"))
        button_layout.addWidget(retry_btn)
        
        # Emergency stop for critical conditions
        if self.condition in [AbnormalCondition.COMMUNICATION_FAILURE, 
                             AbnormalCondition.SORTING_MECHANISM_FAILURE]:
            emergency_btn = QPushButton("Emergency Stop")
            emergency_btn.setStyleSheet("""
                QPushButton {
                    background-color: #d32f2f; 
                    color: white; 
                    font-weight: bold; 
                    padding: 8px 16px;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #b71c1c;
                }
            """)
            emergency_btn.clicked.connect(lambda: self.select_action("EMERGENCY_STOP"))
            button_layout.addWidget(emergency_btn)
        
        layout.addLayout(button_layout)
        
    def get_condition_description(self) -> str:
        """Get human-readable description of the condition"""
        descriptions = {
            AbnormalCondition.NO_WOOD_PRESENT: "No Wood Plank Detected on Conveyor",
            AbnormalCondition.CAMERA_FEED_BLOCKED: "Camera Feed Disconnected or Blocked",
            AbnormalCondition.LOW_CONFIDENCE_DETECTION: "Low Confidence Detection (Poor Image Quality)",
            AbnormalCondition.WOOD_MISALIGNMENT: "Wood Plank Misalignment Detected",
            AbnormalCondition.COMMUNICATION_FAILURE: "Communication Error with Sorting Mechanism",
            AbnormalCondition.DETECTION_FAILURE: "Detection Algorithm Failure",
            AbnormalCondition.CLASSIFICATION_ERROR: "Knot Classification Error",
            AbnormalCondition.SORTING_MECHANISM_FAILURE: "Sorting Mechanism Hardware Failure"
        }
        return descriptions.get(self.condition, "Unknown Condition")
        
    def get_recommended_response(self) -> str:
        """Get recommended system response for the condition"""
        responses = {
            AbnormalCondition.NO_WOOD_PRESENT: "System should pause and wait for wood plank placement. Alert operator if condition persists.",
            AbnormalCondition.CAMERA_FEED_BLOCKED: "System should pause immediately. Check camera connections and clear any obstructions.",
            AbnormalCondition.LOW_CONFIDENCE_DETECTION: "Reroute plank for manual inspection. Check lighting conditions and camera positioning.",
            AbnormalCondition.WOOD_MISALIGNMENT: "Attempt realignment or reroute for manual handling. Check conveyor positioning.",
            AbnormalCondition.COMMUNICATION_FAILURE: "Pause system and check Arduino/sorting mechanism connections. May require emergency stop.",
            AbnormalCondition.DETECTION_FAILURE: "Retry detection with different parameters. If failure persists, reroute for manual inspection.",
            AbnormalCondition.CLASSIFICATION_ERROR: "Reroute plank for manual classification. Review detection confidence thresholds.",
            AbnormalCondition.SORTING_MECHANISM_FAILURE: "Emergency stop required. Check hardware connections and mechanical components."
        }
        return responses.get(self.condition, "Contact system administrator.")
        
    def select_action(self, action: str):
        """Handle action selection"""
        self.action_selected.emit(action)
        self.accept()

class EnhancedErrorHandler(QObject):
    """Enhanced error handler with modal notifications for abnormal conditions"""
    
    abnormal_condition_detected = pyqtSignal(AbnormalCondition, str)
    system_paused = pyqtSignal()
    system_resumed = pyqtSignal()
    
    def __init__(self, log_directory="logs", parent=None):
        super().__init__(parent)
        self.log_directory = log_directory
        self.setup_logging()
        self.error_counts = {}
        self.last_errors = {}
        self.system_paused = False
        self.abnormal_conditions_history = []
        self.parent_widget = None
        
        # Thresholds for triggering abnormal conditions
        self.confidence_threshold = 0.6
        self.no_wood_timeout = 30  # seconds
        self.communication_retry_limit = 3
        
    def set_parent_widget(self, parent):
        """Set parent widget for modal dialogs"""
        self.parent_widget = parent
        
    def setup_logging(self):
        """Setup centralized logging system"""
        try:
            os.makedirs(self.log_directory, exist_ok=True)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Setup file handler for all logs
            log_file = os.path.join(self.log_directory, "system.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            
            # Setup file handler for abnormal conditions
            abnormal_file = os.path.join(self.log_directory, "abnormal_conditions.log")
            abnormal_handler = logging.FileHandler(abnormal_file)
            abnormal_handler.setLevel(logging.ERROR)
            abnormal_handler.setFormatter(formatter)
            
            # Setup console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            
            # Configure logger
            self.logger = logging.getLogger("WoodSortingEnhanced")
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(abnormal_handler)
            self.logger.addHandler(console_handler)
            
            self.logger.info("Enhanced error handling system initialized")
            
        except Exception as e:
            print(f"Failed to setup logging: {e}")
            self.logger = logging.getLogger("WoodSortingEnhanced")
            
    def detect_abnormal_condition(self, 
                                 condition: AbnormalCondition, 
                                 details: str,
                                 context: Optional[Dict[str, Any]] = None):
        """Detect and handle abnormal conditions with modal notification"""
        
        # Log the condition
        self.logger.error(f"ABNORMAL CONDITION: {condition.value} - {details}")
        
        # Record in history
        condition_record = {
            'condition': condition,
            'details': details,
            'timestamp': datetime.now(),
            'context': context or {}
        }
        self.abnormal_conditions_history.append(condition_record)
        
        # Show modal notification if parent widget is available
        if self.parent_widget:
            dialog = AbnormalConditionDialog(condition, details, self.parent_widget)
            dialog.action_selected.connect(self.handle_operator_response)
            dialog.exec_()
        else:
            # Fallback: emit signal for handling elsewhere
            self.abnormal_condition_detected.emit(condition, details)
            
    def handle_operator_response(self, action: str):
        """Handle operator response to abnormal condition"""
        self.logger.info(f"Operator selected action: {action}")
        
        if action == "PAUSE_SYSTEM":
            self.pause_system()
        elif action == "REROUTE_FOR_MANUAL":
            self.reroute_for_manual_inspection()
        elif action == "RETRY_OPERATION":
            self.retry_current_operation()
        elif action == "EMERGENCY_STOP":
            self.emergency_stop()
            
    def pause_system(self):
        """Pause the system operation"""
        self.system_paused = True
        self.logger.warning("System paused due to abnormal condition")
        self.system_paused.emit()
        
    def resume_system(self):
        """Resume system operation"""
        self.system_paused = False
        self.logger.info("System resumed from pause")
        self.system_resumed.emit()
        
    def reroute_for_manual_inspection(self):
        """Reroute current plank for manual inspection"""
        self.logger.info("Plank rerouted for manual inspection")
        # This would trigger hardware signals to reroute the plank
        
    def retry_current_operation(self):
        """Retry the current operation"""
        self.logger.info("Retrying current operation")
        # This would restart the detection/classification process
        
    def emergency_stop(self):
        """Emergency stop of all system operations"""
        self.system_paused = True
        self.logger.critical("EMERGENCY STOP activated")
        # This would immediately stop all hardware operations
        
    # Specific detection methods for CASE 3.1 conditions
    
    def check_wood_presence(self, detection_result) -> bool:
        """Check if wood is present and detect NO_WOOD_PRESENT condition"""
        if not detection_result or not detection_result.get('wood_present', False):
            self.detect_abnormal_condition(
                AbnormalCondition.NO_WOOD_PRESENT,
                "No wood plank detected on conveyor belt. System waiting for wood placement.",
                {'detection_result': detection_result}
            )
            return False
        return True
        
    def check_camera_feed(self, frame) -> bool:
        """Check camera feed quality and detect CAMERA_FEED_BLOCKED condition"""
        if frame is None:
            self.detect_abnormal_condition(
                AbnormalCondition.CAMERA_FEED_BLOCKED,
                "Camera feed is disconnected or blocked. No image data received.",
                {'frame_status': 'None'}
            )
            return False
            
        # Check if frame is mostly black (blocked camera)
        import numpy as np
        if np.mean(frame) < 10:  # Very dark image
            self.detect_abnormal_condition(
                AbnormalCondition.CAMERA_FEED_BLOCKED,
                "Camera feed appears to be blocked or very dark. Check for obstructions.",
                {'mean_intensity': np.mean(frame)}
            )
            return False
            
        return True
        
    def check_detection_confidence(self, defects) -> bool:
        """Check detection confidence and detect LOW_CONFIDENCE_DETECTION condition"""
        if not defects:
            return True
            
        low_confidence_defects = [d for d in defects if d.get('confidence', 1.0) < self.confidence_threshold]
        
        if low_confidence_defects:
            confidence_values = [d.get('confidence', 0) for d in low_confidence_defects]
            avg_confidence = sum(confidence_values) / len(confidence_values)
            
            self.detect_abnormal_condition(
                AbnormalCondition.LOW_CONFIDENCE_DETECTION,
                f"Low confidence detection detected. Average confidence: {avg_confidence:.2f}, "
                f"Threshold: {self.confidence_threshold}. Poor image quality or unclear defects.",
                {
                    'low_confidence_count': len(low_confidence_defects),
                    'average_confidence': avg_confidence,
                    'threshold': self.confidence_threshold
                }
            )
            return False
            
        return True
        
    def check_wood_alignment(self, detection_result) -> bool:
        """Check wood alignment and detect WOOD_MISALIGNMENT condition"""
        if not detection_result:
            return True
            
        # Check if wood piece is significantly off-center or rotated
        wood_bbox = detection_result.get('wood_bbox')
        if wood_bbox:
            center_x = (wood_bbox[0] + wood_bbox[2]) / 2
            center_y = (wood_bbox[1] + wood_bbox[3]) / 2
            
            # Expected center (assuming 1920x1080 frame)
            expected_center_x = 960
            expected_center_y = 540
            
            # Calculate displacement
            displacement_x = abs(center_x - expected_center_x)
            displacement_y = abs(center_y - expected_center_y)
            
            # Thresholds for misalignment (in pixels)
            max_displacement_x = 200
            max_displacement_y = 150
            
            if displacement_x > max_displacement_x or displacement_y > max_displacement_y:
                self.detect_abnormal_condition(
                    AbnormalCondition.WOOD_MISALIGNMENT,
                    f"Wood plank misalignment detected. Displacement: X={displacement_x:.1f}px, "
                    f"Y={displacement_y:.1f}px. Thresholds: X={max_displacement_x}px, Y={max_displacement_y}px.",
                    {
                        'displacement_x': displacement_x,
                        'displacement_y': displacement_y,
                        'wood_center': (center_x, center_y),
                        'expected_center': (expected_center_x, expected_center_y)
                    }
                )
                return False
                
        return True
        
    def check_communication_status(self, arduino_response) -> bool:
        """Check Arduino communication and detect COMMUNICATION_FAILURE condition"""
        if arduino_response is None:
            self.detect_abnormal_condition(
                AbnormalCondition.COMMUNICATION_FAILURE,
                "Arduino communication failure. No response received from sorting mechanism.",
                {'response': None}
            )
            return False
            
        if isinstance(arduino_response, str) and "ERROR" in arduino_response.upper():
            self.detect_abnormal_condition(
                AbnormalCondition.COMMUNICATION_FAILURE,
                f"Arduino communication error: {arduino_response}",
                {'response': arduino_response}
            )
            return False
            
        return True
        
    def check_classification_validity(self, grade_result) -> bool:
        """Check classification validity and detect CLASSIFICATION_ERROR condition"""
        if not grade_result:
            self.detect_abnormal_condition(
                AbnormalCondition.CLASSIFICATION_ERROR,
                "Classification failed to produce valid result.",
                {'grade_result': None}
            )
            return False
            
        grade = grade_result.get('grade')
        confidence = grade_result.get('confidence', 0)
        
        # Check if grade is valid (2-0, 2-1, 2-2, 2-3, 2-4)
        valid_grades = ['2-0', '2-1', '2-2', '2-3', '2-4']
        
        if grade not in valid_grades:
            self.detect_abnormal_condition(
                AbnormalCondition.CLASSIFICATION_ERROR,
                f"Invalid grade classification: '{grade}'. Expected one of: {valid_grades}",
                {'invalid_grade': grade, 'valid_grades': valid_grades}
            )
            return False
            
        # Check classification confidence
        if confidence < 0.5:
            self.detect_abnormal_condition(
                AbnormalCondition.CLASSIFICATION_ERROR,
                f"Low classification confidence: {confidence:.2f}. Grade: {grade}",
                {'grade': grade, 'confidence': confidence}
            )
            return False
            
        return True
        
    def get_abnormal_conditions_summary(self) -> Dict[str, Any]:
        """Get summary of all abnormal conditions for monitoring"""
        recent_conditions = [
            c for c in self.abnormal_conditions_history 
            if (datetime.now() - c['timestamp']).total_seconds() < 3600  # Last hour
        ]
        
        condition_counts = {}
        for condition in recent_conditions:
            condition_type = condition['condition'].value
            condition_counts[condition_type] = condition_counts.get(condition_type, 0) + 1
            
        return {
            'total_conditions': len(self.abnormal_conditions_history),
            'recent_conditions': len(recent_conditions),
            'condition_counts': condition_counts,
            'system_paused': self.system_paused,
            'last_condition': self.abnormal_conditions_history[-1] if self.abnormal_conditions_history else None
        }

# Global enhanced error handler instance
enhanced_error_handler = EnhancedErrorHandler()

# Convenience functions for abnormal condition detection
def check_wood_presence(detection_result) -> bool:
    return enhanced_error_handler.check_wood_presence(detection_result)

def check_camera_feed(frame) -> bool:
    return enhanced_error_handler.check_camera_feed(frame)

def check_detection_confidence(defects) -> bool:
    return enhanced_error_handler.check_detection_confidence(defects)

def check_wood_alignment(detection_result) -> bool:
    return enhanced_error_handler.check_wood_alignment(detection_result)

def check_communication_status(arduino_response) -> bool:
    return enhanced_error_handler.check_communication_status(arduino_response)

def check_classification_validity(grade_result) -> bool:
    return enhanced_error_handler.check_classification_validity(grade_result)

def get_abnormal_conditions_summary():
    return enhanced_error_handler.get_abnormal_conditions_summary()

def set_error_handler_parent(parent):
    enhanced_error_handler.set_parent_widget(parent)
