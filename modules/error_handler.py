import logging
import os
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any

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
    GENERAL = "GENERAL"

class ErrorHandler:
    """Centralized error handling and logging system"""
    
    def __init__(self, log_directory="wood_sorting_app/logs"):
        self.log_directory = log_directory
        self.setup_logging()
        self.error_counts = {}
        self.last_errors = {}
        
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
            
            # Setup file handler for errors only
            error_file = os.path.join(self.log_directory, "errors.log")
            error_handler = logging.FileHandler(error_file)
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            
            # Setup console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            
            # Configure root logger
            self.logger = logging.getLogger("WoodSortingSystem")
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(error_handler)
            self.logger.addHandler(console_handler)
            
            self.logger.info("Error handling system initialized")
            
        except Exception as e:
            print(f"Failed to setup logging: {e}")
            # Fallback: create a basic logger
            self.logger = logging.getLogger("WoodSortingSystem")
            
    def log_error(self, 
                  component: SystemComponent, 
                  severity: ErrorSeverity,
                  message: str,
                  exception: Optional[Exception] = None,
                  context: Optional[Dict[str, Any]] = None):
        """Log an error with context information"""
        
        # Build detailed error message
        error_msg = f"[{component.value}] {message}"
        
        if context:
            context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
            error_msg += f" (Context: {context_str})"
            
        if exception:
            error_msg += f" (Exception: {str(exception)})"
        
        # Log based on severity
        if severity == ErrorSeverity.INFO:
            self.logger.info(error_msg)
        elif severity == ErrorSeverity.WARNING:
            self.logger.warning(error_msg)
        elif severity == ErrorSeverity.ERROR:
            self.logger.error(error_msg)
        elif severity == ErrorSeverity.CRITICAL:
            self.logger.critical(error_msg)
            
        # Track error counts
        error_key = f"{component.value}_{severity.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.last_errors[component.value] = {
            'severity': severity.value,
            'message': message,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    def log_camera_error(self, camera_name: str, error_msg: str, exception: Optional[Exception] = None):
        """Specific logging for camera errors"""
        self.log_error(
            SystemComponent.CAMERA,
            ErrorSeverity.ERROR,
            f"Camera '{camera_name}': {error_msg}",
            exception,
            {"camera": camera_name}
        )
        
    def log_detection_error(self, error_msg: str, exception: Optional[Exception] = None):
        """Specific logging for detection errors"""
        self.log_error(
            SystemComponent.DETECTION,
            ErrorSeverity.ERROR,
            error_msg,
            exception
        )
        
    def log_arduino_error(self, error_msg: str, exception: Optional[Exception] = None):
        """Specific logging for Arduino communication errors"""
        self.log_error(
            SystemComponent.ARDUINO,
            ErrorSeverity.ERROR,
            error_msg,
            exception
        )
        
    def log_info(self, component: SystemComponent, message: str, context: Optional[Dict[str, Any]] = None):
        """Log informational message"""
        self.log_error(component, ErrorSeverity.INFO, message, None, context)
        
    def log_warning(self, component: SystemComponent, message: str, context: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self.log_error(component, ErrorSeverity.WARNING, message, None, context)
        
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors for system monitoring"""
        return {
            'error_counts': self.error_counts.copy(),
            'last_errors': self.last_errors.copy(),
            'total_errors': sum(count for key, count in self.error_counts.items() if 'ERROR' in key or 'CRITICAL' in key)
        }
        
    def clear_error_counts(self):
        """Clear error counts (useful for session resets)"""
        self.error_counts.clear()
        self.last_errors.clear()
        self.log_info(SystemComponent.GENERAL, "Error counts cleared")

# Global error handler instance
error_handler = ErrorHandler()

# Convenience functions for easy access
def log_camera_error(camera_name: str, error_msg: str, exception: Optional[Exception] = None):
    error_handler.log_camera_error(camera_name, error_msg, exception)

def log_detection_error(error_msg: str, exception: Optional[Exception] = None):
    error_handler.log_detection_error(error_msg, exception)

def log_arduino_error(error_msg: str, exception: Optional[Exception] = None):
    error_handler.log_arduino_error(error_msg, exception)

def log_info(component: SystemComponent, message: str, context: Optional[Dict[str, Any]] = None):
    error_handler.log_info(component, message, context)

def log_warning(component: SystemComponent, message: str, context: Optional[Dict[str, Any]] = None):
    error_handler.log_warning(component, message, context)

def log_error(component: SystemComponent, message: str, exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
    error_handler.log_error(component, ErrorSeverity.ERROR, message, exception, context)

def get_error_summary():
    return error_handler.get_error_summary()
