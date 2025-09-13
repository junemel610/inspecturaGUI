print("=== DETECTION_MODULE LOADED ===")
import cv2
import numpy as np
import json
import os
import time
import threading
import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from queue import Queue
from collections import defaultdict

from modules.utils_module import calculate_defect_size, map_model_output_to_standard
# from modules.alignment_module import AlignmentModule, AlignmentResult, AlignmentStatus

try:
    import degirum as dg
    import degirum_tools
    DEGIRUM_AVAILABLE = True
except ImportError:
    print("Warning: 'degirum' module not found. Running in mock mode for detection.")
    DEGIRUM_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print("DEBUG: onnxruntime imported successfully")
except ImportError as e:
    print(f"Warning: 'onnxruntime' module not found. ONNX models will not be available. Error: {e}")
    ONNX_AVAILABLE = False
    print("DEBUG: onnxruntime import failed")

# Check for ultralytics (required for YOLOv8/YOLOE models)
try:
    import ultralytics
    ULTRALYTICS_AVAILABLE = True
    print("DEBUG: ultralytics imported successfully")
except ImportError as e:
    print(f"Warning: 'ultralytics' module not found. YOLOv8/YOLOE models may not load properly. Error: {e}")
    ULTRALYTICS_AVAILABLE = False
    print("DEBUG: ultralytics import failed")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data classes and enums
class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class ValidationResult:
    def __init__(self, is_valid: bool, message: str = "", details: Dict = None):
        self.is_valid = is_valid
        self.message = message
        self.details = details or {}

class BenchmarkResult:
    def __init__(self, avg_inference_time: float, throughput: float, memory_usage: float):
        self.avg_inference_time = avg_inference_time
        self.throughput = throughput
        self.memory_usage = memory_usage

class CameraStatus(Enum):
    AVAILABLE = "available"
    IN_USE = "in_use"
    ERROR = "error"

@dataclass
class CameraHandle:
    camera_name: str
    lock: threading.Lock
    acquired_at: float

class RecoveryAction(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    ABORT = "abort"

class StreamResult:
    def __init__(self, success: bool, data: Any = None, error: str = ""):
        self.success = success
        self.data = data
        self.error = error

# Configuration Manager
class ConfigurationManager:
    def __init__(self, config_file: str = 'config/degirum_config.json'):
        self.config_file = config_file
        self.config = self.load_config()
        self.validator = ConfigValidator()

    def load_config(self) -> Dict:
        """Load configuration from JSON file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            else:
                return self.get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self.get_default_config()

    def get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "models": {
                "defect_detector": {
                    "type": "defect",
                    "path": "/home/inspectura/Desktop/InspecturaGUI/models/UpdatedDefects--640x640_quant_hailort_hailo8_1/UpdatedDefects--640x640_quant_hailort_hailo8_1.hef",
                    "zoo_url": "/home/inspectura/Desktop/InspecturaGUI/models/UpdatedDefects--640x640_quant_hailort_hailo8_1",
                    "model_name": "UpdatedDefects--640x640_quant_hailort_hailo8_1",
                    "confidence_threshold": 0.5,
                    "input_shape": [640, 640, 3],
                    "health_check_interval": 300
                }
            },
            "inference": {
                "fps": 30,
                "batch_size": 1,
                "timeout": 5000,
                "retry_attempts": 3
            },
            "cameras": {
                "top": {
                    "index": 0,
                    "resolution": [1920, 1080],
                    "fps": 30
                },
                "bottom": {
                    "index": 2,
                    "resolution": [1920, 1080],
                    "fps": 30
                }
            }
        }

    def get_model_config(self, model_name: str) -> Dict:
        """Get configuration for specific model"""
        return self.config.get("models", {}).get(model_name, {})

    def update_model_config(self, model_name: str, updates: Dict) -> bool:
        """Update model configuration with validation"""
        try:
            if model_name not in self.config["models"]:
                self.config["models"][model_name] = {}

            self.config["models"][model_name].update(updates)

            if self.validate_config(self.config):
                self.save_config()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update model config: {e}")
            return False

    def get_inference_config(self, camera_name: str) -> Dict:
        """Get inference configuration for camera"""
        base_config = self.config.get("inference", {})
        camera_config = self.config.get("cameras", {}).get(camera_name, {})
        return {**base_config, **camera_config}

    def validate_config(self, config: Dict) -> ValidationResult:
        """Validate configuration against schema"""
        return self.validator.validate(config)

    def save_config(self):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

# Config Validator
class ConfigValidator:
    def __init__(self):
        self.required_fields = {
            "models": ["type", "path", "model_name"],
            "inference": ["fps", "batch_size"],
            "cameras": ["index", "resolution"]
        }

    def validate(self, config: Dict) -> ValidationResult:
        """Validate configuration structure"""
        try:
            # Check required sections
            for section in ["models", "inference", "cameras"]:
                if section not in config:
                    return ValidationResult(False, f"Missing required section: {section}")

            # Validate models section
            if "models" in config:
                for model_name, model_config in config["models"].items():
                    for field in self.required_fields["models"]:
                        if field not in model_config:
                            return ValidationResult(False, f"Missing field '{field}' in model '{model_name}'")

            # Validate inference section
            if "inference" in config:
                for field in self.required_fields["inference"]:
                    if field not in config["inference"]:
                        return ValidationResult(False, f"Missing field '{field}' in inference config")

            # Validate cameras section
            if "cameras" in config:
                for camera_name, camera_config in config["cameras"].items():
                    for field in self.required_fields["cameras"]:
                        if field not in camera_config:
                            return ValidationResult(False, f"Missing field '{field}' in camera '{camera_name}'")

            return ValidationResult(True, "Configuration is valid")

        except Exception as e:
            return ValidationResult(False, f"Validation error: {str(e)}")

# Model Health Monitor
class ModelHealthMonitor:
    def __init__(self):
        self.metrics = {}
        self.thresholds = {
            'inference_time': 1000,  # ms
            'error_rate': 0.05,      # 5%
            'memory_usage': 1024,    # MB
        }

    def track_inference(self, model_name: str, inference_time: float, success: bool):
        """Track inference performance metrics"""
        if model_name not in self.metrics:
            self.metrics[model_name] = {
                'inference_times': [],
                'success_count': 0,
                'total_count': 0,
                'last_check': time.time()
            }

        metrics = self.metrics[model_name]
        metrics['inference_times'].append(inference_time)
        metrics['total_count'] += 1
        if success:
            metrics['success_count'] += 1

        # Keep only last 100 measurements
        if len(metrics['inference_times']) > 100:
            metrics['inference_times'] = metrics['inference_times'][-100:]

    def check_health(self, model_name: str) -> HealthStatus:
        """Check model health against thresholds"""
        if model_name not in self.metrics:
            return HealthStatus.UNKNOWN

        metrics = self.metrics[model_name]
        if metrics['total_count'] == 0:
            return HealthStatus.UNKNOWN

        # Calculate metrics
        avg_inference_time = sum(metrics['inference_times']) / len(metrics['inference_times'])
        error_rate = 1 - (metrics['success_count'] / metrics['total_count'])

        # Check thresholds
        if avg_inference_time > self.thresholds['inference_time'] or error_rate > self.thresholds['error_rate']:
            return HealthStatus.UNHEALTHY
        elif avg_inference_time > self.thresholds['inference_time'] * 0.8 or error_rate > self.thresholds['error_rate'] * 0.8:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def get_performance_report(self, model_name: str) -> Dict:
        """Generate performance report"""
        if model_name not in self.metrics:
            return {}

        metrics = self.metrics[model_name]
        if not metrics['inference_times']:
            return {}

        return {
            'avg_inference_time': sum(metrics['inference_times']) / len(metrics['inference_times']),
            'min_inference_time': min(metrics['inference_times']),
            'max_inference_time': max(metrics['inference_times']),
            'success_rate': metrics['success_count'] / metrics['total_count'] if metrics['total_count'] > 0 else 0,
            'total_inferences': metrics['total_count'],
            'health_status': self.check_health(model_name).value
        }

# Model Validator
class ModelValidator:
    def __init__(self):
        self.test_cases = self.load_test_cases()

    def load_test_cases(self) -> List[Dict]:
        """Load test cases for validation"""
        return [
            {
                'name': 'basic_inference',
                'input_shape': (640, 640, 3),
                'expected_output_keys': ['results', 'image_overlay']
            }
        ]

    def validate_inference(self, model, test_input) -> ValidationResult:
        """Validate model inference capabilities"""
        try:
            start_time = time.time()
            result = model(test_input)
            inference_time = (time.time() - start_time) * 1000  # ms

            # Check if result has expected structure
            if not hasattr(result, 'results'):
                return ValidationResult(False, "Model result missing 'results' attribute")

            if not hasattr(result, 'image_overlay'):
                return ValidationResult(False, "Model result missing 'image_overlay' attribute")

            return ValidationResult(True, f"Inference successful in {inference_time:.2f}ms")

        except Exception as e:
            return ValidationResult(False, f"Inference failed: {str(e)}")

    def validate_output_format(self, result) -> bool:
        """Validate model output format"""
        try:
            if not hasattr(result, 'results'):
                return False

            for detection in result.results:
                if not isinstance(detection, dict):
                    return False
                required_keys = ['label', 'confidence', 'bbox']
                if not all(key in detection for key in required_keys):
                    return False

            return True
        except:
            return False

    def benchmark_performance(self, model, iterations: int = 100) -> BenchmarkResult:
        """Benchmark model performance"""
        if model is None:
            return BenchmarkResult(0, 0, 0)

        inference_times = []
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)

        for i in range(iterations):
            try:
                start_time = time.time()
                result = model(dummy_frame)
                inference_time = (time.time() - start_time) * 1000
                inference_times.append(inference_time)
            except Exception as e:
                logger.error(f"Benchmark iteration {i} failed: {e}")
                continue

        if not inference_times:
            return BenchmarkResult(0, 0, 0)

        avg_time = sum(inference_times) / len(inference_times)
        throughput = 1000 / avg_time if avg_time > 0 else 0  # fps

        return BenchmarkResult(avg_time, throughput, 0)  # Memory usage not implemented

# Model Manager
class ModelManager:
    def __init__(self, config_manager):
        self.config = config_manager
        self.models = {}
        self.health_monitor = ModelHealthMonitor()
        self.validator = ModelValidator()

    def load_model(self, model_name: str, model_type: str = 'defect') -> Optional[object]:
        """Load model with validation and health checks"""
        try:
            model_config = self.config.get_model_config(model_name)
            if not model_config:
                logger.error(f"No configuration found for model: {model_name}")
                return None

            # Try loading from HEF file first
            model_path = model_config.get('path')
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading model from HEF file: {model_path}")
                model = self._load_from_hef(model_config)
            else:
                # Fallback to zoo URL
                zoo_url = model_config.get('zoo_url')
                if zoo_url:
                    logger.info(f"Loading model from zoo URL: {zoo_url}")
                    model = self._load_from_zoo(model_config)
                else:
                    logger.error(f"No valid path or zoo URL for model: {model_name}")
                    return None

            if model:
                # Validate model
                validation = self.validator.validate_inference(model, np.zeros((640, 640, 3), dtype=np.uint8))
                if validation.is_valid:
                    self.models[model_name] = model
                    logger.info(f"Model {model_name} loaded and validated successfully")
                    return model
                else:
                    logger.error(f"Model validation failed: {validation.message}")
                    return None
            else:
                logger.error(f"Failed to load model: {model_name}")
                return None

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None

    def _load_from_hef(self, model_config: Dict):
        """Load model from HEF file"""
        try:
            if not DEGIRUM_AVAILABLE:
                logger.warning("Degirum not available, cannot load HEF model")
                return None

            model_path = model_config['path']
            zoo_url = model_config.get('zoo_url', '')
            model_name = model_config.get('model_name', '')
            inference_host = model_config.get('inference_host', "@local")

            # For HEF files, use the directory as zoo_url and model_name for identification
            if zoo_url and model_name:
                logger.info(f"Loading HEF model with zoo_url: {zoo_url}, model_name: {model_name}")
                model = dg.load_model(
                    model_name=model_name,
                    inference_host_address=inference_host,
                    zoo_url=zoo_url
                )
            else:
                # Fallback: try to use the model path directly
                logger.warning("No zoo_url or model_name provided, attempting direct HEF loading")
                if hasattr(dg, 'load_model_from_file'):
                    model = dg.load_model_from_file(model_path, inference_host_address=inference_host)
                else:
                    # Extract directory and filename for zoo-style loading
                    model_dir = os.path.dirname(model_path)
                    model_filename = os.path.basename(model_path)
                    model = dg.load_model(
                        model_name=model_filename,
                        inference_host_address=inference_host,
                        zoo_url=model_dir
                    )

            return model

        except Exception as e:
            logger.error(f"Failed to load HEF model: {e}")
            return None

    def _load_from_zoo(self, model_config: Dict):
        """Load model from zoo URL"""
        try:
            if not DEGIRUM_AVAILABLE:
                logger.warning("Degirum not available, cannot load zoo model")
                return None

            zoo_url = model_config['zoo_url']
            model_name = model_config['model_name']
            inference_host = model_config.get('inference_host', "@local")

            model = dg.load_model(
                model_name=model_name,
                inference_host_address=inference_host,
                zoo_url=zoo_url
            )

            return model

        except Exception as e:
            logger.error(f"Failed to load zoo model: {e}")
            return None

    def validate_model(self, model, test_data=None) -> ValidationResult:
        """Validate model functionality and performance"""
        if test_data is None:
            test_data = np.zeros((640, 640, 3), dtype=np.uint8)

        return self.validator.validate_inference(model, test_data)

    def get_model_health(self, model_name: str) -> HealthStatus:
        """Get current health status of model"""
        return self.health_monitor.check_health(model_name)

    def reload_model(self, model_name: str) -> bool:
        """Reload model with error recovery"""
        try:
            if model_name in self.models:
                del self.models[model_name]

            model_config = self.config.get_model_config(model_name)
            if model_config:
                new_model = self.load_model(model_name, model_config.get('type', 'defect'))
                if new_model:
                    return True

            return False
        except Exception as e:
            logger.error(f"Failed to reload model {model_name}: {e}")
            return False

# Camera Coordinator
class CameraCoordinator:
    def __init__(self, camera_configs: Dict):
        self.cameras = {}
        self.locks = {}
        self.usage_tracker = {}
        self.initialize_cameras(camera_configs)

    def initialize_cameras(self, camera_configs: Dict):
        """Initialize camera locks and tracking"""
        for camera_name, config in camera_configs.items():
            self.locks[camera_name] = threading.Lock()
            self.usage_tracker[camera_name] = {
                'in_use': False,
                'last_used': None,
                'usage_count': 0
            }

    def acquire_camera(self, camera_name: str, requester: str) -> Optional[CameraHandle]:
        """Acquire camera with conflict prevention"""
        if camera_name not in self.locks:
            logger.error(f"Unknown camera: {camera_name}")
            return None

        lock = self.locks[camera_name]
        if lock.acquire(timeout=5.0):  # 5 second timeout
            self.usage_tracker[camera_name]['in_use'] = True
            self.usage_tracker[camera_name]['last_used'] = time.time()
            self.usage_tracker[camera_name]['usage_count'] += 1

            return CameraHandle(
                camera_name=camera_name,
                lock=lock,
                acquired_at=time.time()
            )
        else:
            logger.warning(f"Failed to acquire camera {camera_name} - timeout")
            return None

    def release_camera(self, handle: CameraHandle):
        """Release camera handle"""
        try:
            handle.lock.release()
            self.usage_tracker[handle.camera_name]['in_use'] = False
            logger.debug(f"Released camera {handle.camera_name}")
        except Exception as e:
            logger.error(f"Error releasing camera {handle.camera_name}: {e}")

    def get_camera_status(self, camera_name: str) -> CameraStatus:
        """Get current camera status"""
        if camera_name not in self.usage_tracker:
            return CameraStatus.ERROR

        tracker = self.usage_tracker[camera_name]
        if tracker['in_use']:
            return CameraStatus.IN_USE
        else:
            return CameraStatus.AVAILABLE

    def prevent_conflicts(self, camera_name: str, operation: str) -> bool:
        """Check and prevent camera access conflicts"""
        status = self.get_camera_status(camera_name)
        if status == CameraStatus.IN_USE:
            logger.warning(f"Camera {camera_name} conflict prevented for operation: {operation}")
            return False
        return True

# Stream Processor
class StreamProcessor:
    def __init__(self, model_manager, camera_coordinator, error_handler=None):
        self.model_manager = model_manager
        self.camera_coordinator = camera_coordinator
        self.error_handler = error_handler
        self.recovery_strategies = {
            'retry': RetryStrategy(),
            'fallback': FallbackStrategy(),
            'circuit_breaker': CircuitBreakerStrategy()
        }

    def process_stream(self, camera_name: str, model_name: str,
                      analyzer: callable = None) -> StreamResult:
        """Process video stream with comprehensive error recovery"""
        try:
            # Acquire camera
            camera_handle = self.camera_coordinator.acquire_camera(camera_name, "stream_processor")
            if not camera_handle:
                return StreamResult(False, error=f"Failed to acquire camera {camera_name}")

            try:
                # Get model
                model = self.model_manager.models.get(model_name)
                if not model:
                    return StreamResult(False, error=f"Model {model_name} not available")

                # Start stream processing
                result = self._process_with_recovery(model, camera_name, analyzer)
                return result

            finally:
                # Always release camera
                self.camera_coordinator.release_camera(camera_handle)

        except Exception as e:
            logger.error(f"Stream processing error: {e}")
            return StreamResult(False, error=str(e))

    def _process_with_recovery(self, model, camera_name: str, analyzer: callable) -> StreamResult:
        """Process with error recovery"""
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # For now, simulate frame processing
                # In real implementation, this would use predict_stream
                if hasattr(model, 'predict_stream'):
                    # Use Degirum's predict_stream if available
                    result = model.predict_stream(callback=self._frame_callback)
                else:
                    # Fallback to single frame processing
                    result = self._simulate_stream_processing(model, camera_name)

                return StreamResult(True, data=result)

            except Exception as e:
                retry_count += 1
                logger.warning(f"Stream processing attempt {retry_count} failed: {e}")

                if retry_count >= max_retries:
                    return StreamResult(False, error=f"Stream processing failed after {max_retries} attempts")

                # Apply recovery strategy
                recovery_action = self.handle_stream_error(e, {'camera': camera_name, 'retry': retry_count})
                if recovery_action == RecoveryAction.ABORT:
                    return StreamResult(False, error="Recovery strategy aborted processing")

                time.sleep(1)  # Brief pause before retry

        return StreamResult(False, error="Max retries exceeded")

    def _simulate_stream_processing(self, model, camera_name: str):
        """Simulate stream processing for models without predict_stream"""
        # This is a placeholder - in real implementation would process actual stream
        return {"frames_processed": 1, "camera": camera_name}

    def _frame_callback(self, frame_result):
        """Callback for frame processing in stream"""
        # Track performance
        inference_time = getattr(frame_result, 'inference_time', 100)
        success = getattr(frame_result, 'success', True)
        self.model_manager.health_monitor.track_inference("defect_detector", inference_time, success)

    def handle_stream_error(self, error: Exception, context: Dict) -> RecoveryAction:
        """Handle stream processing errors with appropriate recovery"""
        error_type = type(error).__name__

        if "timeout" in str(error).lower():
            return RecoveryAction.RETRY
        elif "connection" in str(error).lower():
            return RecoveryAction.FALLBACK
        else:
            return RecoveryAction.ABORT

    def validate_stream_result(self, result) -> ValidationResult:
        """Validate stream processing results"""
        if not result or not isinstance(result, dict):
            return ValidationResult(False, "Invalid result format")

        return ValidationResult(True, "Result validation passed")

# Recovery Strategies
class RetryStrategy:
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def execute_with_retry(self, operation: callable, *args, **kwargs):
        """Execute operation with exponential backoff retry"""
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    sleep_time = self.backoff_factor ** attempt
                    time.sleep(sleep_time)
                else:
                    raise last_exception

class FallbackStrategy:
    def __init__(self):
        self.fallback_model = None

    def execute_with_fallback(self, operation: callable, *args, **kwargs):
        """Execute with fallback mechanism"""
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary operation failed, attempting fallback: {e}")
            if self.fallback_model:
                # Implement fallback logic
                return self.fallback_model(*args, **kwargs)
            else:
                raise e

class CircuitBreakerStrategy:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def call(self, operation: callable, *args, **kwargs):
        """Execute operation with circuit breaker pattern"""
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = operation(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
            raise e

class DetectionModule:
    def __init__(self, dev_mode=False, inference_host_address="@local", config=None):
        print("DEBUG: DetectionModule __init__ called")
        self.dev_mode = dev_mode
        self.inference_host_address = inference_host_address
        self.config = config

        # Initialize new components
        self.config_manager = ConfigurationManager()
        self.model_manager = ModelManager(self.config_manager)
        self.camera_coordinator = CameraCoordinator(self.config_manager.config.get("cameras", {}))
        self.stream_processor = StreamProcessor(self.model_manager, self.camera_coordinator)

        # Legacy model paths for backward compatibility
        self.defect_model_path = "/home/inspectura/Desktop/InspecturaGUI/models/UpdatedDefects--640x640_quant_hailort_hailo8_1/UpdatedDefects--640x640_quant_hailort_hailo8_1.hef"
        self.defect_model_zoo_url = "/home/inspectura/Desktop/InspecturaGUI/models/UpdatedDefects--640x640_quant_hailort_hailo8_1"
        self.defect_model_name = "UpdatedDefects--640x640_quant_hailort_hailo8_1"

        # Model instances (legacy)
        self.defect_model = None
        self.onnx_wood_session = None
        self.ultralytics_wood_model = None

        # Detection thresholds
        self.wood_confidence_threshold = 0.3
        self.defect_confidence_threshold = 0.5

        # Load models using new system
        self.load_models()

    def load_models(self):
        """Load models using the new ModelManager system."""
        print("DEBUG: load_models() called with new ModelManager")

        # Load defect detection model using new system
        self.defect_model = self.model_manager.load_model("defect_detector", "defect")

        # For backward compatibility, also try legacy loading if new system fails
        if self.defect_model is None:
            print("DEBUG: New model loading failed, falling back to legacy method")
            self._load_models_legacy()

        # Update configuration with inference host
        if self.defect_model:
            model_config = self.config_manager.get_model_config("defect_detector")
            model_config['inference_host'] = self.inference_host_address
            self.config_manager.update_model_config("defect_detector", model_config)

    def _load_models_legacy(self):
        """Legacy model loading for backward compatibility."""
        print("DEBUG: Using legacy model loading")

        # Load DeGirum defect detection model (legacy method)
        if DEGIRUM_AVAILABLE:
            try:
                # Load defect detection model
                try:
                    self.defect_model = dg.load_model(
                        model_name=self.defect_model_name,
                        inference_host_address=self.inference_host_address,
                        zoo_url=self.defect_model_zoo_url
                    )
                    print("DeGirum defect detection model loaded successfully (legacy).")
                except Exception as model_error:
                    print(f"Failed to load model with dg.load_model: {model_error}")
                    # Try alternative loading method for HEF files
                    try:
                        import os
                        if os.path.exists(self.defect_model_path):
                            print(f"Model file exists at {self.defect_model_path}, trying alternative loading...")
                            # For now, create a mock model for testing
                            self.defect_model = None
                            print("WARNING: Using mock model for testing - replace with proper HEF loading")
                        else:
                            print(f"Model file not found at {self.defect_model_path}")
                            self.defect_model = None
                    except Exception as alt_error:
                        print(f"Alternative loading also failed: {alt_error}")
                        self.defect_model = None

            except Exception as e:
                print(f"Error loading DeGirum models: {e}")
                self.defect_model = None
        else:
            print("DeGirum not available, defect detection will not be available.")

    # COMMENTED OUT: Wood detection method - focusing on defect detection only
    # def detect_wood_presence(self, frame):
    #     """
    #     Stage 1: Detect if wood is present in the frame using the wood detection model.
    #     Returns (wood_detected, confidence, wood_bbox)
    #     """
    #     print(f"DEBUG: Starting wood detection on frame shape: {frame.shape}")
    #     print(f"DEBUG: ultralytics_wood_model: {self.ultralytics_wood_model}")
    #     print(f"DEBUG: onnx_wood_session: {self.onnx_wood_session}")
    #     print(f"DEBUG: wood_model: {self.wood_model}")

    #     # COMMENTED OUT: ONNX/YOLO wood detection - focusing on DeGirum only
    #     # if self.ultralytics_wood_model is not None or self.onnx_wood_session is not None:
    #     #     print("DEBUG: Trying YOLO/ONNX model first")
    #     #     try:
    #     #         result = self._detect_wood_with_onnx(frame)
    #     #         print(f"DEBUG: YOLO/ONNX result: {result}")
    #     #         return result
    #     #     except Exception as e:
    #     #         print(f"Error in YOLO/ONNX wood detection: {e}")
    #     #         print("DEBUG: Falling back due to YOLO/ONNX error")

    #     # Focus on DeGirum model for wood detection
    #     if self.wood_model is not None:
    #         print("DEBUG: Using DeGirum model for wood detection")
    #         try:
    #             result = self._detect_wood_with_degirum(frame)
    #             print(f"DEBUG: DeGirum result: {result}")
    #             return result
    #         except Exception as e:
    #         print(f"Error in DeGirum wood detection: {e}")
    #         print("DEBUG: Falling back due to DeGirum error")

    #     # Final fallback to basic detection
    #     print("DEBUG: No ML models available, using fallback detection")
    #     result = self._fallback_wood_detection(frame)
    #     print(f"DEBUG: Fallback detection result: {result}")
    #     return result

    # STUB: Simple replacement for detect_wood_presence to avoid GUI errors
    def detect_wood_presence(self, frame):
        """Stub method - wood detection commented out, always return wood detected"""
        print("DEBUG: detect_wood_presence called but wood detection is commented out")
        # Return dummy values to prevent GUI errors
        return True, 0.9, [100, 100, 500, 300]

    # REMOVED: All wood detection methods - focusing on defect detection only
    # def _detect_wood_with_onnx(self, frame):
    # def _detect_wood_with_degirum(self, frame):
    # def _fallback_wood_detection(self, frame):
    # def detect_defects_in_wood_region(self, frame, wood_bbox, camera_name="top"):
        """
        Stage 2: Detect defects within the identified wood region.
        Returns (annotated_frame, defect_dict, defect_measurements)
        """
        if self.defect_model is None:
            print("Defect detection model not available")
            return frame, {}, []
        
        try:
            # Crop frame to wood region if bbox is provided
            if wood_bbox:
                x1, y1, x2, y2 = wood_bbox
                # Add some padding around the wood region
                padding = 20
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)
                
                wood_region = frame[y1:y2, x1:x2]
            else:
                # Use full frame if no wood bbox
                wood_region = frame
                x1, y1 = 0, 0
            
            # Run defect detection on wood region
            inference_result = self.defect_model(wood_region)
            
            # Get annotated frame for the wood region
            annotated_region = inference_result.image_overlay
            
            # Create full annotated frame
            annotated_frame = frame.copy()
            if wood_bbox:
                # Place the annotated region back into the full frame
                annotated_frame[y1:y2, x1:x2] = annotated_region
                # Note: Wood bounding box will be drawn at the end of analyze_frame
            else:
                annotated_frame = annotated_region
            
            # Process defect detections
            final_defect_dict = {}
            defect_measurements = []
            detections = inference_result.results
            
            print(f"DEBUG: Processing {len(detections)} defect detections")
            
            for det in detections:
                confidence = det.get('confidence', 0)
                print(f"DEBUG: Defect detection - confidence: {confidence:.3f}, threshold: {self.defect_confidence_threshold}")
                
                if confidence < self.defect_confidence_threshold:
                    print(f"DEBUG: Skipping low confidence detection: {confidence:.3f}")
                    continue
                
                model_label = det['label']
                print(f"DEBUG: Raw model label: '{model_label}'")
                standard_defect_type = map_model_output_to_standard(model_label)
                print(f"DEBUG: Mapped to standard type: '{standard_defect_type}'")
                
                # Adjust bbox coordinates if we cropped the frame
                bbox = det['bbox'].copy()
                if wood_bbox:
                    bbox[0] += x1  # Adjust x1
                    bbox[1] += y1  # Adjust y1
                    bbox[2] += x1  # Adjust x2
                    bbox[3] += y1  # Adjust y2
                
                bbox_info = {'bbox': bbox}
                size_mm, percentage = calculate_defect_size(bbox_info, camera_name)
                
                # Store measurements for grading
                defect_measurements.append((standard_defect_type, size_mm, percentage))
                
                # Count defects by type
                if standard_defect_type in final_defect_dict:
                    final_defect_dict[standard_defect_type] += 1
                else:
                    final_defect_dict[standard_defect_type] = 1
            
            return annotated_frame, final_defect_dict, defect_measurements
            
        except Exception as e:
            print(f"Error during defect detection on {camera_name} camera: {e}")
            return frame, {}, []

    def detect_defects_in_full_frame(self, frame, camera_name="top"):
        """
        Detect defects on the full frame with enhanced error recovery and monitoring
        Returns (annotated_frame, defect_dict, defect_measurements)
        """
        if self.defect_model is None:
            print("Defect detection model not available")
            return frame, {}, []

        try:
            # Track inference start time for performance monitoring
            start_time = time.time()

            # Run defect detection on full frame
            inference_result = self.defect_model(frame)

            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # ms

            # Track performance metrics
            self.model_manager.health_monitor.track_inference("defect_detector", inference_time, True)

            # Get annotated frame
            annotated_frame = inference_result.image_overlay

            # Process defect detections
            final_defect_dict = {}
            defect_measurements = []
            detections = inference_result.results

            print(f"DEBUG: Processing {len(detections)} defect detections on full frame")

            for det in detections:
                confidence = det.get('confidence', 0)
                print(f"DEBUG: Defect detection - confidence: {confidence:.3f}, threshold: {self.defect_confidence_threshold}")

                if confidence < self.defect_confidence_threshold:
                    print(f"DEBUG: Skipping low confidence detection: {confidence:.3f}")
                    continue

                model_label = det['label']
                print(f"DEBUG: Raw model label: '{model_label}'")
                standard_defect_type = map_model_output_to_standard(model_label)
                print(f"DEBUG: Mapped to standard type: '{standard_defect_type}'")

                # Adjust bbox coordinates (no cropping, so coordinates are already correct)
                bbox = det['bbox'].copy()

                bbox_info = {'bbox': bbox}
                size_mm, percentage = calculate_defect_size(bbox_info, camera_name)

                # Store measurements for grading
                defect_measurements.append((standard_defect_type, size_mm, percentage))

                # Count defects by type
                if standard_defect_type in final_defect_dict:
                    final_defect_dict[standard_defect_type] += 1
                else:
                    final_defect_dict[standard_defect_type] = 1

            print(f"DEBUG: Final defect counts: {final_defect_dict}")
            return annotated_frame, final_defect_dict, defect_measurements

        except Exception as e:
            # Track failed inference
            self.model_manager.health_monitor.track_inference("defect_detector", 0, False)

            print(f"Error during defect detection on full frame for {camera_name} camera: {e}")

            # Check model health and attempt recovery if needed
            health_status = self.model_manager.get_model_health("defect_detector")
            if health_status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]:
                print(f"Model health is {health_status.value}, attempting recovery...")
                if self.model_manager.reload_model("defect_detector"):
                    print("Model reloaded successfully")
                else:
                    print("Model reload failed")

            return frame, {}, []

    def analyze_frame(self, frame, camera_name="top"):
        """
        Main analysis function - FOCUS ON DEFECT DETECTION ONLY:
        Skip wood detection, go directly to defect detection on full frame
        """
        print(f"DEBUG: analyze_frame - Focusing on defect detection only (wood detection commented out)")
        
        # COMMENTED OUT: Wood detection - focusing on defect detection only
        # Stage 1: Wood detection
        # wood_detected, wood_confidence, wood_bbox = self.detect_wood_presence(frame)
        # Store confidence for visualization
        # self.last_wood_confidence = wood_confidence

        # Debug logging for bounding box
        # print(f"DEBUG: analyze_frame - wood_detected: {wood_detected}, wood_bbox: {wood_bbox}")
        # if wood_detected and wood_bbox:
        #     print(f"DEBUG: Wood detected with bbox: {wood_bbox}")
        # elif wood_detected:
        #     print("DEBUG: Wood detected but no bbox returned")
        # else:
        #     print("DEBUG: No wood detected")

        # if not wood_detected:
        #     # No wood detected, return original frame with alignment overlay
        #     print("DEBUG: analyze_frame - No wood detected path")
        #     annotated_frame = frame.copy()
        #     cv2.putText(annotated_frame, "No Wood Detected", (50, 50),
        #                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #     # Add alignment overlay for reference
        #     alignment_result = self.alignment_module.check_wood_alignment(frame, None)
        #     annotated_frame = self.alignment_module.draw_alignment_overlay(annotated_frame, alignment_result)

        #     return annotated_frame, {}, [], alignment_result

        # # Stage 2: Alignment checking
        # alignment_result = self.alignment_module.check_wood_alignment(frame, wood_bbox)

        # # Stage 3: Defect detection in wood region (only if aligned)
        # if alignment_result.status == AlignmentStatus.ALIGNED:
        #     print("DEBUG: analyze_frame - Aligned path")
        #     annotated_frame, defect_dict, defect_measurements = self.detect_defects_in_wood_region(frame, wood_bbox, camera_name)
        # else:
        #     # Wood detected but misaligned - still show wood region but skip defect detection
        #     print("DEBUG: analyze_frame - Misaligned path")
        #     annotated_frame = frame.copy()
        #     if wood_bbox:
        #         x1, y1, x2, y2 = wood_bbox
        #         # Enhanced visualization for misaligned wood - use RED for better visibility
        #         cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Thicker RED border
        #         # Add corner markers for misaligned wood
        #         corner_size = 25
        #         # Top-left corner
        #         cv2.line(annotated_frame, (x1, y1), (x1 + corner_size, y1), (0, 0, 255), 4)
        #         cv2.line(annotated_frame, (x1, y1), (x1, y1 + corner_size), (0, 0, 255), 4)
        #         # Top-right corner
        #         cv2.line(annotated_frame, (x2, y1), (x2 - corner_size, y1), (0, 0, 255), 4)
        #         cv2.line(annotated_frame, (x2, y1), (x2, y1 + corner_size), (0, 0, 255), 4)
        #         # Bottom-left corner
        #         cv2.line(annotated_frame, (x1, y2), (x1 + corner_size, y2), (0, 0, 255), 4)
        #         cv2.line(annotated_frame, (x1, y2), (x1, y2 - corner_size), (0, 0, 255), 4)
        #         # Bottom-right corner
        #         cv2.line(annotated_frame, (x2, y2), (x2 - corner_size, y2), (0, 0, 255), 4)
        #         cv2.line(annotated_frame, (x2, y2), (x2, y2 - corner_size), (0, 0, 255), 4)

        #     # Add alignment overlay
        #     annotated_frame = self.alignment_module.draw_alignment_overlay(annotated_frame, alignment_result)
        #     return annotated_frame, {}, [], alignment_result

        # FOCUS: Go directly to defect detection on full frame
        print("DEBUG: analyze_frame - Direct defect detection on full frame")
        annotated_frame, defect_dict, defect_measurements = self.detect_defects_in_full_frame(frame, camera_name)

        # Create a dummy alignment result for compatibility
        from enum import Enum
        class DummyAlignmentStatus(Enum):
            ALIGNED = "aligned"

        class DummyAlignmentResult:
            def __init__(self):
                self.status = DummyAlignmentStatus.ALIGNED
                self.top_overlap_percent = 1.0
                self.bottom_overlap_percent = 1.0
                self.wood_bbox = None
                self.confidence_score = 1.0
                self.details = {"message": "Full-frame defect detection - alignment not required"}

        alignment_result = DummyAlignmentResult()

        return annotated_frame, defect_dict, defect_measurements, alignment_result

    def detect_wood(self, frame):
        """
        Enhanced wood detection using the wood detection model.
        Falls back to visual detection if model is not available.
        Returns True if wood is detected, False otherwise.
        """
        wood_detected, confidence, _ = self.detect_wood_presence(frame)
        return wood_detected

    # New methods for enhanced functionality
    def get_model_health_status(self, model_name: str = "defect_detector") -> HealthStatus:
        """Get the health status of a model"""
        return self.model_manager.get_model_health(model_name)

    def get_model_performance_report(self, model_name: str = "defect_detector") -> Dict:
        """Get performance report for a model"""
        return self.model_manager.health_monitor.get_performance_report(model_name)

    def reload_model(self, model_name: str = "defect_detector") -> bool:
        """Reload a model with error recovery"""
        return self.model_manager.reload_model(model_name)

    def process_stream(self, camera_name: str, model_name: str = "defect_detector") -> StreamResult:
        """Process video stream with error recovery"""
        return self.stream_processor.process_stream(camera_name, model_name)

    def get_camera_status(self, camera_name: str) -> CameraStatus:
        """Get the status of a camera"""
        return self.camera_coordinator.get_camera_status(camera_name)

    def acquire_camera(self, camera_name: str, requester: str = "detection_module") -> Optional[CameraHandle]:
        """Acquire a camera for exclusive use"""
        return self.camera_coordinator.acquire_camera(camera_name, requester)

    def release_camera(self, handle: CameraHandle):
        """Release a camera handle"""
        self.camera_coordinator.release_camera(handle)

    def update_model_config(self, model_name: str, updates: Dict) -> bool:
        """Update model configuration"""
        return self.config_manager.update_model_config(model_name, updates)

    def get_model_config(self, model_name: str) -> Dict:
        """Get model configuration"""
        return self.config_manager.get_model_config(model_name)

    def validate_configuration(self) -> ValidationResult:
        """Validate current configuration"""
        return self.config_manager.validate_config(self.config_manager.config)

    def benchmark_model(self, model_name: str = "defect_detector", iterations: int = 100) -> BenchmarkResult:
        """Benchmark model performance"""
        model = self.model_manager.models.get(model_name)
        if model:
            return self.model_manager.validator.benchmark_performance(model, iterations)
        else:
            return BenchmarkResult(0, 0, 0)

    # REMOVED: All wood detection methods - focusing on defect detection only
    # def _detect_wood_by_color(self, frame):
    # def _detect_wood_by_texture(self, frame):
    # def _detect_wood_by_shape(self, frame):
    # def detect_wood_triggered_by_ir(self, frame, ir_triggered=False):
