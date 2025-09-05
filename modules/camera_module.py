import cv2
import time
import threading
from modules.error_handler import log_camera_error, log_info, log_warning, SystemComponent

class CameraModule:
    def __init__(self, dev_mode=False):
        self.dev_mode = dev_mode
        self.cap_top = None
        self.cap_bottom = None
        self.camera_width = 1280
        self.camera_height = 720
        
        # Camera status tracking for enhanced error handling
        self.camera_status = {
            "top": {"connected": False, "last_error": None, "error_count": 0, "last_successful_read": None},
            "bottom": {"connected": False, "last_error": None, "error_count": 0, "last_successful_read": None}
        }
        
        # Connection monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.reconnection_attempts = {"top": 0, "bottom": 0}
        self.max_reconnection_attempts = 5
        
        log_info(SystemComponent.CAMERA, f"CameraModule initialized (dev_mode={dev_mode})")

    def initialize_cameras(self):
        """Initialize cameras with enhanced error handling and status tracking."""
        if self.dev_mode:
            log_info(SystemComponent.CAMERA, "Skipping physical camera initialization in development mode")
            return True

        success = True
        
        # Initialize top camera
        try:
            log_info(SystemComponent.CAMERA, "Initializing top camera (index 0)")
            self.cap_top = cv2.VideoCapture(0)
            
            if not self.cap_top.isOpened():
                raise Exception("Camera at index 0 failed to open")
                
            # Configure camera settings
            self.cap_top.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap_top.set(cv2.CAP_PROP_FPS, 30)
            
            # Test frame read
            ret, frame = self.cap_top.read()
            if not ret or frame is None:
                raise Exception("Failed to read test frame from top camera")
                
            actual_width = self.cap_top.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap_top.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            self.camera_status["top"]["connected"] = True
            self.camera_status["top"]["error_count"] = 0
            self.camera_status["top"]["last_successful_read"] = time.time()
            
            log_info(SystemComponent.CAMERA, "Top camera initialized successfully", 
                    {"width": actual_width, "height": actual_height, "index": 0})
            
        except Exception as e:
            success = False
            self.camera_status["top"]["connected"] = False
            self.camera_status["top"]["last_error"] = str(e)
            self.camera_status["top"]["error_count"] += 1
            log_camera_error("top", f"Initialization failed: {str(e)}", e)
            if self.cap_top:
                self.cap_top.release()
                self.cap_top = None

        # Initialize bottom camera
        try:
            log_info(SystemComponent.CAMERA, "Initializing bottom camera (index 2)")
            self.cap_bottom = cv2.VideoCapture(2)
            
            if not self.cap_bottom.isOpened():
                raise Exception("Camera at index 2 failed to open")
                
            # Configure camera settings
            self.cap_bottom.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap_bottom.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap_bottom.set(cv2.CAP_PROP_FPS, 30)
            
            # Test frame read
            ret, frame = self.cap_bottom.read()
            if not ret or frame is None:
                raise Exception("Failed to read test frame from bottom camera")
                
            actual_width = self.cap_bottom.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap_bottom.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            self.camera_status["bottom"]["connected"] = True
            self.camera_status["bottom"]["error_count"] = 0
            self.camera_status["bottom"]["last_successful_read"] = time.time()
            
            log_info(SystemComponent.CAMERA, "Bottom camera initialized successfully",
                    {"width": actual_width, "height": actual_height, "index": 2})
            
        except Exception as e:
            success = False
            self.camera_status["bottom"]["connected"] = False
            self.camera_status["bottom"]["last_error"] = str(e)
            self.camera_status["bottom"]["error_count"] += 1
            log_camera_error("bottom", f"Initialization failed: {str(e)}", e)
            if self.cap_bottom:
                self.cap_bottom.release()
                self.cap_bottom = None
                
        # Start connection monitoring if any camera is connected
        if self.camera_status["top"]["connected"] or self.camera_status["bottom"]["connected"]:
            self.start_connection_monitoring()
        elif not success:
            log_warning(SystemComponent.CAMERA, "No cameras could be initialized - system will run in degraded mode")
            
        return success
    def read_frame(self, camera_name):
        """Read a frame from the specified camera with enhanced error handling."""
        if self.dev_mode:
            # Return dummy frame for dev mode
            dummy_frame = self._create_dummy_frame()
            return True, dummy_frame
            
        camera = self.cap_top if camera_name == "top" else self.cap_bottom
        
        if camera is None or not camera.isOpened():
            # Camera not available
            if self.camera_status[camera_name]["error_count"] % 30 == 1:  # Log every 30th error
                log_camera_error(camera_name, "Camera not available or not opened")
            self.camera_status[camera_name]["error_count"] += 1
            self.camera_status[camera_name]["connected"] = False
            return False, None
            
        try:
            ret, frame = camera.read()
            
            if not ret or frame is None:
                # Frame read failed
                self.camera_status[camera_name]["connected"] = False
                self.camera_status[camera_name]["error_count"] += 1
                
                if self.camera_status[camera_name]["error_count"] % 20 == 1:  # Log every 20th error
                    log_camera_error(camera_name, "Failed to read frame from camera")
                
                # Attempt reconnection if error count is high
                if (self.camera_status[camera_name]["error_count"] > 50 and 
                    self.reconnection_attempts[camera_name] < self.max_reconnection_attempts):
                    self._attempt_camera_reconnection(camera_name)
                
                return False, None
            else:
                # Frame read successful
                current_time = time.time()
                if not self.camera_status[camera_name]["connected"]:
                    # Camera recovered
                    self.camera_status[camera_name]["connected"] = True
                    self.camera_status[camera_name]["error_count"] = 0
                    self.reconnection_attempts[camera_name] = 0
                    log_info(SystemComponent.CAMERA, f"Camera '{camera_name}' recovered and reconnected")
                
                self.camera_status[camera_name]["last_successful_read"] = current_time
                return True, frame
                
        except Exception as e:
            self.camera_status[camera_name]["connected"] = False
            self.camera_status[camera_name]["last_error"] = str(e)
            self.camera_status[camera_name]["error_count"] += 1
            log_camera_error(camera_name, f"Exception during frame read: {str(e)}", e)
            return False, None

    def _create_dummy_frame(self):
        """Create a dummy frame for dev mode"""
        import numpy as np
        frame = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        frame[:] = (64, 64, 64)  # Dark gray
        
        # Add dev mode indicators
        cv2.putText(frame, "DEV MODE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(frame, f"{self.camera_width}x{self.camera_height}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame

    def _attempt_camera_reconnection(self, camera_name):
        """Attempt to reconnect a failed camera"""
        try:
            log_info(SystemComponent.CAMERA, f"Attempting to reconnect camera '{camera_name}' (attempt {self.reconnection_attempts[camera_name] + 1})")
            
            # Release current camera
            camera = self.cap_top if camera_name == "top" else self.cap_bottom
            if camera is not None:
                camera.release()
            
            # Wait before reconnection
            time.sleep(1)
            
            # Try to reconnect
            camera_index = 0 if camera_name == "top" else 2
            new_camera = cv2.VideoCapture(camera_index)
            
            if new_camera.isOpened():
                # Configure settings
                new_camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                new_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                new_camera.set(cv2.CAP_PROP_FPS, 30)
                
                # Test read
                ret, frame = new_camera.read()
                if ret and frame is not None:
                    # Successful reconnection
                    if camera_name == "top":
                        self.cap_top = new_camera
                    else:
                        self.cap_bottom = new_camera
                    
                    self.camera_status[camera_name]["connected"] = True
                    self.camera_status[camera_name]["error_count"] = 0
                    self.reconnection_attempts[camera_name] = 0
                    self.camera_status[camera_name]["last_successful_read"] = time.time()
                    log_info(SystemComponent.CAMERA, f"Successfully reconnected camera '{camera_name}'")
                    return True
                else:
                    new_camera.release()
                    
            self.reconnection_attempts[camera_name] += 1
            log_warning(SystemComponent.CAMERA, 
                       f"Failed to reconnect camera '{camera_name}' (attempt {self.reconnection_attempts[camera_name]})")
            return False
            
        except Exception as e:
            self.reconnection_attempts[camera_name] += 1
            log_camera_error(camera_name, f"Exception during reconnection: {str(e)}", e)
            return False

    def start_connection_monitoring(self):
        """Start background thread for connection monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._connection_monitor_loop, daemon=True)
        self.monitoring_thread.start()
        log_info(SystemComponent.CAMERA, "Started camera connection monitoring")

    def _connection_monitor_loop(self):
        """Background monitoring loop for camera connections"""
        while self.monitoring_active:
            try:
                # Check camera status every 10 seconds
                time.sleep(10)
                
                current_time = time.time()
                for camera_name in ["top", "bottom"]:
                    status = self.camera_status[camera_name]
                    
                    # Check if camera has been silent for too long
                    if (status["last_successful_read"] and 
                        current_time - status["last_successful_read"] > 30):  # 30 seconds
                        if status["connected"]:
                            log_warning(SystemComponent.CAMERA, 
                                       f"Camera '{camera_name}' may be unresponsive - no successful reads for 30 seconds")
                    
                    # Attempt periodic reconnection for failed cameras
                    if (not status["connected"] and 
                        status["error_count"] > 100 and 
                        self.reconnection_attempts[camera_name] < self.max_reconnection_attempts and
                        status["error_count"] % 200 == 0):  # Try every 200 errors
                        self._attempt_camera_reconnection(camera_name)
                            
            except Exception as e:
                log_camera_error("monitor", f"Error in connection monitoring: {str(e)}", e)

    def release_cameras(self):
        """Release camera resources with proper cleanup."""
        log_info(SystemComponent.CAMERA, "Releasing cameras and stopping monitoring")
        
        # Stop monitoring thread
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=3)
        
        # Release cameras
        try:
            if self.cap_top:
                self.cap_top.release()
                self.cap_top = None
                log_info(SystemComponent.CAMERA, "Top camera released")
        except Exception as e:
            log_camera_error("top", f"Error releasing camera: {str(e)}", e)
            
        try:
            if self.cap_bottom:
                self.cap_bottom.release()
                self.cap_bottom = None
                log_info(SystemComponent.CAMERA, "Bottom camera released")
        except Exception as e:
            log_camera_error("bottom", f"Error releasing camera: {str(e)}", e)
            
        # Reset status
        for camera_name in self.camera_status:
            self.camera_status[camera_name]["connected"] = False
            
        log_info(SystemComponent.CAMERA, "All cameras released and monitoring stopped")

    def get_camera_status(self):
        """Get current status of all cameras for monitoring"""
        return self.camera_status.copy()

    def is_camera_connected(self, camera_name):
        """Check if specific camera is connected and working"""
        return self.camera_status.get(camera_name, {}).get("connected", False)

    def get_available_cameras(self):
        """Get list of currently available and working cameras"""
        available = []
        for camera_name in ["top", "bottom"]:
            if self.camera_status[camera_name]["connected"]:
                available.append(camera_name)
        return available

    def get_system_health(self):
        """Get overall camera system health summary"""
        total_errors = sum(status["error_count"] for status in self.camera_status.values())
        connected_cameras = len(self.get_available_cameras())
        
        health_status = "HEALTHY"
        if connected_cameras == 0:
            health_status = "CRITICAL"
        elif connected_cameras == 1:
            health_status = "DEGRADED"
        elif total_errors > 100:
            health_status = "UNSTABLE"
            
        return {
            "status": health_status,
            "connected_cameras": connected_cameras,
            "total_cameras": 2,
            "total_errors": total_errors,
            "camera_details": self.camera_status.copy()
        }

    def apply_roi(self, frame, camera_name, roi_enabled, roi_coordinates):
        """Apply Region of Interest (ROI) to frame for focused detection."""
        if not roi_enabled.get(camera_name, False):
            return frame, None
        
        roi_coords = roi_coordinates.get(camera_name, {})
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

    def draw_roi_overlay(self, frame, camera_name, roi_enabled, roi_coordinates):
        """Draw ROI rectangle overlay on frame for visualization."""
        if not roi_enabled.get(camera_name, False):
            return frame
        
        roi_coords = roi_coordinates.get(camera_name, {})
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
        cv2.putText(frame_copy, f"ROI - {camera_name.upper()}", 
                   (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return frame_copy