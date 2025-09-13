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

        # Try to disable MSMF backend to prevent issues
        self._disable_msmf_backend()

        log_info(SystemComponent.CAMERA, f"CameraModule initialized (dev_mode={dev_mode})")

    def _disable_msmf_backend(self):
        """Attempt to disable MSMF backend to prevent Windows-specific issues."""
        try:
            import os
            # Set environment variable to prefer other backends over MSMF
            os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
            os.environ['OPENCV_VIDEOIO_DEBUG'] = '1'  # Enable debug logging

            # Try to disable MSMF by setting priority to 0
            if hasattr(cv2, 'CAP_MSMF'):
                # This is a best-effort attempt to avoid MSMF
                log_info(SystemComponent.CAMERA, "Attempted to disable MSMF backend")
        except Exception as e:
            log_warning(SystemComponent.CAMERA, f"Could not disable MSMF backend: {str(e)}")

    def initialize_cameras(self):
        """Initialize cameras with simple direct approach like the test program."""
        if self.dev_mode:
            log_info(SystemComponent.CAMERA, "Initializing laptop webcam for development mode")
            return self._initialize_dev_cameras()

        success = True

        # Initialize top camera (video0) - direct approach like test program
        try:
            log_info(SystemComponent.CAMERA, "Initializing top camera (video0)")
            self.cap_top = cv2.VideoCapture(0)  # Direct camera index like test program

            if self.cap_top.isOpened():
                # Test the camera
                ret, frame = self.cap_top.read()
                if ret and frame is not None:
                    log_info(SystemComponent.CAMERA, "Successfully initialized top camera (video0)")
                else:
                    self.cap_top.release()
                    log_warning(SystemComponent.CAMERA, "Top camera opened but failed to read frame")
                    self.cap_top = None
            else:
                log_warning(SystemComponent.CAMERA, "Failed to open top camera (video0)")
                self.cap_top = None

        except Exception as e:
            log_warning(SystemComponent.CAMERA, f"Exception initializing top camera: {str(e)}")
            if self.cap_top:
                self.cap_top.release()
                self.cap_top = None

        # Initialize bottom camera (video2) - direct approach like test program
        try:
            log_info(SystemComponent.CAMERA, "Initializing bottom camera (video2)")
            self.cap_bottom = cv2.VideoCapture(2)  # Direct camera index like test program

            if self.cap_bottom.isOpened():
                # Test the camera
                ret, frame = self.cap_bottom.read()
                if ret and frame is not None:
                    log_info(SystemComponent.CAMERA, "Successfully initialized bottom camera (video2)")
                else:
                    self.cap_bottom.release()
                    log_warning(SystemComponent.CAMERA, "Bottom camera opened but failed to read frame")
                    self.cap_bottom = None
            else:
                log_warning(SystemComponent.CAMERA, "Failed to open bottom camera (video2)")
                self.cap_bottom = None

        except Exception as e:
            log_warning(SystemComponent.CAMERA, f"Exception initializing bottom camera: {str(e)}")
            if self.cap_bottom:
                self.cap_bottom.release()
                self.cap_bottom = None

        # Configure camera settings if cameras are available
        if self.cap_top and self.cap_top.isOpened():
            self.cap_top.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap_top.set(cv2.CAP_PROP_FPS, 30)

            actual_width = self.cap_top.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap_top.get(cv2.CAP_PROP_FRAME_HEIGHT)

            self.camera_status["top"]["connected"] = True
            self.camera_status["top"]["error_count"] = 0
            self.camera_status["top"]["last_successful_read"] = time.time()

            log_info(SystemComponent.CAMERA, "Top camera configured successfully",
                    {"width": actual_width, "height": actual_height})

        if self.cap_bottom and self.cap_bottom.isOpened():
            self.cap_bottom.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap_bottom.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap_bottom.set(cv2.CAP_PROP_FPS, 30)

            actual_width = self.cap_bottom.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap_bottom.get(cv2.CAP_PROP_FRAME_HEIGHT)

            self.camera_status["bottom"]["connected"] = True
            self.camera_status["bottom"]["error_count"] = 0
            self.camera_status["bottom"]["last_successful_read"] = time.time()

            log_info(SystemComponent.CAMERA, "Bottom camera configured successfully",
                    {"width": actual_width, "height": actual_height})

        # Start connection monitoring if any camera is connected
        if self.camera_status["top"]["connected"] or self.camera_status["bottom"]["connected"]:
            self.start_connection_monitoring()
        elif not success:
            log_warning(SystemComponent.CAMERA, "No cameras could be initialized - system will run in degraded mode")

        return success

    def _detect_available_cameras(self):
        """Detect available cameras by testing different indices and backends."""
        available_cameras = []
        max_test_index = 5  # Test up to index 5

        # Skip known problematic indices that often cause obsensor errors
        skip_indices = [0]  # Skip index 0 as it often has obsensor issues

        for index in range(max_test_index):
            if index in skip_indices:
                log_info(SystemComponent.CAMERA, f"Skipping camera index {index} (known problematic)")
                continue

            try:
                # Try default backend first (more reliable)
                cap_default = cv2.VideoCapture(index)
                if cap_default.isOpened():
                    # Test multiple reads to ensure stability
                    success_count = 0
                    for _ in range(3):
                        ret, frame = cap_default.read()
                        if ret and frame is not None:
                            success_count += 1

                    if success_count >= 2:  # Require at least 2 successful reads
                        available_cameras.append({
                            'index': index,
                            'backend': 'DEFAULT',
                            'resolution': f"{frame.shape[1]}x{frame.shape[0]}"
                        })
                        log_info(SystemComponent.CAMERA, f"Found working camera at index {index} with DEFAULT backend")
                    else:
                        log_warning(SystemComponent.CAMERA, f"Camera at index {index} opened but unstable ({success_count}/3 successful reads)")

                    cap_default.release()
                    continue

                # Try DSHOW backend if default fails
                try:
                    cap_dshow = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                    if cap_dshow.isOpened():
                        ret, frame = cap_dshow.read()
                        if ret and frame is not None:
                            available_cameras.append({
                                'index': index,
                                'backend': 'DSHOW',
                                'resolution': f"{frame.shape[1]}x{frame.shape[0]}"
                            })
                            log_info(SystemComponent.CAMERA, f"Found working camera at index {index} with DSHOW backend")
                        cap_dshow.release()
                except Exception as dshow_error:
                    log_warning(SystemComponent.CAMERA, f"DSHOW backend failed for index {index}: {str(dshow_error)}")

            except Exception as e:
                log_warning(SystemComponent.CAMERA, f"Error testing camera at index {index}: {str(e)}")

        return available_cameras

    def _validate_frame(self, frame):
        """Validate frame data to prevent matrix assertion errors."""
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

            # Check data type
            if frame.dtype != 'uint8':
                return False

            # Check if frame data is accessible
            try:
                # Try to access a small portion of the frame
                _ = frame[0:1, 0:1, 0:1]
            except:
                return False

            return True

        except Exception as e:
            log_warning(SystemComponent.CAMERA, f"Frame validation failed: {str(e)}")
            return False

    def read_frame(self, camera_name):
        """Read a frame from the specified camera with enhanced error handling."""
        if self.dev_mode:
            # Use laptop webcam for dev mode
            return self._read_dev_frame(camera_name)
            
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
                # Validate frame data before returning
                if not self._validate_frame(frame):
                    log_warning(SystemComponent.CAMERA, f"Invalid frame data received from camera '{camera_name}'")
                    self.camera_status[camera_name]["error_count"] += 1
                    return False, None

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

    def _initialize_dev_cameras(self):
        """Initialize cameras for development mode using direct camera indices like test program."""
        try:
            # Initialize top camera (video0) - direct approach
            log_info(SystemComponent.CAMERA, "Initializing top camera (video0) for dev mode")
            self.cap_top = cv2.VideoCapture(0)

            if self.cap_top.isOpened():
                # Test top camera
                ret, frame = self.cap_top.read()
                if not ret or frame is None:
                    log_warning(SystemComponent.CAMERA, "Top camera opened but can't read frame")
                    self.cap_top.release()
                    self.cap_top = None
                else:
                    log_info(SystemComponent.CAMERA, "Top camera initialized successfully for dev mode")
            else:
                log_warning(SystemComponent.CAMERA, "Failed to open top camera (video0)")
                self.cap_top = None

            # Initialize bottom camera (video2) - direct approach
            log_info(SystemComponent.CAMERA, "Initializing bottom camera (video2) for dev mode")
            self.cap_bottom = cv2.VideoCapture(2)

            if self.cap_bottom.isOpened():
                # Test bottom camera
                ret, frame = self.cap_bottom.read()
                if not ret or frame is None:
                    log_warning(SystemComponent.CAMERA, "Bottom camera opened but can't read frame")
                    self.cap_bottom.release()
                    self.cap_bottom = None
                else:
                    log_info(SystemComponent.CAMERA, "Bottom camera initialized successfully for dev mode")
            else:
                log_warning(SystemComponent.CAMERA, "Failed to open bottom camera (video2)")
                self.cap_bottom = None

            # Configure camera settings for both cameras
            if self.cap_top and self.cap_top.isOpened():
                self.cap_top.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                self.cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                self.cap_top.set(cv2.CAP_PROP_FPS, 30)

            if self.cap_bottom and self.cap_bottom.isOpened():
                self.cap_bottom.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                self.cap_bottom.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                self.cap_bottom.set(cv2.CAP_PROP_FPS, 30)

            # Mark cameras as connected if available
            if self.cap_top and self.cap_top.isOpened():
                self.camera_status["top"]["connected"] = True
                self.camera_status["top"]["error_count"] = 0
                self.camera_status["top"]["last_successful_read"] = time.time()

            if self.cap_bottom and self.cap_bottom.isOpened():
                self.camera_status["bottom"]["connected"] = True
                self.camera_status["bottom"]["error_count"] = 0
                self.camera_status["bottom"]["last_successful_read"] = time.time()

            log_info(SystemComponent.CAMERA, "Development cameras initialization completed")

            return True

        except Exception as e:
            log_camera_error("webcam", f"Failed to initialize development cameras: {str(e)}", e)
            self.camera_status["top"]["connected"] = False
            self.camera_status["bottom"]["connected"] = False
            return False

    def _read_dev_frame(self, camera_name):
        """Read frame from appropriate camera in dev mode - simplified approach."""
        try:
            # Use the appropriate camera capture
            camera = self.cap_top if camera_name == "top" else self.cap_bottom

            if not camera or not camera.isOpened():
                self.camera_status[camera_name]["connected"] = False
                self.camera_status[camera_name]["error_count"] += 1
                log_camera_error(camera_name, f"Camera not available in dev mode")
                return False, None

            ret, frame = camera.read()

            if not ret or frame is None:
                self.camera_status[camera_name]["connected"] = False
                self.camera_status[camera_name]["error_count"] += 1
                log_camera_error(camera_name, f"Failed to read frame from {camera_name} camera in dev mode")
                return False, None

            # Frame read successful
            current_time = time.time()
            if not self.camera_status[camera_name]["connected"]:
                self.camera_status[camera_name]["connected"] = True
                self.camera_status[camera_name]["error_count"] = 0
                log_info(SystemComponent.CAMERA, f"Dev camera '{camera_name}' recovered")

            self.camera_status[camera_name]["last_successful_read"] = current_time

            # Add dev mode indicator
            cv2.putText(frame, f"DEV MODE - {camera_name.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            return True, frame

        except Exception as e:
            self.camera_status[camera_name]["connected"] = False
            self.camera_status[camera_name]["last_error"] = str(e)
            self.camera_status[camera_name]["error_count"] += 1
            log_camera_error(camera_name, f"Exception reading {camera_name} camera in dev mode: {str(e)}", e)
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
        """Attempt to reconnect a failed camera using simple direct approach."""
        try:
            log_info(SystemComponent.CAMERA, f"Attempting to reconnect camera '{camera_name}' (attempt {self.reconnection_attempts[camera_name] + 1})")

            # Release current camera
            camera = self.cap_top if camera_name == "top" else self.cap_bottom
            if camera is not None:
                camera.release()

            # Wait before reconnection
            time.sleep(1)

            # Try to reconnect using direct camera index like test program
            camera_index = 0 if camera_name == "top" else 2  # video0 for top, video2 for bottom

            log_info(SystemComponent.CAMERA, f"Reconnecting camera {camera_name} at index {camera_index}")

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

    def get_top_frame(self):
        """Get a frame from the top camera"""
        success, frame = self.read_frame("top")
        return frame if success else None

    def get_bottom_frame(self):
        """Get a frame from the bottom camera"""
        success, frame = self.read_frame("bottom")
        return frame if success else None

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

    # COMMENTED OUT: ROI functions no longer needed for full-frame defect detection
    # def apply_roi(self, frame, camera_name, roi_enabled, roi_coordinates):
    #     """Apply Region of Interest (ROI) to frame for focused detection."""
    #     if not roi_enabled.get(camera_name, False):
    #         return frame, None
    #
    #     roi_coords = roi_coordinates.get(camera_name, {})
    #     if not roi_coords:
    #         return frame, None
    #
    #     x1, y1 = roi_coords.get("x1", 0), roi_coords.get("y1", 0)
    #     x2, y2 = roi_coords.get("x2", frame.shape[1]), roi_coords.get("y2", frame.shape[0])
    #
    #     # Ensure coordinates are within frame bounds
    #     x1 = max(0, min(x1, frame.shape[1]))
    #     y1 = max(0, min(y1, frame.shape[0]))
    #     x2 = max(x1, min(x2, frame.shape[1]))
    #     y2 = max(y1, min(y2, frame.shape[0]))
    #
    #     # Extract ROI
    #     roi_frame = frame[y1:y2, x1:x2]
    #     roi_info = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    #
    #     return roi_frame, roi_info

    # def draw_roi_overlay(self, frame, camera_name, roi_enabled, roi_coordinates):
    #     """Draw ROI rectangle overlay(s) on frame for visualization."""
    #     frame_copy = frame.copy()

    #     # Draw all enabled ROIs
    #     for roi_name, enabled in roi_enabled.items():
    #         if not enabled:
    #         continue

    #     roi_coords = roi_coordinates.get(roi_name, {})
    #     if not roi_coords:
    #         continue

    #         x1, y1 = roi_coords.get("x1", 0), roi_coords.get("y1", 0)
    #         x2, y2 = roi_coords.get("x2", frame.shape[1]), roi_coords.get("y2", frame.shape[0])

    #         # Ensure coordinates are within frame bounds
    #         x1 = max(0, min(x1, frame.shape[1]))
    #         y1 = max(0, min(y1, frame.shape[0]))
    #         x2 = max(x1, min(x2, frame.shape[1]))
    #         y2 = max(y1, min(y2, frame.shape[0]))

    #         # Draw ROI rectangle (yellow border)
    #         cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 255), 3)

    #         # Add ROI label
    #         cv2.putText(frame_copy, f"{roi_name.upper()} ALIGNMENT",
    #                    (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    #     return frame_copy