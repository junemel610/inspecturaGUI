#!/usr/bin/env python3
"""
ROI-Based Wood Detection and Grading System Module

This module implements the ROI-based wood detection system according to the
roi_based_wood_detection_design.md specification. It provides:

1. Interactive ROI definition system for camera feeds
2. ROI configuration management with JSON persistence
3. Overlap detection algorithms between wood bounding boxes and ROIs
4. Multiple ROI support per camera
5. Real-time ROI activation/deactivation
6. Integration methods with camera and wood detection systems
7. Comprehensive error handling and performance optimizations

Author: Kilo Code
"""

import cv2
import numpy as np
import json
import os
import time
import threading
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import copy

# Import existing modules for integration
from modules.wood_detection_module import (
    WoodDetectionEngine, CannyEdgeDetector, ColorRecognitionEngine,
    ContourAnalyzer, WoodDetectionResult
)
from modules.error_handler import log_info, log_warning, log_error, SystemComponent
from modules.utils_module import calculate_defect_size

# Setup logging
logger = logging.getLogger(__name__)

class ROIStatus(Enum):
    """ROI status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    OVERLAPPING = "overlapping"
    ERROR = "error"

class ROISessionStatus(Enum):
    """ROI session status enumeration"""
    ACTIVE = "active"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"

@dataclass
class ROIConfig:
    """ROI configuration data class"""
    coordinates: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    active: bool = True
    name: str = ""
    overlap_threshold: float = 0.3
    camera_name: str = ""
    roi_id: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ROIConfig':
        """Create from dictionary"""
        return cls(**data)

@dataclass
class ROISession:
    """Enhanced ROI session data class for tracking wood-ROI interactions with defect accumulation"""
    session_id: str
    camera_name: str
    roi_id: str
    start_time: float
    defects_accumulated: List[Dict] = None
    frame_count: int = 0
    wood_detections: List[Dict] = None
    status: ROISessionStatus = ROISessionStatus.ACTIVE
    end_time: Optional[float] = None
    wood_piece_id: Optional[str] = None  # Track specific wood piece
    accumulated_defect_measurements: List[Tuple] = None  # (defect_type, size_mm, percentage)
    performance_metrics: Dict = None

    def __post_init__(self):
        if self.defects_accumulated is None:
            self.defects_accumulated = []
        if self.wood_detections is None:
            self.wood_detections = []
        if self.accumulated_defect_measurements is None:
            self.accumulated_defect_measurements = []
        if self.performance_metrics is None:
            self.performance_metrics = {
                'processing_times': [],
                'detection_confidences': [],
                'frame_rates': []
            }

    def add_defects(self, defects: Dict, wood_detection: Dict, frame_id: int,
                   defect_measurements: List[Tuple] = None, processing_time: float = 0.0):
        """Add defects from a frame to the session with enhanced tracking"""
        frame_data = {
            'timestamp': time.time(),
            'defects': defects,
            'wood_detection': wood_detection,
            'frame_id': frame_id,
            'defect_measurements': defect_measurements or [],
            'processing_time': processing_time
        }

        self.defects_accumulated.append(frame_data)
        self.frame_count += 1

        # Accumulate defect measurements for grading
        if defect_measurements:
            self.accumulated_defect_measurements.extend(defect_measurements)

        # Track performance metrics
        if processing_time > 0:
            self.performance_metrics['processing_times'].append(processing_time)

        # Track detection confidence if available
        if wood_detection and 'confidence' in wood_detection:
            self.performance_metrics['detection_confidences'].append(wood_detection['confidence'])

    def get_accumulated_results(self) -> Dict:
        """Get consolidated results for grading with enhanced defect analysis"""
        # Aggregate defects across all frames
        total_defects = {}
        for frame_data in self.defects_accumulated:
            for defect_type, count in frame_data['defects'].items():
                total_defects[defect_type] = total_defects.get(defect_type, 0) + count

        # Calculate session performance metrics
        avg_processing_time = (sum(self.performance_metrics['processing_times']) /
                             len(self.performance_metrics['processing_times'])
                             if self.performance_metrics['processing_times'] else 0)

        avg_confidence = (sum(self.performance_metrics['detection_confidences']) /
                        len(self.performance_metrics['detection_confidences'])
                        if self.performance_metrics['detection_confidences'] else 0)

        return {
            'session_id': self.session_id,
            'wood_piece_id': self.wood_piece_id,
            'duration': (self.end_time or time.time()) - self.start_time,
            'total_frames': self.frame_count,
            'total_defects': total_defects,
            'defect_measurements': self.accumulated_defect_measurements,
            'detailed_frame_data': self.defects_accumulated,
            'wood_detections': self.wood_detections,
            'performance_metrics': {
                'avg_processing_time': avg_processing_time,
                'avg_detection_confidence': avg_confidence,
                'total_processing_time': sum(self.performance_metrics['processing_times']),
                'frame_rate': self.frame_count / max(self.duration, 0.001) if self.end_time else 0
            }
        }

    def get_grading_ready_data(self) -> List[Tuple]:
        """Get defect measurements in format ready for SS-EN 1611-1 grading"""
        return self.accumulated_defect_measurements.copy()

    @property
    def duration(self) -> float:
        """Get session duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def end_session(self, status: ROISessionStatus = ROISessionStatus.COMPLETED):
        """End the session with final performance calculations"""
        self.status = status
        self.end_time = time.time()

        # Final performance calculations
        duration = self.duration
        if duration > 0:
            self.performance_metrics['frame_rate'] = self.frame_count / duration

class ROIManager:
    """
    ROI Management System

    Handles ROI definition, activation/deactivation, and persistence.
    Supports multiple ROIs per camera with real-time status tracking.
    """

    def __init__(self, config_file: str = 'config/roi_config.json'):
        self.config_file = config_file
        self.rois: Dict[str, Dict[str, ROIConfig]] = {}  # {camera_name: {roi_id: ROIConfig}}
        self.active_rois: Dict[str, set] = {}  # {camera_name: set of active roi_ids}
        self.roi_states: Dict[str, Dict[str, ROIStatus]] = {}  # {camera_name: {roi_id: status}}
        self.lock = threading.RLock()

        # Load configuration
        self.load_config()

        log_info(SystemComponent.CAMERA, f"ROIManager initialized with {len(self.rois)} cameras")

    def define_roi(self, camera_name: str, roi_id: str, coordinates: Tuple[int, int, int, int],
                   name: str = "", overlap_threshold: float = 0.3) -> bool:
        """Define a new ROI for a camera feed"""
        try:
            with self.lock:
                if camera_name not in self.rois:
                    self.rois[camera_name] = {}
                    self.active_rois[camera_name] = set()
                    self.roi_states[camera_name] = {}

                # Validate coordinates
                if not self._validate_coordinates(coordinates):
                    log_error(SystemComponent.CAMERA, f"Invalid ROI coordinates: {coordinates}")
                    return False

                # Create ROI config
                roi_config = ROIConfig(
                    coordinates=coordinates,
                    active=True,
                    name=name or f"ROI_{roi_id}",
                    overlap_threshold=overlap_threshold,
                    camera_name=camera_name,
                    roi_id=roi_id
                )

                self.rois[camera_name][roi_id] = roi_config
                self.active_rois[camera_name].add(roi_id)
                self.roi_states[camera_name][roi_id] = ROIStatus.ACTIVE

                # Save configuration
                self.save_config()

                log_info(SystemComponent.CAMERA,
                        f"Defined ROI {roi_id} for camera {camera_name}: {coordinates}")
                return True

        except Exception as e:
            log_error(SystemComponent.CAMERA, f"Error defining ROI {roi_id}: {e}")
            return False

    def activate_roi(self, camera_name: str, roi_id: str) -> bool:
        """Activate an ROI for detection"""
        try:
            with self.lock:
                if camera_name in self.rois and roi_id in self.rois[camera_name]:
                    self.rois[camera_name][roi_id].active = True
                    self.active_rois[camera_name].add(roi_id)
                    self.roi_states[camera_name][roi_id] = ROIStatus.ACTIVE
                    self.save_config()
                    log_info(SystemComponent.CAMERA, f"Activated ROI {roi_id} for camera {camera_name}")
                    return True
                return False
        except Exception as e:
            log_error(SystemComponent.CAMERA, f"Error activating ROI {roi_id}: {e}")
            return False

    def deactivate_roi(self, camera_name: str, roi_id: str) -> bool:
        """Deactivate an ROI"""
        try:
            with self.lock:
                if camera_name in self.rois and roi_id in self.rois[camera_name]:
                    self.rois[camera_name][roi_id].active = False
                    self.active_rois[camera_name].discard(roi_id)
                    self.roi_states[camera_name][roi_id] = ROIStatus.INACTIVE
                    self.save_config()
                    log_info(SystemComponent.CAMERA, f"Deactivated ROI {roi_id} for camera {camera_name}")
                    return True
                return False
        except Exception as e:
            log_error(SystemComponent.CAMERA, f"Error deactivating ROI {roi_id}: {e}")
            return False

    def get_active_rois(self, camera_name: str) -> List[str]:
        """Get list of active ROI IDs for a camera"""
        with self.lock:
            return list(self.active_rois.get(camera_name, set()))

    def get_roi_config(self, camera_name: str, roi_id: str) -> Optional[ROIConfig]:
        """Get ROI configuration"""
        with self.lock:
            return self.rois.get(camera_name, {}).get(roi_id)

    def update_roi_coordinates(self, camera_name: str, roi_id: str,
                              coordinates: Tuple[int, int, int, int]) -> bool:
        """Update ROI coordinates"""
        try:
            with self.lock:
                if camera_name in self.rois and roi_id in self.rois[camera_name]:
                    if not self._validate_coordinates(coordinates):
                        return False

                    self.rois[camera_name][roi_id].coordinates = coordinates
                    self.save_config()
                    log_info(SystemComponent.CAMERA,
                            f"Updated ROI {roi_id} coordinates: {coordinates}")
                    return True
                return False
        except Exception as e:
            log_error(SystemComponent.CAMERA, f"Error updating ROI coordinates: {e}")
            return False

    def delete_roi(self, camera_name: str, roi_id: str) -> bool:
        """Delete an ROI"""
        try:
            with self.lock:
                if camera_name in self.rois and roi_id in self.rois[camera_name]:
                    del self.rois[camera_name][roi_id]
                    self.active_rois[camera_name].discard(roi_id)
                    if roi_id in self.roi_states[camera_name]:
                        del self.roi_states[camera_name][roi_id]
                    self.save_config()
                    log_info(SystemComponent.CAMERA, f"Deleted ROI {roi_id} for camera {camera_name}")
                    return True
                return False
        except Exception as e:
            log_error(SystemComponent.CAMERA, f"Error deleting ROI {roi_id}: {e}")
            return False

    def load_config(self):
        """Load ROI configuration from JSON file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)

                # Reconstruct ROI configurations
                for camera_name, camera_rois in data.get('rois', {}).items():
                    self.rois[camera_name] = {}
                    self.active_rois[camera_name] = set()
                    self.roi_states[camera_name] = {}

                    for roi_id, roi_data in camera_rois.items():
                        roi_config = ROIConfig.from_dict(roi_data)
                        self.rois[camera_name][roi_id] = roi_config

                        if roi_config.active:
                            self.active_rois[camera_name].add(roi_id)
                            self.roi_states[camera_name][roi_id] = ROIStatus.ACTIVE
                        else:
                            self.roi_states[camera_name][roi_id] = ROIStatus.INACTIVE

                log_info(SystemComponent.CAMERA, f"Loaded ROI configuration from {self.config_file}")
            else:
                log_info(SystemComponent.CAMERA, f"No ROI configuration file found at {self.config_file}")

        except Exception as e:
            log_error(SystemComponent.CAMERA, f"Error loading ROI configuration: {e}")

    def save_config(self):
        """Save ROI configuration to JSON file"""
        try:
            # Convert to serializable format
            data = {'rois': {}}
            for camera_name, camera_rois in self.rois.items():
                data['rois'][camera_name] = {}
                for roi_id, roi_config in camera_rois.items():
                    data['rois'][camera_name][roi_id] = roi_config.to_dict()

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

            # Save to file
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)

            log_info(SystemComponent.CAMERA, f"Saved ROI configuration to {self.config_file}")

        except Exception as e:
            log_error(SystemComponent.CAMERA, f"Error saving ROI configuration: {e}")

    def _validate_coordinates(self, coordinates: Tuple[int, int, int, int]) -> bool:
        """Validate ROI coordinates"""
        try:
            x1, y1, x2, y2 = coordinates
            return x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0
        except (ValueError, TypeError):
            return False

    def get_all_rois(self) -> Dict[str, Dict[str, ROIConfig]]:
        """Get all ROIs for all cameras"""
        with self.lock:
            return copy.deepcopy(self.rois)

class OverlapDetector:
    """
    Overlap Detection System

    Calculates overlap between wood bounding boxes and ROIs.
    Tracks overlap events for workflow triggering.
    """

    def __init__(self, roi_manager: ROIManager):
        self.roi_manager = roi_manager
        self.overlap_history: Dict[str, List[Dict]] = {}  # {roi_id: [overlap_events]}
        self.performance_stats = {
            'total_calculations': 0,
            'avg_calculation_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        # Caching for performance
        self._overlap_cache: Dict[str, Dict] = {}
        self._cache_timeout = 1.0  # 1 second cache timeout

    def detect_overlaps(self, wood_detections: List[WoodDetectionResult],
                       camera_name: str) -> Dict[str, List[str]]:
        """
        Detect overlaps between wood detections and active ROIs

        Returns: {wood_bbox_id: [overlapping_roi_ids]}
        """
        start_time = time.time()
        overlaps = {}

        try:
            active_rois = self.roi_manager.get_active_rois(camera_name)

            for i, wood_detection in enumerate(wood_detections):
                if not wood_detection.detected:
                    continue

                wood_bbox = wood_detection.bbox
                wood_bbox_id = f"wood_{i}"
                overlapping_rois = []

                for roi_id in active_rois:
                    roi_config = self.roi_manager.get_roi_config(camera_name, roi_id)
                    if not roi_config:
                        continue

                    # Check overlap with caching
                    cache_key = f"{camera_name}_{roi_id}_{wood_bbox}"
                    if cache_key in self._overlap_cache:
                        cached_result = self._overlap_cache[cache_key]
                        if time.time() - cached_result['timestamp'] < self._cache_timeout:
                            overlap_percentage = cached_result['overlap']
                            self.performance_stats['cache_hits'] += 1
                        else:
                            del self._overlap_cache[cache_key]
                            overlap_percentage = self.calculate_overlap_percentage(wood_bbox, roi_config.coordinates)
                            self.performance_stats['cache_misses'] += 1
                    else:
                        overlap_percentage = self.calculate_overlap_percentage(wood_bbox, roi_config.coordinates)
                        self.performance_stats['cache_misses'] += 1

                    # Cache the result
                    self._overlap_cache[cache_key] = {
                        'overlap': overlap_percentage,
                        'timestamp': time.time()
                    }

                    # Check if overlap meets threshold
                    if overlap_percentage >= roi_config.overlap_threshold:
                        overlapping_rois.append(roi_id)

                        # Track overlap event
                        self._track_overlap_event(camera_name, roi_id, wood_bbox, overlap_percentage)

                if overlapping_rois:
                    overlaps[wood_bbox_id] = overlapping_rois

            # Update performance stats
            calculation_time = time.time() - start_time
            self.performance_stats['total_calculations'] += 1
            self.performance_stats['avg_calculation_time'] = (
                (self.performance_stats['avg_calculation_time'] *
                 (self.performance_stats['total_calculations'] - 1) +
                 calculation_time) / self.performance_stats['total_calculations']
            )

            return overlaps

        except Exception as e:
            log_error(SystemComponent.CAMERA, f"Error detecting overlaps: {e}")
            return {}

    def calculate_overlap_percentage(self, bbox1: Tuple[int, int, int, int],
                                   bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate percentage overlap between two bounding boxes (IoU)"""
        try:
            x1_min, y1_min, x1_max, y1_max = bbox1
            x2_min, y2_min, x2_max, y2_max = bbox2

            # Calculate intersection
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            # Check if there's intersection
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0

            # Calculate areas
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
            bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)

            # Calculate IoU
            union_area = bbox1_area + bbox2_area - inter_area
            if union_area == 0:
                return 0.0

            return inter_area / union_area

        except Exception as e:
            log_error(SystemComponent.CAMERA, f"Error calculating overlap: {e}")
            return 0.0

    def _track_overlap_event(self, camera_name: str, roi_id: str, wood_bbox: Tuple,
                           overlap_percentage: float):
        """Track overlap events for analytics"""
        try:
            full_roi_id = f"{camera_name}_{roi_id}"

            if full_roi_id not in self.overlap_history:
                self.overlap_history[full_roi_id] = []

            event = {
                'timestamp': time.time(),
                'wood_bbox': wood_bbox,
                'overlap_percentage': overlap_percentage,
                'camera_name': camera_name,
                'roi_id': roi_id
            }

            self.overlap_history[full_roi_id].append(event)

            # Keep only last 100 events per ROI
            if len(self.overlap_history[full_roi_id]) > 100:
                self.overlap_history[full_roi_id] = self.overlap_history[full_roi_id][-100:]

        except Exception as e:
            log_error(SystemComponent.CAMERA, f"Error tracking overlap event: {e}")

    def get_overlap_history(self, camera_name: str, roi_id: str) -> List[Dict]:
        """Get overlap history for a specific ROI"""
        full_roi_id = f"{camera_name}_{roi_id}"
        return self.overlap_history.get(full_roi_id, [])

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return self.performance_stats.copy()

    def clear_cache(self):
        """Clear overlap calculation cache"""
        self._overlap_cache.clear()
        log_info(SystemComponent.CAMERA, "Overlap detection cache cleared")

class ROIBasedWorkflowManager:
    """
    Enhanced ROI-Based Workflow Manager

    Manages session-based defect accumulation and workflow triggering
    based on wood-ROI overlap detection with SS-EN 1611-1 grading integration.
    """

    def __init__(self, detection_module, grading_module, arduino_module):
        self.detection_module = detection_module
        self.grading_module = grading_module
        self.arduino_module = arduino_module

        self.active_sessions: Dict[str, ROISession] = {}  # {session_id: ROISession}
        self.completed_sessions: Dict[str, Dict] = {}  # Store completed session results
        self.session_timeout = 30.0  # 30 seconds
        self.max_sessions_per_camera = 5
        self.wood_piece_tracker: Dict[str, str] = {}  # {wood_bbox_id: session_id}

        # Performance tracking
        self.session_stats = {
            'total_sessions': 0,
            'completed_sessions': 0,
            'timeout_sessions': 0,
            'error_sessions': 0,
            'avg_session_duration': 0.0,
            'avg_defects_per_session': 0.0,
            'grading_success_rate': 0.0
        }

        # Error handling
        self.error_counts = defaultdict(int)
        self.last_errors = []

        # Start session cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_sessions, daemon=True)
        self.cleanup_thread.start()

        log_info(SystemComponent.CAMERA, "Enhanced ROIBasedWorkflowManager initialized")

    def start_roi_session(self, camera_name: str, roi_id: str, wood_piece_id: str = None) -> str:
        """Start a detection session when wood enters ROI with wood piece tracking"""
        try:
            # Generate unique session ID with higher precision and random component
            import random
            timestamp = int(time.time() * 1000000)  # microsecond precision
            random_suffix = random.randint(1000, 9999)  # Add randomness
            session_id = f"{camera_name}_{roi_id}_{timestamp}_{random_suffix}"

            # Check session limits
            camera_sessions = [s for s in self.active_sessions.values()
                              if s.camera_name == camera_name]
            if len(camera_sessions) >= self.max_sessions_per_camera:
                log_warning(SystemComponent.CAMERA,
                            f"Maximum sessions ({self.max_sessions_per_camera}) reached for camera {camera_name}")
                return ""

            # Check if wood piece already has an active session
            if wood_piece_id and wood_piece_id in self.wood_piece_tracker:
                existing_session_id = self.wood_piece_tracker[wood_piece_id]
                if existing_session_id in self.active_sessions:
                    log_info(SystemComponent.CAMERA,
                             f"Wood piece {wood_piece_id} already has active session {existing_session_id}")
                    return existing_session_id

            # Create new session
            session = ROISession(
                session_id=session_id,
                camera_name=camera_name,
                roi_id=roi_id,
                start_time=time.time(),
                wood_piece_id=wood_piece_id
            )

            self.active_sessions[session_id] = session
            self.session_stats['total_sessions'] += 1

            # Track wood piece if provided
            if wood_piece_id:
                self.wood_piece_tracker[wood_piece_id] = session_id

            log_info(SystemComponent.CAMERA,
                     f"Started ROI session {session_id} for camera {camera_name}, ROI {roi_id}, wood_piece: {wood_piece_id}")
            return session_id

        except Exception as e:
            self._handle_error("start_roi_session", e)
            return ""

    def accumulate_defects(self, session_id: str, defects: Dict, wood_detection: Dict,
                          defect_measurements: List[Tuple] = None, processing_time: float = 0.0) -> bool:
        """Accumulate defects during ROI overlap period with enhanced tracking"""
        try:
            if session_id not in self.active_sessions:
                error_msg = f"Session {session_id} not found for defect accumulation"
                log_warning(SystemComponent.CAMERA, error_msg)
                # Track this as an error for monitoring purposes
                self.error_counts["accumulate_defects_invalid_session"] += 1
                self.last_errors.append({
                    'operation': "accumulate_defects",
                    'error': error_msg,
                    'timestamp': time.time(),
                    'context': {"session_id": session_id}
                })
                return False

            session = self.active_sessions[session_id]

            # Add defects with enhanced tracking
            session.add_defects(
                defects=defects,
                wood_detection=wood_detection,
                frame_id=session.frame_count,
                defect_measurements=defect_measurements,
                processing_time=processing_time
            )

            # Update wood detections with deduplication
            if wood_detection not in session.wood_detections:
                session.wood_detections.append(wood_detection)

            log_info(SystemComponent.CAMERA,
                     f"Accumulated defects in session {session_id}: {defects}, frame {session.frame_count}")
            return True

        except Exception as e:
            self._handle_error("accumulate_defects", e, {"session_id": session_id})
            return False

    def end_roi_session(self, session_id: str, end_reason: str = "normal") -> Optional[Dict]:
        """End session and process accumulated defects with SS-EN 1611-1 grading"""
        try:
            if session_id not in self.active_sessions:
                log_warning(SystemComponent.CAMERA, f"Session {session_id} not found for ending")
                return None

            session = self.active_sessions[session_id]

            # Determine end status based on reason
            if end_reason == "timeout":
                status = ROISessionStatus.TIMEOUT
            elif end_reason == "error":
                status = ROISessionStatus.ERROR
            else:
                status = ROISessionStatus.COMPLETED

            session.end_session(status)
            results = session.get_accumulated_results()

            # Process grading if we have accumulated defects
            grading_results = None
            if results['total_frames'] > 0 and results['defect_measurements']:
                grading_results = self._process_grading_workflow(session, results)

            # Enhanced results with grading
            enhanced_results = {
                **results,
                'grading_results': grading_results,
                'end_reason': end_reason,
                'session_status': status.value
            }

            # Update statistics
            self._update_session_stats(results, grading_results)

            # Store completed session results
            self.completed_sessions[session_id] = enhanced_results

            # Clean up tracking
            if session.wood_piece_id and session.wood_piece_id in self.wood_piece_tracker:
                del self.wood_piece_tracker[session.wood_piece_id]

            # Remove from active sessions
            del self.active_sessions[session_id]

            log_info(SystemComponent.CAMERA,
                     f"Ended ROI session {session_id} - Duration: {results['duration']:.2f}s, "
                     f"Frames: {results['total_frames']}, Defects: {sum(results['total_defects'].values())}, "
                     f"Grade: {grading_results.get('grade', 'N/A') if grading_results else 'N/A'}")
            return enhanced_results

        except Exception as e:
            self._handle_error("end_roi_session", e, {"session_id": session_id})
            if session_id in self.active_sessions:
                self.active_sessions[session_id].end_session(ROISessionStatus.ERROR)
                del self.active_sessions[session_id]
            return None

    def _process_grading_workflow(self, session: ROISession, session_results: Dict) -> Optional[Dict]:
        """Process accumulated defects through SS-EN 1611-1 grading system"""
        try:
            # Get defect measurements ready for grading
            defect_measurements = session.get_grading_ready_data()

            if not defect_measurements:
                log_info(SystemComponent.CAMERA, f"No defect measurements for session {session.session_id}")
                return None

            # Perform SS-EN 1611-1 grading
            from modules.grading_module import determine_surface_grade
            grade = determine_surface_grade(defect_measurements)

            # Calculate additional grading metrics
            total_defects = sum(session_results['total_defects'].values())
            grading_info = {
                'grade': grade,
                'total_defects': total_defects,
                'defect_breakdown': session_results['total_defects'],
                'defect_measurements_count': len(defect_measurements),
                'grading_standard': 'SS-EN 1611-1'
            }

            # Send to Arduino if connected
            arduino_success = False
            if self.arduino_module and hasattr(self.arduino_module, 'is_connected') and self.arduino_module.is_connected():
                try:
                    from modules.grading_module import convert_grade_to_arduino_command
                    arduino_command = convert_grade_to_arduino_command(grade)
                    arduino_success = self.arduino_module.send_grade_command(arduino_command)
                    grading_info['arduino_command'] = arduino_command
                    grading_info['arduino_success'] = arduino_success
                except Exception as arduino_error:
                    log_error(SystemComponent.CAMERA, f"Arduino communication error: {arduino_error}")
                    grading_info['arduino_error'] = str(arduino_error)

            log_info(SystemComponent.CAMERA,
                     f"Grading completed for session {session.session_id}: Grade {grade}, "
                     f"{total_defects} defects, Arduino: {'Success' if arduino_success else 'Failed/N/A'}")

            return grading_info

        except Exception as e:
            self._handle_error("_process_grading_workflow", e, {"session_id": session.session_id})
            return None

    def _update_session_stats(self, session_results: Dict, grading_results: Optional[Dict]):
        """Update session statistics with new completed session"""
        try:
            duration = session_results['duration']
            total_defects = sum(session_results['total_defects'].values())

            self.session_stats['completed_sessions'] += 1

            # Update average duration
            self.session_stats['avg_session_duration'] = (
                (self.session_stats['avg_session_duration'] *
                 (self.session_stats['completed_sessions'] - 1) +
                 duration) / self.session_stats['completed_sessions']
            )

            # Update average defects per session
            self.session_stats['avg_defects_per_session'] = (
                (self.session_stats['avg_defects_per_session'] *
                 (self.session_stats['completed_sessions'] - 1) +
                 total_defects) / self.session_stats['completed_sessions']
            )

            # Update grading success rate
            if grading_results:
                current_successes = int(self.session_stats['grading_success_rate'] *
                                      (self.session_stats['completed_sessions'] - 1))
                self.session_stats['grading_success_rate'] = (current_successes + 1) / self.session_stats['completed_sessions']
            else:
                current_successes = int(self.session_stats['grading_success_rate'] *
                                      (self.session_stats['completed_sessions'] - 1))
                self.session_stats['grading_success_rate'] = current_successes / self.session_stats['completed_sessions']

        except Exception as e:
            self._handle_error("_update_session_stats", e)

    def _handle_error(self, operation: str, error: Exception, context: Dict = None):
        """Centralized error handling with tracking"""
        error_info = {
            'operation': operation,
            'error': str(error),
            'timestamp': time.time(),
            'context': context or {}
        }

        self.error_counts[operation] += 1
        self.last_errors.append(error_info)

        # Keep only last 50 errors
        if len(self.last_errors) > 50:
            self.last_errors = self.last_errors[-50:]

        log_error(SystemComponent.CAMERA, f"Error in {operation}: {error}")
        if context:
            log_error(SystemComponent.CAMERA, f"Context: {context}")

    def trigger_grading_workflow(self, session_results: Dict) -> bool:
        """Legacy method for backward compatibility"""
        try:
            # Extract defect measurements for grading
            defect_measurements = session_results.get('defect_measurements', [])

            # Convert to grading format
            grading_defects = []
            for defect_type, size_mm, percentage in defect_measurements:
                grading_defects.append((defect_type, size_mm, percentage))

            # Perform grading
            from modules.grading_module import determine_surface_grade
            grade = determine_surface_grade(grading_defects)

            # Send to Arduino
            if self.arduino_module and hasattr(self.arduino_module, 'is_connected') and self.arduino_module.is_connected():
                try:
                    from modules.grading_module import convert_grade_to_arduino_command
                    arduino_command = convert_grade_to_arduino_command(grade)
                    success = self.arduino_module.send_grade_command(arduino_command)
                    if success:
                        log_info(SystemComponent.CAMERA,
                                f"Sent grade {grade} to Arduino for session {session_results['session_id']}")
                        return True
                    else:
                        log_error(SystemComponent.CAMERA,
                                  f"Failed to send grade {grade} to Arduino")
                        return False
                except Exception as arduino_error:
                    log_error(SystemComponent.CAMERA, f"Arduino communication error: {arduino_error}")
                    return False
            else:
                log_warning(SystemComponent.CAMERA,
                            f"Arduino not connected - grade {grade} calculated but not sent")
                return False

        except Exception as e:
            self._handle_error("trigger_grading_workflow", e)
            return False

    def get_active_sessions(self, camera_name: Optional[str] = None) -> List[ROISession]:
        """Get active sessions, optionally filtered by camera"""
        sessions = list(self.active_sessions.values())
        if camera_name:
            sessions = [s for s in sessions if s.camera_name == camera_name]
        return sessions

    def get_session_stats(self) -> Dict:
        """Get comprehensive session statistics"""
        stats = self.session_stats.copy()
        stats.update({
            'active_sessions_count': len(self.active_sessions),
            'completed_sessions_count': len(self.completed_sessions),
            'error_counts': dict(self.error_counts),
            'recent_errors': self.last_errors[-5:] if self.last_errors else []
        })
        return stats

    def get_active_sessions_for_camera(self, camera_name: str) -> List[ROISession]:
        """Get all active sessions for a specific camera"""
        return [s for s in self.active_sessions.values() if s.camera_name == camera_name]

    def get_session_by_wood_piece(self, wood_piece_id: str) -> Optional[ROISession]:
        """Get active session for a specific wood piece"""
        if wood_piece_id in self.wood_piece_tracker:
            session_id = self.wood_piece_tracker[wood_piece_id]
            return self.active_sessions.get(session_id)
        return None

    def force_end_session(self, session_id: str, reason: str = "forced") -> bool:
        """Force end a session (for manual intervention or error recovery)"""
        try:
            if session_id not in self.active_sessions:
                return False

            results = self.end_roi_session(session_id, reason)
            return results is not None

        except Exception as e:
            self._handle_error("force_end_session", e, {"session_id": session_id})
            return False

    def get_completed_session_results(self, session_id: str) -> Optional[Dict]:
        """Get results from a completed session"""
        return self.completed_sessions.get(session_id)

    def clear_old_completed_sessions(self, max_age_hours: int = 24):
        """Clear completed sessions older than specified hours"""
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            sessions_to_remove = []

            for session_id, results in self.completed_sessions.items():
                if results.get('end_time', 0) < cutoff_time:
                    sessions_to_remove.append(session_id)

            for session_id in sessions_to_remove:
                del self.completed_sessions[session_id]

            if sessions_to_remove:
                log_info(SystemComponent.CAMERA,
                         f"Cleared {len(sessions_to_remove)} old completed sessions")

        except Exception as e:
            self._handle_error("clear_old_completed_sessions", e)

    def _cleanup_expired_sessions(self):
        """Background thread to cleanup expired sessions with enhanced timeout handling"""
        while True:
            try:
                current_time = time.time()
                expired_sessions = []

                for session_id, session in self.active_sessions.items():
                    time_since_start = current_time - session.start_time
                    if time_since_start > self.session_timeout:
                        expired_sessions.append((session_id, time_since_start))

                # End expired sessions
                for session_id, timeout_duration in expired_sessions:
                    try:
                        session = self.active_sessions[session_id]
                        session.end_session(ROISessionStatus.TIMEOUT)
                        results = session.get_accumulated_results()

                        log_warning(SystemComponent.CAMERA,
                                    f"Session {session_id} timed out after {timeout_duration:.2f}s "
                                    f"({results['total_frames']} frames, "
                                    f"{sum(results['total_defects'].values())} defects)")

                        # Trigger grading for timed out sessions with defect data
                        if results['total_frames'] > 0 and results['defect_measurements']:
                            grading_success = self.trigger_grading_workflow(results)
                            log_info(SystemComponent.CAMERA,
                                     f"Timeout grading for session {session_id}: "
                                     f"{'Success' if grading_success else 'Failed'}")
                        else:
                            log_info(SystemComponent.CAMERA,
                                     f"No defect data for timeout session {session_id}")

                        # Clean up tracking
                        if session.wood_piece_id and session.wood_piece_id in self.wood_piece_tracker:
                            del self.wood_piece_tracker[session.wood_piece_id]

                        del self.active_sessions[session_id]
                        self.session_stats['timeout_sessions'] += 1

                    except Exception as session_error:
                        self._handle_error("_cleanup_expired_sessions", session_error,
                                         {"session_id": session_id, "timeout_duration": timeout_duration})
                        # Force removal even if cleanup fails
                        if session_id in self.active_sessions:
                            del self.active_sessions[session_id]

                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                self._handle_error("_cleanup_expired_sessions", e)
                time.sleep(10)

class ROIVisualizer:
    """
    ROI Visual Feedback System

    Provides real-time overlay of ROIs and wood detections on camera feeds.
    """

    def __init__(self, roi_manager: ROIManager):
        self.roi_manager = roi_manager
        self.colors = {
            'active': (0, 255, 0),      # Green for active ROIs
            'inactive': (128, 128, 128), # Gray for inactive ROIs
            'overlap': (0, 165, 255),    # Orange for overlapping ROIs
            'wood': (255, 0, 0),         # Red for detected wood
            'text': (255, 255, 255)      # White for text
        }

        # Performance tracking
        self.render_stats = {
            'total_renders': 0,
            'avg_render_time': 0.0
        }

    def draw_roi_overlays(self, frame: np.ndarray, camera_name: str,
                          overlapping_rois: Optional[List[str]] = None) -> np.ndarray:
        """Draw ROI overlays on camera frame"""
        start_time = time.time()
        overlay_frame = frame.copy()

        try:
            camera_rois = self.roi_manager.rois.get(camera_name, {})
            overlapping_rois = overlapping_rois or []

            for roi_id, roi_data in camera_rois.items():
                if not roi_data.active:
                    continue

                coordinates = roi_data.coordinates
                x1, y1, x2, y2 = coordinates

                # Determine color based on overlap status
                if roi_id in overlapping_rois:
                    color = self.colors['overlap']
                    status_text = "OVERLAP"
                else:
                    color = self.colors['active']
                    status_text = "ACTIVE"

                # Draw ROI rectangle
                cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 3)

                # Draw corner markers
                marker_size = 15
                # Top-left
                cv2.line(overlay_frame, (x1, y1), (x1 + marker_size, y1), color, 2)
                cv2.line(overlay_frame, (x1, y1), (x1, y1 + marker_size), color, 2)
                # Top-right
                cv2.line(overlay_frame, (x2, y1), (x2 - marker_size, y1), color, 2)
                cv2.line(overlay_frame, (x2, y1), (x2, y1 + marker_size), color, 2)
                # Bottom-left
                cv2.line(overlay_frame, (x1, y2), (x1 + marker_size, y2), color, 2)
                cv2.line(overlay_frame, (x1, y2), (x1, y2 - marker_size), color, 2)
                # Bottom-right
                cv2.line(overlay_frame, (x2, y2), (x2 - marker_size, y2), color, 2)
                cv2.line(overlay_frame, (x2, y2), (x2, y2 - marker_size), color, 2)

                # Add ROI label
                label = f"{roi_data.name} ({roi_id}) - {status_text}"
                cv2.putText(overlay_frame, label, (x1 + 10, y1 + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Add overlap threshold info
                threshold_text = f"Threshold: {roi_data.overlap_threshold:.2f}"
                cv2.putText(overlay_frame, threshold_text, (x1 + 10, y1 + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)

            # Update performance stats
            render_time = time.time() - start_time
            self.render_stats['total_renders'] += 1
            self.render_stats['avg_render_time'] = (
                (self.render_stats['avg_render_time'] *
                 (self.render_stats['total_renders'] - 1) +
                 render_time) / self.render_stats['total_renders']
            )

            return overlay_frame

        except Exception as e:
            log_error(SystemComponent.CAMERA, f"Error drawing ROI overlays: {e}")
            return frame

    def draw_wood_detections(self, frame: np.ndarray,
                           wood_detections: List[WoodDetectionResult]) -> np.ndarray:
        """Draw wood detection bounding boxes"""
        overlay_frame = frame.copy()

        try:
            for i, detection in enumerate(wood_detections):
                if not detection.detected:
                    continue

                bbox = detection.bbox
                confidence = detection.confidence

                x1, y1, x2, y2 = bbox
                color = self.colors['wood']

                # Draw bounding box
                cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 3)

                # Draw corner markers
                marker_size = 10
                # Top-left
                cv2.line(overlay_frame, (x1, y1), (x1 + marker_size, y1), color, 2)
                cv2.line(overlay_frame, (x1, y1), (x1, y1 + marker_size), color, 2)
                # Top-right
                cv2.line(overlay_frame, (x2, y1), (x2 - marker_size, y1), color, 2)
                cv2.line(overlay_frame, (x2, y1), (x2, y1 + marker_size), color, 2)
                # Bottom-left
                cv2.line(overlay_frame, (x1, y2), (x1 + marker_size, y2), color, 2)
                cv2.line(overlay_frame, (x1, y2), (x1, y2 - marker_size), color, 2)
                # Bottom-right
                cv2.line(overlay_frame, (x2, y2), (x2 - marker_size, y2), color, 2)
                cv2.line(overlay_frame, (x2, y2), (x2, y2 - marker_size), color, 2)

                # Add confidence label
                label = f"Wood {i+1}: {confidence:.2f}"
                cv2.putText(overlay_frame, label, (x1 + 10, y1 + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Add feature info if available
                if detection.features:
                    dominant_color = detection.features.get('dominant_color', 'unknown')
                    color_text = f"Color: {dominant_color}"
                    cv2.putText(overlay_frame, color_text, (x1 + 10, y1 + 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)

            return overlay_frame

        except Exception as e:
            log_error(SystemComponent.CAMERA, f"Error drawing wood detections: {e}")
            return frame

    def draw_combined_overlay(self, frame: np.ndarray, camera_name: str,
                            wood_detections: List[WoodDetectionResult],
                            overlapping_rois: Optional[List[str]] = None) -> np.ndarray:
        """Draw combined ROI and wood detection overlays"""
        # Draw ROI overlays first
        frame_with_rois = self.draw_roi_overlays(frame, camera_name, overlapping_rois)

        # Draw wood detections on top
        final_frame = self.draw_wood_detections(frame_with_rois, wood_detections)

        return final_frame

    def get_render_stats(self) -> Dict:
        """Get rendering performance statistics"""
        return self.render_stats.copy()

# Integration functions for existing systems
def integrate_with_camera_module(camera_module, roi_manager: ROIManager,
                               roi_visualizer: ROIVisualizer):
    """Integration method for camera module"""
    def get_frame_with_roi_overlay(camera_name: str) -> Optional[np.ndarray]:
        """Get frame with ROI overlay"""
        try:
            success, frame = camera_module.read_frame(camera_name)
            if not success or frame is None:
                return None

            return roi_visualizer.draw_roi_overlays(frame, camera_name)
        except Exception as e:
            log_error(SystemComponent.CAMERA, f"Error getting frame with ROI overlay: {e}")
            return None

    # Add method to camera module
    camera_module.get_frame_with_roi_overlay = get_frame_with_roi_overlay
    return camera_module

def integrate_with_detection_module(detection_module, wood_detector: WoodDetectionEngine,
                                  overlap_detector: OverlapDetector,
                                  workflow_manager: ROIBasedWorkflowManager):
    """Integration method for detection module"""
    def analyze_frame_with_roi_workflow(frame: np.ndarray, camera_name: str) -> Dict:
        """Enhanced frame analysis with ROI-based workflow"""
        try:
            results = {
                'annotated_frame': frame,
                'wood_detections': [],
                'overlaps': {},
                'sessions_triggered': [],
                'grading_results': None
            }

            # Step 1: Detect wood
            wood_detections = wood_detector.detect_wood(frame)
            results['wood_detections'] = wood_detections

            # Step 2: Detect overlaps
            overlaps = overlap_detector.detect_overlaps(wood_detections, camera_name)
            results['overlaps'] = overlaps

            # Step 3: Manage ROI sessions
            sessions_triggered = []
            for wood_bbox_id, overlapping_roi_ids in overlaps.items():
                for roi_id in overlapping_roi_ids:
                    # Check if session already exists
                    session_id = f"{camera_name}_{roi_id}_{int(time.time())}"
                    existing_sessions = workflow_manager.get_active_sessions(camera_name)
                    active_session = None

                    for session in existing_sessions:
                        if session.roi_id == roi_id and session.status == ROISessionStatus.ACTIVE:
                            active_session = session
                            break

                    # Start new session if none exists
                    if not active_session:
                        session_id = workflow_manager.start_roi_session(camera_name, roi_id)
                        if session_id:
                            sessions_triggered.append(session_id)
                            active_session = workflow_manager.active_sessions.get(session_id)

                    # Accumulate defects in session
                    if active_session:
                        # Get defect detection results for wood region
                        wood_detection = wood_detections[int(wood_bbox_id.split('_')[1])]
                        if wood_detection.detected:
                            x1, y1, x2, y2 = wood_detection.bbox
                            wood_region = frame[y1:y2, x1:x2]

                            # Run defect detection
                            annotated_region, defects, defect_measurements = detection_module.detect_defects_in_full_frame(wood_region, camera_name)

                            # Accumulate in session
                            workflow_manager.accumulate_defects(
                                active_session.session_id, defects,
                                wood_detection.features or {}
                            )

            results['sessions_triggered'] = sessions_triggered

            # Step 4: Check for session completion
            active_sessions = workflow_manager.get_active_sessions(camera_name)
            for session in active_sessions:
                # Simple completion check - end session if no overlaps in this frame
                session_overlaps = any(roi_id in overlaps.get(wood_bbox_id, [])
                                     for wood_bbox_id, roi_ids in overlaps.items()
                                     for roi_id in roi_ids)
                if not session_overlaps:
                    # End session and trigger grading
                    session_results = workflow_manager.end_roi_session(session.session_id)
                    if session_results:
                        grading_success = workflow_manager.trigger_grading_workflow(session_results)
                        results['grading_results'] = {
                            'session_id': session.session_id,
                            'success': grading_success,
                            'results': session_results
                        }

            return results

        except Exception as e:
            log_error(SystemComponent.CAMERA, f"Error in ROI workflow analysis: {e}")
            return {
                'annotated_frame': frame,
                'wood_detections': [],
                'overlaps': {},
                'sessions_triggered': [],
                'grading_results': None,
                'error': str(e)
            }

    # Add method to detection module
    detection_module.analyze_frame_with_roi_workflow = analyze_frame_with_roi_workflow
    return detection_module

# Default configuration
DEFAULT_ROI_CONFIG = {
    "top_camera": {
        "inspection_zone": {
            "coordinates": [64, 0, 1216, 108],
            "active": True,
            "name": "Top Inspection Zone",
            "overlap_threshold": 0.3
        }
    },
    "bottom_camera": {
        "inspection_zone": {
            "coordinates": [64, 612, 1216, 720],
            "active": True,
            "name": "Bottom Inspection Zone",
            "overlap_threshold": 0.3
        }
    }
}

def create_default_roi_config(config_file: str = 'config/roi_config.json'):
    """Create default ROI configuration file"""
    try:
        os.makedirs(os.path.dirname(config_file), exist_ok=True)

        with open(config_file, 'w') as f:
            json.dump({"rois": DEFAULT_ROI_CONFIG}, f, indent=2)

        log_info(SystemComponent.CAMERA, f"Created default ROI configuration at {config_file}")
        return True

    except Exception as e:
        log_error(SystemComponent.CAMERA, f"Error creating default ROI configuration: {e}")
        return False

# Main ROI Module class for easy integration
class ROIModule:
    """
    Main ROI Module class providing unified access to all ROI functionality
    """

    def __init__(self, config_file: str = 'config/roi_config.json'):
        self.config_file = config_file

        # Initialize components
        self.roi_manager = ROIManager(config_file)
        self.overlap_detector = OverlapDetector(self.roi_manager)
        self.wood_detector = WoodDetectionEngine()  # Use existing wood detection engine
        self.roi_visualizer = ROIVisualizer(self.roi_manager)

        # Workflow manager (needs to be initialized with other modules)
        self.workflow_manager = None

        log_info(SystemComponent.CAMERA, "ROI Module initialized")

    def initialize_workflow_manager(self, detection_module, grading_module, arduino_module):
        """Initialize workflow manager with required modules"""
        self.workflow_manager = ROIBasedWorkflowManager(
            detection_module, grading_module, arduino_module
        )
        log_info(SystemComponent.CAMERA, "ROI Workflow Manager initialized")

    def process_frame(self, frame: np.ndarray, camera_name: str) -> Dict:
        """Process a frame through the complete enhanced ROI pipeline"""
        start_time = time.time()

        try:
            results = {
                'annotated_frame': frame,
                'wood_detections': [],
                'overlaps': {},
                'active_sessions': [],
                'session_events': [],
                'performance_stats': {},
                'processing_time': 0.0
            }

            # Detect wood
            wood_detections = self.wood_detector.detect_wood(frame)
            results['wood_detections'] = wood_detections

            # Detect overlaps
            overlaps = self.overlap_detector.detect_overlaps(wood_detections, camera_name)
            results['overlaps'] = overlaps

            # Get overlapping ROI IDs for visualization
            overlapping_rois = []
            for roi_ids in overlaps.values():
                overlapping_rois.extend(roi_ids)

            # Draw overlays
            results['annotated_frame'] = self.roi_visualizer.draw_combined_overlay(
                frame, camera_name, wood_detections, overlapping_rois
            )

            # Handle workflow if manager is available
            if self.workflow_manager:
                workflow_results = self._process_frame_workflow_enhanced(
                    frame, camera_name, wood_detections, overlaps
                )
                results.update(workflow_results)
                results['active_sessions'] = [s.session_id for s in self.workflow_manager.get_active_sessions(camera_name)]

            # Add performance stats
            results['performance_stats'] = {
                'overlap_detector': self.overlap_detector.get_performance_stats(),
                'visualizer': self.roi_visualizer.get_render_stats(),
                'wood_detector': self.wood_detector.get_performance_stats() if hasattr(self.wood_detector, 'get_performance_stats') else {}
            }

            results['processing_time'] = time.time() - start_time
            return results

        except Exception as e:
            log_error(SystemComponent.CAMERA, f"Error processing frame in ROI module: {e}")
            return {
                'annotated_frame': frame,
                'wood_detections': [],
                'overlaps': {},
                'active_sessions': [],
                'session_events': [],
                'performance_stats': {},
                'processing_time': time.time() - start_time,
                'error': str(e)
            }

    def _process_frame_workflow_enhanced(self, frame: np.ndarray, camera_name: str,
                                       wood_detections: List, overlaps: Dict) -> Dict:
        """Enhanced frame workflow processing with defect accumulation and session management"""
        workflow_results = {
            'session_events': [],
            'grading_events': [],
            'active_session_count': 0
        }

        try:
            # Track active sessions for this camera
            active_sessions = self.workflow_manager.get_active_sessions_for_camera(camera_name)
            workflow_results['active_session_count'] = len(active_sessions)

            # Process each wood detection
            for i, wood_detection in enumerate(wood_detections):
                if not wood_detection.detected:
                    continue

                wood_bbox_id = f"wood_{i}"
                overlapping_roi_ids = overlaps.get(wood_bbox_id, [])

                # Check if this wood piece has overlapping ROIs
                if overlapping_roi_ids:
                    # Generate wood piece ID for tracking
                    wood_piece_id = f"{camera_name}_wood_{int(time.time() * 1000)}_{i}"

                    # Check if we already have a session for this wood piece
                    existing_session = self.workflow_manager.get_session_by_wood_piece(wood_piece_id)

                    if not existing_session:
                        # Start new session for first overlapping ROI
                        roi_id = overlapping_roi_ids[0]  # Use first overlapping ROI
                        session_id = self.workflow_manager.start_roi_session(
                            camera_name, roi_id, wood_piece_id
                        )

                        if session_id:
                            workflow_results['session_events'].append({
                                'event': 'session_started',
                                'session_id': session_id,
                                'wood_piece_id': wood_piece_id,
                                'roi_id': roi_id
                            })

                            existing_session = self.workflow_manager.active_sessions.get(session_id)

                    # Accumulate defects in the session
                    if existing_session:
                        # Perform defect detection on the wood region
                        x1, y1, x2, y2 = wood_detection.bbox
                        wood_region = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else frame

                        # Run defect detection
                        start_detection = time.time()
                        annotated_region, defects, defect_measurements = self.detection_module.detect_defects_in_full_frame(
                            wood_region, camera_name
                        )
                        detection_time = time.time() - start_detection

                        # Adjust defect coordinates back to full frame
                        adjusted_measurements = []
                        for defect_type, size_mm, percentage in defect_measurements:
                            adjusted_measurements.append((defect_type, size_mm, percentage))

                        # Accumulate in session
                        success = self.workflow_manager.accumulate_defects(
                            existing_session.session_id,
                            defects,
                            wood_detection.__dict__ if hasattr(wood_detection, '__dict__') else {'bbox': wood_detection.bbox},
                            adjusted_measurements,
                            detection_time
                        )

                        if success:
                            workflow_results['session_events'].append({
                                'event': 'defects_accumulated',
                                'session_id': existing_session.session_id,
                                'defect_count': sum(defects.values()),
                                'frame_id': existing_session.frame_count
                            })

                else:
                    # No overlapping ROIs - check if we need to end sessions for wood pieces that moved out
                    # This is a simplified check - in practice you'd track wood piece movement
                    pass

            # Check for sessions that should end (wood pieces that moved out of all ROIs)
            sessions_to_end = []
            for session in active_sessions:
                # Check if this session's wood piece still has overlaps
                wood_still_overlapping = False
                for wood_bbox_id, roi_ids in overlaps.items():
                    if session.roi_id in roi_ids:
                        wood_still_overlapping = True
                        break

                if not wood_still_overlapping:
                    sessions_to_end.append(session.session_id)

            # End sessions for wood pieces that exited ROI
            for session_id in sessions_to_end:
                results = self.workflow_manager.end_roi_session(session_id, "wood_exited_roi")
                if results and results.get('grading_results'):
                    workflow_results['grading_events'].append({
                        'event': 'grading_completed',
                        'session_id': session_id,
                        'grade': results['grading_results'].get('grade'),
                        'total_defects': results['grading_results'].get('total_defects')
                    })

                    workflow_results['session_events'].append({
                        'event': 'session_completed',
                        'session_id': session_id,
                        'reason': 'wood_exited_roi'
                    })

        except Exception as e:
            log_error(SystemComponent.CAMERA, f"Error in enhanced frame workflow: {e}")
            workflow_results['error'] = str(e)

        return workflow_results

    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        stats = {
            'roi_manager': {
                'total_rois': sum(len(camera_rois) for camera_rois in self.roi_manager.rois.values()),
                'active_rois': sum(len(active) for active in self.roi_manager.active_rois.values())
            },
            'overlap_detector': self.overlap_detector.get_performance_stats(),
            'visualizer': self.roi_visualizer.get_render_stats(),
            'wood_detector': self.wood_detector.get_performance_stats() if hasattr(self.wood_detector, 'get_performance_stats') else {}
        }

        if self.workflow_manager:
            stats['workflow_manager'] = self.workflow_manager.get_session_stats()

        return stats

# Export main classes
__all__ = [
    'ROIManager', 'OverlapDetector', 'ROIBasedWorkflowManager', 'ROIVisualizer',
    'ROIModule', 'ROIConfig', 'ROISession', 'ROIStatus', 'ROISessionStatus',
    'integrate_with_camera_module', 'integrate_with_detection_module',
    'create_default_roi_config', 'DEFAULT_ROI_CONFIG'
]