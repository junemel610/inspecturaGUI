#!/usr/bin/env python3
"""
Check Alignment Module for Wood Sorting Application

This module handles alignment checking between detected wood pieces and predefined
regions of interest (ROIs) for top and bottom alignment verification.

Key Features:
- Dynamic ROI definition based on frame dimensions
- Overlap detection algorithms
- Alignment status tracking
- Integration with visualization pipeline
- Configurable thresholds and positions
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from modules.error_handler import log_info, log_warning, log_error, SystemComponent


class AlignmentStatus(Enum):
    """Enumeration of possible alignment statuses"""
    ALIGNED = "aligned"
    MISALIGNED_TOP = "misaligned_top"
    MISALIGNED_BOTTOM = "misaligned_bottom"
    MISALIGNED_BOTH = "misaligned_both"
    NO_WOOD_DETECTED = "no_wood_detected"
    INSUFFICIENT_OVERLAP = "insufficient_overlap"


@dataclass
class AlignmentROI:
    """Data structure for alignment ROI definition"""
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    required_overlap_percent: float
    color: Tuple[int, int, int] = (255, 255, 0)  # Yellow by default


@dataclass
class AlignmentResult:
    """Data structure for alignment check results"""
    status: AlignmentStatus
    top_overlap_percent: float
    bottom_overlap_percent: float
    wood_bbox: Optional[Tuple[int, int, int, int]]
    top_roi: AlignmentROI
    bottom_roi: AlignmentROI
    confidence_score: float
    details: Dict[str, Any]


class AlignmentModule:
    """
    Main alignment checking module for wood sorting system.

    Handles ROI definition, overlap detection, and alignment status determination.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the alignment module with configuration.

        Args:
            config: Configuration dictionary containing alignment settings
        """
        self.config = config
        self.alignment_config = config.get('alignment', {})

        # Default configuration values
        self.top_roi_margin_percent = self.alignment_config.get('top_roi_margin_percent', 0.15)
        self.bottom_roi_margin_percent = self.alignment_config.get('bottom_roi_margin_percent', 0.15)
        self.min_overlap_threshold = self.alignment_config.get('min_overlap_threshold', 0.6)
        self.alignment_tolerance_percent = self.alignment_config.get('alignment_tolerance_percent', 0.1)

        # Color scheme for visualization
        self.aligned_color = (0, 255, 0)      # Green
        self.misaligned_color = (0, 0, 255)   # Red
        self.roi_color = (255, 255, 0)        # Yellow

        log_info(SystemComponent.DETECTION, "AlignmentModule initialized")

    def define_alignment_rois(self, frame_width: int, frame_height: int) -> Tuple[AlignmentROI, AlignmentROI]:
        """
        Define top and bottom alignment ROIs based on frame dimensions.

        Args:
            frame_width: Width of the camera frame
            frame_height: Height of the camera frame

        Returns:
            Tuple of (top_roi, bottom_roi) AlignmentROI objects
        """
        # Calculate ROI dimensions based on frame size
        roi_margin_pixels = int(frame_height * self.top_roi_margin_percent)  # 15% of frame height
        roi_width_margin = int(frame_width * 0.05)  # 5% margin on sides

        # Define top ROI (top 15% of frame)
        top_roi = AlignmentROI(
            name="top_alignment",
            x1=roi_width_margin,                    # 5% from left
            y1=0,                                   # Top of frame
            x2=frame_width - roi_width_margin,      # 5% from right
            y2=roi_margin_pixels,                   # 15% down from top
            required_overlap_percent=self.min_overlap_threshold,
            color=self.roi_color
        )

        # Define bottom ROI (bottom 15% of frame)
        bottom_roi = AlignmentROI(
            name="bottom_alignment",
            x1=roi_width_margin,                    # 5% from left
            y1=frame_height - roi_margin_pixels,    # 15% up from bottom
            x2=frame_width - roi_width_margin,      # 5% from right
            y2=frame_height,                        # Bottom of frame
            required_overlap_percent=self.min_overlap_threshold,
            color=self.roi_color
        )

        log_info(SystemComponent.DETECTION,
                f"Defined dynamic alignment ROIs for {frame_width}x{frame_height} frame - "
                f"Top: ({top_roi.x1},{top_roi.y1}) to ({top_roi.x2},{top_roi.y2}), "
                f"Bottom: ({bottom_roi.x1},{bottom_roi.y1}) to ({bottom_roi.x2},{bottom_roi.y2})")

        return top_roi, bottom_roi

    def calculate_bbox_overlap(self, bbox1: Tuple[int, int, int, int],
                             bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate the overlap percentage between two bounding boxes.

        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)

        Returns:
            Overlap percentage (0.0 to 1.0)
        """
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        # Check if there's an intersection
        if x1_inter >= x2_inter or y1_inter >= y2_inter:
            return 0.0

        # Calculate intersection area
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate area of second bbox (ROI)
        roi_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Avoid division by zero
        if roi_area == 0:
            return 0.0

        # Return overlap percentage
        return inter_area / roi_area

    def check_wood_alignment(self, frame: np.ndarray,
                           wood_bbox: Optional[Tuple[int, int, int, int]]) -> AlignmentResult:
        """
        Main alignment checking function.

        Args:
            frame: Camera frame as numpy array
            wood_bbox: Detected wood bounding box (x1, y1, x2, y2) or None

        Returns:
            AlignmentResult object with detailed alignment information
        """
        frame_height, frame_width = frame.shape[:2]

        # Define ROIs based on current frame dimensions
        top_roi, bottom_roi = self.define_alignment_rois(frame_width, frame_height)

        # Handle case where no wood is detected
        if wood_bbox is None:
            return AlignmentResult(
                status=AlignmentStatus.NO_WOOD_DETECTED,
                top_overlap_percent=0.0,
                bottom_overlap_percent=0.0,
                wood_bbox=None,
                top_roi=top_roi,
                bottom_roi=bottom_roi,
                confidence_score=0.0,
                details={"error": "No wood detected in frame"}
            )

        # Calculate overlap with top ROI
        top_overlap = self.calculate_bbox_overlap(wood_bbox, (top_roi.x1, top_roi.y1, top_roi.x2, top_roi.y2))

        # Calculate overlap with bottom ROI
        bottom_overlap = self.calculate_bbox_overlap(wood_bbox, (bottom_roi.x1, bottom_roi.y1, bottom_roi.x2, bottom_roi.y2))

        # Determine alignment status
        status = self._determine_alignment_status(top_overlap, bottom_overlap)

        # Calculate confidence score based on overlap percentages
        confidence_score = (top_overlap + bottom_overlap) / 2.0

        # Create detailed results
        details = {
            "top_overlap_percent": top_overlap,
            "bottom_overlap_percent": bottom_overlap,
            "wood_bbox_area": (wood_bbox[2] - wood_bbox[0]) * (wood_bbox[3] - wood_bbox[1]),
            "top_roi_area": (top_roi.x2 - top_roi.x1) * (top_roi.y2 - top_roi.y1),
            "bottom_roi_area": (bottom_roi.x2 - bottom_roi.x1) * (bottom_roi.y2 - bottom_roi.y1),
            "frame_dimensions": (frame_width, frame_height)
        }

        result = AlignmentResult(
            status=status,
            top_overlap_percent=top_overlap,
            bottom_overlap_percent=bottom_overlap,
            wood_bbox=wood_bbox,
            top_roi=top_roi,
            bottom_roi=bottom_roi,
            confidence_score=confidence_score,
            details=details
        )

        log_info(SystemComponent.DETECTION,
                f"Alignment check result: {status.value}, Top: {top_overlap:.2f}, Bottom: {bottom_overlap:.2f}")

        return result

    def _determine_alignment_status(self, top_overlap: float, bottom_overlap: float) -> AlignmentStatus:
        """
        Determine the overall alignment status based on overlap percentages.

        Args:
            top_overlap: Overlap percentage with top ROI
            bottom_overlap: Overlap percentage with bottom ROI

        Returns:
            AlignmentStatus enumeration value
        """
        top_aligned = top_overlap >= self.min_overlap_threshold
        bottom_aligned = bottom_overlap >= self.min_overlap_threshold

        if top_aligned and bottom_aligned:
            return AlignmentStatus.ALIGNED
        elif not top_aligned and not bottom_aligned:
            return AlignmentStatus.MISALIGNED_BOTH
        elif not top_aligned:
            return AlignmentStatus.MISALIGNED_TOP
        elif not bottom_aligned:
            return AlignmentStatus.MISALIGNED_BOTTOM
        else:
            return AlignmentStatus.INSUFFICIENT_OVERLAP

    def draw_alignment_overlay(self, frame: np.ndarray, alignment_result: AlignmentResult) -> np.ndarray:
        """
        Draw alignment visualization overlay on the frame.

        Args:
            frame: Input frame
            alignment_result: AlignmentResult from check_wood_alignment

        Returns:
            Frame with alignment overlay drawn
        """
        overlay_frame = frame.copy()

        # Draw ROIs
        self._draw_roi(overlay_frame, alignment_result.top_roi)
        self._draw_roi(overlay_frame, alignment_result.bottom_roi)

        # Draw wood bounding box with color based on alignment status
        if alignment_result.wood_bbox is not None:
            bbox_color = self._get_bbox_color(alignment_result.status)
            self._draw_wood_bbox(overlay_frame, alignment_result.wood_bbox, bbox_color)

        # Draw alignment status text
        self._draw_status_text(overlay_frame, alignment_result)

        return overlay_frame

    def _draw_roi(self, frame: np.ndarray, roi: AlignmentROI) -> None:
        """Draw ROI rectangle on frame"""
        cv2.rectangle(frame, (roi.x1, roi.y1), (roi.x2, roi.y2), roi.color, 2)

        # Add ROI label
        label = f"{roi.name.replace('_', ' ').title()}"
        cv2.putText(frame, label, (roi.x1 + 10, roi.y1 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi.color, 2)

    def _draw_wood_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                        color: Tuple[int, int, int]) -> None:
        """Draw wood bounding box with specified color"""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Add wood label
        cv2.putText(frame, "Wood Detected", (x1 + 10, y1 + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def _get_bbox_color(self, status: AlignmentStatus) -> Tuple[int, int, int]:
        """Get bounding box color based on alignment status"""
        if status == AlignmentStatus.ALIGNED:
            return self.aligned_color
        else:
            return self.misaligned_color

    def _draw_status_text(self, frame: np.ndarray, result: AlignmentResult) -> None:
        """Draw alignment status text on frame"""
        # Position text in top-left corner
        text_lines = [
            f"Alignment: {result.status.value.replace('_', ' ').title()}",
            ".1f",
            ".1f",
            ".1f"
        ]

        y_offset = 30
        for line in text_lines:
            cv2.putText(frame, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_offset += 25

    def update_configuration(self, new_config: Dict[str, Any]) -> None:
        """
        Update alignment module configuration.

        Args:
            new_config: New configuration dictionary
        """
        self.alignment_config.update(new_config)

        # Update instance variables
        self.top_roi_margin_percent = self.alignment_config.get('top_roi_margin_percent', 0.15)
        self.bottom_roi_margin_percent = self.alignment_config.get('bottom_roi_margin_percent', 0.15)
        self.min_overlap_threshold = self.alignment_config.get('min_overlap_threshold', 0.6)
        self.alignment_tolerance_percent = self.alignment_config.get('alignment_tolerance_percent', 0.1)

        log_info(SystemComponent.DETECTION, "Alignment module configuration updated")

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get current configuration summary"""
        return {
            "top_roi_margin_percent": self.top_roi_margin_percent,
            "bottom_roi_margin_percent": self.bottom_roi_margin_percent,
            "min_overlap_threshold": self.min_overlap_threshold,
            "alignment_tolerance_percent": self.alignment_tolerance_percent,
            "roi_color": self.roi_color,
            "aligned_color": self.aligned_color,
            "misaligned_color": self.misaligned_color
        }