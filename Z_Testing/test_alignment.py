#!/usr/bin/env python3
"""
Test script for the alignment module functionality
"""

from modules.alignment_module import AlignmentModule, AlignmentStatus, AlignmentResult
from config.settings import get_config

def test_alignment_module():
    """Test the alignment module with different scenarios"""
    print("Testing Alignment Module...")

    # Get configuration
    config = get_config()

    # Initialize alignment module
    alignment_module = AlignmentModule(config)

    print(f"Alignment Module initialized with config:")
    print(f"  Frame dimensions: {alignment_module.frame_width}x{alignment_module.frame_height}")
    print(f"  Top ROI margin: {alignment_module.top_roi_margin_percent * 100}%")
    print(f"  Bottom ROI margin: {alignment_module.bottom_roi_margin_percent * 100}%")
    print(f"  Min overlap threshold: {alignment_module.min_overlap_threshold * 100}%")

    # Test 1: Properly aligned wood
    print("\n=== Test 1: Properly Aligned Wood ===")
    # Wood bbox that overlaps well with both ROIs
    wood_bbox1 = [300, 200, 700, 500]  # Centered wood piece
    result1 = alignment_module.check_wood_alignment(None, wood_bbox1)
    print(f"Status: {result1.status.value}")
    print(".1f")
    print(f"Message: {result1.message}")

    # Test 2: Misaligned wood (shifted to the right)
    print("\n=== Test 2: Misaligned Wood (Shifted Right) ===")
    # Shift wood to the right - should have poor overlap with left ROIs
    wood_bbox2 = [600, 200, 1000, 500]  # Right-shifted wood piece
    result2 = alignment_module.check_wood_alignment(None, wood_bbox2)
    print(f"Status: {result2.status.value}")
    print(".1f")
    print(f"Message: {result2.message}")

    # Test 3: No wood detected
    print("\n=== Test 3: No Wood Detected ===")
    result3 = alignment_module.check_wood_alignment(None, None)
    print(f"Status: {result3.status.value}")
    print(f"Message: {result3.message}")

    # Test 4: Partially aligned wood
    print("\n=== Test 4: Partially Aligned Wood ===")
    # Wood that overlaps with one ROI but not the other
    wood_bbox4 = [200, 200, 600, 500]  # Left-shifted wood piece
    result4 = alignment_module.check_wood_alignment(None, wood_bbox4)
    print(f"Status: {result4.status.value}")
    print(".1f")
    print(f"Message: {result4.message}")

    print("\n=== Test Results Summary ===")
    print(f"Test 1 (Centered/Aligned): {result1.status.value}")
    print(f"Test 2 (Right-shifted/Misaligned): {result2.status.value}")
    print(f"Test 3 (No Wood): {result3.status.value}")
    print(f"Test 4 (Left-shifted/Partial): {result4.status.value}")

    print("\n=== ROI Information ===")
    print(f"Top ROI: x={alignment_module.top_roi[0]}-{alignment_module.top_roi[2]}, y={alignment_module.top_roi[1]}-{alignment_module.top_roi[3]}")
    print(f"Bottom ROI: x={alignment_module.bottom_roi[0]}-{alignment_module.bottom_roi[2]}, y={alignment_module.bottom_roi[1]}-{alignment_module.bottom_roi[3]}")

    print("\nAlignment module test completed successfully!")

if __name__ == "__main__":
    test_alignment_module()