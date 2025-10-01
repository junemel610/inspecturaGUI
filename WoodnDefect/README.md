# WoodnDefect

This folder contains Python scripts for wood detection and defect analysis using computer vision and AI models.

## Files

### WoodnDefect.py

A comprehensive wood detection and defect analysis script that integrates color-based wood detection with AI-powered defect detection.

**Features:**
- **Wood Detection**: Uses RGB color segmentation and contour analysis to detect wood planks
- **Defect Detection**: Employs DeGirum AI model (V2DefectCombined) to identify wood defects within detected wood regions
- **Real-time Processing**: Processes video streams from dual cameras (top and bottom views)
- **ROI-based Analysis**: Automatically generates regions of interest (ROI) around detected wood for targeted defect inspection
- **Visualization**: Displays bounding boxes for wood detection (green) and defects (red) on live video feed
- **Defect Classes**: Detects Dead Knots, Knots with Crack, Live Knots, and Missing Knots

**Key Components:**
- `ColorWoodDetector` class: Main detector with wood and defect detection methods
- `CameraHandler` class: Manages camera initialization and settings
- Real-time video processing loop with dual camera support

**Usage:**
```bash
python WoodnDefect.py
```

**Requirements:**
- OpenCV
- NumPy
- DeGirum (degirum, degirum_tools)
- Dual camera setup (indices 0 and 2)

### rgb_wood_detector.py

Similar to WoodnDefect.py, this script provides the same functionality with integrated wood and defect detection capabilities.

**Features:**
- Identical functionality to WoodnDefect.py
- Color-based wood detection with automatic ROI generation
- DeGirum model integration for defect analysis
- Real-time dual camera processing
- Enhanced visualization with defect bounding boxes

**Usage:**
```bash
python rgb_wood_detector.py
```

## Model Configuration

- **Defect Model**: V2DefectCombined--640x640_quant_hailort_hailo8_1
- **Confidence Threshold**: 0.40 (40%)
- **Defect Classes**: Dead Knots, Knots with Crack, Live Knots, Missing Knots

## Output

- **Video Feed**: Real-time display with wood (green boxes) and defect (red boxes) annotations
- **Console Logs**: Detection results, confidence scores, and defect information every 30 frames
- **Status Overlays**: Camera status, detection counts, and confidence scores on video feed

## Controls

- **'q'**: Quit application
- **'d'**: Toggle detection on/off
- **'c'**: Toggle between video and mask view
- **'s'**: Save current frames

## Dependencies

Install required packages:
```bash
pip install opencv-python numpy degirum
```

## Notes

- Requires cameras at indices 0 (top) and 2 (bottom)
- Optimized for 1280x720 resolution
- Defect detection is performed only within automatically generated wood ROIs
- Multiple defects in the same area are resolved by selecting the highest confidence detection