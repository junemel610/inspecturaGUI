# Wood Lane Alignment Detection - Highway Lane Style

## Overview
This feature implements a highway lane-style misalignment detection system for wood pieces. Two "lane" ROIs (left and right) are placed inside the main TOP and BOTTOM camera ROIs. If the wood detection bounding box touches these lanes, it indicates misalignment and triggers a warning.

## Configuration

### Lane ROI Coordinates
The lane ROIs are defined in the `ALIGNMENT_LANE_ROIS` configuration (lines 163-194 in testIR.py):

```python
ALIGNMENT_LANE_ROIS = {
    "top": {
        "left_lane": {
            "x1": 345,  # Start from left edge of main ROI
            "y1": 0,
            "x2": 400,  # Left lane width (55 pixels)
            "y2": 720
        },
        "right_lane": {
            "x1": 825,  # Start near right edge of main ROI
            "y1": 0,
            "x2": 880,  # Right lane boundary (55 pixels)
            "y2": 720
        }
    },
    "bottom": {
        "left_lane": {
            "x1": 350,  # Start from left edge of main ROI
            "y1": 0,
            "x2": 405,  # Left lane width (55 pixels)
            "y2": 720
        },
        "right_lane": {
            "x1": 910,  # Start near right edge of main ROI
            "y1": 0,
            "x2": 965,  # Right lane boundary (55 pixels)
            "y2": 720
        }
    }
}
```

### Customization
You can adjust the lane positions and widths by modifying:
- **x1, x2**: Horizontal boundaries of each lane
- **y1, y2**: Vertical boundaries (typically full height: 0 to 720)
- **Lane width**: Difference between x2 and x1 (currently 55 pixels for all lanes)

## Implementation

### 1. Lane Alignment Check Function
**Location**: App class, around line 7244

```python
def check_wood_lane_alignment(self, camera_name):
    """Check if wood detection bounding box touches the lane ROIs (highway lane style check)"""
```

This function:
- Gets the wood detection bounding box for the specified camera
- Checks if the wood bbox intersects with left or right lane ROIs
- Displays a warning message if misalignment is detected
- Uses a non-blocking warning dialog to notify the user

### 2. Bounding Box Intersection Helper
**Location**: App class, around line 7304

```python
def _check_bbox_intersection(self, box1_x1, box1_y1, box1_x2, box1_y2,
                              box2_x1, box2_y1, box2_x2, box2_y2):
    """Check if two bounding boxes intersect (overlap)"""
```

This helper function determines if two bounding boxes overlap using standard intersection logic.

### 3. Integration into Detection Pipeline
**Location**: Around line 4033 in the main detection loop

The lane alignment check is called immediately after wood detection results are stored:

```python
# Store wood detection results for overlay display
self.wood_detection_results[camera_name] = wood_detection

# Check for wood lane alignment (highway lane style check)
self.check_wood_lane_alignment(camera_name)
```

### 4. Visual Overlay
**Location**: `draw_wood_detection_overlay` function, around line 1917

The lane ROIs are visualized with:
- **Red borders**: Marking the left and right lane boundaries
- **Labels**: "LEFT LANE" and "RIGHT LANE" text at the top of each lane
- **Wood bounding boxes**: Green boxes showing detected wood position

## How It Works

### Detection Flow
1. **Wood Detection**: System detects wood and gets bounding box coordinates
2. **Lane Check**: Compares wood bbox with left and right lane ROIs
3. **Intersection Test**: Uses bbox intersection algorithm to detect overlap
4. **Warning Display**: If wood touches either lane, shows warning dialog

### Warning System
When misalignment is detected:
- **Console Output**: Prints detailed warning message
- **Warning Dialog**: Shows non-blocking `messagebox.showwarning` dialog
- **Message Content**: Specifies which camera and which lane(s) were touched

Example warning:
```
⚠️ Wood Misalignment Detected on TOP camera!
Wood is touching the left lane boundary.
```

## Visual Indicators

When viewing the camera feed, you'll see:
- **Red rectangles**: Left and right lane boundaries (no-go zones)
- **Green rectangle**: Wood detection bounding box (should stay in center)
- **Yellow rectangle**: Auto ROI for defect detection (if available)

### Proper Alignment
```
|RED|                GREEN WOOD BOX                |RED|
|   |                                              |   |
```

### Misalignment (Left)
```
|RED|                                              |RED|
|GRN|                                              |   |
```

### Misalignment (Right)
```
|RED|                                              |RED|
|   |                                              |GRN|
```

## Benefits

1. **Early Detection**: Identifies misaligned wood before it causes processing issues
2. **Non-Intrusive**: Warning-only system doesn't stop the process
3. **Visual Feedback**: Lane boundaries clearly visible on camera feed
4. **Dual Camera**: Monitors both top and bottom cameras independently
5. **Customizable**: Easy to adjust lane positions and widths

## Adjusting Lane Sensitivity

To make the detection more or less sensitive:

### Wider Lanes (More Sensitive)
Increase the lane width by adjusting the x-coordinates:
```python
"left_lane": {
    "x1": 345,
    "x2": 450,  # Increased from 400 (now 105 pixels wide)
}
```

### Narrower Lanes (Less Sensitive)
Decrease the lane width:
```python
"left_lane": {
    "x1": 345,
    "x2": 375,  # Decreased from 400 (now 30 pixels wide)
}
```

### Center Position
Move lanes closer or further from center by adjusting both boundaries proportionally.

## Testing

To test the lane alignment detection:
1. Run the application in detection mode
2. Place wood at different positions (left, center, right)
3. Observe the visual overlays on the camera feed
4. Check for warning dialogs when wood touches lane boundaries
5. Verify console output for detailed alignment information

## Notes

- **Python-side only**: No Arduino commands are sent
- **Warning only**: Process continues normally after warning
- **Camera-specific**: Each camera has independent lane ROIs
- **Real-time**: Checks performed during every wood detection cycle
- **Non-blocking**: Warning dialog doesn't pause detection
