# Lane Alignment Collision Detection

## Overview
The wood sorting system now includes **automatic misalignment detection** when the wood (AUTO ROI) touches the red alignment lanes (highway-style lanes).

## Feature Description

### What It Does
- **Monitors** the AUTO ROI (wood detection bounding box) position in real-time
- **Detects** when the wood intersects with either the TOP or BOTTOM alignment lanes
- **Warns** the user visually and via console messages when misalignment occurs

### Visual Feedback

#### Normal Aligned Wood
- **AUTO ROI**: Yellow border with "AUTO ROI" label
- **Lanes**: Semi-transparent red lanes at top and bottom
- **Status**: No warnings

```
┌─────────────────────────────┐
│     TOP LANE (RED)         │ ← Red semi-transparent zone
├─────────────────────────────┤
│                             │
│    ┌──────────────┐        │
│    │  AUTO ROI    │ ← Yellow border (aligned)
│    │  (YELLOW)    │
│    └──────────────┘        │
│                             │
├─────────────────────────────┤
│   BOTTOM LANE (RED)        │ ← Red semi-transparent zone
└─────────────────────────────┘
```

#### Misaligned Wood (Touching Top Lane)
- **AUTO ROI**: Changes to RED border with red semi-transparent overlay
- **Warning Text**: "⚠ MISALIGNED - TOP LANE" appears above AUTO ROI
- **Console**: Prints warning message

```
┌─────────────────────────────┐
│     TOP LANE (RED)         │
├─────────────────────────────┤
│  ⚠ MISALIGNED - TOP LANE   │ ← Warning text (RED)
│  ┌──────────────┐          │
│  │  AUTO ROI    │ ← RED border + red overlay (WARNING!)
│  │  (RED)       │
│  └──────────────┘          │
│                             │
├─────────────────────────────┤
│   BOTTOM LANE (RED)        │
└─────────────────────────────┘
```

#### Misaligned Wood (Touching Bottom Lane)
- **AUTO ROI**: Changes to RED border with red semi-transparent overlay
- **Warning Text**: "⚠ MISALIGNED - BOTTOM LANE" appears above AUTO ROI
- **Console**: Prints warning message

```
┌─────────────────────────────┐
│     TOP LANE (RED)         │
├─────────────────────────────┤
│                             │
│  ┌──────────────┐          │
│  │  AUTO ROI    │ ← RED border + red overlay (WARNING!)
│  │  (RED)       │
│  └──────────────┘          │
│  ⚠ MISALIGNED - BOTTOM LANE│ ← Warning text (RED)
├─────────────────────────────┤
│   BOTTOM LANE (RED)        │
└─────────────────────────────┘
```

## Technical Implementation

### Collision Detection Algorithm
Uses **AABB (Axis-Aligned Bounding Box)** collision detection:

```python
def check_roi_collision(self, roi1_x, roi1_y, roi1_w, roi1_h, 
                        roi2_x, roi2_y, roi2_w, roi2_h):
    """
    Check if two rectangular ROIs overlap/collide.
    
    Two rectangles overlap if:
    1. Left edge of rect1 is to the left of right edge of rect2
    2. Right edge of rect1 is to the right of left edge of rect2
    3. Top edge of rect1 is above bottom edge of rect2
    4. Bottom edge of rect1 is below top edge of rect2
    """
    roi1_x2 = roi1_x + roi1_w
    roi1_y2 = roi1_y + roi1_h
    roi2_x2 = roi2_x + roi2_w
    roi2_y2 = roi2_y + roi2_h
    
    overlap = (roi1_x < roi2_x2 and 
              roi1_x2 > roi2_x and 
              roi1_y < roi2_y2 and 
              roi1_y2 > roi2_y)
    
    return overlap
```

### Integration Points

#### 1. First `draw_wood_detection_overlay` Function (Line ~2006)
- Used for visualization in wood detection module
- Checks collision after drawing AUTO ROI from `wood_detection.get('auto_roi')`

#### 2. Second `draw_wood_detection_overlay` Function (Line ~4111)
- Used for live detection feed
- Checks collision after drawing dynamic ROI from `self.dynamic_roi[camera_name]`

### Warning Display Logic

When collision detected:
1. **Create warning overlay**: Semi-transparent red fill on AUTO ROI
2. **Blend overlay**: 30% red overlay + 70% original frame
3. **Draw red border**: 3px thick red rectangle around AUTO ROI
4. **Add warning text**: Red text above AUTO ROI indicating which lane
5. **Console output**: Print warning message for logging

```python
if self.check_roi_collision(roi_x, roi_y, roi_w, roi_h, ...):
    # Draw WARNING overlay
    warning_overlay = overlay_frame.copy()
    cv2.rectangle(warning_overlay, (roi_x, roi_y), 
                 (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), -1)
    cv2.addWeighted(warning_overlay, 0.3, overlay_frame, 0.7, 0, overlay_frame)
    
    # Draw red border
    cv2.rectangle(overlay_frame, (roi_x, roi_y), 
                 (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 3)
    
    # Add warning text
    cv2.putText(overlay_frame, "⚠ MISALIGNED - TOP LANE", ...)
    
    # Console warning
    print(f"⚠ WARNING [{camera_name}]: Wood is MISALIGNED - touching TOP lane!")
```

## Configuration

### Lane ROI Dimensions (from config.py)
```python
ALIGNMENT_LANE_ROIS = {
    "top": {
        "top_lane": {"x1": 0, "y1": 0, "x2": 1280, "y2": 100},      # Top 100px
        "bottom_lane": {"x1": 0, "y1": 620, "x2": 1280, "y2": 720}  # Bottom 100px
    },
    "bottom": {
        "top_lane": {"x1": 0, "y1": 0, "x2": 1280, "y2": 100},
        "bottom_lane": {"x1": 0, "y1": 620, "x2": 1280, "y2": 720}
    }
}
```

### Checkbox Control
- **Lane ROI Checkbox**: Enable/disable lane visibility AND collision detection
- **Default**: Enabled (checked)
- **Location**: ROI panel in GUI

## Usage

### For Operators
1. **Enable Lane ROI**: Check the "Lane ROI" checkbox in the ROI panel
2. **Run detection**: Start conveyor (ON button)
3. **Monitor alignment**: Watch for red warnings on AUTO ROI
4. **Adjust wood position**: If warnings appear, wood is misaligned

### Console Warnings
Monitor terminal output for alignment warnings:
```
⚠ WARNING [top]: Wood is MISALIGNED - touching TOP lane!
⚠ WARNING [bottom]: Wood is MISALIGNED - touching BOTTOM lane!
```

### Expected Behavior
- **Aligned wood**: AUTO ROI stays yellow, no warnings
- **Misaligned wood**: AUTO ROI turns red, warning text appears
- **Multiple cameras**: Both top and bottom cameras detect independently

## Troubleshooting

### No Warnings Appearing
1. **Check Lane ROI checkbox**: Must be enabled
2. **Verify wood detection**: AUTO ROI must be visible (yellow box)
3. **Check console**: Should see debug messages if lanes are drawing

### False Warnings
- **Lane dimensions too large**: Adjust `ALIGNMENT_LANE_ROIS` in config.py
- **Wood too wide**: May be normal for wider wood pieces

### Warnings Not Clearing
- **Persistent collision**: Wood is still touching lanes
- **Frame lag**: Wait for next frame update (60 FPS)

## Future Enhancements

### Possible Improvements
1. **Alignment score**: Calculate percentage of misalignment
2. **Historical tracking**: Log alignment issues over time
3. **Automatic rejection**: Trigger Arduino to reject misaligned pieces
4. **Adjustable sensitivity**: Configure lane width dynamically
5. **Audio alerts**: Add sound warnings for critical misalignment

## Related Files
- `testIR.py` (lines 1165-1195): `check_roi_collision()` function
- `testIR.py` (lines 2006-2048): First collision detection implementation
- `testIR.py` (lines 4111-4157): Second collision detection implementation
- `config.py` (lines 165-195): `ALIGNMENT_LANE_ROIS` configuration

## Testing Recommendations

### Test Cases
1. **Perfectly aligned wood**: No warnings should appear
2. **Wood touching top lane**: Should show "MISALIGNED - TOP LANE"
3. **Wood touching bottom lane**: Should show "MISALIGNED - BOTTOM LANE"
4. **Wood touching both lanes**: Should show both warnings
5. **Lane ROI disabled**: No warnings, no lanes visible
6. **No wood detected**: No warnings (no AUTO ROI to check)

### Validation
- ✅ AUTO ROI turns from yellow to red when misaligned
- ✅ Warning text appears above AUTO ROI
- ✅ Console prints warning messages
- ✅ Semi-transparent red overlay appears on AUTO ROI
- ✅ Collision detection works for both cameras independently

---

**Last Updated**: October 16, 2025  
**Status**: ✅ Implemented and tested
