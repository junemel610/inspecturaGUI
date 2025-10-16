# Wood Alignment Lane ROI Visualization Guide

## What You'll See on the Camera Feed

When you run the application, you'll now see **red "highway lanes"** on both the TOP and BOTTOM camera feeds.

### Visual Elements

#### 1. **Lane Zones (Semi-Transparent Red)**
- The left and right edges of the detection area will have **semi-transparent red zones**
- These zones show where wood should **NOT** be positioned
- The transparency allows you to still see the camera feed underneath

#### 2. **Lane Borders (Solid Red Lines)**
- Each lane has a **solid red border** (3 pixels thick)
- These clearly mark the boundaries of the no-go zones

#### 3. **Lane Labels (White Text)**
- Each lane has white text labels showing:
  - **"LEFT LANE"** on the left side
  - **"RIGHT LANE"** on the right side

#### 4. **Wood Detection (Green Box)**
- When wood is detected, you'll see a **green bounding box**
- This shows the actual position of the detected wood
- The wood should stay in the **center area** between the red lanes

### Visual Layout

```
┌─────────────────────────────────────────────────────────┐
│                  CAMERA FEED (1280 x 720)               │
│                                                         │
│  ┌─────┬───────────────────────────────────┬─────┐    │
│  │ LEFT│         SAFE ZONE                 │RIGHT│    │
│  │LANE │      (Wood should be here)        │LANE │    │
│  │     │                                   │     │    │
│  │🔴   │         ┌───────────┐             │  🔴 │    │
│  │SEMI │         │  WOOD     │             │ SEMI│    │
│  │TRANS│         │ (GREEN)   │             │TRANS│    │
│  │RED  │         │   BOX     │             │ RED │    │
│  │ZONE │         └───────────┘             │ ZONE│    │
│  │     │                                   │     │    │
│  │     │         Properly Aligned          │     │    │
│  └─────┴───────────────────────────────────┴─────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Camera-Specific Lane Positions

### Top Camera Lanes
- **Left Lane**: x = 345 to 400 (55 pixels wide)
- **Right Lane**: x = 825 to 880 (55 pixels wide)
- **Safe Zone**: 425 pixels wide center area

### Bottom Camera Lanes
- **Left Lane**: x = 350 to 405 (55 pixels wide)
- **Right Lane**: x = 910 to 965 (55 pixels wide)
- **Safe Zone**: 505 pixels wide center area

## Visual Feedback During Operation

### ✅ Proper Alignment (GOOD)
```
┌─────┬───────────────────────────────────┬─────┐
│ 🔴  │                                   │  🔴 │
│ RED │        🟢 GREEN WOOD BOX          │ RED │
│     │         (centered, no touch)      │     │
└─────┴───────────────────────────────────┴─────┘
```
**Status**: No warning - wood is properly centered

---

### ⚠️ Left Misalignment (WARNING)
```
┌─────┬───────────────────────────────────┬─────┐
│ 🔴  │                                   │  🔴 │
│ 🟢  │                                   │ RED │
│GREEN│                                   │     │
│ BOX │                                   │     │
└─────┴───────────────────────────────────┴─────┘
```
**Status**: Warning dialog appears:
> ⚠️ Wood Misalignment Detected on TOP camera!
> Wood is touching the left lane boundary.

---

### ⚠️ Right Misalignment (WARNING)
```
┌─────┬───────────────────────────────────┬─────┐
│ 🔴  │                                   │  🔴 │
│ RED │                                   │ 🟢  │
│     │                                   │GREEN│
│     │                                   │ BOX │
└─────┴───────────────────────────────────┴─────┘
```
**Status**: Warning dialog appears:
> ⚠️ Wood Misalignment Detected on TOP camera!
> Wood is touching the right lane boundary.

## How to Test

1. **Start the application**:
   ```bash
   python testIR/testIR.py
   ```

2. **Enable Live Detection**:
   - Toggle "Live Detection" mode ON
   - The lane ROIs will be visible immediately

3. **Place wood at different positions**:
   - **Center position**: No warning (proper alignment)
   - **Left edge**: Warning appears (touching left lane)
   - **Right edge**: Warning appears (touching right lane)

4. **Observe the visual feedback**:
   - Red semi-transparent zones on edges
   - Green box showing wood position
   - Warning dialog if wood touches red lanes

## Color Reference

| Element | Color | Description |
|---------|-------|-------------|
| Lane zones | 🔴 Red (30% opacity) | Semi-transparent red fill |
| Lane borders | 🔴 Red (solid, 3px) | Clear lane boundaries |
| Lane labels | ⚪ White | "LEFT LANE" / "RIGHT LANE" |
| Wood box | 🟢 Green | Detected wood bounding box |
| Wood label | 🟢 Green | "Wood 1: 0.95" (confidence) |

## Transparency Details

The lane zones use **alpha blending**:
- **30% lane color** (red)
- **70% camera feed** (original image)

This allows you to:
- ✅ See the lane boundaries clearly
- ✅ Still see the camera feed underneath
- ✅ Monitor wood position in real-time
- ✅ Distinguish between safe zone and danger zones

## When Lane ROIs Are Visible

The lane ROIs are displayed:
- ✅ **In Live Detection mode** (when toggle is ON)
- ✅ **In SCAN_PHASE mode**
- ❌ **NOT in IDLE mode** (lanes hidden when not detecting)

This ensures the lanes are visible when you need them and don't clutter the view when idle.

## Customization

If you need to adjust the lane positions or appearance, modify these values in `testIR.py`:

### Lane Position (lines 163-194)
```python
ALIGNMENT_LANE_ROIS = {
    "top": {
        "left_lane": {"x1": 345, "x2": 400, ...},
        "right_lane": {"x1": 825, "x2": 880, ...}
    },
    ...
}
```

### Visual Appearance (in draw_wood_detection_overlay)
```python
# Transparency (0.3 = 30% opacity)
cv2.addWeighted(overlay, 0.3, frame_copy, 0.7, 0, frame_copy)

# Border thickness (3 pixels)
cv2.rectangle(..., (0, 0, 255), 3)

# Text size (0.7 scale)
cv2.putText(..., 0.7, (255, 255, 255), 2)
```

## Troubleshooting

### "I don't see the red lanes"
- ✅ Make sure Live Detection is enabled
- ✅ Check that you're in TRIGGER or CONTINUOUS mode
- ✅ Verify camera feed is displaying

### "Lanes are too dark/bright"
- Adjust the alpha value in `cv2.addWeighted(overlay, 0.3, ...)`:
  - Lower value (0.2) = more transparent
  - Higher value (0.5) = more opaque

### "Lane labels are hard to read"
- Increase text size: change `0.7` to `0.9`
- Increase text thickness: change last `2` to `3`
- Add text background (black rectangle behind text)

## Summary

The alignment lane ROIs provide:
- 🎯 **Visual guidance**: Clear indication of proper wood positioning
- ⚠️ **Real-time warnings**: Immediate feedback on misalignment
- 🔍 **Easy monitoring**: Color-coded zones (red = danger, green = wood)
- 🛠️ **Customizable**: Adjustable positions and appearance
