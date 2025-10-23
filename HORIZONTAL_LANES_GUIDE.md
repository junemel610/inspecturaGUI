# Horizontal Lane Alignment - Visual Guide

## Updated Lane Orientation: TOP and BOTTOM Lanes

The lanes are now positioned **horizontally** at the **top and bottom** of the frame, similar to highway lanes running across the road.

## Visual Layout

```
┌─────────────────────────────────────────────────────────┐
│               CAMERA FEED (1280 x 720)                  │
│                                                         │
│  ═══════════════════════════════════════════════════   │
│  ║  🔴 TOP LANE (Semi-transparent Red)          ║   │
│  ║         TOP LANE (label)                     ║   │
│  ═══════════════════════════════════════════════════   │
│  ┌─────────────────────────────────────────────┐       │
│  │                                             │       │
│  │             SAFE ZONE                       │       │
│  │        (Wood should be here)                │       │
│  │                                             │       │
│  │         ┌───────────────┐                   │       │
│  │         │  WOOD BOX     │                   │       │
│  │         │  (GREEN)      │                   │       │
│  │         └───────────────┘                   │       │
│  │                                             │       │
│  │        Properly Aligned                     │       │
│  │                                             │       │
│  └─────────────────────────────────────────────┘       │
│  ═══════════════════════════════════════════════════   │
│  ║  🔴 BOTTOM LANE (Semi-transparent Red)       ║   │
│  ║         BOTTOM LANE (label)                  ║   │
│  ═══════════════════════════════════════════════════   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Configuration

### Top Camera Lanes
```python
"top": {
    "top_lane": {
        "x1": 345,   # Left edge
        "y1": 0,     # Top of frame
        "x2": 880,   # Right edge
        "y2": 100    # 100 pixels tall
    },
    "bottom_lane": {
        "x1": 345,   # Left edge
        "y1": 620,   # Start 100px from bottom
        "x2": 880,   # Right edge
        "y2": 720    # Bottom of frame (100 pixels tall)
    }
}
```

### Bottom Camera Lanes
```python
"bottom": {
    "top_lane": {
        "x1": 350,   # Left edge
        "y1": 0,     # Top of frame
        "x2": 965,   # Right edge
        "y2": 100    # 100 pixels tall
    },
    "bottom_lane": {
        "x1": 350,   # Left edge
        "y1": 620,   # Start 100px from bottom
        "x2": 965,   # Right edge
        "y2": 720    # Bottom of frame (100 pixels tall)
    }
}
```

## Lane Dimensions

### Top Camera
- **Top Lane**: 100 pixels tall (y: 0 to 100)
- **Bottom Lane**: 100 pixels tall (y: 620 to 720)
- **Safe Zone**: 520 pixels tall (y: 100 to 620)

### Bottom Camera
- **Top Lane**: 100 pixels tall (y: 0 to 100)
- **Bottom Lane**: 100 pixels tall (y: 620 to 720)
- **Safe Zone**: 520 pixels tall (y: 100 to 620)

## Alignment Examples

### ✅ Proper Alignment (GOOD)
```
═══════════════════════════════════
║  🔴 TOP LANE (Red Zone)      ║
═══════════════════════════════════
┌─────────────────────────────────┐
│                                 │
│      ┌───────────────┐          │
│      │  🟢 WOOD BOX  │          │
│      │   (Centered)  │          │
│      └───────────────┘          │
│                                 │
└─────────────────────────────────┘
═══════════════════════════════════
║  🔴 BOTTOM LANE (Red Zone)   ║
═══════════════════════════════════
```
**Status**: ✅ No warning - wood is properly centered

---

### ⚠️ Top Misalignment (WARNING)
```
═══════════════════════════════════
║  🔴 TOP LANE (Red Zone)      ║
║  ┌───────────────┐            ║
═══╪═══════════════╪═════════════╪═
   │  🟢 WOOD BOX  │
   │  (Too High!)  │
   └───────────────┘
```
**Status**: ⚠️ Warning appears:
> Wood Misalignment Detected on TOP camera!
> Wood is touching the top lane boundary.

---

### ⚠️ Bottom Misalignment (WARNING)
```
   ┌───────────────┐
   │  🟢 WOOD BOX  │
   │  (Too Low!)   │
═══╪═══════════════╪═════════════╪═
║  └───────────────┘            ║
║  🔴 BOTTOM LANE (Red Zone)   ║
═══════════════════════════════════
```
**Status**: ⚠️ Warning appears:
> Wood Misalignment Detected on TOP camera!
> Wood is touching the bottom lane boundary.

---

### ⚠️⚠️ Both Lanes (SEVERE)
```
═══════════════════════════════════
║  ┌─────────────────────────┐ ║
═══╪═════════════════════════╪═╪═
   │  🟢 WOOD BOX (TOO TALL) │
   │                         │
   │                         │
═══╪═════════════════════════╪═╪═
║  └─────────────────────────┘ ║
═══════════════════════════════════
```
**Status**: ⚠️⚠️ Warning appears:
> Wood Misalignment Detected on TOP camera!
> Wood is touching the both lane boundary.

## Visual Indicators

### Colors
- 🔴 **Red semi-transparent zones** (30% opacity) - Top and bottom lanes
- 🔴 **Red solid borders** (3px thick) - Lane boundaries
- ⚪ **White text** - "TOP LANE" and "BOTTOM LANE" labels
- 🟢 **Green box** - Detected wood

### Text Labels
- **"TOP LANE"** - Centered in top red zone
- **"BOTTOM LANE"** - Centered in bottom red zone
- Horizontal orientation (easy to read)
- Large font (0.8 scale)

## How It Works

### Detection Logic
1. **Wood detected** → Get bounding box coordinates
2. **Check top lane** → Does wood bbox intersect top lane ROI?
3. **Check bottom lane** → Does wood bbox intersect bottom lane ROI?
4. **Show warning** → If any intersection detected

### Intersection Detection
Wood bbox intersects lane if:
- Wood's top edge (y1) < Lane's bottom edge (y2) **AND**
- Wood's bottom edge (y2) > Lane's top edge (y1) **AND**
- Wood overlaps horizontally with lane

## Use Cases

### Perfect for Detecting:
✅ **Wood too high** - Touching top lane  
✅ **Wood too low** - Touching bottom lane  
✅ **Wood too tall** - Touching both lanes  
✅ **Vertical misalignment** - Off-center vertically  

### Safe Zone
The center area (y: 100 to 620) is where wood should be:
- **Height**: 520 pixels
- **Position**: Between top and bottom lanes
- **No warnings** when wood stays here

## Customization

### Adjust Lane Height
```python
# Make lanes taller (more sensitive)
"y2": 150,  # Top lane: was 100, now 150px tall
"y1": 570,  # Bottom lane: starts higher

# Make lanes shorter (less sensitive)
"y2": 50,   # Top lane: now 50px tall
"y1": 670,  # Bottom lane: starts lower
```

### Adjust Safe Zone
```python
# Larger safe zone (less sensitive)
"y2": 80,   # Top lane shorter
"y1": 640,  # Bottom lane starts lower

# Smaller safe zone (more sensitive)
"y2": 150,  # Top lane taller
"y1": 570,  # Bottom lane starts higher
```

## Advantages

✅ **Natural orientation** - Matches horizontal wood movement  
✅ **Easy to understand** - Top/bottom is intuitive  
✅ **Better coverage** - Full width of detection area  
✅ **Vertical alignment** - Detects up/down misalignment  
✅ **Professional appearance** - Clean horizontal lanes  

## Testing

1. **Run the application**:
   ```bash
   python testIR/testIR.py
   ```

2. **Enable Live Detection**

3. **Look for horizontal red lanes**:
   - Top red zone at top of frame
   - Bottom red zone at bottom of frame
   - "TOP LANE" and "BOTTOM LANE" labels

4. **Test alignment**:
   - Wood centered → No warning
   - Wood high → Top lane warning
   - Wood low → Bottom lane warning
   - Wood very tall → Both lanes warning

## Summary

The lanes are now positioned **horizontally** at the **top and bottom** of the camera frame, providing clear visual feedback for vertical wood alignment! 🎉

```
═══ TOP LANE ═══
     (RED)
                
    SAFE ZONE
   (Wood Here)
                
═══ BOTTOM LANE ═══
     (RED)
```
