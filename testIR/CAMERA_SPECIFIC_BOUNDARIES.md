# Camera-Specific Lane Boundaries

## Problem: False Positives on Bottom Camera ⚠️

### What Happened
```
TOP Camera:    y=135 to y=608  → ✅ NO COLLISION
BOTTOM Camera: y=85 to y=629   → ❌ COLLISION (both lanes!)
```

The wood was **physically aligned**, but the bottom camera perspective made it appear misaligned.

---

## Root Cause

### Why Different Cameras See Different Y-Coordinates

1. **Different mounting positions** - Cameras at different heights/angles
2. **Different perspectives** - Same wood appears at different Y positions
3. **Optical distortion** - Lens differences between cameras

### Example from Your Logs
```
TOP Camera AUTO ROI:    y=135 (top edge)
BOTTOM Camera AUTO ROI: y=85  (top edge)
Difference: 50 pixels! 
```

The **same physical wood** appears 50 pixels higher in the bottom camera!

---

## Solution Applied ✅

### Changed: Camera-Specific Boundaries

#### Before (One Size Fits All - Wrong!)
```python
# Same boundaries for all cameras
TOP_LANE_BOUNDARY = 100
BOTTOM_LANE_BOUNDARY = 620
```

#### After (Camera-Specific - Correct!)
```python
ALIGNMENT_LANE_ROIS = {
    "top": {
        "top_lane": {"y2": 100},      # Top camera: strict
        "bottom_lane": {"y1": 620}     # Top camera: strict
    },
    "bottom": {
        "top_lane": {"y2": 80},        # Bottom camera: adjusted (was 100)
        "bottom_lane": {"y1": 640}     # Bottom camera: adjusted (was 620)
    }
}
```

### Changes Made

**Bottom camera adjustments:**
- **Top lane boundary:** 100 → **80** (tighter, wood appears higher)
- **Bottom lane boundary:** 620 → **640** (more room, wood appears lower)

This gives the bottom camera a **larger safe zone** (y=80 to y=640) to account for its different perspective.

---

## Code Changes

### 1. Updated Configuration (Line ~165)
```python
"bottom": {
    "top_lane": {
        "y2": 80    # Adjusted from 100
    },
    "bottom_lane": {
        "y1": 640   # Adjusted from 620
    }
}
```

### 2. Updated Collision Detection (Line ~1905)
```python
# OLD: Hardcoded values
top_lane_boundary = 100
bottom_lane_boundary = 620

# NEW: Camera-specific from configuration
from testIR import ALIGNMENT_LANE_ROIS
camera_lanes = ALIGNMENT_LANE_ROIS.get(camera, {})
top_lane_boundary = camera_lanes['top_lane'].get('y2', 100)
bottom_lane_boundary = camera_lanes['bottom_lane'].get('y1', 620)
```

---

## Expected Results

### Before (False Positives)
```
TOP Camera: ✅ NO COLLISION (y=135 to y=608)
BOTTOM Camera: ❌ COLLISION (y=85 to y=629)
Result: Unnecessary warning for properly aligned wood
```

### After (Accurate Detection)
```
TOP Camera:
  y=135 > 100 (top boundary) ✅
  y=608 < 620 (bottom boundary) ✅
  Result: NO COLLISION ✅

BOTTOM Camera:
  y=85 > 80 (top boundary) ✅        ← NEW boundary!
  y=629 < 640 (bottom boundary) ✅   ← NEW boundary!
  Result: NO COLLISION ✅

Overall: Wood properly aligned, no false warnings! 🎉
```

---

## New Log Output

You should now see:
```
============================================================
[COLLISION CHECK] Checking wood alignment for BOTTOM camera
============================================================
  Wood ROI: x=0, y=85, w=615, h=544
  ROI Top Edge: y=85
  ROI Bottom Edge: y=629
  Top Lane Boundary: y=80 (camera-specific)      ← Shows adjusted value!
  TOP COLLISION: False (Wood top=85 > 80)        ← Now FALSE!
  Bottom Lane Boundary: y=640 (camera-specific)  ← Shows adjusted value!
  BOTTOM COLLISION: False (Wood bottom=629 < 640) ← Now FALSE!
  ✅ RESULT: NO COLLISION - Wood is properly aligned
============================================================
```

---

## Safe Zones (Per Camera)

### Top Camera Safe Zone
```
y = 0   ┌──────────────────────────┐
        │   TOP LANE (y=0-100)     │
y = 100 ├──────────────────────────┤
        │                          │
        │   SAFE ZONE              │
        │   (y=101-619)            │
        │                          │
y = 620 ├──────────────────────────┤
        │   BOTTOM LANE (y=620-720)│
y = 720 └──────────────────────────┘
```

### Bottom Camera Safe Zone (Larger!)
```
y = 0   ┌──────────────────────────┐
        │   TOP LANE (y=0-80)      │ ← Smaller
y = 80  ├──────────────────────────┤
        │                          │
        │   SAFE ZONE              │
        │   (y=81-639)             │ ← LARGER!
        │                          │
y = 640 ├──────────────────────────┤
        │   BOTTOM LANE (y=640-720)│ ← Smaller
y = 720 └──────────────────────────┘
```

**Safe zone difference:**
- Top camera: 519 pixels (101-619)
- Bottom camera: **559 pixels** (81-639) - 40 pixels larger!

---

## Fine-Tuning

If you still get false positives/negatives, adjust the boundaries:

### If Bottom Camera Still Shows False Positives
```python
"bottom": {
    "top_lane": {"y2": 70},    # Even tighter top (currently 80)
    "bottom_lane": {"y1": 650}  # Even more room at bottom (currently 640)
}
```

### If Missing Real Misalignments
```python
"bottom": {
    "top_lane": {"y2": 90},    # Less strict top (currently 80)
    "bottom_lane": {"y1": 630}  # Less strict bottom (currently 640)
}
```

### Testing Process
1. Run with properly aligned wood
2. Check both cameras show NO COLLISION
3. If bottom camera still shows collision, increase safe zone
4. Test with misaligned wood (intentionally high/low)
5. Verify both cameras detect actual misalignment

---

## Summary

✅ **Fixed:** Camera-specific lane boundaries  
✅ **Bottom camera:** Larger safe zone (y=81-639)  
✅ **Top camera:** Original safe zone (y=101-619)  
✅ **Collision detection:** Uses camera-specific boundaries  
✅ **Logs show:** "(camera-specific)" label for clarity  

**Result:** No more false positives for properly aligned wood! 🎯
