# Collision Detection Fix - Parent App Reference

## Problem
The collision detection logs were not appearing because `ColorWoodDetector` class didn't have access to:
- `lane_roi_var` (GUI checkbox state)
- `show_alignment_warning()` (notification trigger method)
- `clear_alignment_warning()` (warning clear method)

## Solution
Pass a reference to the parent application when creating the `ColorWoodDetector` instance.

---

## Code Changes

### 1. ColorWoodDetector.__init__() - Accept Parent Reference
```python
class ColorWoodDetector:
    def __init__(self, parent_app=None):
        self.parent_app = parent_app  # ← NEW: Reference to main app
        # ... rest of initialization
```

### 2. Main App - Pass Self Reference
```python
# Initialize RGB Wood Detector for dynamic ROI generation
self.rgb_wood_detector = ColorWoodDetector(parent_app=self)  # ← Pass self
```

### 3. Collision Detection - Use Parent Reference
```python
# OLD: Checked self (doesn't exist in ColorWoodDetector)
if auto_roi and hasattr(self, 'lane_roi_var') and self.lane_roi_var.get():

# NEW: Check parent_app
if auto_roi and self.parent_app and hasattr(self.parent_app, 'lane_roi_var') and self.parent_app.lane_roi_var.get():
    # ... collision detection ...
    
    # Call parent methods
    self.parent_app.show_alignment_warning(camera, "TOP")
    self.parent_app.clear_alignment_warning(camera)
```

### 4. Debug Logging - Show Why Skipped
```python
else:
    # Lane ROI is disabled or parent_app not available
    if auto_roi:
        if not self.parent_app:
            print(f"[COLLISION CHECK] Skipped - parent_app not set")
        elif not hasattr(self.parent_app, 'lane_roi_var'):
            print(f"[COLLISION CHECK] Skipped - lane_roi_var not available")
        elif not self.parent_app.lane_roi_var.get():
            print(f"[COLLISION CHECK] Skipped - Lane ROI checkbox is unchecked")
```

---

## Now You Should See

### When Wood is Detected:
```
🎯 Auto ROI generated: (0, 45, 535, 497)

============================================================
[COLLISION CHECK] Checking wood alignment for TOP camera
============================================================
  Wood ROI: x=0, y=45, w=535, h=497
  ROI Top Edge: y=45
  ROI Bottom Edge: y=542
  Top Lane Boundary: y=100
  TOP COLLISION: False (Wood top=45 > 100)
  Bottom Lane Boundary: y=620
  BOTTOM COLLISION: False (Wood bottom=542 < 620)
  ✅ RESULT: NO COLLISION - Wood is properly aligned
============================================================

✅ Detection complete: wood_detected=True, count=1, confidence=0.20
```

### If Lane ROI is Unchecked:
```
🎯 Auto ROI generated: (0, 45, 535, 497)
[COLLISION CHECK] Skipped - Lane ROI checkbox is unchecked
✅ Detection complete: wood_detected=True, count=1, confidence=0.20
```

---

## Test Now

**Run the application:**
```bash
cd /home/inspectura/Desktop/InspecturaGUI
python testIR/testIR.py
```

**What to verify:**
1. ✅ `[COLLISION CHECK]` section appears after `🎯 Auto ROI generated`
2. ✅ Shows exact Y-coordinates for wood position
3. ✅ Shows True/False for TOP and BOTTOM collision
4. ✅ Shows final result (COLLISION or NO COLLISION)

**If still not appearing:**
- Check if you see "[COLLISION CHECK] Skipped - ..." message
- This will tell you why collision check is being skipped

---

## Architecture

```
Main Application (GUI)
  ├─> self.lane_roi_var (BooleanVar for checkbox)
  ├─> self.show_alignment_warning() (notification method)
  ├─> self.clear_alignment_warning() (clear method)
  │
  └─> self.rgb_wood_detector = ColorWoodDetector(parent_app=self)
        │
        └─> self.parent_app = reference to Main Application
              │
              └─> Can access:
                    - self.parent_app.lane_roi_var.get()
                    - self.parent_app.show_alignment_warning()
                    - self.parent_app.clear_alignment_warning()
```

This creates a **two-way communication** between the detector and the main app! 🔄
