# Collision Detection Architecture - Refactored

## Overview
The collision detection logic has been moved from the overlay drawing functions to the **wood detection function** (`detect_wood_comprehensive`). This ensures collision checking happens **once** when wood is detected, using the correct coordinates directly from the detection result.

---

## Architecture Changes

### ❌ OLD Architecture (Confusing)
```
detect_wood_comprehensive()
  └─> Detects wood, creates AUTO ROI
  
draw_wood_detection_overlay()  ← Called every frame
  └─> Reads stored detection result
  └─> Re-checks collision using ROI coordinates
  └─> Draws visual overlay
  └─> Triggers notifications

Problem: Collision logic mixed with drawing logic, 
         checking coordinates multiple times, confusing variable names
```

### ✅ NEW Architecture (Clean)
```
detect_wood_comprehensive()  ← Called when detecting wood
  └─> Detects wood, creates AUTO ROI
  └─> ✨ CHECKS COLLISION immediately with fresh coordinates
  └─> Stores collision result in detection result
  └─> Triggers notification if collision detected
  
draw_wood_detection_overlay()  ← Called every frame
  └─> Reads stored detection result
  └─> Reads collision status from result
  └─> Draws visual overlay (red if collision, yellow if not)
  └─> NO collision checking, NO notification triggering

Benefit: Collision logic in one place, 
         uses source coordinates directly,
         no confusion between detection and drawing
```

---

## Code Flow

### Step 1: Wood Detection (with Collision Check)

**Location:** `detect_wood_comprehensive()` in `ColorWoodDetector` class

```python
def detect_wood_comprehensive(self, image, profile_names=None, roi=None, camera='top'):
    # ... detect wood, create AUTO ROI ...
    
    if result['wood_detected']:
        # Store AUTO ROI
        auto_roi = self.generate_auto_roi(wood_candidates, image.shape)
        result['auto_roi'] = auto_roi
        
        # ✨ CHECK COLLISION HERE - Right after detection!
        if auto_roi and hasattr(self, 'lane_roi_var') and self.lane_roi_var.get():
            roi_x, roi_y, roi_w, roi_h = auto_roi
            roi_y_bottom = roi_y + roi_h
            
            # Check top lane collision
            if roi_y <= 100:
                result['lane_collision'] = 'TOP'
                self.show_alignment_warning(camera, "TOP")
            
            # Check bottom lane collision
            elif roi_y_bottom >= 620:
                result['lane_collision'] = 'BOTTOM'
                self.show_alignment_warning(camera, "BOTTOM")
            
            else:
                result['lane_collision'] = None
                self.clear_alignment_warning(camera)
    
    return result
```

### Step 2: Visual Overlay (based on Collision Result)

**Location:** `draw_wood_detection_overlay()` methods

```python
def draw_wood_detection_overlay(self, frame, camera_name):
    # ... draw lanes ...
    
    # Get stored detection result
    wood_detection = self.wood_detection_results.get(camera_name)
    
    if wood_detection and wood_detection.get('auto_roi'):
        roi_x, roi_y, roi_w, roi_h = wood_detection['auto_roi']
        
        # ✨ READ collision status (no checking, just reading!)
        lane_collision = wood_detection.get('lane_collision')
        
        if lane_collision:
            # Draw RED warning overlay
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 3)
            cv2.putText(frame, f"⚠ MISALIGNED - {lane_collision} LANE", ...)
        else:
            # Draw YELLOW normal ROI
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 2)
            cv2.putText(frame, "AUTO ROI", ...)
    
    return frame
```

---

## Benefits of New Architecture

### 1. **Single Source of Truth**
- Collision detection happens **once** in `detect_wood_comprehensive()`
- Uses the **original coordinates** from wood detection
- No re-reading, re-parsing, or re-checking coordinates

### 2. **Clear Separation of Concerns**
- **Detection function**: Detects wood + checks collision + triggers notifications
- **Drawing function**: Reads result + draws visual feedback
- Each function has one clear responsibility

### 3. **No Variable Confusion**
- In detection: `roi_x, roi_y, roi_w, roi_h` come directly from `auto_roi` generation
- In drawing: Read `lane_collision` status from stored result
- No mixing of detection coordinates with drawing coordinates

### 4. **Easier Debugging**
- All collision logic in one place: `detect_wood_comprehensive()`
- Log messages show exactly when collision is checked
- No duplicate logging from multiple draw calls

### 5. **Better Performance**
- Collision checked once per detection (not every frame)
- Drawing function just reads boolean flag
- No redundant coordinate calculations

---

## Logging Output

### New Log Format (from Detection Function)

```
🪵 Starting comprehensive wood detection on image shape: (720, 1280, 3)
🎨 Color mask: 45678 pixels (4.9%)
📐 Found 1 wood candidates after contour filtering
🎯 Auto ROI generated: (450, 85, 200, 300)

============================================================
[COLLISION CHECK] Checking wood alignment for TOP camera
============================================================
  Wood ROI: x=450, y=85, w=200, h=300
  ROI Top Edge: y=85
  ROI Bottom Edge: y=385
  Top Lane Boundary: y=100
  TOP COLLISION: True (Wood top=85 <= 100)
  ⚠️  MISALIGNMENT DETECTED: Wood is TOO HIGH (touching TOP lane)
  Bottom Lane Boundary: y=620
  BOTTOM COLLISION: False (Wood bottom=385 < 620)
  🚨 RESULT: COLLISION DETECTED - Wood is MISALIGNED!
============================================================

[DEBUG] show_alignment_warning called: camera=top, lane=TOP
[DEBUG] Showing new warning: TOP_LANE
[DEBUG] Putting warning message into queue...

✅ Detection complete: wood_detected=True, count=1, confidence=0.85
```

### What Drawing Function Does (Silent)

Drawing function just reads `wood_detection['lane_collision']` and draws:
- If `'TOP'` or `'BOTTOM'`: Draw red overlay + warning text
- If `None`: Draw yellow AUTO ROI

**No logging from drawing function** - keeps terminal clean!

---

## Detection Result Structure

The `detect_wood_comprehensive()` function returns:

```python
{
    'wood_detected': True,
    'wood_count': 1,
    'wood_candidates': [...],
    'auto_roi': (x, y, w, h),
    'confidence': 0.85,
    'lane_collision': 'TOP' | 'BOTTOM' | None,  # ✨ NEW!
    ...
}
```

The `lane_collision` field is added by the collision detection logic:
- `'TOP'`: Wood touching top lane (y ≤ 100)
- `'BOTTOM'`: Wood touching bottom lane (y+h ≥ 620)
- `None`: No collision, wood properly aligned

---

## Coordinate Clarity

### In Detection Function
```python
# AUTO ROI is freshly generated
auto_roi = self.generate_auto_roi(wood_candidates, image.shape)
roi_x, roi_y, roi_w, roi_h = auto_roi

# Coordinates are DIRECT from detection
# No confusion, no re-reading from storage
```

### In Drawing Function
```python
# Read stored detection result
wood_detection = self.wood_detection_results.get(camera_name)
auto_roi = wood_detection.get('auto_roi')
roi_x, roi_y, roi_w, roi_h = auto_roi

# ❌ OLD: Would check collision again here (CONFUSING!)
# ✅ NEW: Just read the collision result
lane_collision = wood_detection.get('lane_collision')
```

---

## Notification Timing

### When Notifications Are Triggered

1. **Wood detected** → `detect_wood_comprehensive()` called
2. **AUTO ROI generated** → collision check runs
3. **Collision detected** → `show_alignment_warning()` called immediately
4. **Notification queued** → message added to queue
5. **GUI updates** → `process_message_queue()` shows popup

### When Notifications Are NOT Triggered

- **Wood not detected**: No AUTO ROI, no collision check
- **Lane ROI disabled**: Collision check skipped
- **Cooldown active**: Notification suppressed (5 seconds)
- **Same warning**: Duplicate prevention blocks notification

---

## Testing the New Architecture

### Run the Application

```bash
cd /home/inspectura/Desktop/InspecturaGUI
python testIR/testIR.py
```

### What to Look For

**Terminal Output:**
```
============================================================
[COLLISION CHECK] Checking wood alignment for TOP camera
============================================================
```

This section **only appears** when:
1. Wood is detected (`wood_detected=True`)
2. AUTO ROI is generated
3. Lane ROI checkbox is checked

**If you DON'T see this section:**
- Wood is not being detected, OR
- AUTO ROI is not being generated, OR
- Lane ROI checkbox is unchecked

**Visual Feedback:**
- Yellow "AUTO ROI" box = No collision, properly aligned
- Red box with "⚠ MISALIGNED - TOP/BOTTOM LANE" = Collision detected

---

## Summary

| Aspect | OLD | NEW |
|--------|-----|-----|
| **Collision Logic Location** | In drawing function | In detection function ✅ |
| **Coordinates Source** | Re-read from storage | Direct from detection ✅ |
| **Checks Per Second** | ~10+ (every frame) | ~3 (when detecting) ✅ |
| **Variable Confusion** | High (mixed contexts) | Low (single context) ✅ |
| **Debugging** | Multiple locations | One location ✅ |
| **Notification Trigger** | From drawing | From detection ✅ |
| **Performance** | Lower (redundant checks) | Higher (single check) ✅ |

**Result:** Clean, efficient, easy to understand collision detection! 🎯
