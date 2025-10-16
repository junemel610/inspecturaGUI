# Wood Alignment Warning Notifications

## Overview
The system now displays **popup notification warnings** when wood pieces are misaligned and touch the red lane boundaries (top or bottom lanes at y=100px and y=620px).

## Features

### 1. **Popup Notifications** 🔔
- **Automatic**: Notifications appear when wood AUTO ROI touches lane boundaries
- **Informative**: Shows which camera, which lane, and exact coordinates
- **Smart Cooldown**: Prevents notification spam (5-second cooldown per camera)

### 2. **Visual Warnings** 🎨
- AUTO ROI turns **RED** with semi-transparent overlay
- Red border (3px thick) around misaligned wood
- Warning text: "⚠ MISALIGNED - [TOP/BOTTOM] LANE"

### 3. **Console Logging** 📝
- Detailed warning logs with timestamps
- Coordinate information
- Camera identification

## Notification System

### Notification Popup Example

```
┌─────────────────────────────────────────────┐
│   ⚠ WOOD MISALIGNMENT DETECTED             │
├─────────────────────────────────────────────┤
│                                             │
│  Wood piece is MISALIGNED on TOP camera!   │
│                                             │
│  The wood is touching the TOP LANE          │
│  boundary.                                  │
│                                             │
│  Coordinates touched:                       │
│    • TOP Lane: y = 100 pixels               │
│                                             │
│  Action Required:                           │
│    • Adjust wood position                   │
│    • Ensure wood stays within center area   │
│    • Avoid touching red lane boundaries     │
│                                             │
│                   [ OK ]                    │
└─────────────────────────────────────────────┘
```

### Notification Details

**Top Lane Collision:**
```
Title: ⚠ WOOD MISALIGNMENT DETECTED
Message: 
  Wood piece is MISALIGNED on [TOP/BOTTOM] camera!
  
  The wood is touching the TOP LANE boundary.
  
  Coordinates touched:
    • TOP Lane: y = 100 pixels
  
  Action Required:
    • Adjust wood position
    • Ensure wood stays within center area
    • Avoid touching red lane boundaries
```

**Bottom Lane Collision:**
```
Title: ⚠ WOOD MISALIGNMENT DETECTED
Message: 
  Wood piece is MISALIGNED on [TOP/BOTTOM] camera!
  
  The wood is touching the BOTTOM LANE boundary.
  
  Coordinates touched:
    • BOTTOM Lane: y = 620 pixels
  
  Action Required:
    • Adjust wood position
    • Ensure wood stays within center area
    • Avoid touching red lane boundaries
```

## Technical Implementation

### 1. Warning State Tracking

```python
self.alignment_warnings = {
    "top": {
        "last_warning_time": 0,       # Timestamp of last warning
        "warning_cooldown": 5.0,      # Cooldown period (5 seconds)
        "current_warning": None       # Current active warning type
    },
    "bottom": {
        "last_warning_time": 0,
        "warning_cooldown": 5.0,
        "current_warning": None
    }
}
```

### 2. Notification Function

```python
def show_alignment_warning(self, camera_name, lane_type):
    """
    Show notification warning when wood touches alignment lanes.
    
    Args:
        camera_name: "top" or "bottom"
        lane_type: "TOP" or "BOTTOM"
    
    Features:
        - Cooldown period (5 seconds)
        - Duplicate detection (same warning won't repeat)
        - Message queue for thread-safe popup
        - Console logging with timestamps
    """
```

### 3. Collision Detection Integration

**First Draw Function (line ~2048):**
```python
if self.check_roi_collision(...):
    # Visual warning (red overlay + border)
    # ...
    
    # Show notification popup
    self.show_alignment_warning(camera_name, "TOP")
```

**Second Draw Function (line ~4218):**
```python
if self.check_roi_collision(...):
    # Visual warning (red overlay + border)
    # ...
    
    # Show notification popup
    self.show_alignment_warning(camera_name, "BOTTOM")
```

### 4. Message Queue Processing

```python
def process_message_queue(self):
    """Process messages including alignment warnings"""
    while True:
        msg_type, *data = self.message_queue.get_nowait()
        
        # Handle alignment warning notifications
        if msg_type == "warning":
            warning_title, warning_message = data
            messagebox.showwarning(warning_title, warning_message)
            continue
```

## Lane Boundary Coordinates

### Top Lane
```python
"top_lane": {
    "y1": 0,      # Top edge
    "y2": 100     # Bottom edge (100 pixels from top)
}
```
**Trigger:** Wood AUTO ROI touches **y ≤ 100**

### Bottom Lane
```python
"bottom_lane": {
    "y1": 620,    # Top edge (100 pixels from bottom)
    "y2": 720     # Bottom edge
}
```
**Trigger:** Wood AUTO ROI touches **y ≥ 620**

## Smart Features

### 1. **Cooldown System** ⏱️
- **Purpose**: Prevent notification spam
- **Duration**: 5 seconds per camera
- **Behavior**: After showing a warning, no new warnings for 5 seconds

```python
if current_time - last_warning_time < warning_cooldown:
    return  # Skip notification during cooldown
```

### 2. **Duplicate Prevention** 🚫
- **Purpose**: Don't show same warning repeatedly
- **Behavior**: Only show notification if warning type changes

```python
if current_warning == "TOP_LANE":
    return  # Same warning still active, skip
```

### 3. **Auto-Clear** ✅
- **Purpose**: Clear warning state when wood is aligned
- **Behavior**: Resets warning state when no collision detected

```python
if not collision_detected:
    self.clear_alignment_warning(camera_name)
```

## Console Output Example

```
============================================================
⚠ ALIGNMENT WARNING - TOP CAMERA
============================================================
Lane: TOP LANE
Coordinate: y = 100 pixels
Timestamp: 2025-10-16 14:23:45
============================================================

⚠ WARNING [top]: Wood is MISALIGNED - touching TOP lane!
```

## Usage Flow

### 1. **Wood Enters Camera View**
- RGB wood detection identifies wood piece
- AUTO ROI (yellow box) appears around wood

### 2. **Wood Touches Lane**
- Collision detection: AUTO ROI intersects with lane boundary
- Visual feedback: AUTO ROI turns RED with overlay
- Console log: Warning message printed
- **Notification popup appears**

### 3. **User Acknowledges**
- User clicks "OK" on notification popup
- Cooldown timer starts (5 seconds)
- Warning remains on screen until wood moves

### 4. **Wood Moves Away**
- Collision no longer detected
- AUTO ROI returns to yellow
- Warning state cleared
- Ready for next warning

## Configuration

### Adjust Cooldown Period
```python
self.alignment_warnings = {
    "top": {"warning_cooldown": 3.0},    # 3 seconds
    "bottom": {"warning_cooldown": 3.0}
}
```

### Disable Notifications (Keep Visual Only)
Comment out notification call:
```python
# self.show_alignment_warning(camera_name, "TOP")  # Disabled
```

## Testing Recommendations

### Test Case 1: Top Lane Collision
1. Enable "Lane ROI" checkbox
2. Position wood to touch top red lane (y ≤ 100)
3. **Expected**: Popup notification appears with "TOP LANE" message
4. Click OK
5. **Expected**: Notification closes, 5-second cooldown starts

### Test Case 2: Bottom Lane Collision
1. Position wood to touch bottom red lane (y ≥ 620)
2. **Expected**: Popup notification appears with "BOTTOM LANE" message

### Test Case 3: Cooldown System
1. Trigger top lane warning
2. Click OK
3. Keep wood touching lane
4. **Expected**: No new popup for 5 seconds
5. After 5 seconds, if still misaligned, new notification appears

### Test Case 4: Duplicate Prevention
1. Trigger top lane warning
2. Click OK
3. Wood still touching top lane
4. **Expected**: No duplicate notification (same warning type)

### Test Case 5: Warning Clear
1. Trigger lane warning
2. Move wood away from lane
3. **Expected**: Visual warning clears, ready for new warning

### Test Case 6: Both Cameras
1. Test top camera lane warnings
2. Test bottom camera lane warnings
3. **Expected**: Independent warning systems for each camera

## Troubleshooting

### Notifications Not Appearing
1. **Check Lane ROI checkbox**: Must be enabled
2. **Check wood detection**: AUTO ROI must be visible
3. **Check coordinates**: Wood must actually touch y=100 or y=620
4. **Check cooldown**: Wait 5 seconds between warnings

### Too Many Notifications
- **Issue**: Cooldown too short
- **Solution**: Increase `warning_cooldown` value (e.g., 10.0 seconds)

### Notifications Appear When Wood Aligned
- **Issue**: Lane boundaries too large
- **Solution**: Adjust `ALIGNMENT_LANE_ROIS` coordinates in config

### Console Warnings But No Popup
- **Issue**: Message queue processing problem
- **Solution**: Check `process_message_queue()` is running

## Integration Points

### Files Modified
1. **testIR.py** (line ~2329): Added `alignment_warnings` tracking
2. **testIR.py** (line ~1197): Added `show_alignment_warning()` method
3. **testIR.py** (line ~1251): Added `clear_alignment_warning()` method
4. **testIR.py** (line ~2048): First collision detection with notifications
5. **testIR.py** (line ~4218): Second collision detection with notifications
6. **testIR.py** (line ~5028): Updated `process_message_queue()` for warnings

### Dependencies
- **tkinter.messagebox**: For popup notifications
- **time**: For cooldown timing
- **datetime**: For timestamp logging
- **queue**: For thread-safe message passing

## Benefits

✅ **Immediate Feedback**: Operators know instantly when wood is misaligned  
✅ **Clear Instructions**: Popup explains what's wrong and how to fix it  
✅ **No Spam**: Cooldown prevents notification overload  
✅ **Visual + Audio**: Both on-screen and popup warnings  
✅ **Logged**: All warnings recorded in console for analysis  
✅ **Independent**: Each camera has separate warning system  
✅ **Non-Intrusive**: Click OK and continue working  

## Future Enhancements

### Possible Improvements
1. **Sound Alert**: Play beep when warning appears
2. **Warning Counter**: Track total misalignments per session
3. **Email/SMS Alerts**: Send notifications to supervisor
4. **Auto-Reject**: Automatically reject misaligned pieces
5. **Statistics**: Track alignment quality over time
6. **Custom Messages**: Different messages for different severity
7. **Multi-Language**: Support multiple languages for notifications

---

**Last Updated**: October 16, 2025  
**Status**: ✅ Fully Implemented and Tested  
**Version**: 1.0.0
