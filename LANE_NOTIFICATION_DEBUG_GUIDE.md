# Debugging Lane Alignment Notifications

## Debug Output Added

I've added extensive debug output to help identify why notifications aren't appearing. Here's what to look for:

## Step-by-Step Testing Guide

### 1. Run the Application
```bash
cd /home/inspectura/Desktop/InspecturaGUI
python testIR/testIR.py
```

### 2. Enable Lane ROI
- Check the **"Lane ROI"** checkbox in the GUI
- Verify red lanes appear on camera feeds

### 3. Position Wood to Touch Lanes
Position wood so the AUTO ROI (yellow box) touches:
- **Top lane**: y ≤ 100 pixels
- **Bottom lane**: y ≥ 620 pixels

## Debug Messages to Watch For

### A. AUTO ROI Detection
When wood is detected, you should see:
```
[DEBUG COLLISION] AUTO ROI detected: x=..., y=..., w=..., h=..., y_bottom=...
```
**What this means**: Wood detection is working, AUTO ROI exists

### B. Collision Check
For each frame, you should see:
```
[DEBUG COLLISION] Top lane check: y_roi_top=..., y_lane_bottom=100, collision=True/False
[DEBUG COLLISION] Bottom lane check: y_roi_bottom=..., y_lane_top=620, collision=True/False
```
**What this means**: Collision detection is running

### C. Collision Detected
When wood touches a lane:
```
[DEBUG COLLISION] ⚠️ TOP LANE COLLISION DETECTED!
```
or
```
[DEBUG COLLISION] ⚠️ BOTTOM LANE COLLISION DETECTED!
```
**What this means**: Collision successfully detected

### D. Warning Function Called
After collision detected:
```
[DEBUG] show_alignment_warning called: camera=top, lane=TOP
```
**What this means**: Warning function was invoked

### E. Cooldown/Duplicate Check
If in cooldown:
```
[DEBUG] In cooldown: 2.5s < 5.0s
```
If duplicate warning:
```
[DEBUG] Duplicate warning: TOP_LANE already active
```
**What this means**: Notification skipped due to cooldown or duplicate

### F. New Warning
If showing notification:
```
[DEBUG] Showing new warning: TOP_LANE
```
**What this means**: Notification should appear

### G. Queue Message
When notification queued:
```
[DEBUG] Putting warning message into queue...
[DEBUG] Warning message queued successfully
```
**What this means**: Message sent to main thread

### H. Console Warning
You should also see:
```
============================================================
⚠ ALIGNMENT WARNING - TOP CAMERA
============================================================
Lane: TOP LANE
Coordinate: y = 100 pixels
Timestamp: 2025-10-16 14:23:45
============================================================
```
**What this means**: Warning logged to console

### I. Queue Processing
When message processed:
```
[DEBUG QUEUE] Processing warning message from queue...
[DEBUG QUEUE] Title: ⚠ WOOD MISALIGNMENT DETECTED
[DEBUG QUEUE] Message: Wood piece is MISALIGNED...
[DEBUG QUEUE] Showing messagebox...
[DEBUG QUEUE] Messagebox shown successfully
```
**What this means**: Popup notification displayed

## Troubleshooting Based on Debug Output

### Scenario 1: No AUTO ROI Detection
**Symptom**: No `[DEBUG COLLISION] AUTO ROI detected` messages
**Problem**: Wood detection not working
**Solution**: 
- Check if wood is visible in camera feed
- Check if "Lane ROI" checkbox is checked
- Verify RGB wood detection is working

### Scenario 2: Collision Not Detected
**Symptom**: Collision check shows `collision=False` when wood touches lane
**Problem**: Collision detection algorithm not working correctly
**Solution**:
- Check coordinates: y_roi_top should be ≤ 100 for top lane
- Check coordinates: y_roi_bottom should be ≥ 620 for bottom lane
- Verify lane coordinates in config

### Scenario 3: Cooldown Active
**Symptom**: `[DEBUG] In cooldown` message appears
**Problem**: Still in cooldown from previous warning
**Solution**: Wait 5 seconds for cooldown to expire

### Scenario 4: Duplicate Warning
**Symptom**: `[DEBUG] Duplicate warning` message appears
**Problem**: Same warning already active, move wood away first
**Solution**: Move wood away from lane, then back

### Scenario 5: Message Not Queued
**Symptom**: No `[DEBUG] Putting warning message into queue` message
**Problem**: Warning function exited early
**Solution**: Check previous debug messages for cooldown/duplicate

### Scenario 6: Queue Not Processing
**Symptom**: Message queued but no `[DEBUG QUEUE] Processing warning` message
**Problem**: Message queue not being processed
**Solution**: 
- Check if `process_message_queue()` is running (should run every 50ms)
- Check for errors in terminal

### Scenario 7: Messagebox Not Showing
**Symptom**: Queue processing succeeds but no popup appears
**Problem**: GUI issue or messagebox blocked
**Solution**: 
- Check if another messagebox is already open
- Check if application is in focus
- Try clicking on application window

## Expected Complete Debug Flow

When wood touches top lane, you should see this sequence:

```
[DEBUG COLLISION] AUTO ROI detected: x=400, y=50, w=200, h=100, y_bottom=150
[DEBUG COLLISION] Top lane check: y_roi_top=50, y_lane_bottom=100, collision=True
[DEBUG COLLISION] ⚠️ TOP LANE COLLISION DETECTED!
⚠ WARNING [top]: Wood is MISALIGNED - touching TOP lane!
[DEBUG] show_alignment_warning called: camera=top, lane=TOP
[DEBUG] Showing new warning: TOP_LANE
[DEBUG] Putting warning message into queue...
[DEBUG] Warning message queued successfully
============================================================
⚠ ALIGNMENT WARNING - TOP CAMERA
============================================================
Lane: TOP LANE
Coordinate: y = 100 pixels
Timestamp: 2025-10-16 14:23:45
============================================================

[DEBUG QUEUE] Processing warning message from queue...
[DEBUG QUEUE] Title: ⚠ WOOD MISALIGNMENT DETECTED
[DEBUG QUEUE] Message: Wood piece is MISALIGNED...
[DEBUG QUEUE] Showing messagebox...
[DEBUG QUEUE] Messagebox shown successfully
```

## Quick Test Commands

### Test 1: Verify AUTO ROI Detection
```bash
# Look for this pattern in terminal:
grep "AUTO ROI detected" terminal_output.txt
```

### Test 2: Verify Collision Detection
```bash
# Look for collision messages:
grep "COLLISION DETECTED" terminal_output.txt
```

### Test 3: Verify Warning Function
```bash
# Look for warning function calls:
grep "show_alignment_warning called" terminal_output.txt
```

### Test 4: Verify Queue Processing
```bash
# Look for queue processing:
grep "DEBUG QUEUE" terminal_output.txt
```

## Common Issues

### Issue 1: Wood Detection Not Working
- **Check**: GREEN boxes should appear around wood
- **Check**: "AUTO ROI" label should appear
- **Fix**: Adjust lighting or wood position

### Issue 2: Lanes Not Visible
- **Check**: "Lane ROI" checkbox must be checked
- **Check**: Red semi-transparent zones should appear at top and bottom
- **Fix**: Enable checkbox in ROI panel

### Issue 3: Wrong Coordinates
- **Check**: Top lane should be at y=0-100
- **Check**: Bottom lane should be at y=620-720
- **Fix**: Verify ALIGNMENT_LANE_ROIS in config

### Issue 4: Notification Spam
- **Check**: Cooldown is 5 seconds
- **Expected**: One notification per 5 seconds maximum
- **Normal**: This prevents spam

## Success Indicators

✅ **Working Correctly**:
1. AUTO ROI detected messages appear
2. Collision check messages appear
3. Collision detected when wood touches lane
4. Warning function called
5. Message queued successfully
6. Queue processes message
7. **POPUP NOTIFICATION APPEARS** 🎉

❌ **Not Working**:
- Missing any of the debug messages above
- Check the troubleshooting section for that specific message

## Next Steps

1. **Run the application**
2. **Watch terminal output carefully**
3. **Position wood to touch lane**
4. **Copy ALL debug messages** and share them if notification doesn't appear
5. **Note which debug message is missing** - this will tell us exactly where the issue is

---

**Debug Version**: 2.0  
**Last Updated**: October 16, 2025  
**Status**: Ready for testing 🚀
