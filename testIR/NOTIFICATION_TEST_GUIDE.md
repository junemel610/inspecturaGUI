# Lane Alignment Notification Test Guide

## Purpose
This guide explains how to test the lane alignment notification system using the **Test Notification** button.

---

## Test Button Location
In the GUI, look for the **ROI** panel (below the bottom camera feed):
```
┌─ ROI ─────────────────┐
│ ☑ Top ROI             │
│ ☑ Bottom ROI          │
│ ☑ Lane ROI            │
│ 🧪 Test Notification  │  ← Click this button!
└───────────────────────┘
```

---

## How to Test

### Step 1: Click the Test Button
1. Start the application: `python3 testIR.py`
2. Wait for cameras to initialize
3. Click the **🧪 Test Notification** button in the ROI panel

### Step 2: Observe the Results

#### ✅ SUCCESS - Notification System Working
You should see:
1. **Popup Window** appears with:
   - Title: "⚠ WOOD MISALIGNMENT DETECTED"
   - Message with lane information and action required
   - Click "OK" to close the popup

2. **Terminal Output**:
   ```
   ============================================================
   🧪 TEST: Triggering alignment notification
   ============================================================
   [TEST] Calling show_alignment_warning('top', 'TOP')...
   [DEBUG] show_alignment_warning called: camera=top, lane=TOP
   [DEBUG] Showing new warning: TOP_LANE
   [DEBUG] Putting warning message into queue...
   [DEBUG] Warning message queued successfully
   
   ============================================================
   ⚠ ALIGNMENT WARNING - TOP CAMERA
   ============================================================
   Lane: TOP LANE
   Coordinate: y = 100 pixels
   Timestamp: 2025-01-XX XX:XX:XX
   ============================================================
   
   [DEBUG QUEUE] Processing warning message from queue...
   [DEBUG QUEUE] Title: ⚠ WOOD MISALIGNMENT DETECTED
   [DEBUG QUEUE] Message: Wood piece is MISALIGNED on TOP camera!...
   [DEBUG QUEUE] Showing messagebox...
   [DEBUG QUEUE] Messagebox shown successfully
   [TEST] If popup appears, notification system is working!
   [TEST] If no popup, check terminal for debug messages
   ============================================================
   ```

#### ❌ FAILURE - Notification System NOT Working

**Scenario A: Popup appears, debug messages incomplete**
- **What you see**: Popup works, but missing some debug messages
- **Meaning**: Message queue is working, but collision detection may have issues
- **Action**: Focus on the collision detection logic in `draw_wood_detection_overlay()`

**Scenario B: No popup, debug messages stop at "Putting warning message into queue"**
- **What you see**: Terminal shows debug up to "Warning message queued successfully" but nothing after
- **Meaning**: Message queue not being processed
- **Check**:
  1. Is `process_message_queue()` being called? Search terminal for "[DEBUG QUEUE]"
  2. Is the message queue blocked?
  3. Is tkinter mainloop running?

**Scenario C: No popup, debug messages stop at "show_alignment_warning called"**
- **What you see**: Only first debug message appears
- **Meaning**: Function is being called but execution stops inside
- **Check**:
  1. Look for Python exceptions/errors in terminal
  2. Check if cooldown/duplicate logic is blocking

**Scenario D: No popup, no debug messages at all**
- **What you see**: Nothing happens when clicking button
- **Meaning**: Button click not connected or method not found
- **Check**:
  1. Look for Python errors: "AttributeError: ... has no attribute 'test_alignment_notification'"
  2. Ensure the test method exists in the class

---

## Understanding the Test

### What the Test Does
```python
def test_alignment_notification(self):
    # 1. Reset cooldown (allow immediate testing)
    self.alignment_warnings["top"]["last_warning_time"] = 0
    self.alignment_warnings["top"]["current_warning"] = None
    
    # 2. Trigger notification
    self.show_alignment_warning("top", "TOP")
    
    # 3. Report success/failure based on popup appearance
```

### Why This Test is Important
1. **Isolates the notification system**: Tests popups independently from collision detection
2. **Verifies message queue**: Confirms thread-safe popup display works
3. **Identifies bottlenecks**: Debug output shows exactly where the system fails
4. **Quick iteration**: No need to position wood physically to test notifications

---

## Troubleshooting Based on Test Results

### ✅ Test Button Works → Collision Detection Issue
If the test button shows popups successfully, but real collisions don't:

**Possible Causes:**
1. **Wood detection not working**: No AUTO ROI being generated
   - Check terminal for "[DEBUG COLLISION] AUTO ROI detected"
   - Verify RGB wood detector is active

2. **Wood not touching boundaries**: ROI coordinates don't meet conditions
   - Check terminal for collision debug output
   - Ensure wood position actually crosses y=100 or y≥620

3. **Checkbox unchecked**: Lane ROI checkbox is disabled
   - Verify "☑ Lane ROI" is checked in GUI
   - Check terminal for "lane_roi_var value: True"

**Debug Steps:**
```
Run application → Enable "Lane ROI" → Place wood to cross boundaries → Check terminal:
- Look for: "[DEBUG COLLISION] AUTO ROI detected: roi_y=XXX"
- Look for: "[DEBUG COLLISION] Top/Bottom lane check: roi_y (XXX) <= 100"
- Look for: "[DEBUG COLLISION] ⚠️ COLLISION DETECTED!"
```

### ❌ Test Button Doesn't Work → Notification System Issue
If the test button itself doesn't show popups:

**Possible Causes:**
1. **Message queue not processing**: `process_message_queue()` not called
2. **Tkinter issue**: GUI thread blocked or mainloop not running
3. **Python error**: Exception preventing popup display

**Debug Steps:**
1. Check terminal immediately after clicking test button
2. Search for "[DEBUG QUEUE]" - if missing, message queue not processing
3. Look for Python exceptions or stack traces
4. Try direct messagebox test:
   ```python
   # Add to test_alignment_notification():
   from tkinter import messagebox
   messagebox.showinfo("Direct Test", "If you see this, tkinter works!")
   ```

---

## Next Steps After Testing

### If Test Succeeds ✅
1. **Enable Lane ROI checkbox** in GUI
2. **Position wood** so it crosses boundaries:
   - Top boundary: y ≤ 100 pixels
   - Bottom boundary: y ≥ 620 pixels
3. **Check terminal** for collision debug messages
4. **Verify popup** appears during real wood detection

### If Test Fails ❌
1. **Copy all terminal output** from clicking test button
2. **Report exact failure point** based on debug messages
3. **Check for Python errors** in terminal
4. **Test basic tkinter**:
   ```python
   from tkinter import Tk, messagebox
   root = Tk()
   root.withdraw()
   messagebox.showinfo("Test", "Works!")
   ```

---

## Expected vs. Actual Behavior

| Action | Expected Behavior | If Different... |
|--------|------------------|-----------------|
| Click test button | Popup appears immediately | Notification system broken |
| Terminal shows all debug messages | Complete pipeline visible | Identify where pipeline breaks |
| Popup shows lane info | Message content correct | Check warning message formatting |
| Second click shows popup again | Cooldown reset works | Test multiple triggers |

---

## Code References

**Test Button:**
- Location: `testIR.py` line ~2569
- Method: `test_alignment_notification()` at line ~1265

**Notification Flow:**
1. `test_alignment_notification()` → Resets cooldown, triggers warning
2. `show_alignment_warning()` → Creates message, puts in queue
3. `process_message_queue()` → Reads queue, displays popup
4. `messagebox.showwarning()` → Shows actual popup window

---

## Summary

The **🧪 Test Notification** button is your **diagnostic tool** for the lane alignment notification system:

- ✅ **Test passes** → Notification system works, focus on collision detection
- ❌ **Test fails** → Notification system broken, fix message queue/popup display

**Use this test button FIRST** before debugging collision detection!
