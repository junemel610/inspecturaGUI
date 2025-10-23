# Collision Detection Logging Guide

## Enhanced Logging Format

The collision detection now uses clear, easy-to-read logging that will help you troubleshoot.

---

## Log Format Examples

### When Wood is Detected and Checked

Every time the system detects wood and checks for lane collision, you'll see logs like this:

```
[COLLISION CHECK] Camera=top | ROI Top=150 | Top Lane Boundary=100 | TOP COLLISION=False
[COLLISION CHECK] Camera=top | ROI Bottom=450 | Bottom Lane Boundary=620 | BOTTOM COLLISION=False
[COLLISION CHECK] Camera=top | ✅ NO COLLISION - Wood is properly aligned
```

### When Top Lane Collision Occurs

```
[COLLISION CHECK] Camera=top | ROI Top=85 | Top Lane Boundary=100 | TOP COLLISION=True
[DEBUG COLLISION] ⚠️ TOP LANE COLLISION DETECTED!
⚠ WARNING [top]: Wood is MISALIGNED - touching TOP lane!
[DEBUG] show_alignment_warning called: camera=top, lane=TOP
[COLLISION CHECK] Camera=top | ⚠️ COLLISION DETECTED - Notification triggered
```

### When Bottom Lane Collision Occurs

```
[COLLISION CHECK] Camera=top | ROI Bottom=650 | Bottom Lane Boundary=620 | BOTTOM COLLISION=True
[DEBUG COLLISION] ⚠️ BOTTOM LANE COLLISION DETECTED!
⚠ WARNING [top]: Wood is MISALIGNED - touching BOTTOM lane!
[DEBUG] show_alignment_warning called: camera=top, lane=BOTTOM
[COLLISION CHECK] Camera=top | ⚠️ COLLISION DETECTED - Notification triggered
```

---

## Understanding the Logs

### Key Log Messages

| Log Prefix | Meaning |
|-----------|---------|
| `[COLLISION CHECK]` | Summary of collision detection for this frame |
| `[DEBUG COLLISION]` | Detailed collision event (when collision happens) |
| `[DEBUG]` | Notification system debug messages |
| `[DEBUG QUEUE]` | Message queue processing debug |

### Collision Logic

**Top Lane Collision:**
- Top lane boundary: y = 100 pixels
- Collision if: `ROI Top ≤ 100`
- Example: ROI Top = 85 → COLLISION (wood too high)
- Example: ROI Top = 150 → NO COLLISION (wood OK)

**Bottom Lane Collision:**
- Bottom lane boundary: y = 620 pixels
- Collision if: `ROI Bottom ≥ 620`
- Example: ROI Bottom = 650 → COLLISION (wood too low)
- Example: ROI Bottom = 450 → NO COLLISION (wood OK)

---

## Troubleshooting with Logs

### Scenario 1: No Logs at All
**Problem:** You don't see any `[COLLISION CHECK]` logs

**Possible Causes:**
1. Wood not detected by RGB detector
2. AUTO ROI not being created
3. Lane ROI checkbox unchecked

**What to Check:**
```bash
# Look for wood detection messages:
grep "Wood" terminal_output.txt

# Check if AUTO ROI is being created:
grep "AUTO ROI" terminal_output.txt

# Verify lane ROI is enabled:
grep "lane_roi_var" terminal_output.txt
```

### Scenario 2: Logs Show "NO COLLISION" But You Expect Collision
**Problem:** Wood appears to touch lanes but logs show `NO COLLISION`

**Check These Values:**
```
[COLLISION CHECK] Camera=top | ROI Top=105 | Top Lane Boundary=100 | TOP COLLISION=False
```

**Analysis:**
- ROI Top = 105 pixels
- Top lane boundary = 100 pixels
- 105 > 100 → NO COLLISION (correct!)
- Wood needs to be **at or above** y=100 to trigger top collision

**Solution:** Position wood higher (y ≤ 100) or lower (y ≥ 620)

### Scenario 3: Logs Show Collision But No Notification Popup
**Problem:** Logs show `COLLISION DETECTED` but no popup appears

**Expected Log Sequence:**
```
1. [COLLISION CHECK] ... | TOP COLLISION=True
2. [DEBUG COLLISION] ⚠️ TOP LANE COLLISION DETECTED!
3. [DEBUG] show_alignment_warning called: camera=top, lane=TOP
4. [DEBUG] Showing new warning: TOP_LANE
5. [DEBUG] Putting warning message into queue...
6. [DEBUG] Warning message queued successfully
7. [DEBUG QUEUE] Processing warning message from queue...
8. [DEBUG QUEUE] Showing messagebox...
9. [DEBUG QUEUE] Messagebox shown successfully
```

**Diagnose the Issue:**

**If logs stop at step 3:**
- Check: Is it in cooldown? Look for "In cooldown" message
- Check: Is it duplicate? Look for "Duplicate warning" message
- Solution: Wait 5 seconds or move wood away and back

**If logs stop at step 6:**
- Problem: Message queue not processing
- Check: Search for any `[DEBUG QUEUE]` messages
- Solution: Message queue thread may not be running

**If logs reach step 8 but no popup:**
- Problem: Tkinter messagebox blocked or hidden
- Check: Are there other popups already open?
- Solution: Close any existing popups, check window focus

### Scenario 4: Too Many Notifications
**Problem:** Notifications appearing too frequently

**What You'll See:**
```
[DEBUG] In cooldown: 2.3s < 5.0s
```

**This is Normal:** Cooldown system prevents spam
- Notifications limited to once per 5 seconds per camera
- This is intentional to avoid annoying the operator

---

## Real-Time Monitoring

### Watch Logs in Real-Time

```bash
# Run application and grep for collision checks
python testIR/testIR.py 2>&1 | grep "COLLISION CHECK"

# Watch for collision events only
python testIR/testIR.py 2>&1 | grep "COLLISION DETECTED"

# Full debug output
python testIR/testIR.py 2>&1 | tee debug_output.txt
```

### Quick Test Procedure

1. **Start application:**
   ```bash
   cd /home/inspectura/Desktop/InspecturaGUI
   python testIR/testIR.py
   ```

2. **Verify Lane ROI is enabled:**
   - Look for "☑ Lane ROI" checkbox checked in GUI

3. **Position wood to trigger collision:**
   - **Top collision:** Move wood to top of frame (y ≤ 100)
   - **Bottom collision:** Move wood to bottom of frame (y ≥ 620)

4. **Watch terminal for logs:**
   - Should see `[COLLISION CHECK]` messages
   - Should see `TOP COLLISION=True` or `BOTTOM COLLISION=True`
   - Should see warning sequence

5. **Verify popup appears:**
   - Popup window should appear with warning
   - Click OK to dismiss

---

## Expected Y-Coordinate Ranges

### Frame Coordinate System
```
y = 0   ┌──────────────────────────────┐
        │     TOP LANE (Red)           │
y = 100 ├──────────────────────────────┤
        │                              │
        │     SAFE ZONE (Center)       │
        │     Wood should stay here    │
        │                              │
y = 620 ├──────────────────────────────┤
        │     BOTTOM LANE (Red)        │
y = 720 └──────────────────────────────┘
```

### Collision Triggers

**Top Lane Collision:**
- Triggers when: Wood's top edge ≤ 100 pixels
- Safe range: Wood's top edge > 100 pixels

**Bottom Lane Collision:**
- Triggers when: Wood's bottom edge ≥ 620 pixels
- Safe range: Wood's bottom edge < 620 pixels

**Safe Zone:**
- Wood's top edge: 101 - 619 pixels
- Wood positioned properly in center
- No collision, no warning

---

## Sample Log Analysis

### Example 1: Wood Properly Aligned
```
[COLLISION CHECK] Camera=top | ROI Top=250 | Top Lane Boundary=100 | TOP COLLISION=False
[COLLISION CHECK] Camera=top | ROI Bottom=400 | Bottom Lane Boundary=620 | BOTTOM COLLISION=False
[COLLISION CHECK] Camera=top | ✅ NO COLLISION - Wood is properly aligned
```

**Analysis:** 
- ROI Top = 250 (> 100) ✅
- ROI Bottom = 400 (< 620) ✅
- Wood is in safe zone, no warning needed

### Example 2: Wood Too High (Top Collision)
```
[COLLISION CHECK] Camera=top | ROI Top=75 | Top Lane Boundary=100 | TOP COLLISION=True
[DEBUG COLLISION] ⚠️ TOP LANE COLLISION DETECTED!
⚠ WARNING [top]: Wood is MISALIGNED - touching TOP lane!
[COLLISION CHECK] Camera=top | ⚠️ COLLISION DETECTED - Notification triggered
```

**Analysis:**
- ROI Top = 75 (≤ 100) ⚠️
- Wood is touching top lane
- Notification triggered correctly

### Example 3: Wood Too Low (Bottom Collision)
```
[COLLISION CHECK] Camera=top | ROI Top=400 | Top Lane Boundary=100 | TOP COLLISION=False
[COLLISION CHECK] Camera=top | ROI Bottom=650 | Bottom Lane Boundary=620 | BOTTOM COLLISION=True
[DEBUG COLLISION] ⚠️ BOTTOM LANE COLLISION DETECTED!
⚠ WARNING [top]: Wood is MISALIGNED - touching BOTTOM lane!
[COLLISION CHECK] Camera=top | ⚠️ COLLISION DETECTED - Notification triggered
```

**Analysis:**
- ROI Top = 400 (> 100) ✅
- ROI Bottom = 650 (≥ 620) ⚠️
- Wood is touching bottom lane
- Notification triggered correctly

---

## Quick Reference

### Collision Logic Summary

```python
# Top lane collision
if ROI_Top_Y <= 100:
    TOP_COLLISION = True  # Wood too high!

# Bottom lane collision  
if ROI_Bottom_Y >= 620:
    BOTTOM_COLLISION = True  # Wood too low!
```

### Log Search Commands

```bash
# Find all collision checks
grep "\[COLLISION CHECK\]" debug.log

# Find only collisions detected
grep "COLLISION=True" debug.log

# Find notification triggers
grep "show_alignment_warning called" debug.log

# Find queue processing
grep "\[DEBUG QUEUE\]" debug.log
```

---

## Summary

The enhanced logging provides:

1. ✅ **Clear collision status** for every frame with wood detection
2. ✅ **Exact Y-coordinates** showing wood position vs lane boundaries
3. ✅ **Explicit True/False** for collision detection
4. ✅ **Summary message** indicating collision or no collision
5. ✅ **Notification tracking** showing when popups are triggered

**Use these logs to:**
- Verify collision detection is working
- Confirm Y-coordinate logic is correct
- Debug notification system issues
- Understand why collisions are/aren't detected

**No need for test buttons** - the logs tell you everything! 📊
