# Notification Debugging - Complete Flow Trace

## What We're Looking For

When you run the application and a collision is detected, you should see this **complete sequence** of debug messages:

---

## Expected Debug Output (Complete Flow)

### 1. Collision Detection
```
============================================================
[COLLISION CHECK] Checking wood alignment for BOTTOM camera
============================================================
  Wood ROI: x=0, y=53, w=615, h=423
  ROI Top Edge: y=53
  ROI Bottom Edge: y=476
  Top Lane Boundary: y=100
  TOP COLLISION: True (Wood top=53 <= 100)
  ⚠️  MISALIGNMENT DETECTED: Wood is TOO HIGH (touching TOP lane)
```

### 2. Method Call Attempt (NEW DEBUG LOGS)
```
  📞 Checking if parent_app has show_alignment_warning method...
  📞 Calling parent_app.show_alignment_warning('bottom', 'TOP')...
```

### 3. Inside show_alignment_warning Method
```
[DEBUG] show_alignment_warning called: camera=bottom, lane=TOP
[DEBUG] Showing new warning: TOP_LANE
```

### 4. Message Queue
```
[DEBUG] Putting warning message into queue...
[DEBUG] Warning message queued successfully
```

### 5. Console Warning
```
============================================================
⚠ ALIGNMENT WARNING - BOTTOM CAMERA
============================================================
Lane: TOP LANE
Coordinate: y = 100 pixels
Timestamp: 2025-10-16 XX:XX:XX
============================================================
```

### 6. Method Call Completed
```
  📞 show_alignment_warning() call completed
```

### 7. Collision Summary
```
  🚨 RESULT: COLLISION DETECTED - Wood is MISALIGNED!
============================================================
```

### 8. Message Queue Processing (Should appear shortly after)
```
[DEBUG QUEUE] Processing warning message from queue...
[DEBUG QUEUE] Title: ⚠ WOOD MISALIGNMENT DETECTED
[DEBUG QUEUE] Message: Wood piece is MISALIGNED on BOTTOM camera!...
[DEBUG QUEUE] Showing messagebox...
[DEBUG QUEUE] Messagebox shown successfully
```

### 9. Popup Window
**A popup window should appear with:**
- Title: ⚠ WOOD MISALIGNMENT DETECTED
- Message explaining the misalignment
- OK button

---

## Diagnostic Questions

### Q1: Do you see steps 1-7?
**If YES:** Collision detection and method calling works ✅

**If NO:** 
- Which step is the last one you see?
- Do you see the "📞" messages?

### Q2: Do you see step 8 ([DEBUG QUEUE] messages)?
**If NO:** Message queue is not processing
- Check if `process_message_queue()` is running
- This should appear within 50ms of step 7

**If YES but no popup:** Continue to Q3

### Q3: Do you see "Messagebox shown successfully"?
**If YES but no popup visible:**
- Popup may be behind another window
- Check all windows/desktops
- Try Alt+Tab to switch windows

**If NO:**
- `messagebox.showwarning()` is failing
- Python error may be occurring (check for exceptions)

### Q4: Is this the first collision or a repeated one?
**If repeated:**
You might see:
```
[DEBUG] In cooldown: 2.3s < 5.0s
```
OR
```
[DEBUG] Duplicate warning: TOP_LANE already active
```

**This is normal** - cooldown prevents spam. Wait 5 seconds and trigger again.

---

## Troubleshooting Scenarios

### Scenario A: Logs stop at "📞 Calling parent_app.show_alignment_warning..."
**Problem:** Method call is hanging or crashing
**Check:** Look for Python exceptions or error messages after this line

### Scenario B: See all logs up to "Warning message queued successfully" but NO [DEBUG QUEUE]
**Problem:** Message queue not processing
**Cause:** `process_message_queue()` not running or queue blocked
**Solution:** Check if GUI mainloop is running

### Scenario C: See [DEBUG QUEUE] but NOT "Messagebox shown successfully"
**Problem:** `messagebox.showwarning()` is failing
**Possible causes:**
- Tkinter not properly initialized
- Display/X11 issues
- Python error in messagebox call

### Scenario D: See "Messagebox shown successfully" but NO visible popup
**Problem:** Popup is hidden or minimized
**Try:**
- Alt+Tab through windows
- Check taskbar
- Check if popup is behind main window
- Try `messagebox.showinfo()` instead (simpler)

---

## Test Instructions

1. **Run application:**
   ```bash
   cd /home/inspectura/Desktop/InspecturaGUI
   python testIR/testIR.py 2>&1 | tee notification_debug.log
   ```

2. **Trigger collision** (position wood to touch top/bottom lane)

3. **Copy ALL terminal output** starting from:
   ```
   [COLLISION CHECK] Checking wood alignment...
   ```
   up to and including any `[DEBUG QUEUE]` messages

4. **Report back:**
   - Which steps (1-9) do you see?
   - Where does the output stop?
   - Do you see any Python errors?
   - Does a popup appear?

---

## Key Debug Markers

| Marker | Meaning |
|--------|---------|
| `[COLLISION CHECK]` | In collision detection | 
| `📞` | Method call trace (NEW!) |
| `[DEBUG]` | Inside show_alignment_warning |
| `[DEBUG QUEUE]` | Message queue processing |
| `⚠ ALIGNMENT WARNING` | Console warning output |
| `🚨 RESULT` | Collision summary |

---

## Success Criteria

✅ All 9 steps appear in terminal  
✅ Popup window appears on screen  
✅ Popup shows correct camera and lane  
✅ Can click OK to dismiss  

If popup still doesn't appear after seeing all debug messages, we'll try:
1. Direct messagebox call (bypass queue)
2. Alternative notification method
3. Check tkinter/display configuration

---

## Next Steps Based on Results

**Tell me:**
1. Full terminal output from collision to queue processing
2. Which numbered steps appear
3. Any error messages
4. Whether popup appears

This will pinpoint exactly where the notification pipeline breaks! 🔍
