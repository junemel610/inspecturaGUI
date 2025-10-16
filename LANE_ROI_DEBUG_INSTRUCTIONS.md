# Lane ROI Debug Instructions

## The Lane ROI is Not Showing - Let's Debug!

I've added debug print statements to help us figure out why the lanes aren't appearing.

### Steps to Debug:

1. **Run the application:**
   ```bash
   cd /home/inspectura/Desktop/InspecturaGUI
   python testIR/testIR.py
   ```

2. **Watch the terminal output** - You should see messages like:
   ```
   [DEBUG] draw_wood_detection_overlay called for top
   [DEBUG] lane_roi_var value: True
   [DEBUG] camera_name in ALIGNMENT_LANE_ROIS: True
   [DEBUG] Drawing lanes for top!
   ```

3. **Check the "Lane ROI" checkbox** in the ROI panel

4. **Look for these debug messages** in the terminal:

### What the Debug Output Will Tell Us:

#### ✅ **If you see:**
```
[DEBUG] draw_wood_detection_overlay called for top
[DEBUG] lane_roi_var value: True
[DEBUG] camera_name in ALIGNMENT_LANE_ROIS: True
[DEBUG] Drawing lanes for top!
```
**This means:** The function is being called AND the checkbox is checked AND it's trying to draw lanes.
**Problem:** The drawing code itself might have an issue.

#### ❌ **If you see:**
```
[DEBUG] draw_wood_detection_overlay called for top
[DEBUG] lane_roi_var value: False
```
**This means:** The checkbox is NOT checked or not working.
**Solution:** Click the "Lane ROI" checkbox.

#### ❌ **If you see:**
```
[DEBUG] draw_wood_detection_overlay called for top
[DEBUG] lane_roi_var value: True
[DEBUG] camera_name in ALIGNMENT_LANE_ROIS: False
```
**This means:** The camera name doesn't match the configuration.
**Solution:** Check if camera_name is "top" or "bottom" (lowercase).

####  **If you DON'T see any debug messages:**
**This means:** The function `draw_wood_detection_overlay` is NOT being called at all.
**Solution:** Check if the camera feed is updating.

### After Getting Debug Output:

Once you see the debug output in the terminal, **report back what you see** and I'll help you fix the specific issue!

### Quick Test:

1. Make sure the application is running
2. Check the terminal for `[DEBUG]` messages
3. Try clicking the "Lane ROI" checkbox on/off
4. Watch for any changes in the debug output

**The debug messages will tell us exactly where the problem is!** 🔍
