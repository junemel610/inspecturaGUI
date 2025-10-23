# Lane ROI Checkbox Control Guide

## Overview

The **Lane ROI** checkbox allows you to toggle the visibility of the horizontal lane alignment ROIs on the camera feeds. This helps verify that the alignment detection zones are active and correctly positioned.

## Location

The **Lane ROI** checkbox is located in the **ROI** panel, alongside the existing checkboxes:

```
┌────────────────────────────┐
│         ROI Panel          │
├────────────────────────────┤
│ ☑ Top ROI                  │
│ ☑ Bottom ROI               │
│ ☑ Lane ROI     ← NEW!      │
└────────────────────────────┘
```

## Functionality

### ✅ Checked (Default State)
When the **Lane ROI** checkbox is **CHECKED**:
- ✅ Horizontal lane ROIs are **VISIBLE** on camera feeds
- ✅ Red semi-transparent bands appear at top and bottom
- ✅ "TOP LANE" and "BOTTOM LANE" labels are displayed
- ✅ Lane borders are drawn with solid red lines
- ✅ Misalignment detection is **ACTIVE**

**Visual Example:**
```
═══════════════════════════════════
║  🔴 TOP LANE (Red Zone)      ║
═══════════════════════════════════
┌─────────────────────────────────┐
│                                 │
│      ┌───────────────┐          │
│      │  🟢 WOOD BOX  │          │
│      └───────────────┘          │
│                                 │
└─────────────────────────────────┘
═══════════════════════════════════
║  🔴 BOTTOM LANE (Red Zone)   ║
═══════════════════════════════════
```

### ❌ Unchecked
When the **Lane ROI** checkbox is **UNCHECKED**:
- ❌ Lane ROIs are **HIDDEN** from camera feeds
- ❌ No red zones displayed
- ❌ No lane labels shown
- ✅ Misalignment detection still **ACTIVE** (runs in background)
- ✅ Warnings will still appear if wood touches lanes

**Visual Example:**
```
┌─────────────────────────────────┐
│                                 │
│                                 │
│      ┌───────────────┐          │
│      │  🟢 WOOD BOX  │          │
│      └───────────────┘          │
│                                 │
│                                 │
└─────────────────────────────────┘
```

## Use Cases

### When to Enable (✅ Checked)

1. **Setup & Calibration**
   - Verifying lane positions are correct
   - Adjusting ALIGNMENT_LANE_ROIS configuration
   - Visual confirmation of detection zones

2. **Testing & Debugging**
   - Checking if wood is approaching lane boundaries
   - Understanding why misalignment warnings appear
   - Visual troubleshooting

3. **Operator Training**
   - Showing operators where safe zones are
   - Demonstrating alignment requirements
   - Educational purposes

### When to Disable (❌ Unchecked)

1. **Normal Operation**
   - Cleaner camera view without overlay graphics
   - Reduced visual clutter during production
   - Focus on wood detection results only

2. **Presentations/Reports**
   - Clean camera feed for documentation
   - Screenshots without debug overlays
   - Professional appearance

3. **Performance (Minimal Impact)**
   - Slightly faster rendering (minor difference)
   - Less GPU usage for overlay drawing

## Behavior Details

### Default State
- **Enabled by default** (`value=True`)
- Lane ROIs are visible when application starts
- Ensures operators can see alignment zones immediately

### Toggle Action
Clicking the checkbox:
1. Instantly shows/hides lane overlays
2. Prints status to console: `"Lane ROI display enabled"` or `"Lane ROI display disabled"`
3. No camera restart required
4. Changes apply immediately to both cameras

### Console Output
```bash
# When checked
Lane ROI display enabled

# When unchecked
Lane ROI display disabled
```

## Technical Details

### Implementation
```python
# Checkbox variable (line ~2361)
self.lane_roi_var = tk.BooleanVar(value=True)

# Checkbox widget (line ~2362)
ttk.Checkbutton(roi_frame, text="Lane ROI", 
                variable=self.lane_roi_var,
                command=self.toggle_lane_roi).pack(anchor="w")

# Toggle function (line ~3622)
def toggle_lane_roi(self):
    """Toggle lane ROI visibility for alignment detection"""
    status = "enabled" if self.lane_roi_var.get() else "disabled"
    print(f"Lane ROI display {status}")

# Drawing logic (line ~3928)
if self.lane_roi_var.get() and camera_name in ALIGNMENT_LANE_ROIS:
    # Draw lane overlays...
```

### Lane Configuration
The lanes being controlled are defined in `ALIGNMENT_LANE_ROIS`:

**Top Camera:**
```python
"top": {
    "top_lane": {"x1": 345, "y1": 0, "x2": 880, "y2": 100},
    "bottom_lane": {"x1": 345, "y1": 620, "x2": 880, "y2": 720}
}
```

**Bottom Camera:**
```python
"bottom": {
    "top_lane": {"x1": 350, "y1": 0, "x2": 965, "y2": 100},
    "bottom_lane": {"x1": 350, "y1": 620, "x2": 965, "y2": 720}
}
```

## ROI Panel Overview

All three checkboxes work independently:

| Checkbox      | Controls                    | Affects                     |
|---------------|-----------------------------|-----------------------------|
| **Top ROI**   | Top camera detection zone   | TOP camera analysis         |
| **Bottom ROI**| Bottom camera detection zone| BOTTOM camera analysis      |
| **Lane ROI**  | Lane alignment overlays     | Both cameras (visual only)  |

### Combined States Example

```
✅ Top ROI      → Top camera active
✅ Bottom ROI   → Bottom camera active  
✅ Lane ROI     → Lane overlays visible

Result: Full system operation with visual lane guides
```

```
✅ Top ROI      → Top camera active
✅ Bottom ROI   → Bottom camera active
❌ Lane ROI     → Lane overlays hidden

Result: Full detection, clean camera view
```

## Verification Steps

### 1. Start Application
```bash
python testIR/testIR.py
```

### 2. Enable Live Detection
- Check the **Live Detection** checkbox
- Wait for beam to be blocked (or use scan mode)

### 3. Test Lane ROI Checkbox

**Enable Lanes (✅):**
1. ✅ Check "Lane ROI"
2. Look at camera feeds
3. Should see red horizontal bands at top and bottom
4. "TOP LANE" and "BOTTOM LANE" labels visible

**Disable Lanes (❌):**
1. ❌ Uncheck "Lane ROI"
2. Look at camera feeds
3. Red bands should disappear
4. Labels should disappear
5. Clean camera view

### 4. Verify Warnings Still Work
Even with Lane ROI unchecked:
- Misalignment detection still runs
- Warnings still appear if wood touches lanes
- Only the **visual overlay** is hidden

## Troubleshooting

### ❌ Checkbox Not Visible
**Problem:** Lane ROI checkbox doesn't appear  
**Solution:** 
- Restart application
- Check if GUI was modified correctly
- Verify line ~2361-2363 in testIR.py

### ❌ Lanes Don't Show When Checked
**Problem:** Checkbox is checked but no lanes visible  
**Solution:**
- Enable "Live Detection" checkbox first
- Block IR beam to trigger detection mode
- Lanes only show during active detection

### ❌ Lanes Show When Unchecked
**Problem:** Lanes visible even when checkbox is unchecked  
**Solution:**
- Restart application
- Check console for errors
- Verify draw_wood_detection_overlay was updated correctly

### ❌ Console Message Not Appearing
**Problem:** No "Lane ROI display enabled/disabled" message  
**Solution:**
- Check if toggle_lane_roi function was added
- Verify callback is connected to checkbox
- Look for Python errors in console

## Best Practices

### Recommended Workflow

1. **Initial Setup:**
   - ✅ Enable Lane ROI
   - Verify lanes appear in correct positions
   - Test with sample wood pieces

2. **Calibration:**
   - ✅ Keep Lane ROI enabled
   - Adjust ALIGNMENT_LANE_ROIS if needed
   - Confirm safe zone is appropriate

3. **Production:**
   - ❌ Disable Lane ROI for clean view
   - ✅ Keep Top ROI and Bottom ROI enabled
   - Warnings still active in background

4. **Troubleshooting:**
   - ✅ Re-enable Lane ROI
   - Visually inspect alignment
   - Adjust wood positioning

### Configuration Tips

To adjust lane sensitivity, modify `ALIGNMENT_LANE_ROIS` (lines ~163-194):

**More Sensitive (Taller Lanes):**
```python
"top_lane": {"x1": 345, "y1": 0, "x2": 880, "y2": 150},    # Was 100
"bottom_lane": {"x1": 345, "y1": 570, "x2": 880, "y2": 720} # Was 620
```

**Less Sensitive (Shorter Lanes):**
```python
"top_lane": {"x1": 345, "y1": 0, "x2": 880, "y2": 50},     # Was 100
"bottom_lane": {"x1": 345, "y1": 670, "x2": 880, "y2": 720} # Was 620
```

## Summary

✅ **Lane ROI checkbox added successfully!**

**Features:**
- ✅ Toggle lane visibility on/off
- ✅ Verify lane positions during setup
- ✅ Clean camera view when disabled
- ✅ Detection still runs in background
- ✅ Instant toggle (no restart needed)

**Location:** ROI Panel → Lane ROI checkbox

**Default:** ✅ Enabled (visible)

**Use:** Toggle as needed for visual confirmation or clean view!

🎉 You can now see and control the horizontal lane ROIs!
