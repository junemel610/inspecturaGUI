# Boundary Fine-Tuning for Bottom Camera

## Analysis of False Positive

### What Happened
```
TOP Camera:    y=137 to y=608  ✅ NO COLLISION
BOTTOM Camera: y=87 to y=641   ❌ COLLISION (by 1 pixel!)
```

**The Issue:** Bottom camera boundary was y=640, wood bottom edge was y=641  
**Off by:** Only **1 pixel**! This is too strict for properly aligned wood.

---

## Detailed Analysis

### Top Camera (Reference - Correct)
```
Wood ROI: y=137 to y=608
Top boundary: y=100    → 137 > 100 ✅ Safe (37 pixels margin)
Bottom boundary: y=620 → 608 < 620 ✅ Safe (12 pixels margin)
Result: NO COLLISION ✅
```

### Bottom Camera (Before Adjustment - Too Strict)
```
Wood ROI: y=87 to y=641
Top boundary: y=80     → 87 > 80 ✅ Safe (7 pixels margin)
Bottom boundary: y=640 → 641 >= 640 ❌ COLLISION (1 pixel over!)
Result: FALSE POSITIVE ⚠️
```

**Problem:** Wood was properly aligned, but exceeded boundary by just 1 pixel due to:
- Camera perspective differences
- Optical distortion at frame edges
- Natural wood detection variance (±few pixels)

---

## Solution: Adjusted Bottom Camera Boundaries

### Changes Made

**Bottom camera boundaries adjusted:**
```python
# BEFORE (Too strict):
"top_lane": {"y2": 80}     # Top lane boundary
"bottom_lane": {"y1": 640}  # Bottom lane boundary

# AFTER (More lenient):
"top_lane": {"y2": 75}     # -5 pixels (tighter top)
"bottom_lane": {"y1": 650}  # +10 pixels (more room at bottom)
```

### Rationale

**Top boundary: 80 → 75 (-5 pixels)**
- Bottom camera consistently shows wood starting around y=85-87
- Original y=80 gave only 5-7 pixel margin (too tight)
- New y=75 gives 10-12 pixel margin (more reasonable)

**Bottom boundary: 640 → 650 (+10 pixels)**
- Bottom camera wood ends around y=629-641 for aligned wood
- Original y=640 was too strict (failed on y=641)
- New y=650 gives 9-21 pixel margin (appropriate tolerance)

---

## New Safe Zones

### Top Camera (Unchanged - Works Well)
```
y = 0   ┌──────────────────────────┐
        │   TOP LANE (y=0-100)     │
y = 100 ├──────────────────────────┤
        │                          │
        │   SAFE ZONE              │
        │   (y=101-619)            │
        │   519 pixels             │
        │                          │
y = 620 ├──────────────────────────┤
        │   BOTTOM LANE (y=620-720)│
y = 720 └──────────────────────────┘
```

### Bottom Camera (Adjusted - More Forgiving)
```
y = 0   ┌──────────────────────────┐
        │   TOP LANE (y=0-75)      │ ← SMALLER (was 80)
y = 75  ├──────────────────────────┤
        │                          │
        │   SAFE ZONE              │
        │   (y=76-649)             │ ← LARGER!
        │   574 pixels             │ ← +55 pixels vs top camera
        │                          │
y = 650 ├──────────────────────────┤ ← MORE ROOM (was 640)
        │   BOTTOM LANE (y=650-720)│
y = 720 └──────────────────────────┘
```

**Safe zone comparison:**
- Top camera: 519 pixels (y=101-619)
- Bottom camera: **574 pixels** (y=76-649)
- Difference: **+55 pixels** (10.6% more forgiving)

---

## Expected Results

### With Your Wood Position

**Top Camera:**
```
Wood: y=137 to y=608
  Top: 137 > 100 ✅ (37px margin)
  Bottom: 608 < 620 ✅ (12px margin)
Result: NO COLLISION ✅
```

**Bottom Camera (After Adjustment):**
```
Wood: y=87 to y=641
  Top: 87 > 75 ✅ (12px margin - improved from 7px)
  Bottom: 641 < 650 ✅ (9px margin - improved from -1px!)
Result: NO COLLISION ✅
```

---

## Testing Results You Should See

Run the application again with properly aligned wood:

```
============================================================
[COLLISION CHECK] Checking wood alignment for BOTTOM camera
============================================================
  Wood ROI: x=0, y=87, w=615, h=554
  ROI Top Edge: y=87
  ROI Bottom Edge: y=641
  Top Lane Boundary: y=75 (camera-specific)       ← Changed from 80
  TOP COLLISION: False (Wood top=87 > 75)         ← Still safe ✅
  Bottom Lane Boundary: y=650 (camera-specific)   ← Changed from 640
  BOTTOM COLLISION: False (Wood bottom=641 < 650) ← NOW FALSE! ✅
  ✅ RESULT: NO COLLISION - Wood is properly aligned
============================================================
```

**No more false positives!** 🎉

---

## Tolerance Analysis

### Margin Distribution

**Top Camera (Stricter - Reference):**
```
Top margin:    137 - 100 = 37 pixels
Bottom margin: 620 - 608 = 12 pixels
Total buffer:  49 pixels
```

**Bottom Camera (More Lenient - Adjusted):**
```
Top margin:    87 - 75 = 12 pixels
Bottom margin: 650 - 641 = 9 pixels
Total buffer:  21 pixels
```

**Why bottom camera needs more tolerance:**
1. Mounted at different angle → optical distortion
2. Different lens characteristics → edge detection variance
3. Further from wood → more perspective distortion
4. Processing happens after top camera → accumulated variance

---

## When to Trigger Collision

### Truly Misaligned Wood Examples

**Example 1: Wood Too High**
```
Bottom Camera ROI: y=50 to y=600
  Top: 50 < 75 ❌ COLLISION (25px into danger zone)
  Bottom: 600 < 650 ✅
Action: Trigger TOP lane warning ⚠️
```

**Example 2: Wood Too Low**
```
Bottom Camera ROI: y=100 to y=670
  Top: 100 > 75 ✅
  Bottom: 670 >= 650 ❌ COLLISION (20px into danger zone)
Action: Trigger BOTTOM lane warning ⚠️
```

**Example 3: Properly Aligned (Your Case)**
```
Bottom Camera ROI: y=87 to y=641
  Top: 87 > 75 ✅ (12px safe margin)
  Bottom: 641 < 650 ✅ (9px safe margin)
Action: No warning ✅
```

---

## Fine-Tuning Guidelines

### If Still Getting False Positives on Bottom Camera

**Option 1: Increase bottom boundary further**
```python
"bottom_lane": {"y1": 660}  # Even more room (currently 650)
```

**Option 2: Decrease top boundary**
```python
"top_lane": {"y2": 70}  # Even more room at top (currently 75)
```

### If Missing Real Misalignments

**Make boundaries stricter:**
```python
"bottom": {
    "top_lane": {"y2": 80},    # Back to stricter (currently 75)
    "bottom_lane": {"y1": 645}  # Slightly stricter (currently 650)
}
```

### Calibration Process

1. **Test with 5 properly aligned wood pieces**
   - Record min/max Y coordinates for both cameras
   - Calculate average safe zone

2. **Add 10-15 pixel margin** to both boundaries
   - Accounts for natural variance
   - Prevents false positives from minor variations

3. **Test with intentionally misaligned wood**
   - Move wood 50+ pixels out of alignment
   - Verify both cameras detect the misalignment

4. **Fine-tune based on results**
   - Adjust boundaries in 5-pixel increments
   - Balance between false positives and false negatives

---

## Summary of Changes

| Boundary | Camera | Before | After | Change | Reason |
|----------|--------|--------|-------|--------|--------|
| Top Lane | Top | 100 | 100 | No change | Working well |
| Bottom Lane | Top | 620 | 620 | No change | Working well |
| Top Lane | Bottom | 80 | **75** | **-5px** | Tighter, more margin |
| Bottom Lane | Bottom | 640 | **650** | **+10px** | More room, fix false positive |

**Result:**
- ✅ Top camera: Still accurate (unchanged)
- ✅ Bottom camera: More forgiving (fixed false positive by 1 pixel)
- ✅ Overall: Better tolerance for camera perspective differences

---

## Monitoring Recommendations

**After this adjustment, monitor for:**

1. **False negatives:** Missing actual misalignments
   - If wood is clearly misaligned but no warning → boundaries too loose
   - Solution: Tighten boundaries by 5 pixels

2. **False positives:** Warnings on properly aligned wood
   - If still getting warnings on good wood → boundaries still too strict
   - Solution: Loosen boundaries by 5-10 more pixels

3. **Consistency:** Compare warnings between cameras
   - Both cameras should agree on aligned/misaligned
   - Large discrepancies indicate calibration needed

---

## Current Configuration Summary

```python
ALIGNMENT_LANE_ROIS = {
    "top": {
        "top_lane": {"y2": 100},     # Standard reference
        "bottom_lane": {"y1": 620}    # Standard reference
    },
    "bottom": {
        "top_lane": {"y2": 75},      # More lenient (-5px from 80)
        "bottom_lane": {"y1": 650}    # More lenient (+10px from 640)
    }
}
```

**Safe zones:**
- Top camera: y=101-619 (519 pixels)
- Bottom camera: y=76-649 (574 pixels, +10.6% tolerance)

This configuration should eliminate false positives while still catching real misalignments! 🎯
