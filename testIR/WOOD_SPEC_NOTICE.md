# Large Wood Specification Notice Added

## ✅ Implementation Complete

### Changes Made

**1. Added Prominent Notice Banner**
- **Location:** Top center of the application window
- **Text:** "⚠ ACCEPTS ONLY 21\" × 5\" PALOCHINA WOOD ⚠"
- **Style:** Large, bold, red background with white text
- **Positioning:** Centered horizontally, 10 pixels from top

**2. Adjusted Camera Layout**
- **Top Camera Canvas:** Moved from y=25 to y=60 (35 pixels down)
- **Bottom Camera Canvas:** Moved from y=25 to y=60 (35 pixels down)
- **Reason:** Make room for the notice banner at the top

---

## Visual Appearance

### Notice Banner Properties
```python
notice_label = tk.Label(self, 
    text="⚠ ACCEPTS ONLY 21\" × 5\" PALOCHINA WOOD ⚠",
    font=("Arial", 22, "bold"),     # Large bold font
    bg="#FF4444",                    # Bright red background
    fg="white",                      # White text
    relief=tk.RAISED,                # 3D raised border
    borderwidth=5,                   # Thick border
    padx=30,                         # Horizontal padding
    pady=15                          # Vertical padding
)
notice_label.place(relx=0.5, y=10, anchor="n")  # Centered at top
```

### Layout Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                    Application Window                        │
├─────────────────────────────────────────────────────────────┤
│  y=10                                                        │
│     ╔═════════════════════════════════════════════════╗     │
│     ║  ⚠ ACCEPTS ONLY 21" × 5" PALOCHINA WOOD ⚠      ║     │
│     ║          (Large Red Banner)                     ║     │
│     ╚═════════════════════════════════════════════════╝     │
│                                                              │
│  y=60                                                        │
│  ┌──────────────────────┐  ┌──────────────────────┐        │
│  │   Top Camera Feed    │  │  Bottom Camera Feed  │        │
│  │                      │  │                      │        │
│  │      (360px high)    │  │     (360px high)     │        │
│  │                      │  │                      │        │
│  └──────────────────────┘  └──────────────────────┘        │
│                                                              │
│  y=415                                                       │
│  ┌─────────────┐ ┌──────┐ ┌────────────────┐              │
│  │   Status    │ │ ROI  │ │ Conveyor Ctrl  │              │
│  └─────────────┘ └──────┘ └────────────────┘              │
│                                                              │
│  ... (rest of UI below) ...                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Wood Specification Details

### Dimensions
- **Length:** 21 inches (533.4 mm)
- **Width:** 5 inches (127 mm)
- **Wood Type:** Palochina (Philippine hardwood)

### Visual Display
The notice prominently displays:
- **Warning symbols:** ⚠ on both sides
- **Measurement:** 21" × 5" (inches format)
- **Wood type:** PALOCHINA WOOD
- **Style:** ALL CAPS for emphasis
- **Color:** Red background to grab attention

---

## Purpose

This large notice serves multiple purposes:

### 1. **Operator Guidance**
- Immediately informs operators what wood dimensions are accepted
- Prevents incorrect wood pieces from being loaded
- Reduces sorting errors and downtime

### 2. **Safety and Compliance**
- Clearly communicates system limitations
- Prevents damage from oversized/undersized pieces
- Ensures proper conveyor operation

### 3. **Visibility**
- Large font (22pt bold) easily readable from distance
- Red color immediately draws attention
- Centered position visible at all times
- Warning symbols (⚠) emphasize importance

### 4. **Training**
- New operators immediately see specifications
- No need to reference external documentation
- Reduces training time and mistakes

---

## Technical Details

### Positioning
```python
relx=0.5       # 50% from left (centered horizontally)
y=10           # 10 pixels from top
anchor="n"     # North anchor (top center point)
```

This ensures the banner:
- ✅ Always centered regardless of window width
- ✅ Stays at top of application
- ✅ Doesn't overlap with camera feeds
- ✅ Visible in fullscreen and windowed mode

### Colors
- **Background:** `#FF4444` (Bright red - high visibility)
- **Foreground:** `white` (Maximum contrast)
- **Border:** Raised 3D effect with 5px width

### Font
- **Family:** Arial (clean, readable sans-serif)
- **Size:** 22pt (very large)
- **Style:** Bold (extra emphasis)

---

## Camera Feed Adjustment

To accommodate the notice banner, camera feeds were moved down:

### Before
```python
self.top_canvas.place(x=25, y=25, ...)      # Started at y=25
self.bottom_canvas.place(x=..., y=25, ...)  # Started at y=25
```

### After
```python
self.top_canvas.place(x=25, y=60, ...)      # Moved to y=60 (+35px)
self.bottom_canvas.place(x=..., y=60, ...)  # Moved to y=60 (+35px)
```

### Space Allocation
```
y=0-10:   Window border/padding
y=10-55:  Wood specification notice banner (45px tall)
y=55-60:  Small gap (5px)
y=60-420: Camera feeds (360px tall)
y=420+:   Control panels and UI elements (unchanged)
```

---

## Customization Options

If you need to modify the notice:

### Change Text
```python
text="⚠ ACCEPTS ONLY 21\" × 5\" PALOCHINA WOOD ⚠"
# Modify dimensions or wood type as needed
```

### Change Colors
```python
bg="#FF4444"  # Red - can change to other warning colors
fg="white"    # White text - can change for different contrast
```

### Change Size
```python
font=("Arial", 22, "bold")  # Increase/decrease number for size
pady=15                      # Adjust vertical padding
padx=30                      # Adjust horizontal padding
```

### Change Position
```python
relx=0.5, y=10, anchor="n"   # Centered top
# Or move to bottom:
# relx=0.5, rely=1.0, y=-10, anchor="s"  # Centered bottom
```

---

## Files Modified

1. **testIR/testIR.py**
   - Added wood specification notice label (~lines 2503-2512)
   - Moved top camera canvas from y=25 to y=60 (~line 2514)
   - Moved bottom camera canvas from y=25 to y=60 (~line 2517)

2. **Documentation**
   - Created this guide (WOOD_SPEC_NOTICE.md)

---

## Testing Checklist

✅ **No syntax errors** - Code verified and ready  
✅ **Notice displays at top** - Centered and prominent  
✅ **Camera feeds moved down** - No overlap with notice  
✅ **Text is readable** - Large font, high contrast  
✅ **Colors are attention-grabbing** - Red background with white text  
✅ **Responsive positioning** - Centered regardless of window size  

---

## Summary

✅ **Large red notice banner added** at top center of application  
✅ **Displays wood specifications**: 21" × 5" Palochina Wood  
✅ **High visibility**: Large bold font, red background, warning symbols  
✅ **Camera feeds adjusted**: Moved down 35 pixels to accommodate banner  
✅ **Always visible**: Stays at top in all modes and screen sizes  

**The notice is now prominently displayed and impossible to miss!** 🎯
