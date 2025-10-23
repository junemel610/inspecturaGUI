# Rotated Lane Labels - Visual Guide

## Updated Lane Label Orientation

The "LEFT LANE" and "RIGHT LANE" labels are now **rotated 90 degrees** to display vertically along the lane edges, similar to highway lane markings.

### Visual Layout

```
                    CAMERA FEED
┌─────────────────────────────────────────────────┐
│                                                 │
│  ┌─┬─────────────────────────────────────┬─┐  │
│  │L│                                     │R│  │
│  │E│                                     │I│  │
│  │F│         SAFE ZONE                   │G│  │
│  │T│      (Wood should be here)          │H│  │
│  │ │                                     │T│  │
│  │L│        ┌───────────┐                │ │  │
│  │A│        │  WOOD     │                │L│  │
│  │N│        │ (GREEN)   │                │A│  │
│  │E│        │   BOX     │                │N│  │
│  │ │        └───────────┘                │E│  │
│  │ │                                     │ │  │
│  │ │       Properly Aligned              │ │  │
│  │ │                                     │ │  │
│  └─┴─────────────────────────────────────┴─┴──┘
│  🔴                                       🔴   │
│  RED                                      RED  │
│  ZONE                                     ZONE │
└─────────────────────────────────────────────────┘
```

### Text Orientation

**Before** (Horizontal - WRONG):
```
┌─────┐
│ LEFT│
│ LANE│
└─────┘
```

**After** (Vertical - CORRECT):
```
┌─┐
│L│
│E│
│F│
│T│
│ │
│L│
│A│
│N│
│E│
└─┘
```

## How It Works

The rotated text is achieved by:

1. **Creating temporary image** with the text
2. **Drawing text horizontally** on temp image
3. **Rotating image 90°** using OpenCV's `cv2.rotate()`
4. **Placing rotated text** on the main frame

### Rotation Details

- **Angle**: 90 degrees counter-clockwise
- **Direction**: Bottom to top (reading upward)
- **Position**: Centered vertically on each lane
- **Font**: Hershey Simplex, 0.6 scale
- **Color**: White (RGB: 255, 255, 255)
- **Thickness**: 2 pixels

## Visual Examples

### Left Lane (TOP camera view)
```
Position: x=345 to x=400
Text: "LEFT LANE"

┌──┐
│ L│ ← Letter "L" at bottom
│ E│
│ F│
│ T│
│  │ ← Space
│ L│
│ A│
│ N│
│ E│ ← Letter "E" at top
└──┘
```

### Right Lane (TOP camera view)
```
Position: x=825 to x=880
Text: "RIGHT LANE"

┌──┐
│ R│ ← Letter "R" at bottom
│ I│
│ G│
│ H│
│ T│
│  │ ← Space
│ L│
│ A│
│ N│
│ E│ ← Letter "E" at top
└──┘
```

## Complete Camera View with Rotated Labels

```
        TOP CAMERA (1280 x 720)
┌─────────────────────────────────────────┐
│                                         │
│  ┌──┬─────────────────────────┬──┐    │
│  │  │  🔴 Semi-transparent     │  │    │
│  │ L│     Red Zone            │ R│    │
│  │ E│                         │ I│    │
│  │ F│  ┌──────────────────┐   │ G│    │
│  │ T│  │                  │   │ H│    │
│  │  │  │   🟢 WOOD BOX    │   │ T│    │
│  │ L│  │   (GREEN)        │   │  │    │
│  │ A│  │                  │   │ L│    │
│  │ N│  └──────────────────┘   │ A│    │
│  │ E│                         │ N│    │
│  │  │     Safe Zone           │ E│    │
│  │  │  🔴 Semi-transparent     │  │    │
│  └──┴─────────────────────────┴──┘    │
│   ↑                             ↑      │
│  Left                          Right   │
│  Lane                          Lane    │
└─────────────────────────────────────────┘
```

## Reading Direction

The text reads from **BOTTOM to TOP**:
- Start reading at the bottom of the lane
- Read upward along the vertical lane
- Similar to reading a vertical sign or banner

This matches the physical orientation of highway lanes where text is painted vertically along the road edges.

## Technical Implementation

### Function: `draw_rotated_text()`

Located inside `draw_wood_detection_overlay()` function.

**Steps**:
1. Calculate text size
2. Create temporary black canvas
3. Draw white text horizontally
4. Rotate canvas 90° counter-clockwise
5. Extract text region using mask
6. Composite onto main frame

**Parameters**:
- `img`: Frame to draw on
- `text`: "LEFT LANE" or "RIGHT LANE"
- `position`: (x, y) coordinates
- `font`: cv2.FONT_HERSHEY_SIMPLEX
- `font_scale`: 0.6
- `color`: (255, 255, 255) white
- `thickness`: 2 pixels
- `angle`: 90 degrees

## Testing the Rotated Labels

1. **Start the application**:
   ```bash
   python testIR/testIR.py
   ```

2. **Enable Live Detection mode**

3. **Look for rotated text**:
   - "LEFT LANE" should appear vertically on the left red zone
   - "RIGHT LANE" should appear vertically on the right red zone
   - Text should read from bottom to top

4. **Verify positioning**:
   - Text centered vertically in the middle of each lane
   - Text positioned in the middle of the lane width
   - White color stands out against red background

## Color Scheme

| Element | Color | Description |
|---------|-------|-------------|
| Lane zones | 🔴 Red (30% opacity) | Semi-transparent red fill |
| Lane borders | 🔴 Red (100% opacity, 3px) | Solid red outline |
| Lane labels | ⚪ White (100% opacity) | **ROTATED 90°** |
| Wood box | 🟢 Green | Detected wood |

## Advantages of Rotated Text

✅ **Better use of space**: Fits within narrow lane width
✅ **Clearer labeling**: Text runs along the lane edge
✅ **Professional look**: Similar to highway lane markings
✅ **Less obstruction**: Doesn't block horizontal view
✅ **Easier reading**: Natural eye movement along lane

## Troubleshooting

### "Text is upside down"
- Text should read from bottom to top
- If inverted, the rotation angle may need adjustment
- Currently set to 90° counter-clockwise (correct)

### "Text is cut off"
- The function includes boundary checking
- Text automatically adjusts if near frame edge
- Should not exceed lane boundaries

### "Text is too small/large"
- Adjust `font_scale` parameter (currently 0.6)
- Increase to 0.8 for larger text
- Decrease to 0.4 for smaller text

### "Text is hard to read"
- Increase `thickness` from 2 to 3
- Consider adding black outline (draw text twice with offset)
- Adjust color contrast if needed

## Summary

The lane labels are now properly rotated 90 degrees and will display vertically along each lane edge, providing clear, professional-looking lane identification that matches highway-style lane markings! 🎉
