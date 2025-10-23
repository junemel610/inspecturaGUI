# Wood Lane Alignment - Visual Reference

## Camera View Layout (Top View)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FULL CAMERA FRAME                            │
│  (1280 x 720 pixels)                                                │
│                                                                     │
│  ┌──────┬─────────────────────────────────────────────┬──────┐    │
│  │      │        MAIN DETECTION ROI AREA              │      │    │
│  │ LEFT │                                             │ RIGHT│    │
│  │ LANE │         (Wood should be here)               │ LANE │    │
│  │      │                                             │      │    │
│  │ RED  │              ┌─────────────┐                │ RED  │    │
│  │ ZONE │              │   WOOD      │                │ ZONE │    │
│  │      │              │  BOUNDING   │                │      │    │
│  │  55  │              │    BOX      │                │  55  │    │
│  │  px  │              │  (GREEN)    │                │  px  │    │
│  │      │              └─────────────┘                │      │    │
│  │      │                                             │      │    │
│  │      │         Centered, no lane touch            │      │    │
│  │      │                                             │      │    │
│  └──────┴─────────────────────────────────────────────┴──────┘    │
│  x:345  x:400                                   x:825 x:880        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Proper Alignment (GOOD) ✅

```
┌──────┬──────────────────────────────────────────┬──────┐
│ RED  │                                          │ RED  │
│ LANE │        ┌──────────────────┐              │ LANE │
│      │        │                  │              │      │
│  ❌  │        │   WOOD (GREEN)   │              │  ❌  │
│      │        │                  │              │      │
│      │        └──────────────────┘              │      │
└──────┴──────────────────────────────────────────┴──────┘

✅ Wood is centered
✅ No intersection with lane ROIs
✅ No warning displayed
```

## Left Misalignment (BAD) ⚠️

```
┌──────┬──────────────────────────────────────────┬──────┐
│ RED  │                                          │ RED  │
│ LANE │                                          │ LANE │
│   ┌──┼───────────┐                             │      │
│  ❌│  │   WOOD    │                             │  ❌  │
│   │  │  (GREEN)  │                             │      │
│   └──┼───────────┘                             │      │
└──────┴──────────────────────────────────────────┴──────┘

⚠️ Wood touches LEFT LANE
⚠️ Warning: "Wood is touching the left lane boundary"
```

## Right Misalignment (BAD) ⚠️

```
┌──────┬──────────────────────────────────────────┬──────┐
│ RED  │                                          │ RED  │
│ LANE │                                          │ LANE │
│      │                         ┌───────────┬──┐ │      │
│  ❌  │                         │   WOOD    │  │❌│     │
│      │                         │  (GREEN)  │  │ │      │
│      │                         └───────────┴──┘ │      │
└──────┴──────────────────────────────────────────┴──────┘

⚠️ Wood touches RIGHT LANE
⚠️ Warning: "Wood is touching the right lane boundary"
```

## Both Lanes (SEVERE) ⚠️⚠️

```
┌──────┬──────────────────────────────────────────┬──────┐
│ RED  │                                          │ RED  │
│ LANE │                                          │ LANE │
│ ┌────┼──────────────────────────────────────┬──┼───┐  │
│ │ ❌ │         WOOD (GREEN)                 │  │❌ │  │
│ │    │                                      │  │   │  │
│ └────┼──────────────────────────────────────┴──┼───┘  │
└──────┴──────────────────────────────────────────┴──────┘

⚠️⚠️ Wood touches BOTH LANES
⚠️⚠️ Warning: "Wood is touching the both lane boundary"
```

## Configuration Values

### Top Camera Lane ROIs
```python
"top": {
    "left_lane": {
        "x1": 345,   # Left edge of main ROI
        "y1": 0,     # Top of frame
        "x2": 400,   # 55 pixels wide
        "y2": 720    # Bottom of frame
    },
    "right_lane": {
        "x1": 825,   # Near right edge
        "y1": 0,     # Top of frame
        "x2": 880,   # 55 pixels wide
        "y2": 720    # Bottom of frame
    }
}
```

### Bottom Camera Lane ROIs
```python
"bottom": {
    "left_lane": {
        "x1": 350,   # Left edge of main ROI
        "y1": 0,     # Top of frame
        "x2": 405,   # 55 pixels wide
        "y2": 720    # Bottom of frame
    },
    "right_lane": {
        "x1": 910,   # Near right edge
        "y1": 0,     # Top of frame
        "x2": 965,   # 55 pixels wide
        "y2": 720    # Bottom of frame
    }
}
```

## Lane Width Calculation

For each lane:
- **Width** = x2 - x1
- **Top Left Lane**: 400 - 345 = 55 pixels
- **Top Right Lane**: 880 - 825 = 55 pixels
- **Bottom Left Lane**: 405 - 350 = 55 pixels
- **Bottom Right Lane**: 965 - 910 = 55 pixels

## Center Safe Zone

The area between the lanes is the "safe zone" where wood should be positioned:

### Top Camera Safe Zone
- **Left boundary**: x = 400
- **Right boundary**: x = 825
- **Width**: 425 pixels (centered area)

### Bottom Camera Safe Zone
- **Left boundary**: x = 405
- **Right boundary**: x = 910
- **Width**: 505 pixels (centered area)

## Adjustment Guide

### To Make Lanes Wider (More Sensitive)
```python
# Expand left lane to the right
"x2": 450,  # Was 400, now 50 pixels wider

# Expand right lane to the left
"x1": 775,  # Was 825, now 50 pixels wider
```

### To Make Lanes Narrower (Less Sensitive)
```python
# Shrink left lane
"x2": 370,  # Was 400, now 30 pixels narrower

# Shrink right lane
"x1": 855,  # Was 825, now 30 pixels narrower
```

### To Move Lanes Inward (Smaller Safe Zone)
```python
# Move left lane right
"x1": 395,  # Was 345
"x2": 450,  # Was 400

# Move right lane left
"x1": 775,  # Was 825
"x2": 830,  # Was 880
```

### To Move Lanes Outward (Larger Safe Zone)
```python
# Move left lane left
"x1": 295,  # Was 345
"x2": 350,  # Was 400

# Move right lane right
"x1": 875,  # Was 825
"x2": 930,  # Was 880
```

## Color Legend

When viewing the overlay:
- 🔴 **RED**: Lane boundaries (no-go zones)
- 🟢 **GREEN**: Detected wood (primary candidate)
- 🟡 **YELLOW**: Secondary wood candidates or auto ROI
- 🔵 **BLUE**: (Not used in lane detection)

## Real-World Example

If your wood piece is 100mm wide and the camera resolution is 1280x720:
- **Typical wood bbox width**: ~35-50 pixels (depends on camera distance)
- **Lane width**: 55 pixels
- **Safe zone**: 425+ pixels wide

This means the wood has plenty of room to stay centered without triggering lanes.
