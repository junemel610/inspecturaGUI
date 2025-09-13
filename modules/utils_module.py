import numpy as np

# SS-EN 1611-1 Grading Standards Implementation
# Grade constants
GRADE_G2_0 = "G2-0"
GRADE_G2_1 = "G2-1"
GRADE_G2_2 = "G2-2"
GRADE_G2_3 = "G2-3"
GRADE_G2_4 = "G2-4"

# Camera-specific calibration based on your setup
# Top camera: 37cm distance, Bottom camera: 29cm distance
# Assuming 1280x720 resolution with typical camera FOV
TOP_CAMERA_DISTANCE_CM = 37
BOTTOM_CAMERA_DISTANCE_CM = 29

# Estimated pixel-to-millimeter factors (will be refined with actual measurements)
# These are calculated based on typical camera FOV at your distances
TOP_CAMERA_PIXEL_TO_MM = 0.4  # Adjusted for 37cm distance
BOTTOM_CAMERA_PIXEL_TO_MM = 0.3  # Adjusted for 29cm distance (closer = smaller pixels)

# Your actual wood pallet width
WOOD_PALLET_WIDTH_MM = 115  # 11.5cm = 115mm

# SS-EN 1611-1 Grading thresholds for each defect type
GRADING_THRESHOLDS = {
    "Sound_Knot": {  # Live knots
        GRADE_G2_0: (10, 5),      # (mm, percentage)
        GRADE_G2_1: (30, 15),
        GRADE_G2_2: (50, 25),
        GRADE_G2_3: (70, 35),
        GRADE_G2_4: (float('inf'), float('inf'))
    },
    "Unsound_Knot": {  # Dead knots, missing knots, knots with cracks
        GRADE_G2_0: (7, 3.5),
        GRADE_G2_1: (20, 10),
        GRADE_G2_2: (35, 17.5),
        GRADE_G2_3: (50, 25),
        GRADE_G2_4: (float('inf'), float('inf'))
    }
}

def calibrate_pixel_to_mm(reference_object_width_px, reference_object_width_mm, camera_name="top"):
    """Calibrate the pixel-to-millimeter conversion factor for specific camera"""
    global TOP_CAMERA_PIXEL_TO_MM, BOTTOM_CAMERA_PIXEL_TO_MM
    
    conversion_factor = reference_object_width_mm / reference_object_width_px
    
    if camera_name == "top":
        TOP_CAMERA_PIXEL_TO_MM = conversion_factor
        print(f"Calibrated TOP camera pixel-to-mm factor: {TOP_CAMERA_PIXEL_TO_MM}")
    else:  # bottom camera
        BOTTOM_CAMERA_PIXEL_TO_MM = conversion_factor
        print(f"Calibrated BOTTOM camera pixel-to-mm factor: {BOTTOM_CAMERA_PIXEL_TO_MM}")
    
    return conversion_factor

def calibrate_with_wood_pallet(wood_pallet_width_px_top, wood_pallet_width_px_bottom):
    """Auto-calibrate both cameras using the known wood pallet width"""
    print(f"Auto-calibrating cameras with {WOOD_PALLET_WIDTH_MM}mm wood pallet...")
    
    top_factor = calibrate_pixel_to_mm(wood_pallet_width_px_top, WOOD_PALLET_WIDTH_MM, "top")
    bottom_factor = calibrate_pixel_to_mm(wood_pallet_width_px_bottom, WOOD_PALLET_WIDTH_MM, "bottom") 
    
    print(f"Calibration complete:")
    print(f"  Top camera (37cm): {top_factor:.4f} mm/pixel")
    print(f"  Bottom camera (29cm): {bottom_factor:.4f} mm/pixel")
    
    return top_factor, bottom_factor

def map_model_output_to_standard(model_label):
    """Map your model's output labels to standardized defect types"""
    # Mapping from UpdatedDefects model labels to standard categories
    label_mapping = {
        # UpdatedDefects model outputs
        "Dead Knots": "Unsound_Knot",
        "Knots with Crack": "Unsound_Knot", 
        "Live Knots": "Sound_Knot",
        "Missing Knots": "Unsound_Knot",
        
        # Variations and fallbacks
        "dead knots": "Unsound_Knot",
        "knots with crack": "Unsound_Knot",
        "live knots": "Sound_Knot", 
        "missing knots": "Unsound_Knot",
        
        # Legacy mappings
        "sound_knots": "Sound_Knot",
        "unsound_knots": "Unsound_Knot",
        "sound knots": "Sound_Knot",
        "unsound knots": "Unsound_Knot",
        "live_knot": "Sound_Knot",
        "dead_knot": "Unsound_Knot",
        "missing_knot": "Unsound_Knot",
        "crack_knot": "Unsound_Knot",
        
        # Generic fallback
        "knot": "Unsound_Knot"
    }
    
    # Normalize the label (lowercase, remove extra spaces)
    normalized_label = model_label.lower().strip().replace('_', ' ')
    
    # Return mapped label or default to unsound knot
    mapped_label = label_mapping.get(normalized_label, "Unsound_Knot")
    print(f"DEBUG: Mapped defect '{model_label}' → '{mapped_label}'")
    return mapped_label

def calculate_defect_size(detection_box, camera_name="top"):
    """Calculate defect size in mm and percentage from detection bounding box"""
    try:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = detection_box['bbox']
        
        # Calculate defect dimensions in pixels
        width_px = abs(x2 - x1)
        height_px = abs(y2 - y1)
        
        # Use the larger dimension (worst case for grading)
        max_dimension_px = max(width_px, height_px)
        
        # Use camera-specific conversion factor
        if camera_name == "top":
            pixel_to_mm = TOP_CAMERA_PIXEL_TO_MM
        else:  # bottom camera
            pixel_to_mm = BOTTOM_CAMERA_PIXEL_TO_MM
        
        # Convert to millimeters
        size_mm = max_dimension_px * pixel_to_mm
        
        # Calculate percentage of actual wood pallet width
        percentage = (size_mm / WOOD_PALLET_WIDTH_MM) * 100
        
        return size_mm, percentage
        
    except Exception as e:
        print(f"Error calculating defect size: {e}")
        # Return conservative values if calculation fails
        return 50.0, 35.0  # Assumes large defect for safety