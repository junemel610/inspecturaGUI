from modules.utils_module import GRADE_G2_0, GRADE_G2_1, GRADE_G2_2, GRADE_G2_3, GRADE_G2_4, GRADING_THRESHOLDS, WOOD_PALLET_WIDTH_MM

def grade_individual_defect(defect_type, size_mm, percentage):
    """Grade an individual defect based on SS-EN 1611-1 standards"""
    if defect_type not in GRADING_THRESHOLDS:
        print(f"Unknown defect type: {defect_type}, defaulting to Unsound_Knot")
        defect_type = "Unsound_Knot"
    
    thresholds = GRADING_THRESHOLDS[defect_type]
    
    # Check each grade threshold (size OR percentage can trigger the grade)
    for grade in [GRADE_G2_0, GRADE_G2_1, GRADE_G2_2, GRADE_G2_3]:
        mm_threshold, pct_threshold = thresholds[grade]
        if size_mm <= mm_threshold or percentage <= pct_threshold:
            return grade
    
    # If no threshold met, it's the worst grade
    return GRADE_G2_4

def determine_surface_grade(defect_measurements):
    """Determine overall grade for a surface based on individual defect measurements"""
    if not defect_measurements:
        return GRADE_G2_0
    
    # Get individual grades for each defect
    defect_grades = []
    defect_counts = {}
    
    for defect_type, size_mm, percentage in defect_measurements:
        # Get grade for this individual defect
        grade = grade_individual_defect(defect_type, size_mm, percentage)
        defect_grades.append(grade)
        
        # Count defects by type
        if defect_type not in defect_counts:
            defect_counts[defect_type] = 0
        defect_counts[defect_type] += 1
    
    # Count total defects
    total_defects = sum(defect_counts.values())
    
    # Grade hierarchy for finding worst grade
    grade_hierarchy = [GRADE_G2_0, GRADE_G2_1, GRADE_G2_2, GRADE_G2_3, GRADE_G2_4]
    
    # Find the worst individual defect grade
    worst_grade_index = 0
    for grade in defect_grades:
        if grade in grade_hierarchy:
            grade_index = grade_hierarchy.index(grade)
            worst_grade_index = max(worst_grade_index, grade_index)
    
    worst_individual_grade = grade_hierarchy[worst_grade_index]
    
    # Apply defect count limitations per SS-EN 1611-1
    if total_defects > 6:
        return GRADE_G2_4
    elif total_defects > 4:
        # Maximum G2-3 regardless of individual grades
        return grade_hierarchy[min(3, worst_grade_index)]  # G2-3 is index 3
    elif total_defects > 2:
        # Maximum G2-2 regardless of individual grades  
        return grade_hierarchy[min(2, worst_grade_index)]  # G2-2 is index 2
    
    # Return the worst individual grade if defect count allows
    return worst_individual_grade

def determine_final_grade(top_grade, bottom_grade):
    """Determine final grade based on worst surface (SS-EN 1611-1 standard)"""
    grade_hierarchy = [GRADE_G2_0, GRADE_G2_1, GRADE_G2_2, GRADE_G2_3, GRADE_G2_4]
    
    # Handle None values (no detection)
    if top_grade is None:
        top_grade = GRADE_G2_0
    if bottom_grade is None:
        bottom_grade = GRADE_G2_0
    
    # Get indices for comparison
    top_index = grade_hierarchy.index(top_grade) if top_grade in grade_hierarchy else 0
    bottom_index = grade_hierarchy.index(bottom_grade) if bottom_grade in grade_hierarchy else 0
    
    # Return the worse grade (higher index)
    final_grade = grade_hierarchy[max(top_index, bottom_index)]
    
    print(f"Final grading: Top={top_grade}, Bottom={bottom_grade}, Final={final_grade}")
    return final_grade

def convert_grade_to_arduino_command(standard_grade):
    """Convert SS-EN 1611-1 grade to Arduino sorting command"""
    # Map the 5 standard grades to sorting gates following strict classification:
    # Good: G2-0 | Fair: G2-1, G2-2, G2-3 | Poor: G2-4
    grade_to_command = {
        GRADE_G2_0: 1,    # Good (G2-0) - Gate 1
        GRADE_G2_1: 2,    # Fair (G2-1) - Gate 2  
        GRADE_G2_2: 2,    # Fair (G2-2) - Gate 2
        GRADE_G2_3: 2,    # Fair (G2-3) - Gate 2
        GRADE_G2_4: 3     # Poor (G2-4) - Gate 3
    }
    
    return grade_to_command.get(standard_grade, 3)  # Default to worst gate if unknown

def get_grade_color(grade):
    """Get color coding for grades"""
    color_map = {
        GRADE_G2_0: 'dark green',
        GRADE_G2_1: 'green', 
        GRADE_G2_2: 'orange',
        GRADE_G2_3: 'dark orange',
        GRADE_G2_4: 'red'
    }
    return color_map.get(grade, 'gray')

def calculate_grade(defect_dict):
    """Calculate grade based on defect dictionary and return grade info"""
    total_defects = sum(defect_dict.values()) if defect_dict else 0
    
    if total_defects == 0:
        return {
            'grade': 0,  # Grade 0 for perfect wood
            'text': 'Perfect (No Defects)',
            'total_defects': 0,
            'color': 'dark green'
        }
    elif total_defects <= 2:
        return {
            'grade': 1,
            'text': f'Good (G2-0) - {total_defects} defects',
            'total_defects': total_defects,
            'color': 'green'
        }
    elif total_defects <= 6:
        return {
            'grade': 2,
            'text': f'Fair (G2-1, G2-2, G2-3) - {total_defects} defects',
            'total_defects': total_defects,
            'color': 'orange'
        }
    else:
        return {
            'grade': 3,
            'text': f'Poor (G2-4) - {total_defects} defects',
            'total_defects': total_defects,
            'color': 'red'
        }