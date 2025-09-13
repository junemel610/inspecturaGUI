# ROI Annotation Tool

A Python tool for drawing bounding boxes (Regions of Interest) on images and extracting coordinates.

## Features

- ✅ Load images from file or webcam
- ✅ Mouse-based rectangle drawing
- ✅ Real-time coordinate display
- ✅ Save/load ROI coordinates to JSON
- ✅ Support for multiple ROIs per image
- ✅ Clear instructions and keyboard shortcuts
- ✅ Fullscreen mode support

## Installation

The tool uses the following dependencies (already included in the project):
- OpenCV (`opencv-python`)
- NumPy
- Tkinter (built-in with Python)

## Usage

### Command Line

```bash
# Using an image file
python modules/roi_annotation_tool.py path/to/your/image.jpg

# Using webcam
python modules/roi_annotation_tool.py --webcam

# Show help
python modules/roi_annotation_tool.py --help
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` or `ESC` | Quit the application |
| `s` | Save ROIs to JSON file |
| `l` | Load ROIs from JSON file |
| `c` | Clear all ROIs |
| `d` | Delete last ROI |
| `r` | Reset image (clear all ROIs and reload) |
| `h` | Show/hide help overlay |
| `f` | Toggle fullscreen mode |

### Mouse Controls

- **Click and drag** to draw rectangles
- Rectangles must be at least 10x10 pixels to be saved
- Multiple ROIs can be drawn on the same image

## Output Format

When saving ROIs, a JSON file is created with the following structure:

```json
{
  "image_path": "path/to/image.jpg",
  "image_size": [height, width],
  "rois": [
    {
      "id": 1,
      "x1": 100,
      "y1": 150,
      "x2": 300,
      "y2": 350,
      "width": 200,
      "height": 200
    }
  ],
  "timestamp": "2025-09-12T21:37:39.984Z"
}
```

## Example Usage

1. **Load an image:**
   ```bash
   python modules/roi_annotation_tool.py sample_image.jpg
   ```

2. **Draw ROIs:**
   - Click and drag to create rectangles
   - Each ROI will be labeled (ROI 1, ROI 2, etc.)
   - Coordinates are displayed in real-time

3. **Save ROIs:**
   - Press `s` to save to a JSON file
   - File will be named `image_name_rois.json`

4. **Load ROIs:**
   - Press `l` to load previously saved ROIs
   - ROIs will be displayed on the image

## Integration with Wood Sorting Application

This tool can be used to:
- Define regions of interest for wood defect detection
- Create training data for machine learning models
- Annotate areas for quality inspection
- Extract coordinates for automated processing

## Technical Details

- Built with OpenCV for image processing and GUI
- Uses mouse callbacks for interactive drawing
- Supports both static images and live webcam feed
- JSON format for ROI persistence
- Real-time coordinate calculation and display
- Error handling for file operations and camera access

## Troubleshooting

- **Module not found**: Install dependencies with `pip install opencv-python numpy`
- **Camera not accessible**: Check camera permissions and try different camera indices
- **Image not loading**: Verify file path and image format support
- **GUI not responding**: Ensure display environment supports OpenCV windows

## Future Enhancements

- Support for different ROI shapes (circles, polygons)
- Batch processing of multiple images
- Integration with machine learning pipelines
- Advanced annotation features (labels, categories)
- Export to different formats (COCO, Pascal VOC, etc.)