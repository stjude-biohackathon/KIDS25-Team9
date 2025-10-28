"""
Lasso Annotation Tool - Usage Guide
====================================

This document provides detailed instructions on how to use the lasso annotation
feature in the AI Image Lab plugin.

## Overview

The Lasso Annotation Tool allows you to quickly annotate objects in images by
drawing a rough outline around them. The tool then automatically refines your
drawing to snap to the actual edges of the object.

## Getting Started

### Prerequisites
- AI Image Lab napari plugin installed
- An image loaded in the annotation tab

### Step-by-Step Instructions

1. **Open the AI Image Lab Plugin**
   - Launch napari
   - Open the AI Image Lab plugin from the Plugins menu

2. **Set Up Your Project**
   - In the Project Setup tab, configure your input image and label folders
   - Select a task (e.g., "Semantic Segmentation 2D")
   - Continue to the Annotation tab

3. **Load an Image**
   - Select an image from the list
   - Click "Create/Load Labels" to open it in the viewer
   - This creates two layers: "Anno-Image" and "Anno-Labels"

4. **Activate the Lasso Tool**
   - Click the "Enable Lasso" button
   - You will see a confirmation message with usage tips
   - A "Lasso-Shapes" layer will be created for drawing

5. **Draw Your Annotation**
   - Click points around the object you want to annotate
   - Don't worry about being precise - just get close to the boundary
   - Click near the starting point or press Enter to complete the polygon
   - The tool will automatically:
     * Detect edges in your image
     * Smooth your rough outline
     * Snap the contour to the nearest object boundaries
     * Add the refined annotation to your labels layer

6. **Continue Annotating**
   - Keep the lasso tool enabled to annotate more objects
   - Each completed polygon is automatically added to your labels
   - You can change the label value in the napari labels layer if needed

7. **Save Your Annotations**
   - Click "Disable Lasso" when done
   - Click "Save Label" to save your annotations

## Algorithm Details

The lasso tool uses several image processing techniques:

1. **Edge Detection**: Canny edge detector identifies boundaries in the image
2. **Contour Smoothing**: B-spline interpolation smooths your rough drawing
3. **Active Contour**: Snake algorithm deforms the contour to match edges
4. **Rasterization**: The final contour is converted to a binary mask

### Parameters

The tool uses optimized parameters for most use cases:
- Edge detection sigma: 1.0
- Active contour alpha (elasticity): 0.015
- Active contour beta (stiffness): 10
- Active contour iterations: 100

## Tips for Best Results

1. **Draw Generously**: Your rough outline doesn't need to be exact, but should
   be close to the object boundary

2. **Good Edge Contrast**: The tool works best when objects have clear edges
   with good contrast from the background

3. **Number of Points**: 10-20 points is usually sufficient for most objects

4. **3D Images**: For 3D volumes, the tool operates on the current Z-slice

5. **Canceling**: Press ESC to cancel the current polygon if you make a mistake

6. **Label Values**: Set the desired label value in the napari labels layer
   before drawing (default is 1)

## Troubleshooting

**Problem**: The lasso snapping doesn't work well
- **Solution**: The image may have weak edges. Try adjusting the image contrast
  or using the standard napari tools for difficult cases

**Problem**: "No viewer found" error
- **Solution**: Make sure you've loaded an image first using "Create/Load Labels"

**Problem**: The tool is too slow
- **Solution**: The active contour algorithm runs for 100 iterations. For very
  large images, this might take a few seconds. This is normal.

**Problem**: The refined shape includes too much/too little
- **Solution**: Try drawing your initial outline closer to the desired boundary.
  The algorithm uses your drawing as a starting point.

## Advanced Usage

### Programmatic Usage

You can also use the lasso tool programmatically:

```python
from data_annotation.lasso_tool import LassoAnnotationTool
import napari

# Create viewer and load image
viewer = napari.Viewer()
viewer.add_image(your_image, name="Anno-Image")
viewer.add_labels(your_labels, name="Anno-Labels")

# Create and activate lasso tool
lasso = LassoAnnotationTool()
lasso.activate(viewer=viewer)

# Tool is now ready for drawing
# To deactivate:
lasso.deactivate()
```

### Customizing for Your Data

If the default parameters don't work well for your specific images, you can
modify the LassoAnnotationTool class in `src/data_annotation/lasso_tool.py`:

- Adjust edge detection parameters in `_detect_edges()`
- Modify active contour parameters in `_active_contour_snap()`
- Change smoothing strength in `_smooth_contour()`

## Related Features

- **Interpolate 3D**: Use after annotating a few slices to propagate labels
- **Standard Napari Tools**: Use the paint brush, fill, and erase tools for
  fine-tuning
- **Save Label**: Always save your work before moving to the next image

## Support

For issues or questions:
- Check the README.md for additional information
- Review the code comments in lasso_tool.py
- Open an issue on the GitHub repository
