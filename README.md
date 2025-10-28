# AI-Image-Lab : Framework for developing Deep Learning models for image analysis tasks


**AI-Image-Lab** is a napari plugin designed to make deep learning models for image analysis.

The project’s goal is to provide a user-friendly interface that seamlessly integrates with Napari, allowing :  

- **Annotation** incorporating model assisted annotation on top napari's annotation framework
- **Training** deep learning models on imaging datasets through the GUI 
- **Running inference** using pre-trained or custom-trained models
- **Fine-tuning / adapting** existing models to new imaging conditions or domains


### Supported Tasks & Architectures  

- **Semantic segmentation (2D & 3D)** using **U-Net (2D and 3D variants)**.  
- **Instance segmentation** with **Mask R-CNN**.  
- **Object detection** with **Faster R-CNN**
- **Transformer** networks incoming in next version !











### Annotation Features

#### Lasso Annotation with Edge Snapping

The plugin includes an intelligent lasso annotation tool that automatically snaps drawn shapes to object boundaries:

**How to use:**
1. Load an image using the "Create/Load Labels" button in the Annotation tab
2. Click the "Enable Lasso" button to activate the tool
3. Draw a rough polygon around the object you want to annotate
4. The tool will automatically:
   - Detect edges in the image using Canny edge detection
   - Smooth your rough drawing
   - Snap the contour to nearby object boundaries using active contour algorithms
   - Add the refined annotation to your labels layer

**Features:**
- **Automatic edge detection**: Uses advanced image processing to find object boundaries
- **Smart snapping**: Active contour algorithm refines your rough drawing to match actual edges
- **2D and 3D support**: Works with both 2D images and 3D volumes (slice by slice)
- **Easy to use**: Just draw a rough shape and let the algorithm do the precision work

**Tips:**
- Draw your lasso shape close to but not necessarily exactly on the object boundary
- The algorithm works best when the object has clear edges
- You can adjust the current label value in the napari labels layer before drawing
- Press ESC to cancel the current polygon if needed
- Remember to disable the lasso tool when you're done annotating

