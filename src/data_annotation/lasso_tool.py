"""
Lasso annotation tool with automatic edge snapping.

This module provides a LassoAnnotationTool class that allows users to draw
freehand lasso shapes in napari that automatically snap to object boundaries.
"""

import numpy as np
from typing import Optional, Tuple, List
from skimage import filters, segmentation, measure
from skimage.draw import polygon
from scipy import ndimage
from scipy.interpolate import splprep, splev


class LassoAnnotationTool:
    """
    A tool for creating lasso annotations that automatically snap to image edges.
    
    This tool allows users to draw a rough lasso shape around an object,
    and then automatically refines the shape to snap to the nearest edges
    in the image using active contour algorithms.
    """
    
    def __init__(self, viewer=None):
        """
        Initialize the lasso annotation tool.
        
        Parameters
        ----------
        viewer : napari.Viewer, optional
            The napari viewer instance to use for annotation.
        """
        self.viewer = viewer
        self.shapes_layer = None
        self.current_image = None
        self.current_label_value = 1
        self._active = False
        self._callbacks = {}
        
    def activate(self, viewer=None, image_layer_name: str = "Anno-Image", 
                 labels_layer_name: str = "Anno-Labels"):
        """
        Activate the lasso tool.
        
        Parameters
        ----------
        viewer : napari.Viewer, optional
            The napari viewer to use. If None, uses the viewer from __init__.
        image_layer_name : str
            Name of the image layer to use for edge detection.
        labels_layer_name : str
            Name of the labels layer to add annotations to.
        """
        if viewer is not None:
            self.viewer = viewer
            
        if self.viewer is None:
            raise ValueError("No viewer available. Please provide a viewer.")
            
        # Get the image for edge detection
        if image_layer_name in self.viewer.layers:
            self.current_image = self.viewer.layers[image_layer_name].data
        else:
            # If no image layer, we'll work without edge snapping
            self.current_image = None
            
        # Create or get the shapes layer for drawing
        if "Lasso-Shapes" not in self.viewer.layers:
            self.shapes_layer = self.viewer.add_shapes(
                name="Lasso-Shapes",
                shape_type='polygon',
                edge_width=2,
                edge_color='yellow',
                face_color=[0, 0, 0, 0],  # transparent fill
                opacity=0.7
            )
        else:
            self.shapes_layer = self.viewer.layers["Lasso-Shapes"]
            
        # Set shapes layer to add mode
        self.shapes_layer.mode = 'add_polygon'
        
        # Connect callback for when a shape is added
        self._callbacks['data'] = self.shapes_layer.events.data.connect(
            self._on_shape_added
        )
        
        self._active = True
        print("[LassoTool] Activated - Draw a polygon around the object")
        
    def deactivate(self):
        """Deactivate the lasso tool and clean up."""
        if self._active:
            # Disconnect callbacks
            if 'data' in self._callbacks:
                self._callbacks['data'].disconnect()
                self._callbacks.clear()
                
            # Remove the shapes layer if it exists
            if self.shapes_layer is not None and "Lasso-Shapes" in self.viewer.layers:
                self.viewer.layers.remove("Lasso-Shapes")
                self.shapes_layer = None
                
            self._active = False
            print("[LassoTool] Deactivated")
            
    def _on_shape_added(self, event):
        """
        Callback when a shape is drawn.
        Refines the shape to snap to edges and adds it to the labels layer.
        """
        if not self._active or self.shapes_layer is None:
            return
            
        # Get the last added shape
        if len(self.shapes_layer.data) == 0:
            return
            
        shape_data = self.shapes_layer.data[-1]
        
        # Refine the shape to snap to edges
        refined_shape = self._refine_lasso_to_edges(shape_data)
        
        # Add the refined shape to the labels layer
        self._add_to_labels(refined_shape)
        
        # Remove the temporary shape
        if len(self.shapes_layer.data) > 0:
            self.shapes_layer.data = self.shapes_layer.data[:-1]
            
        print(f"[LassoTool] Added annotation with {len(refined_shape)} points")
        
    def _refine_lasso_to_edges(self, points: np.ndarray) -> np.ndarray:
        """
        Refine lasso points to snap to image edges.
        
        Parameters
        ----------
        points : np.ndarray
            The rough lasso points drawn by the user (N, 2) for 2D or (N, 3) for 3D.
            
        Returns
        -------
        np.ndarray
            The refined points snapping to edges.
        """
        if self.current_image is None:
            # No image available, return original points
            return points
            
        # Handle 2D vs 3D
        if points.shape[1] == 3:
            # 3D case: use the current z-slice
            z_slice = int(points[0, 0])
            points_2d = points[:, 1:]  # Use y, x coordinates
            
            if z_slice >= self.current_image.shape[0]:
                z_slice = self.current_image.shape[0] - 1
            image_slice = self.current_image[z_slice]
        else:
            # 2D case
            points_2d = points
            image_slice = self.current_image
            
        # Ensure we have a 2D grayscale image
        if image_slice.ndim == 3:
            # Convert RGB to grayscale
            image_slice = np.mean(image_slice, axis=-1)
            
        # Detect edges in the image
        edges = self._detect_edges(image_slice)
        
        # Smooth the initial contour
        smoothed_points = self._smooth_contour(points_2d)
        
        # Apply active contour to snap to edges
        refined_points_2d = self._active_contour_snap(smoothed_points, edges, image_slice)
        
        # Convert back to original format
        if points.shape[1] == 3:
            # Add z coordinate back
            z_coords = np.full((refined_points_2d.shape[0], 1), z_slice)
            refined_points = np.hstack([z_coords, refined_points_2d])
        else:
            refined_points = refined_points_2d
            
        return refined_points
        
    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in the image using Canny edge detection.
        
        Parameters
        ----------
        image : np.ndarray
            2D grayscale image.
            
        Returns
        -------
        np.ndarray
            Edge map (binary image).
        """
        # Normalize image to 0-1 range
        image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Apply Gaussian smoothing
        image_smooth = filters.gaussian(image_norm, sigma=1.0)
        
        # Canny edge detection
        edges = filters.canny(image_smooth, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
        
        return edges.astype(np.float64)
        
    def _smooth_contour(self, points: np.ndarray, smoothing: float = 5.0) -> np.ndarray:
        """
        Smooth the contour using spline interpolation.
        
        Parameters
        ----------
        points : np.ndarray
            Input points (N, 2) in (row, col) format.
        smoothing : float
            Smoothing parameter for spline.
            
        Returns
        -------
        np.ndarray
            Smoothed points.
        """
        if len(points) < 4:
            return points
            
        # Close the contour if not already closed
        if not np.allclose(points[0], points[-1]):
            points = np.vstack([points, points[0]])
            
        try:
            # Fit a B-spline to the points
            # Note: splprep expects (x, y) so we need to transpose
            tck, u = splprep([points[:, 1], points[:, 0]], s=smoothing, per=True)
            
            # Evaluate the spline at more points for smoothness
            u_new = np.linspace(0, 1, len(points) * 2)
            smooth_x, smooth_y = splev(u_new, tck)
            
            # Convert back to (row, col) format
            smooth_points = np.column_stack([smooth_y, smooth_x])
            
            return smooth_points
        except Exception as e:
            print(f"[LassoTool] Smoothing failed: {e}, using original points")
            return points
            
    def _active_contour_snap(self, points: np.ndarray, edges: np.ndarray, 
                            image: np.ndarray, iterations: int = 100) -> np.ndarray:
        """
        Snap contour to edges using active contour (snake) algorithm.
        
        Parameters
        ----------
        points : np.ndarray
            Initial contour points (N, 2) in (row, col) format.
        edges : np.ndarray
            Edge map to snap to.
        image : np.ndarray
            Original image for gradient computation.
        iterations : int
            Number of iterations for active contour.
            
        Returns
        -------
        np.ndarray
            Refined contour points.
        """
        # Compute gradient magnitude for edge attraction
        gradient = filters.sobel(image)
        
        # Combine edges and gradient for better snapping
        edge_strength = edges + 0.5 * gradient
        edge_strength = edge_strength / (edge_strength.max() + 1e-8)
        
        # Create external force field (negative gradient points toward edges)
        external_force = -ndimage.gaussian_gradient_magnitude(edge_strength, sigma=1.0)
        
        try:
            # Use scikit-image's active_contour
            # Note: active_contour uses (x, y) coordinates, so swap columns
            init_snake = np.column_stack([points[:, 1], points[:, 0]])
            
            # Run active contour
            snake = segmentation.active_contour(
                external_force,
                init_snake,
                alpha=0.015,  # elasticity (smoothness)
                beta=10,      # stiffness (curvature penalty)
                gamma=0.001,  # step size
                max_iterations=iterations,
                convergence=0.1
            )
            
            # Convert back to (row, col) format
            refined_points = np.column_stack([snake[:, 1], snake[:, 0]])
            
            return refined_points
        except Exception as e:
            print(f"[LassoTool] Active contour failed: {e}, using smoothed points")
            return points
            
    def _add_to_labels(self, points: np.ndarray):
        """
        Add the refined lasso shape to the labels layer.
        
        Parameters
        ----------
        points : np.ndarray
            Refined contour points.
        """
        if self.viewer is None:
            return
            
        # Find or create the labels layer
        labels_layer_name = "Anno-Labels"
        if labels_layer_name not in self.viewer.layers:
            print(f"[LassoTool] Labels layer '{labels_layer_name}' not found")
            return
            
        labels_layer = self.viewer.layers[labels_layer_name]
        labels_data = labels_layer.data.copy()
        
        # Handle 2D vs 3D
        if points.shape[1] == 3:
            # 3D case
            z_slice = int(points[0, 0])
            points_2d = points[:, 1:]
            
            if z_slice >= labels_data.shape[0]:
                z_slice = labels_data.shape[0] - 1
                
            # Create mask for this slice
            rr, cc = polygon(points_2d[:, 0], points_2d[:, 1], 
                           shape=labels_data[z_slice].shape)
            
            # Add to labels
            labels_data[z_slice][rr, cc] = self.current_label_value
        else:
            # 2D case
            rr, cc = polygon(points[:, 0], points[:, 1], shape=labels_data.shape)
            labels_data[rr, cc] = self.current_label_value
            
        # Update the labels layer
        labels_layer.data = labels_data
        
    def set_label_value(self, value: int):
        """
        Set the label value to use for new annotations.
        
        Parameters
        ----------
        value : int
            The label value (must be > 0).
        """
        if value > 0:
            self.current_label_value = value
        else:
            raise ValueError("Label value must be greater than 0")
            
    @property
    def is_active(self) -> bool:
        """Check if the lasso tool is currently active."""
        return self._active
