"""
Example script demonstrating the Lasso Annotation Tool.

This script shows how to use the LassoAnnotationTool programmatically
for testing or automation purposes.
"""

import numpy as np
from data_annotation.lasso_tool import LassoAnnotationTool


def create_test_image():
    """Create a simple test image with a circle."""
    size = 256
    image = np.zeros((size, size), dtype=np.uint8)
    
    # Draw a circle
    center = (size // 2, size // 2)
    radius = 60
    y, x = np.ogrid[:size, :size]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    image[mask] = 255
    
    # Add some noise to make edges interesting
    noise = np.random.normal(0, 10, (size, size))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return image


def test_lasso_tool_basic():
    """Test basic functionality of the lasso tool."""
    print("Testing LassoAnnotationTool...")
    
    # Create tool instance
    tool = LassoAnnotationTool()
    
    # Check initial state
    assert not tool.is_active, "Tool should not be active initially"
    
    # Test label value setting
    tool.set_label_value(2)
    assert tool.current_label_value == 2, "Label value should be 2"
    
    try:
        tool.set_label_value(0)
        assert False, "Should raise ValueError for label value 0"
    except ValueError:
        pass  # Expected
    
    print("✓ Basic tests passed")


def test_edge_detection():
    """Test edge detection functionality."""
    print("Testing edge detection...")
    
    tool = LassoAnnotationTool()
    image = create_test_image()
    
    # Test edge detection
    edges = tool._detect_edges(image)
    
    assert edges.shape == image.shape, "Edge map should have same shape as input"
    assert edges.dtype == np.float64, "Edge map should be float64"
    assert edges.min() >= 0 and edges.max() <= 1, "Edge values should be normalized"
    
    # Check that edges were actually detected
    assert edges.max() > 0, "Should detect some edges"
    
    print("✓ Edge detection tests passed")


def test_contour_smoothing():
    """Test contour smoothing functionality."""
    print("Testing contour smoothing...")
    
    tool = LassoAnnotationTool()
    
    # Create a simple square contour
    points = np.array([
        [50, 50],
        [50, 150],
        [150, 150],
        [150, 50]
    ], dtype=float)
    
    # Test smoothing
    smoothed = tool._smooth_contour(points, smoothing=5.0)
    
    assert smoothed.shape[1] == 2, "Smoothed points should be 2D"
    assert len(smoothed) >= len(points), "Should have at least as many points"
    
    print("✓ Contour smoothing tests passed")


def test_lasso_refinement():
    """Test lasso refinement with edge snapping."""
    print("Testing lasso refinement...")
    
    tool = LassoAnnotationTool()
    tool.current_image = create_test_image()
    
    # Create a rough circular contour around the test circle
    theta = np.linspace(0, 2*np.pi, 20, endpoint=False)
    radius = 70  # Slightly larger than the actual circle (60)
    center = 128
    points = np.column_stack([
        center + radius * np.sin(theta),
        center + radius * np.cos(theta)
    ])
    
    # Refine the lasso
    refined = tool._refine_lasso_to_edges(points)
    
    assert refined.shape[1] == 2, "Refined points should be 2D"
    assert len(refined) > 0, "Should have refined points"
    
    # The refined contour should be closer to the actual edge
    # (This is a simplified test - in reality, we'd check the distance to actual edges)
    
    print("✓ Lasso refinement tests passed")


def run_all_tests():
    """Run all unit tests for the lasso tool."""
    print("\n" + "="*60)
    print("Running LassoAnnotationTool Tests")
    print("="*60 + "\n")
    
    try:
        test_lasso_tool_basic()
        test_edge_detection()
        test_contour_smoothing()
        test_lasso_refinement()
        
        print("\n" + "="*60)
        print("✓ All tests passed successfully!")
        print("="*60 + "\n")
        
        return True
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_all_tests()
