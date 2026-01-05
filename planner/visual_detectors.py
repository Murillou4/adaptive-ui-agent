"""
Visual Detectors for Pixel-Based Element Detection

Pure computer vision (no ML) to detect visual elements from observations.
These detectors convert pixel observations into structured information
that the reward system can use.

Key principle: If you can't measure it by pixels, you can't reward it.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum
import cv2

from planner.goal_dsl import Color, COLOR_RGB, ElementType


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BoundingBox:
    """Bounding box for detected element."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def cx(self) -> int:
        return self.x + self.width // 2
    
    @property
    def cy(self) -> int:
        return self.y + self.height // 2
    
    def overlaps(self, other: "BoundingBox") -> bool:
        """Check if boxes overlap."""
        return not (self.x + self.width < other.x or 
                    other.x + other.width < self.x or
                    self.y + self.height < other.y or 
                    other.y + other.height < self.y)
    
    def distance_to(self, other: "BoundingBox") -> float:
        """Euclidean distance between centers."""
        return np.sqrt((self.cx - other.cx)**2 + (self.cy - other.cy)**2)


@dataclass
class DetectedElement:
    """A detected visual element."""
    type: ElementType
    bbox: BoundingBox
    color: Optional[Color] = None
    confidence: float = 1.0


@dataclass
class AlignmentInfo:
    """Information about alignment of elements."""
    is_aligned_horizontal: bool
    is_aligned_vertical: bool
    horizontal_variance: float  # Lower = more aligned
    vertical_variance: float
    
    def __repr__(self):
        return f"Alignment(h={self.is_aligned_horizontal}, v={self.is_aligned_vertical})"


@dataclass
class SpacingInfo:
    """Information about spacing between elements."""
    spacings: List[float]
    is_equal: bool
    variance: float
    
    @property
    def mean_spacing(self) -> float:
        return np.mean(self.spacings) if self.spacings else 0


# =============================================================================
# CORE DETECTION FUNCTIONS
# =============================================================================

def detect_color_regions(
    obs: np.ndarray, 
    color: Color, 
    tolerance: int = 40,
    min_area: int = 20
) -> List[BoundingBox]:
    """
    Detect regions of a specific color.
    
    Args:
        obs: RGB observation (H, W, 3)
        color: Target color
        tolerance: Color matching tolerance (0-255)
        min_area: Minimum region area to detect
        
    Returns:
        List of bounding boxes for detected regions
    """
    target_rgb = np.array(COLOR_RGB[color])
    
    # Create color mask
    diff = np.abs(obs.astype(np.float32) - target_rgb)
    mask = np.all(diff < tolerance, axis=2).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append(BoundingBox(x, y, w, h))
    
    return boxes


def detect_rectangles(
    obs: np.ndarray,
    min_area: int = 30,
    background_color: Tuple[int, int, int] = (50, 50, 55)
) -> List[DetectedElement]:
    """
    Detect rectangular shapes in observation.
    
    Args:
        obs: RGB observation (H, W, 3)
        min_area: Minimum rectangle area
        background_color: Background color to ignore
        
    Returns:
        List of detected rectangle elements
    """
    # Convert to grayscale
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    
    # Create mask excluding background
    bg_diff = np.abs(obs.astype(np.float32) - np.array(background_color))
    fg_mask = np.any(bg_diff > 30, axis=2).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    elements = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # Approximate polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if it's roughly rectangular (4 corners)
        x, y, w, h = cv2.boundingRect(contour)
        bbox = BoundingBox(x, y, w, h)
        
        # Determine color of this element
        color = _detect_dominant_color(obs, bbox)
        
        elements.append(DetectedElement(
            type=ElementType.RECTANGLE,
            bbox=bbox,
            color=color,
            confidence=1.0 if len(approx) == 4 else 0.8
        ))
    
    return elements


def _detect_dominant_color(obs: np.ndarray, bbox: BoundingBox) -> Optional[Color]:
    """Detect the dominant color in a bounding box region."""
    region = obs[bbox.y:bbox.y+bbox.height, bbox.x:bbox.x+bbox.width]
    if region.size == 0:
        return None
    
    mean_color = np.mean(region, axis=(0, 1))
    
    # Find closest matching color
    min_dist = float('inf')
    best_color = None
    
    for color, rgb in COLOR_RGB.items():
        dist = np.linalg.norm(mean_color - np.array(rgb))
        if dist < min_dist:
            min_dist = dist
            best_color = color
    
    # Only return if reasonably close
    if min_dist < 80:
        return best_color
    return None


def detect_circles(
    obs: np.ndarray,
    min_radius: int = 5,
    max_radius: int = 50
) -> List[DetectedElement]:
    """
    Detect circular shapes in observation.
    
    Args:
        obs: RGB observation (H, W, 3)
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
        
    Returns:
        List of detected circle elements
    """
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    
    # Hough circle detection
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_radius * 2,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    elements = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0]:
            x, y, r = circle
            bbox = BoundingBox(x - r, y - r, r * 2, r * 2)
            color = _detect_dominant_color(obs, bbox)
            
            elements.append(DetectedElement(
                type=ElementType.CIRCLE,
                bbox=bbox,
                color=color
            ))
    
    return elements


# =============================================================================
# ALIGNMENT AND SPACING DETECTION
# =============================================================================

def detect_alignment(
    elements: List[DetectedElement],
    tolerance: float = 10.0
) -> AlignmentInfo:
    """
    Detect if elements are aligned horizontally or vertically.
    
    Args:
        elements: List of detected elements
        tolerance: Pixel tolerance for alignment
        
    Returns:
        AlignmentInfo with alignment details
    """
    if len(elements) < 2:
        return AlignmentInfo(True, True, 0, 0)
    
    centers_x = [e.bbox.cx for e in elements]
    centers_y = [e.bbox.cy for e in elements]
    
    var_x = np.var(centers_x)
    var_y = np.var(centers_y)
    
    # Low variance in Y means horizontal alignment (same row)
    # Low variance in X means vertical alignment (same column)
    is_horizontal = var_y < tolerance ** 2
    is_vertical = var_x < tolerance ** 2
    
    return AlignmentInfo(
        is_aligned_horizontal=is_horizontal,
        is_aligned_vertical=is_vertical,
        horizontal_variance=var_y,
        vertical_variance=var_x
    )


def detect_spacing(
    elements: List[DetectedElement],
    tolerance: float = 10.0
) -> SpacingInfo:
    """
    Detect spacing between consecutive elements.
    
    Args:
        elements: List of detected elements (should be sorted)
        tolerance: Tolerance for "equal" spacing
        
    Returns:
        SpacingInfo with spacing details
    """
    if len(elements) < 2:
        return SpacingInfo([], True, 0)
    
    # Sort by x position for horizontal spacing
    sorted_elements = sorted(elements, key=lambda e: e.bbox.cx)
    
    spacings = []
    for i in range(len(sorted_elements) - 1):
        e1 = sorted_elements[i]
        e2 = sorted_elements[i + 1]
        
        # Space between right edge of e1 and left edge of e2
        space = e2.bbox.x - (e1.bbox.x + e1.bbox.width)
        spacings.append(max(0, space))
    
    variance = np.var(spacings) if spacings else 0
    is_equal = variance < tolerance ** 2
    
    return SpacingInfo(
        spacings=spacings,
        is_equal=is_equal,
        variance=variance
    )


def detect_centering(
    elements: List[DetectedElement],
    canvas_size: Tuple[int, int],
    tolerance: float = 5.0
) -> bool:
    """
    Detect if elements are centered in the canvas.
    
    Args:
        elements: List of detected elements
        canvas_size: (width, height) of canvas
        tolerance: Pixel tolerance
        
    Returns:
        True if centered
    """
    if not elements:
        return True
    
    canvas_cx = canvas_size[0] // 2
    canvas_cy = canvas_size[1] // 2
    
    # Calculate center of all elements
    all_x = [e.bbox.cx for e in elements]
    all_y = [e.bbox.cy for e in elements]
    
    group_cx = np.mean(all_x)
    group_cy = np.mean(all_y)
    
    return (abs(group_cx - canvas_cx) < tolerance and 
            abs(group_cy - canvas_cy) < tolerance)


# =============================================================================
# HIGH-LEVEL DETECTION API
# =============================================================================

class VisualDetector:
    """
    High-level API for visual detection.
    
    Combines all detection functions into a unified interface.
    """
    
    def __init__(
        self,
        canvas_size: Tuple[int, int] = (64, 64),
        background_color: Tuple[int, int, int] = (50, 50, 55),
        min_element_area: int = 20
    ):
        self.canvas_size = canvas_size
        self.background_color = background_color
        self.min_element_area = min_element_area
    
    def detect_all(self, obs: np.ndarray) -> List[DetectedElement]:
        """Detect all elements in observation."""
        elements = []
        elements.extend(detect_rectangles(obs, self.min_element_area, self.background_color))
        # Could add circles, etc.
        return elements
    
    def detect_by_type(
        self, 
        obs: np.ndarray, 
        element_type: ElementType
    ) -> List[DetectedElement]:
        """Detect elements of a specific type."""
        if element_type == ElementType.RECTANGLE:
            return detect_rectangles(obs, self.min_element_area, self.background_color)
        elif element_type == ElementType.CIRCLE:
            return detect_circles(obs)
        else:
            # For now, treat most types as rectangles
            return detect_rectangles(obs, self.min_element_area, self.background_color)
    
    def detect_by_color(
        self, 
        obs: np.ndarray, 
        color: Color
    ) -> List[DetectedElement]:
        """Detect elements of a specific color."""
        boxes = detect_color_regions(obs, color, min_area=self.min_element_area)
        return [
            DetectedElement(type=ElementType.RECTANGLE, bbox=box, color=color)
            for box in boxes
        ]
    
    def count_elements(self, obs: np.ndarray) -> int:
        """Count total detected elements."""
        return len(self.detect_all(obs))
    
    def count_by_color(self, obs: np.ndarray, color: Color) -> int:
        """Count elements of a specific color."""
        return len(self.detect_by_color(obs, color))
    
    def check_alignment(
        self, 
        obs: np.ndarray, 
        horizontal: bool = True
    ) -> bool:
        """Check if elements are aligned."""
        elements = self.detect_all(obs)
        alignment = detect_alignment(elements)
        return alignment.is_aligned_horizontal if horizontal else alignment.is_aligned_vertical
    
    def check_centering(self, obs: np.ndarray) -> bool:
        """Check if elements are centered."""
        elements = self.detect_all(obs)
        return detect_centering(elements, self.canvas_size)
    
    def check_equal_spacing(self, obs: np.ndarray) -> bool:
        """Check if elements have equal spacing."""
        elements = self.detect_all(obs)
        spacing = detect_spacing(elements)
        return spacing.is_equal
    
    def get_state_summary(self, obs: np.ndarray) -> dict:
        """Get a summary of the visual state."""
        elements = self.detect_all(obs)
        alignment = detect_alignment(elements)
        spacing = detect_spacing(elements)
        centered = detect_centering(elements, self.canvas_size)
        
        # Count by color
        color_counts = {}
        for color in Color:
            count = self.count_by_color(obs, color)
            if count > 0:
                color_counts[color.value] = count
        
        return {
            "total_elements": len(elements),
            "color_counts": color_counts,
            "aligned_horizontal": alignment.is_aligned_horizontal,
            "aligned_vertical": alignment.is_aligned_vertical,
            "centered": centered,
            "equal_spacing": spacing.is_equal,
            "mean_spacing": spacing.mean_spacing,
        }


if __name__ == "__main__":
    # Test with dummy image
    print("Testing Visual Detectors...")
    
    # Create test image with colored rectangles
    obs = np.ones((64, 64, 3), dtype=np.uint8) * 50  # Gray background
    
    # Add blue rectangle
    obs[10:22, 10:22] = [60, 120, 220]
    
    # Add red rectangle
    obs[10:22, 30:42] = [220, 60, 60]
    
    # Add green rectangle
    obs[10:22, 50:62] = [60, 180, 60]
    
    detector = VisualDetector()
    
    # Test detection
    elements = detector.detect_all(obs)
    print(f"Detected {len(elements)} elements")
    
    for elem in elements:
        print(f"  - {elem.type.value} at {elem.bbox.center}, color={elem.color}")
    
    # Test alignment
    alignment = detect_alignment(elements)
    print(f"Alignment: {alignment}")
    
    # Test spacing
    spacing = detect_spacing(elements)
    print(f"Spacing: equal={spacing.is_equal}, mean={spacing.mean_spacing:.1f}")
    
    # Test color detection
    blue_count = detector.count_by_color(obs, Color.BLUE)
    print(f"Blue elements: {blue_count}")
    
    # Test state summary
    summary = detector.get_state_summary(obs)
    print(f"State summary: {summary}")
    
    print("\nVisual Detectors test passed!")
