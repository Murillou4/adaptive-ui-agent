"""
YOLO-World Backend for Open-Vocabulary Object Detection

Uses YOLO-World to detect ANY visual element by text description.
This is the core of universal detection - can find "button", "health bar",
"inventory slot", or any other UI element without training.

Key Features:
- Open vocabulary: detect anything by text prompt
- Fast inference: suitable for real-time RL
- Confidence-based filtering
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

logger = logging.getLogger(__name__)


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
        """Get center point."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        """Get area in pixels."""
        return self.width * self.height
    
    @property
    def x2(self) -> int:
        """Right edge."""
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        """Bottom edge."""
        return self.y + self.height
    
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Convert to (x1, y1, x2, y2) format."""
        return (self.x, self.y, self.x2, self.y2)
    
    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> "BoundingBox":
        """Create from (x1, y1, x2, y2) format."""
        return cls(x=int(x1), y=int(y1), width=int(x2 - x1), height=int(y2 - y1))


@dataclass
class Detection:
    """A detected visual element."""
    class_name: str
    bbox: BoundingBox
    confidence: float
    class_id: int = -1
    metadata: dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"Detection({self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox})"


# =============================================================================
# YOLO-WORLD BACKEND
# =============================================================================

class YOLOWorldBackend:
    """
    Open-vocabulary object detection using YOLO-World.
    
    YOLO-World can detect ANY object by text description, making it
    perfect for universal UI detection.
    
    Example:
        backend = YOLOWorldBackend()
        detections = backend.detect(frame, ["button", "text field", "close icon"])
        for det in detections:
            print(f"Found {det.class_name} at {det.bbox.center}")
    """
    
    # Common UI element classes for quick access
    UI_CLASSES = [
        "button", "text", "icon", "menu", "checkbox", "radio button",
        "slider", "progress bar", "input field", "dropdown", "tab",
        "window", "dialog", "toolbar", "scrollbar", "link"
    ]
    
    # Game element classes
    GAME_CLASSES = [
        "health bar", "mana bar", "inventory", "minimap", "character",
        "enemy", "item", "weapon", "armor", "cursor", "crosshair"
    ]
    
    def __init__(
        self,
        model_size: str = "s",
        device: str = "auto",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Initialize YOLO-World backend.
        
        Args:
            model_size: Model size - "s" (small), "m" (medium), "l" (large)
            device: Device to use - "auto", "cpu", "cuda", "mps"
            confidence_threshold: Minimum confidence to report detection
            iou_threshold: IoU threshold for NMS
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        self._model: Optional[YOLO] = None
        self._current_classes: List[str] = []
        
        if not YOLO_AVAILABLE:
            logger.warning(
                "ultralytics not installed. YOLO-World backend will not work. "
                "Install with: pip install ultralytics"
            )
    
    def _ensure_model_loaded(self) -> bool:
        """Lazy load the model."""
        if self._model is not None:
            return True
        
        if not YOLO_AVAILABLE:
            logger.error("Cannot load model: ultralytics not installed")
            return False
        
        try:
            # Load YOLO-World model
            model_name = f"yolov8{self.model_size}-worldv2"
            logger.info(f"Loading YOLO-World model: {model_name}")
            
            self._model = YOLO(model_name)
            
            # Set device
            if self.device != "auto":
                self._model.to(self.device)
            
            logger.info(f"YOLO-World model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load YOLO-World model: {e}")
            return False
    
    def set_classes(self, classes: List[str]) -> None:
        """
        Set the vocabulary of classes to detect.
        
        Args:
            classes: List of class names to detect
        """
        if not self._ensure_model_loaded():
            return
        
        if classes != self._current_classes:
            self._model.set_classes(classes)
            self._current_classes = classes.copy()
            logger.debug(f"Set detection classes: {classes}")
    
    def detect(
        self,
        frame: np.ndarray,
        classes: Optional[List[str]] = None,
        confidence_threshold: Optional[float] = None
    ) -> List[Detection]:
        """
        Detect elements in frame.
        
        Args:
            frame: RGB image as numpy array (H, W, 3)
            classes: Classes to detect. If None, uses previously set classes
            confidence_threshold: Override default confidence threshold
            
        Returns:
            List of Detection objects
        """
        if not self._ensure_model_loaded():
            return []
        
        # Set classes if provided
        if classes is not None:
            self.set_classes(classes)
        
        if not self._current_classes:
            logger.warning("No classes set for detection")
            return []
        
        # Run inference
        conf = confidence_threshold or self.confidence_threshold
        
        try:
            results = self._model.predict(
                frame,
                conf=conf,
                iou=self.iou_threshold,
                verbose=False
            )
        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            return []
        
        # Parse results
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get box coordinates (xyxy format)
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                # Get confidence and class
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                # Get class name
                if cls_id < len(self._current_classes):
                    class_name = self._current_classes[cls_id]
                else:
                    class_name = f"class_{cls_id}"
                
                # Create detection
                detection = Detection(
                    class_name=class_name,
                    bbox=BoundingBox.from_xyxy(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id
                )
                
                detections.append(detection)
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        logger.debug(f"Detected {len(detections)} elements")
        return detections
    
    def detect_ui(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect common UI elements.
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            List of UI element detections
        """
        return self.detect(frame, classes=self.UI_CLASSES)
    
    def detect_game(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect common game elements.
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            List of game element detections
        """
        return self.detect(frame, classes=self.GAME_CLASSES)
    
    def find(
        self,
        frame: np.ndarray,
        query: str,
        max_results: int = 10
    ) -> List[Detection]:
        """
        Find elements matching a query.
        
        Args:
            frame: RGB image as numpy array
            query: What to find (e.g., "blue button", "submit text")
            max_results: Maximum number of results
            
        Returns:
            List of matching detections
        """
        # Parse query into classes
        # For now, just use the query as-is
        # Future: use LLM to expand query into multiple classes
        classes = [query]
        
        detections = self.detect(frame, classes=classes)
        return detections[:max_results]
    
    def is_available(self) -> bool:
        """Check if YOLO-World is available."""
        return YOLO_AVAILABLE and self._ensure_model_loaded()
    
    @property
    def model_info(self) -> dict:
        """Get model information."""
        return {
            "model_size": self.model_size,
            "available": YOLO_AVAILABLE,
            "loaded": self._model is not None,
            "current_classes": self._current_classes,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_yolo_backend(
    model_size: str = "s",
    device: str = "auto"
) -> YOLOWorldBackend:
    """
    Factory function to create YOLO-World backend.
    
    Args:
        model_size: "s", "m", or "l"
        device: "auto", "cpu", "cuda", or "mps"
        
    Returns:
        Configured YOLOWorldBackend
    """
    return YOLOWorldBackend(model_size=model_size, device=device)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("YOLO-World Backend Test")
    print("=" * 50)
    
    backend = YOLOWorldBackend(model_size="s")
    
    print(f"Available: {YOLO_AVAILABLE}")
    print(f"Model info: {backend.model_info}")
    
    if YOLO_AVAILABLE:
        # Create a simple test image
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[100:200, 100:300] = [100, 100, 255]  # Red rectangle (button-like)
        
        detections = backend.detect(test_frame, classes=["button", "rectangle"])
        print(f"Detections: {detections}")
