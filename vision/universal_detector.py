"""
Universal Vision Detector

Combines multiple detection backends (YOLO-World, OCR) into a single
unified API for detecting elements in any visual interface.

This is the main entry point for the vision system.
"""

import numpy as np
from typing import List, Optional, Union, Dict, Any
import logging
from dataclasses import dataclass

from vision.yolo_backend import YOLOWorldBackend, Detection, create_yolo_backend
from vision.ocr_backend import OCRBackend, TextRegion, create_ocr_backend

logger = logging.getLogger(__name__)


class UniversalVisionDetector:
    """
    Detects elements in any visual interface using multiple backends.
    
    Unified API for:
    - Object detection (buttons, icons, game elements) via YOLO-World
    - Text extraction (labels, values, messages) via EasyOCR
    """
    
    def __init__(
        self,
        yolo_model_size: str = "s",
        ocr_languages: List[str] = None,
        device: str = "auto"
    ):
        """
        Initialize detector with all backends.
        
        Args:
            yolo_model_size: YOLO model size ("s", "m", "l")
            ocr_languages: Languages for OCR (default: ["en"])
            device: Device to use ("auto", "cpu", "cuda")
        """
        self.device = device
        
        # Initialize backends
        self.yolo = create_yolo_backend(model_size=yolo_model_size, device=device)
        self.ocr = create_ocr_backend(languages=ocr_languages, gpu=(device != "cpu"))
        
        logger.info("Universal Vision Detector initialized")
    
    def detect(
        self,
        frame: np.ndarray,
        query: str
    ) -> List[Detection]:
        """
        Detect elements matching query in frame.
        
        Intelligently routes the query to the appropriate backend.
        
        Args:
            frame: RGB image as numpy array
            query: What to find (e.g., "blue button", "text: Submit")
            
        Returns:
            List of detections
        """
        # Check if query is text-specific (starts with "text:")
        if query.lower().startswith("text:"):
            text_query = query[5:].strip()
            return self._detect_text(frame, text_query)
        
        # Otherwise use YOLO for general object detection
        return self.yolo.find(frame, query)
    
    def detect_all(
        self,
        frame: np.ndarray,
        classes: List[str]
    ) -> List[Detection]:
        """
        Detect multiple classes at once.
        
        Args:
            frame: RGB image as numpy array
            classes: List of class names to detect
            
        Returns:
            List of all detections
        """
        return self.yolo.detect(frame, classes=classes)
    
    def extract_text(self, frame: np.ndarray) -> List[TextRegion]:
        """
        Extract all text from frame.
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            List of text regions
        """
        return self.ocr.extract_text(frame)
    
    def _detect_text(self, frame: np.ndarray, text: str) -> List[Detection]:
        """Helper to detect text and convert to Detection objects."""
        regions = self.ocr.find_all_text(frame, text)
        
        detections = []
        for region in regions:
            # Convert TextRegion to Detection
            # Import BoundingBox here to avoid circular dependencies if needed
            from vision.yolo_backend import BoundingBox
            
            x, y, w, h = region.bbox
            
            detections.append(Detection(
                class_name=f"text:{text}",
                bbox=BoundingBox(x, y, w, h),
                confidence=region.confidence,
                metadata={"text": region.text}
            ))
            
        return detections
    
    @property
    def status(self) -> Dict[str, Any]:
        """Get status of backends."""
        return {
            "yolo": self.yolo.model_info,
            "ocr": self.ocr.info
        }
