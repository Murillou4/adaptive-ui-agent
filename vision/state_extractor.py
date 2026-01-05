"""
State Extractor

Converts raw detections and text into a structured VisualState object.
This state serves as the observation for the RL agent and the Planner.

It aggregates:
- Detected UI elements (buttons, inputs, etc.)
- Extracted text
- Spatial relationships
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from vision.universal_detector import UniversalVisionDetector, Detection
from vision.ocr_backend import TextRegion
from vision.yolo_backend import BoundingBox


@dataclass
class VisualState:
    """Structured state extracted from any visual interface."""
    detections: List[Detection]
    text_regions: List[TextRegion]
    raw_frame: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_elements_by_class(self, class_name: str) -> List[Detection]:
        """Get all elements of a specific class."""
        return [d for d in self.detections if d.class_name == class_name]
    
    def get_text_content(self) -> str:
        """Get all text content joined by spaces."""
        return " ".join([t.text for t in self.text_regions])
    
    def find_element(self, query: str) -> Optional[Detection]:
        """Find element by class name or text content."""
        # Try checking class name
        for d in self.detections:
            if d.class_name == query:
                return d
        
        # Try checking text content within detections (if metadata has it)
        for d in self.detections:
            if d.metadata.get("text") == query:
                return d
                
        return None


class StateExtractor:
    """Extracts structured state from any visual interface."""
    
    def __init__(self, detector: UniversalVisionDetector):
        self.detector = detector
        
        # Default classes to always detect for "general" state
        self.default_classes = [
            "button", "input field", "text", "icon", 
            "window", "menu", "cursor"
        ]
    
    def extract(
        self, 
        frame: np.ndarray,
        query_classes: Optional[List[str]] = None
    ) -> VisualState:
        """
        Run full state extraction pipeline.
        
        Args:
            frame: RGB image as numpy array
            query_classes: Specific classes to look for (extends defaults)
            
        Returns:
            VisualState object
        """
        classes_to_detect = list(self.default_classes)
        if query_classes:
            classes_to_detect.extend(query_classes)
            # Remove duplicates
            classes_to_detect = list(set(classes_to_detect))
            
        # 1. Detect objects (YOLO)
        detections = self.detector.detect_all(frame, classes=classes_to_detect)
        
        # 2. Extract text (OCR)
        text_regions = self.detector.extract_text(frame)
        
        # 3. Correlate text with detections
        self._enrich_detections_with_text(detections, text_regions)
        
        return VisualState(
            detections=detections,
            text_regions=text_regions,
            raw_frame=frame
        )
    
    def _enrich_detections_with_text(
        self, 
        detections: List[Detection], 
        text_regions: List[TextRegion]
    ):
        """
        Associate text with overlapping visual elements.
        e.g., A "button" detection containing "Submit" text.
        """
        for det in detections:
            det_center = det.bbox.center
            
            # Find texts inside or very close to this detection
            associated_texts = []
            
            for text in text_regions:
                # Check if text center is inside detection bbox
                tx, ty = text.center
                if (det.bbox.x <= tx <= det.bbox.x2 and
                    det.bbox.y <= ty <= det.bbox.y2):
                    associated_texts.append(text.text)
            
            if associated_texts:
                det.metadata["text"] = " ".join(associated_texts)
