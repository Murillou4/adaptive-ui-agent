"""
Tests for Universal Vision System
"""

import pytest
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.yolo_backend import YOLOWorldBackend, Detection, BoundingBox
from vision.ocr_backend import OCRBackend, TextRegion
from vision.universal_detector import UniversalVisionDetector
from vision.state_extractor import StateExtractor, VisualState


class TestDataTypes:
    """Test data structures."""
    
    def test_bounding_box(self):
        bbox = BoundingBox(x=10, y=20, width=50, height=30)
        assert bbox.center == (35, 35)
        assert bbox.area == 1500
        assert bbox.x2 == 60
        assert bbox.y2 == 50
        assert bbox.to_xyxy() == (10, 20, 60, 50)
        
        bbox2 = BoundingBox.from_xyxy(10, 20, 60, 50)
        assert bbox == bbox2

    def test_text_region(self):
        region = TextRegion(text="Hello", bbox=(10, 20, 50, 30), confidence=0.9)
        assert region.center == (35, 35)
        assert region.contains_text("hell")
        assert not region.contains_text("Bye")


class TestUniversalDetector:
    """Test UniversalVisionDetector."""
    
    def test_initialization(self):
        # Initialize without loading heavy models (device="cpu")
        # Note: In a real test environment, we'd mock the backends
        detector = UniversalVisionDetector(device="cpu")
        assert detector.yolo is not None
        assert detector.ocr is not None
        
    def test_backend_routing(self):
        detector = UniversalVisionDetector(device="cpu")
        
        # We need to mock the backends to test routing without actual inference
        class MockYOLO:
            def find(self, frame, query):
                return [Detection("mock_obj", BoundingBox(0,0,10,10), 0.9)]
                
        class MockOCR:
            def find_all_text(self, frame, query):
                return [TextRegion("mock_text", (0,0,10,10), 0.9)]
        
        detector.yolo = MockYOLO()
        detector.ocr = MockOCR()
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test text query routing
        results = detector.detect(frame, "text:Submit")
        assert len(results) == 1
        assert results[0].class_name == "text:Submit"
        
        # Test object query routing
        results = detector.detect(frame, "button")
        assert len(results) == 1
        assert results[0].class_name == "mock_obj"


class TestStateExtractor:
    """Test StateExtractor."""
    
    def test_enrichment(self):
        detector = UniversalVisionDetector(device="cpu")
        extractor = StateExtractor(detector)
        
        # Create mock detections and text
        button_bbox = BoundingBox(100, 100, 100, 50)  # 100,100 to 200,150
        button = Detection("button", button_bbox, 0.9)
        
        text_bbox = (120, 110, 60, 30)  # Center: 150, 125 (Inside button)
        text = TextRegion("Submit", text_bbox, 0.99)
        
        detections = [button]
        text_regions = [text]
        
        extractor._enrich_detections_with_text(detections, text_regions)
        
        assert "text" in button.metadata
        assert button.metadata["text"] == "Submit"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
