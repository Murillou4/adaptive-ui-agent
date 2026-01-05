"""
OCR Backend for Text Extraction

Uses EasyOCR to extract and locate text in any visual interface.
Critical for reading UI labels, game text, error messages, etc.

Key Features:
- Multi-language support
- Bounding box locations for each text region
- Text search by query
- Confidence-based filtering
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import logging

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TextRegion:
    """A detected text region."""
    text: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    language: str = "unknown"
    
    @property
    def x(self) -> int:
        return self.bbox[0]
    
    @property
    def y(self) -> int:
        return self.bbox[1]
    
    @property
    def width(self) -> int:
        return self.bbox[2]
    
    @property
    def height(self) -> int:
        return self.bbox[3]
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def x2(self) -> int:
        """Right edge."""
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        """Bottom edge."""
        return self.y + self.height
    
    def contains_text(self, query: str, case_sensitive: bool = False) -> bool:
        """Check if this region contains the query text."""
        if case_sensitive:
            return query in self.text
        return query.lower() in self.text.lower()
    
    def __repr__(self) -> str:
        return f'TextRegion("{self.text}", conf={self.confidence:.2f})'


# =============================================================================
# OCR BACKEND
# =============================================================================

class OCRBackend:
    """
    OCR backend using EasyOCR for text extraction.
    
    Example:
        ocr = OCRBackend(languages=["en", "pt"])
        
        # Extract all text from frame
        texts = ocr.extract_text(frame)
        for t in texts:
            print(f"Found: '{t.text}' at {t.center}")
        
        # Find specific text
        bbox = ocr.find_text(frame, "Submit")
        if bbox:
            print(f"Submit button at: {bbox}")
    """
    
    def __init__(
        self,
        languages: List[str] = None,
        gpu: bool = True,
        confidence_threshold: float = 0.3
    ):
        """
        Initialize OCR backend.
        
        Args:
            languages: List of language codes (e.g., ["en", "pt"])
            gpu: Whether to use GPU acceleration
            confidence_threshold: Minimum confidence to report text
        """
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.confidence_threshold = confidence_threshold
        
        self._reader: Optional[easyocr.Reader] = None
        
        if not EASYOCR_AVAILABLE:
            logger.warning(
                "easyocr not installed. OCR backend will not work. "
                "Install with: pip install easyocr"
            )
    
    def _ensure_reader_loaded(self) -> bool:
        """Lazy load the OCR reader."""
        if self._reader is not None:
            return True
        
        if not EASYOCR_AVAILABLE:
            logger.error("Cannot load reader: easyocr not installed")
            return False
        
        try:
            logger.info(f"Loading EasyOCR with languages: {self.languages}")
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                verbose=False
            )
            logger.info("EasyOCR loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load EasyOCR: {e}")
            return False
    
    def extract_text(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
        detail: int = 1
    ) -> List[TextRegion]:
        """
        Extract all text from frame.
        
        Args:
            frame: RGB image as numpy array (H, W, 3)
            confidence_threshold: Override default threshold
            detail: 0 = text only, 1 = text + bbox + confidence
            
        Returns:
            List of TextRegion objects
        """
        if not self._ensure_reader_loaded():
            return []
        
        conf_threshold = confidence_threshold or self.confidence_threshold
        
        try:
            # Run OCR
            results = self._reader.readtext(frame, detail=detail)
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return []
        
        # Parse results
        text_regions = []
        
        for result in results:
            if detail == 0:
                # Just text, no bbox
                text_regions.append(TextRegion(
                    text=result,
                    bbox=(0, 0, 0, 0),
                    confidence=1.0
                ))
            else:
                # Full result: (bbox, text, confidence)
                bbox_points, text, confidence = result
                
                if confidence < conf_threshold:
                    continue
                
                # Convert bbox from 4-point polygon to (x, y, w, h)
                # EasyOCR returns [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                if len(bbox_points) == 4:
                    x_coords = [p[0] for p in bbox_points]
                    y_coords = [p[1] for p in bbox_points]
                    
                    x = int(min(x_coords))
                    y = int(min(y_coords))
                    width = int(max(x_coords) - x)
                    height = int(max(y_coords) - y)
                    
                    bbox = (x, y, width, height)
                else:
                    bbox = (0, 0, 0, 0)
                
                text_regions.append(TextRegion(
                    text=text,
                    bbox=bbox,
                    confidence=confidence
                ))
        
        # Sort by position (top-to-bottom, left-to-right)
        text_regions.sort(key=lambda t: (t.y, t.x))
        
        logger.debug(f"Extracted {len(text_regions)} text regions")
        return text_regions
    
    def find_text(
        self,
        frame: np.ndarray,
        query: str,
        case_sensitive: bool = False,
        exact_match: bool = False
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Find text matching query and return its bounding box.
        
        Args:
            frame: RGB image as numpy array
            query: Text to search for
            case_sensitive: Whether to match case
            exact_match: Whether to require exact match (vs contains)
            
        Returns:
            Bounding box (x, y, width, height) or None if not found
        """
        text_regions = self.extract_text(frame)
        
        for region in text_regions:
            if exact_match:
                if case_sensitive:
                    match = region.text == query
                else:
                    match = region.text.lower() == query.lower()
            else:
                match = region.contains_text(query, case_sensitive)
            
            if match:
                return region.bbox
        
        return None
    
    def find_all_text(
        self,
        frame: np.ndarray,
        query: str,
        case_sensitive: bool = False
    ) -> List[TextRegion]:
        """
        Find all text regions containing query.
        
        Args:
            frame: RGB image as numpy array
            query: Text to search for
            case_sensitive: Whether to match case
            
        Returns:
            List of matching TextRegion objects
        """
        text_regions = self.extract_text(frame)
        
        matches = [
            region for region in text_regions
            if region.contains_text(query, case_sensitive)
        ]
        
        return matches
    
    def get_all_text(self, frame: np.ndarray) -> str:
        """
        Extract all text as a single string.
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            Concatenated text from all regions
        """
        text_regions = self.extract_text(frame)
        return " ".join(region.text for region in text_regions)
    
    def is_available(self) -> bool:
        """Check if OCR is available."""
        return EASYOCR_AVAILABLE and self._ensure_reader_loaded()
    
    @property
    def info(self) -> dict:
        """Get backend information."""
        return {
            "available": EASYOCR_AVAILABLE,
            "loaded": self._reader is not None,
            "languages": self.languages,
            "gpu": self.gpu,
            "confidence_threshold": self.confidence_threshold
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_ocr_backend(
    languages: List[str] = None,
    gpu: bool = True
) -> OCRBackend:
    """
    Factory function to create OCR backend.
    
    Args:
        languages: Language codes (default: ["en"])
        gpu: Whether to use GPU
        
    Returns:
        Configured OCRBackend
    """
    return OCRBackend(languages=languages or ["en"], gpu=gpu)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("OCR Backend Test")
    print("=" * 50)
    
    backend = OCRBackend(languages=["en"])
    
    print(f"Available: {EASYOCR_AVAILABLE}")
    print(f"Backend info: {backend.info}")
    
    if EASYOCR_AVAILABLE:
        import cv2
        
        # Create a simple test image with text
        test_frame = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(
            test_frame, "Hello World", (50, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
        
        texts = backend.extract_text(test_frame)
        print(f"Extracted texts: {texts}")
        
        # Find specific text
        bbox = backend.find_text(test_frame, "Hello")
        print(f"Found 'Hello' at: {bbox}")
