"""Content detector: an open-set, per-codepoint "is this glyph what it claims?" model.

One learned anchor ("slot") per codepoint; a glyph is scored one-vs-rest against
every anchor so that a glyph which is none of them can score low everywhere.
See :mod:`hrothgar.content_detector.model` for the architecture and
:mod:`hrothgar.content_detector.dataset` for the positive/negative curation.
"""

from hrothgar.content_detector.config import ContentDetectorConfig
from hrothgar.content_detector.model import ContentDetector

__all__ = ["ContentDetector", "ContentDetectorConfig"]
