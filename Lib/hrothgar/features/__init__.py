"""Feature-recogniser family: glyph-image -> closed-set judgement + "unrecognisable".

Each feature (a-storey, e-crossbar, …) is a :class:`~hrothgar.features.base.FeatureRecognizer`
subclass in its own module, sharing the one-vs-rest + threshold API.
"""

from hrothgar.features.a_shape import AShapeConfig, AShapeRecognizer
from hrothgar.features.a_storey import AStoreyConfig, AStoreyRecognizer
from hrothgar.features.base import FeatureRecognizer, FeatureRecognizerConfig
from hrothgar.features.display import DisplayConfig, DisplayRecognizer
from hrothgar.features.e_crossbar import ECrossbarConfig, ECrossbarRecognizer
from hrothgar.features.finials import FinialsConfig, FinialsRecognizer
from hrothgar.features.has_serifs import HasSerifsConfig, HasSerifsRecognizer
from hrothgar.features.serifs import SerifsConfig, SerifsRecognizer
from hrothgar.features.terminals import TerminalsConfig, TerminalsRecognizer

__all__ = [
    "FeatureRecognizer",
    "FeatureRecognizerConfig",
    "AShapeConfig",
    "AShapeRecognizer",
    "AStoreyConfig",
    "AStoreyRecognizer",
    "ECrossbarConfig",
    "ECrossbarRecognizer",
    "FinialsConfig",
    "FinialsRecognizer",
    "HasSerifsConfig",
    "HasSerifsRecognizer",
    "SerifsConfig",
    "SerifsRecognizer",
    "TerminalsConfig",
    "TerminalsRecognizer",
    "DisplayConfig",
    "DisplayRecognizer",
]
