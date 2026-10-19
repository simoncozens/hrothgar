"""Instance-membership model: "does this glyph belong to this font instance?".

Given a uniform scale-and-pad glyph image, ``SetMembershipModel`` produces an
L2-normalized embedding; an instance's reference is the mean of its evidence
glyphs' embeddings, and membership is the cosine between a query glyph and the
instance reference.  It is used to evaluate the *style* part of glyph
generation (content is judged separately).
"""

from hrothgar.set_membership.calibration import (
    CalibrationScores,
    ConfidenceScorer,
    PlattCalibrator,
    calibration_report,
    collect_scores,
)
from hrothgar.set_membership.config import SetMembershipConfig
from hrothgar.set_membership.dataset import SetMembershipDatasetMaker
from hrothgar.set_membership.losses import glyph_contrastive_loss
from hrothgar.set_membership.model import SetMembershipModel
from hrothgar.set_membership.train import SetMembershipTrainingLoop

__all__ = [
    "CalibrationScores",
    "ConfidenceScorer",
    "PlattCalibrator",
    "SetMembershipConfig",
    "SetMembershipDatasetMaker",
    "SetMembershipModel",
    "SetMembershipTrainingLoop",
    "calibration_report",
    "collect_scores",
    "glyph_contrastive_loss",
]
