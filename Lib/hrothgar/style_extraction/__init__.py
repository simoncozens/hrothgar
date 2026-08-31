"""Style extraction — a *rich, complete* font-style representation.

Unlike :mod:`hrothgar.style_embedding` (a compact contrastive embedding for
similarity/recommendation), this module learns an autoencoding style
representation whose contract is **reconstructability**: the embedding must be
decodable back into high-fidelity glyphs.  Reconstruction (L1 + glyphloss +
LPIPS, optionally adversarial) is both the training objective and the
acceptance signal.
"""

from hrothgar.style_extraction.config import (
    LossSchedule,
    StyleExtractionConfig,
    StyleExtractionLossWeights,
    StyleExtractionV2Config,
    StyleExtractionV3Config,
)
from hrothgar.style_extraction.dataset import StyleExtractionDatasetMaker
from hrothgar.style_extraction.losses import (
    NLayerDiscriminator,
    adversarial_discriminator_loss,
    adversarial_generator_loss,
    ink_coverage_loss,
    position_dependence_loss,
    reconstruction_loss,
    style_contrastive_loss,
    style_token_diversity_loss,
)
from hrothgar.style_extraction.model import StyleExtractionModel
from hrothgar.style_extraction.model_v2 import StyleExtractionModelV2
from hrothgar.style_extraction.model_v3 import StyleExtractionModelV3
from hrothgar.style_extraction.train import StyleExtractionTrainingLoop

__all__ = [
    "StyleExtractionModel",
    "StyleExtractionModelV2",
    "StyleExtractionModelV3",
    "StyleExtractionConfig",
    "StyleExtractionV2Config",
    "StyleExtractionV3Config",
    "LossSchedule",
    "StyleExtractionLossWeights",
    "StyleExtractionTrainingLoop",
    "StyleExtractionDatasetMaker",
    "NLayerDiscriminator",
    "reconstruction_loss",
    "ink_coverage_loss",
    "style_contrastive_loss",
    "style_token_diversity_loss",
    "position_dependence_loss",
    "adversarial_generator_loss",
    "adversarial_discriminator_loss",
    "load_model",
]


def load_model(path, device, strict: bool = False):
    """Load a style-extraction checkpoint (v2 or v3) and return ``(model, config)``.

    Detects the architecture from the sidecar JSON (v3 adds the
    ``decoder_self_attn_layers`` field) and constructs the matching model, so the
    probe scripts don't need to know which version produced a checkpoint.
    """
    import json
    from pathlib import Path

    config_path = Path(str(path)).with_suffix(".conf.json")
    if not config_path.exists():
        config_path = Path(str(path).replace(".pth", ".conf.json"))
    with config_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if "decoder_self_attn_layers" in data:
        config = StyleExtractionV3Config.from_sidecar(path)
        model = StyleExtractionModelV3(config).to(device)
    else:
        config = StyleExtractionV2Config.from_sidecar(path)
        model = StyleExtractionModelV2(config).to(device)

    model.load(path, device=device, strict=strict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, config
