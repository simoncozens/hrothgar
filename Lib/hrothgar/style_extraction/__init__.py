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
    StyleExtractionV4Config,
    StyleExtractionV5Config,
)
from hrothgar.style_extraction.dataset import StyleExtractionDatasetMaker
from hrothgar.style_extraction.losses import (
    NLayerDiscriminator,
    adversarial_discriminator_loss,
    adversarial_generator_loss,
    attention_health,
    ink_coverage_loss,
    position_dependence_loss,
    reconstruction_loss,
    style_contrastive_loss,
    style_token_diversity_loss,
)
from hrothgar.style_extraction.model import StyleExtractionModel
from hrothgar.style_extraction.model_v2 import StyleExtractionModelV2
from hrothgar.style_extraction.model_v3 import StyleExtractionModelV3
from hrothgar.style_extraction.model_v4 import StyleExtractionModelV4
from hrothgar.style_extraction.model_v5 import StyleExtractionModelV5
from hrothgar.style_extraction.train import StyleExtractionTrainingLoop

__all__ = [
    "StyleExtractionModel",
    "StyleExtractionModelV2",
    "StyleExtractionModelV3",
    "StyleExtractionModelV4",
    "StyleExtractionModelV5",
    "StyleExtractionConfig",
    "StyleExtractionV2Config",
    "StyleExtractionV3Config",
    "StyleExtractionV4Config",
    "StyleExtractionV5Config",
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
    "attention_health",
    "adversarial_generator_loss",
    "adversarial_discriminator_loss",
    "load_model",
]


def load_model(path, device, strict: bool = False):
    """Load a style-extraction checkpoint (v2/v3/v4/v5) and return ``(model, config)``.

    Detects the architecture from the sidecar JSON (v4 adds ``two_stage``, v5 adds
    ``slot_basis_size``, v3 adds ``decoder_self_attn_layers``) and constructs the
    matching model.
    """
    import json
    from pathlib import Path

    config_path = Path(str(path)).with_suffix(".conf.json")
    if not config_path.exists():
        config_path = Path(str(path).replace(".pth", ".conf.json"))
    with config_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if data.get("two_stage"):
        config = StyleExtractionV4Config.from_sidecar(path)
        model = StyleExtractionModelV4(config).to(device)
    elif "slot_basis_size" in data:
        config = StyleExtractionV5Config.from_sidecar(path)
        model = StyleExtractionModelV5(config).to(device)
    elif "decoder_self_attn_layers" in data:
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
