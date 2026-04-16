from hrothgar.ar.model import ARModel, ARModelConfig
from hrothgar.googlefonts import GoogleFonts
from hrothgar.ar.dataset import ARPhase1DatasetMaker  # Just for collation
import torch
import random

samples = "adhesionADHESION"
target = "₹"

model = ARModel(ARModelConfig())
model.load("models/ar_visual_model.pth", device=torch.device("mps"))

dataset = GoogleFonts("/Users/simon/others-repos/fonts")
maker = ARPhase1DatasetMaker(
    repo_url="/Users/simon/others-repos/fonts",
    batch_size=5,
    target_codepoints=[ord(target)],
    common_style_codepoints=[ord(c) for c in samples],
)
# Find fonts *without* the target char.
candidate_fonts = [f for f in dataset.fonts if not f.has_codepoint(ord(target))]
samples = random.sample(candidate_fonts, 5)
batch = [{"char": ord(target), "font": f} for f in samples]
collated = maker.collate_fn(batch)
with torch.no_grad():
    out = model.generate(
        content_images=collated["content_rendering"],
        style_reference_images=collated["style_renderings"],
    )
# Make a grid of the first five style chars + output for each font
import matplotlib.pyplot as plt

for i in range(5):
    fig, axs = plt.subplots(1, 6, figsize=(15, 3))
    for j in range(5):
        style_rendering = collated["style_renderings"][i, j].cpu()
        # (3, 128, 128) -> (128, 128, 3)
        style_rendering = style_rendering.permute(1, 2, 0)

        axs[j].imshow(style_rendering, cmap="gray")
        axs[j].set_title(f"Style char: {chr(collated['style_chars'][i, j].item())}")
        axs[j].axis("off")
    reconstructed = out.reconstructed_images[i].cpu()
    reconstructed = reconstructed.permute(1, 2, 0)
    axs[5].imshow(reconstructed, cmap="gray")
    axs[5].set_title("Predicted target")
    axs[5].axis("off")
    plt.show()
