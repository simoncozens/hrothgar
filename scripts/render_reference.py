import argparse

import numpy as np
import tqdm

from hrothgar.ar.config import ARModelConfig
from hrothgar.googlefonts import StandaloneFont


def main():
    parser = argparse.ArgumentParser(
        description="Export reference glyph array from AR config"
    )
    parser.add_argument("config_path", type=str, help="Path to the AR config JSON file")
    parser.add_argument(
        "reference_font_path", type=str, help="Path to the reference font file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="reference_glyphs.npz",
        help="Output Numpy file for reference glyphs",
    )
    args = parser.parse_args()

    config = ARModelConfig.from_sidecar(
        args.config_path.replace(".conf.json", "")
    )  # Extension added by .from_sidecar
    assert (
        config.target_codepoints is not None
    ), "Target codepoints must be defined in the AR config."
    reference_glyphs = config.target_codepoints
    output = []
    for cp in tqdm.tqdm(reference_glyphs):
        glyph = StandaloneFont(args.reference_font_path).render(cp, config.image_size)
        if glyph is not None:
            output.append(glyph)
        else:
            print(f"Warning: Glyph for codepoint {cp} not found in reference font.")
    output = np.array(output)
    print(f"Saving on {args.output}...")
    np.savez_compressed(args.output, reference_glyphs=output)


if __name__ == "__main__":
    main()
