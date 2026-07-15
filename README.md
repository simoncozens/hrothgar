# HrothGAR

HrothGAR is a model for style transfer of glyph images. It takes a number of style reference images from an existing font, and uses them to generate new glyphs in the same style.

## Architecture

HrothGAR was originally based on the [GAR-Font](https://arxiv.org/html/2601.01593v1) architecture but has been substantially modified for better support of Latin glyphs. The model consists of three main components:

* **G-Tok**, a glyph tokenizer. From the GAR-Font paper, G-Tok is described as "a global-aware tokenizer (G-Tok) that fuses local features with global perception, capturing both fine-grained stroke details and font-level style patterns." HrothGAR's G-Tok follows the same hybrid design: CNN–ViT encoder + vector quantizer + causal ViT–CNN decoder.
* **The glyph generation model**. In GAR-Font, this was an autoregressive
  transformer with causal attention. HrothGAR replaces this with a **MaskGIT**
  bidirectional transformer that uses masked token prediction (BERT-style)
  during training and iterative parallel decoding during inference.
* **A glyph super-resolution upscaler** (new in HrothGAR). A lightweight
  CNN model that upscales low-resolution outputs from the glyph generation model, conditioned on high-resolution style reference glyphs via FiLM. The style conditioning allows the upscaler to correctly resolve terminal shape, sharpness and direction from subpixel information, based on the style of other exemplars from the font. This is not present in the GAR-Font paper and was added to improve the vectorization step.

Other architectural modifications:

* **Explicit codepoint identity conditioning.** GAR-Font infers character identity from a content reference image. In HrothGAR, the Unicode codepoint is passed directly as a categorical input with a learned embedding, concatenated into the conditioning map. This was a necessary change because Latin glyph structure varies so widely across fonts that a reference image is an unreliable source of character identity. (For example, without codepoint identity conditioning, the model may believe that a single-story "a" and a double-story "a" are completely different characters.)
* **Bidirectional (MaskGIT) generation.** The paper's AR transformer predicts tokens left-to-right with causal masking. HrothGAR uses bidirectional attention with masked token prediction, which avoids exposure bias and enables parallel decoding (8–16 iterative steps instead of N sequential).
* **Lookahead auxiliary loss.** Auxiliary heads predict tokens k positions ahead of the current step, providing additional training signal for longer-range sequential structure.
* **Gumbel-softmax perceptual (LPIPS) loss.** To address the many-to-one codebook mapping problem, HrothGAR computes LPIPS perceptual loss on Gumbel-softmax-sampled reconstructions, teaching the model that different code sequences can be equivalent.
* **Health metrics integrated into training.** Linear probes (character  identity + font family), autocorrelation (next-token prediction), and oracle AR (single-font GPT) run inline during tokenizer training and report to TensorBoard.
* Both the tokenizer and the generator use the [glyphloss](https://github.com/simoncozens/glyphloss) library for loss calculation, which is more suitable for glyph images than standard image losses.
* **Structural Enhancement removed**. GAR-Font's Structural Enhancement component was never implemented; this relies on optical character recognition, and one of the motivations of HrothGAR was to generate new or uncommon glyphs, which would not be recognized by an OCR model.
* **Class-balanced style oversampling**. The G-Tok dataset applies weighted oversampling for underrepresented font style classifications - for example, fonts with the "DISPLAY" category in the dataset may be underrepresented, so they are oversampled to ensure that the model sees enough examples of diverse styles during training to prevent the dataset being dominated by text faces.

### Limitations

HrothGAR is currently at proof-of-concept stage. Significantly, *currently Hrothgar's renderer crops descenders!* You won't be able to generate any glyphs which descend below the baseline. This is an intentional choice to improve the consistency of placement and scaling of glyphs within the glyph image, but I recognise it's a significant limitation. Addressing it is conceptually simple (you can do it right now! run `git revert 14da32f`) but it will require much, much longer training, and may require a larger glyph image, subsequent hyperparameter tweaks, and a larger model.

### Further work

The GAR-Font architecture allowed for "Novel Font Adaptation" - low-rank adaptation of the model to a new font with only a few reference glyphs. This is not yet implemented in HrothGAR, but it is planned for future work.

We also plan to experiment with token-corpus fine-tuning of the MaskGIT generator, which will help to condition the generator in the context of a "many-shot" generation (we have a thousand glyphs in a font and we want to generate one more in the same style).

GAR-Font's multimodal textual conditioning stage was originally implemented but is no longer used in HrothGAR after the shift to MaskGIT, although it would be worth experimenting with at a later stage. 

## Installation

HrothGAR is best installed in a new Python virtual environment. The following commands will create a new environment and install the required dependencies:

```bash
$ uv venv hrothgar-venv # (or python -m venv hrothgar-venv)
$ source hrothgar-venv/bin/activate
$ uv pip install . # (or pip install .)
```

## Training

HrothGAR is trained on the Google Fonts dataset. The training process consists of three stages:

* Train G-Tok

```bash
python -m hrothgar.gtok.train --dataset-path <path_to_google/fonts_checkout> --image-size 96
```

(By experiment, 96 is the best compromise between speed and quality for the tokenizer.) Additional options can be viewed using the `--help` flag. One particularly useful one is `--targeted-validation-families-file` which takes a file containing a list of font families to use for validation. These families are then sampled and reconstructured separately during validation steps, allowing you to ensure that the tokenizer is learning a good codebook for specific fonts of interest.

You'll want to train this for around 50k-150k steps. The health metrics reported to TensorBoard (see below) will help you determine when to stop training. 

* Train the generation model

Example run:

```bash
python -m hrothgar.ar.train --gtok-model-path models/gtok.pth --dataset-path <path_to_google/fonts_checkout> --batch-size 8 --target-characters 'abccdefghijklmnopqrstuvwxyz₹' --target-only --style-characters 'adhesionADHESIONR5$' 
```

This step has substantially more command line options; again, see `--help` for reference.

You'll want to train this for a *long time*.

* Train the upscaler model

Example run:

```bash
PYTHONPATH=Lib python -m hrothgar.upscaler.train --dataset-path <path_to_google/fonts_checkout>  --model-path models/upscaler.pth --style-conformance-mode --clean-font-only --clean-font-display-score-threshold 45  --low-res-size 96 --upscaling-factor 4
```

This should converge around 30-40k steps.

### Tokenizer health metrics

Without a well-trained tokenizer, the generation component will fail. During training of the G-Tok component, a number of health metrics are computed to ensure that the tokenizer is learning a good codebook and not collapsing or memorizing font-specific patterns. These metrics are reported to TensorBoard and can be used to determine when to stop training the tokenizer or to adjust hyperparameters.

| Metric | Interpretation |
|---|---|
| `Health/Autocorr/Accuracy` | Autocorrelation probe: chance of accurate next-token prediction. |
| `Health/Autocorr/xChance` | Chance of accurate next-token prediction, normalized to chance. |
| `Health/Autocorr/WithinRow` | Chance of accurate next-token prediction for tokens within the same row of the sequence. |
| `Health/Autocorr/CrossRow` | Chance of accurate next-token prediction for tokens in different rows of the sequence. |
| `Health/Autocorr/WithinCrossRatio` | Ratio of within-row to cross-row next-token prediction accuracy. If this is significantly different from 1, the tokenizer may have good local structure but weak global structure |
| `Health/Codebook/MeanPairwiseSimilarity` | Average cosine similarity between all codebook entries. High similarity (>0.7) means entries are redundant — the codebook isn't using its capacity effectively.  Low similarity (<0.3) means entries are diverse. |
| `Health/OracleAR/Accuracy` | Accuracy of an oracle autoregressive model trained on a single font. This is a measure of how well the codebook can represent a single font. If this is low, then the codebook is not expressive enough. |
| `Health/OracleAR/xChance` | Accuracy of an oracle autoregressive model, normalized to chance. |
| `Health/LinearProbe/CharAccuracy` | Accuracy of a linear probe trained to predict character identity from the code sequence. If this is low, then the codebook is not expressive enough. |
| `Health/LinearProbe/FontAccuracy` | Accuracy of a linear probe trained to predict font family from the code sequence. If this is high, then the codebook is memorizing font-specific patterns rather than general glyph structure. |
| `Health/CoreEntropy/Mean` | Average normalized entropy across active codes. 1.0 = codes used uniformly by all fonts. If this drops over training, then codes are becoming font-specific and early stopping is indicated. |
| `Health/CoreEntropy/Median` | A similar metrics but more robust to outliers than mean |
| `Health/CoreEntropy/ActiveCodes` | How many codebook entries are actually used. If this value drops then the codebook is collapsing; something has gone wrong. |
| `Health/CoreEntropy/FractionHighEntropy` | Fraction of codes with entropy > 0.5 × max. A high number suggests that codes are general rather than font-specific |
| `Health/CoreEntropy/FractionLowEntropy` | Fraction of codes with entropy < 0.1 × max. If this rises, then the tokenizer is memorizing per-font patterns |
| `Health/CoreEntropy/Distribution` | Histogram of all per-code entropies.  |

## Inference (Native)

The `generate.py` script contains an end-to-end generation and upscaling pipeline. (Currently it can only be used on a Google Fonts font file, but that's a bug; we should be able to make that an arbitrary font.)

Example run:

```bash
 python3 generate.py \
    --font-path <path_to_google/fonts_checkout>/tangerine/Tangerine-Regular.ttf \
    --target-chars 'G,x,₹' \ # Must be characters in --target-characters if --target-only was given
    --output-dir output \
    --dataset-path <path_to_google/fonts_checkout> \
    --reference-family noto-sans \ # Or noto-serif
    --gtok-model-path models/gtok.pth \
    --ar-model-path models/maskgit_glyph_gen.pth \
    --upscaler-model-path models/upscaler.pth \
    --save-intermediate # To save the low-resolution outputs from the generator before upscaling
```

## Inference (CoreML)

HrothGAR can also be exported to CoreML for inference on Apple devices. The `export_coreml.py` scripts handle the export process:

```bash
python -m hrothgar.ar.export_coreml \
    --model-path models/maskgit_glyph_gen.pth \
    --gtok-model-path models/gtok.pth \
    --output-dir models/coreml \
    --precision float32 \ # float16 doesn't work well, I don't know why not
    --style-reference-count 8 # style reference count becomes fixed when exporting
python -m hrothgar.upscaler.export_coreml \
       --model-path models/upscaler.pth \
       --output-dir models/coreml
```

The HrothGAR library is structured such that it can run with minimal dependencies in a CoreML environment; you can import `hrothgar.ar.inference_coreml` without torch installed - you just need `numpy` and `coremltools`. `Lib/hrothgar/ar/test_coreml.py` and `Lib/hrothgar/upscaler/test_coreml.py` contain example code which will help you integrate HrothGAR into your own CoreML pipeline.

## License

Original code in `Lib/hrothgar/` is licensed under the Apache license with the exception of:

* `Lib/hrothgar/upstream` is taken from the [GAR-Font](https://github.com/xTryer-s/GAR-Font) repository, which has no clear license.
* `Lib/hrothgar/llamagen_cnn.py` and `Lib/hrothgar/gtok/llamagen_lpips.py` are adapted from the [llamagen](https://github.com/FoundationVision/LlamaGen/) repository, which is licensed under the MIT license.
