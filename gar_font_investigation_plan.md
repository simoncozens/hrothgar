# GAR-Font Investigation Plan: Many-Shot Glyph Generation

## Project Goal

Generate missing glyphs for fonts in a catalogue, given:
- Many reference glyphs from the target font (hundreds)
- Textual description of the font
- Quantitative style tags (e.g. "80% happy", "20% formal", "90% script")
- Source fonts that *do* contain the target glyphs
- A downstream neural vectorization pipeline

This is the inverse of the standard "few-shot font generation" problem — we are **many-shot** on references but **few-target** on outputs.

---

## Can We Implement From the Paper?

**Yes, with caveats.** The paper provides enough to build a faithful reproduction. Here's the assessment:

### What the paper tells us (sufficient detail)

| Component | Known Details |
|---|---|
| **G-Tok tokenizer** | Hybrid CNN-ViT. CNN modified from LlamaGen tokenizer (open-source). 6-layer ViT encoder, 6-layer causal ViT decoder. Codebook: 2048 entries, dim 8. Input 64×64 → 64 tokens. Loss weights and training schedule. |
| **AR Generator** | 24-layer Transformer decoder (314M). Content encoder (28.56M CNN, same as G-Tok). Style encoder (2.78M CNN). 3-layer cross-attention aggregator (0.79M, follows FsFont/LF-Font). Soft decoding via codebook. |
| **Multimodal adapter** | Flan-T5 encoder (frozen). 6-layer cross-attention adapter (4.74M). Projection layer (0.52M). L2 alignment loss against visual-only aggregation. |
| **NFA** | LoRA on Transformer decoder. 128 glyphs, 10 epochs, lr 2e-5. |
| **SE** | GRPO with OCR + style rewards. 4 samples/group, 10 epochs, lr 5e-6. |
| **Training hyperparams** | AdamW (β1=0.9, β2=0.95), batch sizes, learning rates, iteration counts for all stages. |
| **Evaluation** | RMSE, SSIM, LPIPS, FID, content accuracy, style accuracy — all reproducible. |

### What we need to infer or find externally

| Gap | Mitigation |
|---|---|
| **Exact CNN architecture** (layers, channels, strides) | LlamaGen tokenizer is open-source — use it directly as the starting point. The content encoder has the same param count (28.56M) as G-Tok's CNN encoder, suggesting shared architecture. |
| **ViT hidden dim / heads** | Inferable: 4.73M params ÷ 6 layers ≈ 788K/layer. Standard ViT-Small (384 dim, 6 heads) is ~750K/layer — this is likely the configuration. |
| **Transformer decoder config** | 314M ÷ 24 layers ≈ 13.1M/layer. This matches a ~1024-dim, 16-head Transformer with 4096 FFN dim (GPT-2 Medium scale). |
| **Loss weight values** (λ_rec, λ_per, λ_vq, λ_CE, λ_pixel) | Not stated explicitly. Standard VQ-GAN values (λ_rec=1, λ_per=0.1, λ_vq=1) are a reasonable starting point. Can tune. |
| **Style encoder architecture** | 2.78M is small — likely a lightweight CNN (4-5 conv layers). The exact architecture can be approximated and tuned. |
| **Cross-attention aggregator details** | Paper cites FsFont and LF-Font, both of which describe this module in detail. Standard cross-attention with content as query, style as key/value. |
| **OCR model for SE** | Any Chinese character recognition model. Many open-source options. |
| **Style discriminator for SE** | Needs to be trained alongside. Could use the style classifier from evaluation (92.72% acc on 3040 fonts). |

### Verdict

The core architecture is **implementable** from the paper + LlamaGen source code + FsFont/LF-Font references. The gaps are in fine-grained hyperparameters that can be tuned experimentally. We don't need to wait for their code release in June — though it would serve as a useful validation checkpoint.

---

## Architecture Adaptation for Our Use Case

Several aspects of GAR-Font align exceptionally well; others need modification:

### Natural fits

1. **NFA is perfect for many-shot.** It was designed for 128 reference glyphs — we have far more. We could use 256-512 glyphs for NFA, getting even better per-font adaptation.

2. **Multimodal style encoder** already ingests text descriptions. Our quantitative style tags ("80% happy") can be prepended to the text description, or encoded as a separate conditioning vector.

3. **The base model generalizes.** Pre-trained on 3000 fonts → works on unseen fonts. Our catalogue fonts will likely be within the distribution of a well-trained model.

4. **Soft decoding** produces smooth, high-fidelity raster output — ideal input for a downstream vectorizer.

### Adaptations needed

1. **Resolution.** GAR-Font works at 64×64. For vectorization quality, we likely want 128×128 or 256×256. This means:
   - More tokens per glyph (256 or 1024 instead of 64)
   - Longer sequences for the AR decoder
   - Could use hierarchical tokenization or patch-based approaches to keep sequence length manageable

2. **Non-Chinese data.** GAR-Font is trained on GB2312 Chinese characters. We need:
   - A multi-script training dataset (Google Fonts, FontSquirrel, commercial catalogues)
   - Content glyphs from a standard reference font per script
   - Careful handling of very different glyph complexities (Latin vs. Chinese vs. Devanagari)

3. **Style tags integration.** Two approaches:
   - **Simple:** Concatenate quantitative tags with text description as a structured string, feed to the Flan-T5 encoder
   - **Richer:** Add a separate MLP that encodes the numeric style vector, fuse with the text embedding in the adapter. This adds a few extra parameters but gives the model a cleaner signal.

4. **Source font conditioning.** In standard few-shot FFG, the "content glyph" comes from a single standard reference font. In our case, we have *multiple source fonts* that contain the target glyph. We could:
   - Use GAR-Font's standard approach: pick one source font for content, target font for style
   - **Better:** Provide multiple source renderings of the target glyph as additional conditioning, letting the model see how different fonts interpret the same character

5. **Vectorization integration.** Your neural vectorization work can be applied as a post-processing step or trained end-to-end. Options:
   - **Pipeline:** Generate raster → vectorize (simplest, works with existing vectorizer)
   - **Joint:** Add a differentiable vectorization head, train with vector-level losses
   - **RL reward:** Add a vectorization quality reward to the SE stage

---

## Implementation Plan

### Phase 0: Infrastructure & Data (Weeks 1-2)

**Goal:** Dataset curation and training infrastructure.

- [X] **Dataset assembly**
  - Collect diverse multi-script font library (Latin, Cyrillic, Greek, Devanagari, etc.)
  - For each font: render all available glyphs at 64×64 (and optionally 128×128)
  - Split into train/test fonts and train/test characters
  - Associate each font with its textual description and style tags
  - Designate a "content reference" font per script (e.g. a clean sans-serif)

- [X] **Training infrastructure**
  - Set up PyTorch training pipeline with distributed training support
  - Implement data loading: font image pairs, text descriptions, style tags
  - Set up evaluation metrics (RMSE, SSIM, LPIPS, FID)
  - Logging and checkpointing (wandb/tensorboard)

- [X] **Baseline reproduction**
  - Download LlamaGen tokenizer code as CNN backbone reference
  - Obtain FsFont/LF-Font cross-attention aggregator reference
  - Set up a small-scale test (e.g. 50 fonts × 200 glyphs) for fast iteration

### Phase 1: G-Tok Tokenizer (Weeks 2-4)

**Goal:** Train a glyph tokenizer that reliably reconstructs glyphs from 64 discrete tokens.

- [x] **Implement G-Tok**
  - Start from LlamaGen's CNN encoder/decoder
  - Add 6-layer ViT encoder after CNN with 2D sinusoidal position embeddings
  - Add 6-layer causal ViT decoder before CNN decoder
  - Vector quantization with 2048-entry, dim-8 codebook
  - Implement entropy regularization for codebook utilization

- [X] **Training**
  - L1 reconstruction + VGG perceptual + VQ loss
  - 200k iterations, batch 16, lr 1e-4
  - AdamW optimizer

- [X] **Validation**
  - Reconstruction quality on held-out fonts (SSIM, LPIPS)
  - Codebook utilization statistics (dead code ratio)
  - Linear probing: can frozen features predict font style and character identity?
  - Robust reconstruction under local noise

- [x] **Key decision: resolution**
  - Initial 64×64 runs train stably and validate the architecture direction
  - 64×64 reconstructions are not sufficient for downstream vectorization quality
  - Move G-Tok to 128×128 with an 8x downsampling tokenizer grid, yielding 16×16 = 256 tokens
  - Re-run tokenizer training and validation at 128×128 before adding probing/noise studies

### Phase 2: AR Generator — Visual Only (Weeks 4-8)

**Goal:** Conditional glyph generation from content + visual style references.

- [ ] **Implement components**
  - Content encoder: CNN (reuse G-Tok's CNN encoder architecture, 28.56M)
  - Visual style encoder: lightweight CNN (2.78M, ~4-5 conv layers + pooling)
  - Content-style aggregator: 3-layer cross-attention (content queries, style keys/values)
  - Transformer decoder: 24 layers, ~1024 hidden dim, 16 heads (314M)
  - Soft decoding: Softmax(logits) · Codebook → G-Tok decoder
  - [x] AR phase-1 dataset maker and collation implemented (`target_rendering`, `content_rendering`, configurable `style_renderings` count)

- [ ] **Training (Stage 1: visual pretraining)**
  - Input: 1 content glyph + 8 style references → predict target glyph tokens
  - Loss: CE over token indices + L1 pixel reconstruction
  - AdamW (β1=0.9, β2=0.95), batch 32, lr 1e-4
  - 600k iterations (small dataset) or 1M (large dataset)
  - Freeze G-Tok during this stage
  - [x] Visual-only AR training loop scaffold implemented in [Lib/hrothgar/ar/train.py](Lib/hrothgar/ar/train.py) with configurable `N_s`

- [ ] **Validation**
  - Unseen-font generation quality (UFSC and UFUC splits)
  - Compare hard vs. soft decoding
  - Visual inspection of generated glyphs
  - Check style transfer: does changing style references change output style?

- [ ] **Experiment: many-shot conditioning**
  - Standard: 8 style references
  - Test: 16, 32, 64, 128 style references — does more help at pre-training?
  - This is a key differentiator for our use case

### Phase 3: Enhanced Multimodal Style Encoder (Weeks 6-10)

**Goal:** Integrate text descriptions and quantitative style tags.

- [ ] **Implement adapter (following paper)**
  - Flan-T5 encoder (frozen) for text embedding
  - Projection layer (0.52M) into visual feature space
  - 6-layer cross-attention adapter between text and visual style features
  - Concatenate text-fused representation with visual style features

- [ ] **Extend for our style tags**
  - Approach A (simple): Format as structured text, e.g. "A font that is 80% happy, 20% formal, 90% script, with..."
  - Approach B (richer): Separate MLP for numeric style vector → fuse with text embedding
  - Approach C (both): Use text + numeric conditioning as parallel inputs to adapter
  - Experiment to determine which gives the best style control

- [ ] **Training**
  - Freeze the AR generator
  - L2 alignment loss: multimodal aggregation ≈ visual-only aggregation
  - 40k iterations, batch 128, lr 1e-4

- [ ] **Validation**
  - Compare GAR-Font(M_2) and GAR-Font(M_4) configurations
  - Ablate text description vs. style tags vs. both
  - Test: can style tags alone (without visual references) drive generation?
  - Test: does richer text (our descriptions) beat the paper's Qwen-VL descriptions?

### Phase 4: Post-Refinement — NFA + SE (Weeks 8-12)

**Goal:** Per-font adaptation and structural enhancement.

- [ ] **NFA (Novel Font Adaptation)**
  - [x] Add LoRA adapters to Transformer decoder layers (implemented in `ARModel` with decoder-layer injection and LoRA-only checkpoint support)
  - [x] Add NFA fine-tuning loop scaffold (single-font dataset maker + training loop in `Lib/hrothgar/ar/nfa.py`)
  - [ ] Fine-tune on N target font glyphs (paper uses 128; try 256/512 with our data)
  - [ ] 10 epochs, lr 2e-5
  - [ ] Measure: Time per font, quality improvement vs. number of adaptation glyphs

- [ ] **SE (Structural Enhancement)**
  - Implement GRPO-based RL
  - OCR reward: use open-source multilingual OCR model
  - Style reward: train a style discriminator (or use style classifier)
  - 4 samples/group, 10 epochs, batch 32, lr 5e-6

- [ ] **Catalogue-scale testing**
  - Benchmark NFA time per font
  - Can we batch NFA across fonts efficiently?
  - Profile: total time to process 100 fonts × 10 missing glyphs each

### Phase 5: Glyph Super-Resolution (Weeks 10-12)

**Goal:** Upscale generated glyph rasters before vectorization while preserving edge fidelity.

- [ ] **Prototype SR baseline**
  - Train a supervised 2x/4x upscaler on Latin core glyph pairs from Google Fonts
  - Use aligned pairs: render high-resolution glyphs, then downsample for low-resolution input
  - Baseline loss: BCE/L1 reconstruction + edge-aware term (Sobel/Canny-weighted)

- [ ] **Glyph-aware conditioning experiments**
  - Condition the upscaler on frozen G-Tok encoder features (CNN only vs. CNN+ViT)
  - Compare against bicubic and non-conditioned SR baseline
  - Evaluate terminal sharpness, stroke continuity, and curve smoothness on held-out fonts

- [ ] **Resolution handoff study**
  - Determine whether 128->256 or 128->512 upscaling is the best quality/compute trade-off
  - Quantify downstream impact on vectorizer outputs (path smoothness, corner stability)

### Phase 6: Vectorization Integration (Weeks 12-14)

**Goal:** Connect raster glyph generation to vector output.

- [ ] **Pipeline approach (baseline)**
  - Generate raster glyphs with GAR-Font
  - Apply neural vectorization as post-processing
  - Evaluate vector output quality (path smoothness, control point count, fidelity)

- [ ] **Joint training (stretch goal)**
  - Add differentiable rasterizer to enable gradient flow through vectorization
  - Train end-to-end with vector-level losses (curvature smoothness, etc.)
  - Or: add vectorization quality as an additional SE reward signal

- [ ] **Resolution experiments**
  - Test vectorization quality from 64×64 vs. 128×128 vs. 256×256 raster input
  - Determine minimum resolution needed for acceptable vector output
  - If higher resolution is needed, revisit G-Tok token count

### Phase 7: Evaluation & Scaling (Weeks 14-16)

**Goal:** End-to-end evaluation on realistic catalogue scenarios.

- [ ] **Glyph quality evaluation**
  - Side-by-side comparison with ground truth (for fonts where we have the target glyph)
  - Expert/user evaluation of generated vs. real glyphs
  - Evaluate across scripts and glyph complexity levels

- [ ] **Catalogue workflow**
  - End-to-end pipeline: font in → missing glyphs as vectors out
  - Measure throughput (fonts/hour)
  - Quality vs. NFA adaptation time tradeoff

- [ ] **Failure analysis**
  - Identify font styles that are hardest to generate for
  - Identify glyphs that are hardest to generate (rare symbols, complex shapes)
  - Determine when to flag a generation for human review

---

## Key Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **CNN architecture mismatch** — LlamaGen CNN doesn't match GAR-Font's intent | Medium | Medium | LlamaGen is explicitly cited as the baseline. Parameter counts should match. If not, adjust channel counts. |
| **Non-Chinese generalization** — model trained on Chinese may not transfer to Latin/etc. | Low | High | We're training from scratch on our multi-script data. The architecture is script-agnostic. |
| **Resolution bottleneck** — 64×64 too low for vectorization | High | High | Addressed in Phase 5 SR and Phase 6 vectorization handoff. Fallback: train at 128×128 from the start with 256 tokens. |
| **Training cost** — 314M Transformer + 1M iterations is expensive | Medium | Medium | Start with small dataset (400 fonts). Single A100 should suffice for the tokenizer; need 2-4 for the generator. |
| **Vectorization quality** — separate pipeline introduces artifacts | Medium | Medium | Your existing vectorization work should handle this. Joint training is the stretch goal. |

---

## Resource Estimates

| Stage | GPU-Hours (est., A100) | Notes |
|---|---|---|
| G-Tok tokenizer | 24-48 | 200k iters, small model, batch 16 |
| AR Generator pretraining | 200-400 | 600k-1M iters, 314M model, batch 32 |
| Multimodal adapter | 8-16 | 40k iters, only adapter trains |
| NFA per font | 0.5-1 | LoRA only, 10 epochs on 128 glyphs |
| SE per font | 2-4 | RL sampling is expensive |
| **Total for core training** | **~250-500** | Excluding per-font adaptation |

---

## Open Questions to Resolve Early

1. **What resolution does the vectorizer need?** This determines whether we can stay at 64×64 or need to scale up the tokenizer. Should be answered in Week 1 by testing the vectorizer on 64×64 glyph images.

2. **What fonts and scripts are in the catalogue?** This determines the training dataset scope and whether we need script-specific content reference fonts.

3. **How many missing glyphs per font, typically?** If it's 1-5 rare symbols, the generation is straightforward. If it's hundreds (e.g. adding an entire script), we may want to optimize batch generation.

4. **What's the acceptable quality bar?** Is this for display-quality fonts (needs to be perfect) or for fallback rendering (good enough is fine)?

5. **Do we have ground truth for evaluation?** Can we take fonts that *do* have the target glyphs, hide them, generate them, and compare?

---

## Suggested First Steps (This Week)

1. **Clone LlamaGen** and inspect the tokenizer architecture — confirm the CNN matches the 28.56M parameter count.
2. **Script to render fonts** — build a dataset pipeline to render glyphs from .ttf/.otf files at 64×64.
3. **Prototype G-Tok** — implement the hybrid CNN-ViT tokenizer and get it training on a small font subset.
4. **Test vectorizer resolution** — feed 64×64 synthetic glyph images to the vectorizer and assess output quality.
