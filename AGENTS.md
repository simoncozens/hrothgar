# Hroth-GAR

* HrothGAR is a clean-room implementation of the GAR-Font paper using Python and Torch.
* Briefly, GAR-Font aims to generate fonts from few-shot examples using a generic image tokenizer based on LlamaGen, an autoregressive generator, a multi-model adaptor to condition the output based on textual descriptions of the font, "Novel Font Adaptation" and "Structural Enhancement" modules to refine the output.
* For your reference, the original paper is available in TeX format in the `beyond-patches` directory. No code is available, so we're going off the description.
* The original paper used GAR-Font for few-shot font generation (FFG) on Chinese input sources. We will be aiming to use the GAR-Font architecture for *many*-shot font generation - i.e. adding additional glyphs to existing fonts where a good range of glyphs exists to estabish the style.
* **Very important**: A plan for this investigation is available at `gar_font_investigation_plan.md`. Please refer to that for the overall plan and next steps, and keep it updated.
* We will be implementing the architecture component by component, in small, testable and reportable phases. See the plan for details.

## Coding expectations

* Unlike most ML research projects, we will be aiming to write *production Python*; in other words:
  - Code should be clean, well-structured and well-documented. Classes and methods should have docstrings, and the overall codebase should be organized in a logical way.
  - We will be using type annotations and static type checking (e.g. with mypy).
  - Dependencies will be declared through `pyproject.toml`. We will use `uv` for virtual environment management and dependency installation.
  - We will be writing unit tests for our code, and aiming for good test coverage.
  - We will be using version control (e.g. git) to manage our codebase, and following good practices for commit messages and branching.
  - We will be using a consistent coding style (e.g. PEP 8) and adhering to it throughout the project.
* We also want to be rigorous during training runs, and will be logging training metrics, hyperparameters and model checkpoints in a structured way using TensorBoard. Training runs should *only* be performed on a clean git status, and the git commit hash of the code used for the run should be logged alongside the training metrics for reproducibility. There should always be a clear and obvious answer to the question "What changed between the previous run and this one?"

## Other notes

* It is an open question whether methodologies for Chinese fonts can be directly applied to Latin fonts. One key difference is that Chinese characters each have the same unit size, filling an em square, with no ascenders and descenders. Latin characters, on the other hand, have varying widths and heights, with ascenders and descenders. This may require some adjustments to the architecture or training process to account for these differences.