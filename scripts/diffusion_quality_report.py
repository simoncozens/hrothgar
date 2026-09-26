#!/usr/bin/env python
"""Analyse diffusion generation quality, bucketed by LPIPS and font stratum.

Loads a trained factorized ``(codepoint, font-instance)`` diffusion checkpoint
and its sidecars (``.conf.json``, ``.codepoints.json``, ``.instances.json``),
then walks every training instance recorded in the sidecar.  For each codepoint
the instance actually draws, it:

  * samples a glyph with the model (DDIM),
  * compares it against the native ground-truth render with LPIPS,
  * buckets the pair by quality (``< 0.1`` / ``0.1-0.2`` / ``>= 0.2``) and by
    font stratum (sans / serif / display / script / handwriting / mono / other),
  * writes GT + generated PNGs into per-stratum subdirectories under
    ``good`` / ``mid`` / ``bad`` (``good`` always; ``mid``/``bad`` opt-in via
    ``--save-mid`` / ``--save-bad``).

It deliberately uses the **sidecar** (not the stratified sampler) so the
conditioning ``(family_id, weight_norm, style_bucket)`` and the per-instance
``axis_position`` match the exact training-time values, regardless of how the
local Google Fonts checkout has drifted relative to the training server.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
import tqdm
from hrothgar.dataset import _has_non_empty_outline, _hb_font_for_face
from hrothgar.diffusion.config import FontIdDiffusionConfig
from hrothgar.diffusion.dataset_fontid import style_bucket
from hrothgar.diffusion.fontid import build_fontid_model
from hrothgar.googlefonts import GoogleFont, GoogleFonts, StandaloneFont
from hrothgar.llamagen_lpips import LPIPS
from hrothgar.render_utils import render_glyph_with_geometry
from hrothgar.utils import pick_device

GOOD_THRESHOLD = 0.1
MID_THRESHOLD = 0.2
STRATA = ("sans", "serif", "display", "script", "handwriting", "mono", "other")


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_codepoints(spec: str | None) -> set[int] | None:
    """Parse a ``--codepoints`` spec into a set of Unicode ords, or ``None``.

    ``None`` / ``"all"`` means "no filter" (evaluate the full vocabulary).
    Otherwise each comma-separated token contributes every one of its characters
    (so both ``"ABC5$"`` and ``"A,B,C,5,$"`` work).
    """
    if spec is None or spec.strip() == "" or spec.strip().lower() == "all":
        return None
    out: set[int] = set()
    for part in spec.split(","):
        for ch in part:
            if not ch.isspace():
                out.add(ord(ch))
    return out


def _sanitize(name: str) -> str:
    """Make a family/path component safe for a filename."""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    return cleaned or "family"


def _save_grayscale(image: torch.Tensor, path: Path) -> None:
    """Save a ``(H, W)`` [0, 1] (ink=0 / white=1) tensor as a grayscale PNG."""
    from PIL import Image

    arr = image.detach().cpu().clamp(0.0, 1.0).numpy()
    Image.fromarray((arr * 255.0).astype("uint8"), mode="L").save(path)


def _available_codepoints(font: StandaloneFont, vocab: list[int]) -> list[int]:
    """Codepoints in ``vocab`` that this font draws with a non-empty outline."""
    hb_font = _hb_font_for_face(font.hb_face)
    out = []
    for cp in sorted(set(font.codepoints) & set(vocab)):
        gid = hb_font.get_nominal_glyph(cp)
        if _has_non_empty_outline(hb_font.get_glyph_extents(gid)):
            out.append(cp)
    return out


def _stratum(gfont: GoogleFont | None) -> str:
    """Replicate :meth:`Unit.stratum` for a single font."""
    if gfont is None:
        return "other"
    bucket = style_bucket(gfont)  # mono / fancy / sans / serif / other
    if bucket != "fancy":
        return bucket
    if gfont.category() == "Script":
        return "script"
    if "HANDWRITING" in gfont.classification():
        return "handwriting"
    return "display"


def _quality_bucket(
    lpips: float,
    good_threshold: float = GOOD_THRESHOLD,
    mid_threshold: float = MID_THRESHOLD,
) -> str:
    if lpips < good_threshold:
        return "good"
    if lpips < mid_threshold:
        return "mid"
    return "bad"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-path", type=str, required=True,
                   help="Path to the trained diffusion checkpoint (sidecars derived)")
    p.add_argument("--repo", type=str, default=os.environ.get("GOOGLE_FONTS_REPO"),
                   help="Google Fonts repo root, to resolve repo-relative instance paths")
    p.add_argument("--output-dir", type=str, default="outputs/quality_report")
    p.add_argument("--codepoints", type=str, default="all",
                   help="Comma/string of characters to evaluate, or 'all' (default)")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process this many instances (for a quick smoke test)")
    p.add_argument("--good-threshold", type=float, default=GOOD_THRESHOLD,
                   help="LPIPS below which a pair is bucketed 'good'")
    p.add_argument("--mid-threshold", type=float, default=MID_THRESHOLD,
                   help="LPIPS below which a pair is bucketed 'mid' (above good)")
    p.add_argument("--max-good", type=int, default=None,
                   help="Cap the number of saved good pairs (default: unlimited)")
    p.add_argument("--save-mid", action="store_true",
                   help="Also save mid pairs into <out>/mid/<stratum>/")
    p.add_argument("--save-bad", action="store_true",
                   help="Also save bad pairs into <out>/bad/<stratum>/")
    p.add_argument("--max-mid", type=int, default=None,
                   help="Cap the number of saved mid pairs (default: unlimited)")
    p.add_argument("--max-bad", type=int, default=None,
                   help="Cap the number of saved bad pairs (default: unlimited)")
    p.add_argument("--seed", type=int, default=0, help="DDIM noise seed")
    args = p.parse_args()

    if not args.repo:
        raise SystemExit("Provide --repo or set GOOGLE_FONTS_REPO")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = pick_device()
    print(f"Using device: {device}")

    model_path = Path(args.model_path)
    config = FontIdDiffusionConfig.from_sidecar(model_path)
    codepoints = _load_json(Path(str(model_path) + ".codepoints.json"))
    instances = _load_json(Path(str(model_path) + ".instances.json"))
    cp_to_idx = {cp: i for i, cp in enumerate(codepoints)}
    repo = Path(args.repo)
    image_size = config.image_size
    codepoint_filter = _parse_codepoints(args.codepoints)

    model = build_fontid_model(config).to(device)
    model.load(str(model_path), device=device)
    model.eval()

    lpips_model = LPIPS().to(device)

    # Load tags + family metadata once, for stratum classification.
    try:
        gf = GoogleFonts(repo)
    except Exception as exc:  # e.g. missing tags csv on a partial checkout
        print(f"Warning: could not load GoogleFonts metadata ({exc}); "
              "stratum will fall back to 'other'.")
        gf = None

    out_dir = Path(args.output_dir)

    # Which quality buckets to write PNGs for, and per-bucket caps.
    save_enabled = {
        "good": True,
        "mid": args.save_mid,
        "bad": args.save_bad,
    }
    save_caps = {
        "good": args.max_good,
        "mid": args.max_mid,
        "bad": args.max_bad,
    }
    saved_count: Counter = Counter()

    def _maybe_save(bucket, stratum, gt, rec, stem):
        """Write a GT/generated pair into ``<bucket>/<stratum>/`` if enabled."""
        if not save_enabled[bucket]:
            return
        cap = save_caps[bucket]
        if cap is not None and saved_count[bucket] >= cap:
            return
        subdir = out_dir / bucket / stratum
        subdir.mkdir(parents=True, exist_ok=True)
        _save_grayscale(gt, subdir / f"{stem}_gt.png")
        _save_grayscale(rec, subdir / f"{stem}_gen.png")
        saved_count[bucket] += 1

    stratum_cache: dict[str, str] = {}

    def stratum_for(family: str, path: str) -> str:
        if family in stratum_cache:
            return stratum_cache[family]
        gfont = None
        if gf is not None:
            gfont = GoogleFonts.families_by_name.get(family)
        if gfont is None:
            try:
                gfont = GoogleFont(repo / path, gf)
            except Exception:
                gfont = None
        stratum = _stratum(gfont)
        stratum_cache[family] = stratum
        return stratum

    lpips_values: list[float] = []
    by_quality: Counter = Counter()
    by_stratum: Counter = Counter()
    by_quality_stratum: Counter = Counter()
    by_codepoint_sum: defaultdict = defaultdict(float)
    by_codepoint_count: Counter = Counter()

    processed_instances = 0
    skipped_instances = 0
    n_pairs = 0

    expected_size = args.limit if args.limit else len(instances)

    random.shuffle(instances)

    for iid, inst in tqdm.tqdm(enumerate(instances), total=expected_size):
        if args.limit is not None and processed_instances >= args.limit:
            break

        abspath = repo / inst["path"]
        if not abspath.exists():
            skipped_instances += 1
            continue
        try:
            font = StandaloneFont(abspath)
        except Exception:
            skipped_instances += 1
            continue

        available = _available_codepoints(font, codepoints)
        if codepoint_filter is not None:
            available = [cp for cp in available if cp in codepoint_filter]
        if not available:
            continue

        processed_instances += 1
        family = inst["family"]
        weight = inst["weight"]
        style = inst["style"]
        axis_position = inst["axis_position"]
        stratum = stratum_for(family, inst["path"])

        meta = torch.tensor(
            [[inst["family_id"], inst["weight_norm"], inst["style_bucket"]]],
            device=device, dtype=torch.float32,
        ).repeat(len(available), 1)
        cp_t = torch.tensor([cp_to_idx[cp] for cp in available],
                            device=device, dtype=torch.long)

        with torch.no_grad():
            recs = model.sample(cp_t, meta).clamp(0.0, 1.0)  # (B, 1, H, W)
            gts = torch.stack(
                [
                    render_glyph_with_geometry(
                        font, cp, image_size, axis_position=axis_position
                    )[0]
                    for cp in available
                ]
            ).unsqueeze(1).to(device)  # (B, 1, H, W)
            per_pair_lpips = lpips_model(recs, gts).flatten()  # (B,)

        for j, cp in enumerate(available):
            lp = float(per_pair_lpips[j])
            bucket = _quality_bucket(lp, args.good_threshold, args.mid_threshold)
            n_pairs += 1
            lpips_values.append(lp)
            by_quality[bucket] += 1
            by_stratum[stratum] += 1
            by_quality_stratum[(bucket, stratum)] += 1
            by_codepoint_sum[cp] += lp
            by_codepoint_count[cp] += 1

            stem = (
                f"{iid:04d}_{_sanitize(family)}_w{weight}{style}"
                f"_U+{cp:04X}_lp{lp:.3f}"
            )
            _maybe_save(bucket, stratum, gts[j, 0], recs[j, 0], stem)

        if processed_instances % 25 == 0:
            mean = sum(lpips_values) / len(lpips_values) if lpips_values else 0.0
            print(f"  ... {processed_instances} instances, {n_pairs} pairs, "
                  f"mean LPIPS {mean:.4f}")

    if n_pairs == 0:
        raise SystemExit("No (instance, codepoint) pairs were generated; "
                         "check --repo, --codepoints, and the sidecar paths.")

    lpips_sorted = sorted(lpips_values)
    mean = sum(lpips_sorted) / len(lpips_sorted)
    median = lpips_sorted[len(lpips_sorted) // 2]
    p10 = lpips_sorted[int(len(lpips_sorted) * 0.10)]
    p90 = lpips_sorted[int(len(lpips_sorted) * 0.90)]

    report = {
        "model": str(model_path),
        "seed": args.seed,
        "instances_processed": processed_instances,
        "instances_skipped": skipped_instances,
        "pairs": n_pairs,
        "lpips": {
            "mean": mean,
            "median": median,
            "p10": p10,
            "p90": p90,
            "min": lpips_sorted[0],
            "max": lpips_sorted[-1],
        },
        "by_quality": dict(by_quality),
        "by_stratum": dict(by_stratum),
        "by_quality_stratum": {
            f"{q} / {s}": by_quality_stratum[(q, s)]
            for q in ("good", "mid", "bad") for s in STRATA
        },
        "by_codepoint": {
            f"{chr(cp)!r} U+{cp:04X}": {
                "mean_lpips": by_codepoint_sum[cp] / by_codepoint_count[cp],
                "count": by_codepoint_count[cp],
            }
            for cp in sorted(
                by_codepoint_count,
                key=lambda c: -(by_codepoint_sum[c] / by_codepoint_count[c]),
            )
        },
        "saved": dict(saved_count),
    }

    report_path = out_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print("\n=== Quality report ===")
    print(f"instances: {processed_instances} processed / {skipped_instances} skipped")
    print(f"pairs: {n_pairs}")
    print(f"LPIPS: mean {mean:.4f}  median {median:.4f}  "
          f"p10 {p10:.4f}  p90 {p90:.4f}  min {lpips_sorted[0]:.4f}  max {lpips_sorted[-1]:.4f}")
    print(f"quality buckets: {dict(by_quality)}")
    print("\nquality x stratum:")
    header = "  " + "".join(f"{s:>12}" for s in STRATA)
    print(header)
    for q in ("good", "mid", "bad"):
        row = "".join(f"{by_quality_stratum[(q, s)]:>12}" for s in STRATA)
        print(f"{q:<6}{row}")
    print(f"\nsaved: good={saved_count['good']}, mid={saved_count['mid']}, bad={saved_count['bad']}")
    print(f"output dir -> {out_dir}")
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
