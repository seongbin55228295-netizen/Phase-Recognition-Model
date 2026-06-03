"""Aggregate per-variant metric JSONs into the README's four comparison axes.

Reads reports/metrics/<variant>.json (produced by scripts/evaluate.py) and emits,
under reports/tables/:
  - comparison_modal_ablation.md   (image_only vs baseline)
  - comparison_fusion_method.md    (co-attention vs concat)
  - comparison_exposure_bias.md    (tf_oracle / tf_freerun / baseline + TF-FR gap)
  - comparison_ss_strength.md      (p_end 0.25 / 0.5 / 0.75)
  - all_metrics.csv                (every variant x {fr,tf} x all metrics)

Each axis reports the metrics under the inference regime that matches that
variant's intended deployment: FR for everything except the TF-oracle upper
bound, which is reported under TF. The TF-FR gap column is the checkpoint-level
gap (same weights, both regimes), independent of the displayed mode.

Usage:
    python scripts/compare_variants.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (column header, json key inside the chosen mode block, formatter)
METRIC_COLS = [
    ("Frame Acc", "frame_accuracy", lambda x: f"{x:.4f}"),
    ("Frame Acc (w)", "frame_accuracy_weighted", lambda x: f"{x:.4f}"),
    ("Macro F1", "macro_f1", lambda x: f"{x:.4f}"),
    ("Seg IoU", "segment_iou", lambda x: f"{x:.4f}"),
    ("Edit", "edit_score", lambda x: f"{x:.1f}"),
]

# axis title -> list of (variant, mode, row_label)
AXES = {
    "modal_ablation": (
        "모달 ablation — image-only vs 융합(baseline)",
        [
            ("image_only", "fr", "image-only"),
            ("baseline", "fr", "융합 (co-attn, 기준)"),
        ],
    ),
    "fusion_method": (
        "융합 기법 — Co-attention vs Concat",
        [
            ("baseline", "fr", "Co-attention (기준)"),
            ("fusion_concat", "fr", "Concat"),
        ],
    ),
    "exposure_bias": (
        "Exposure Bias — TF/FR 분포 차이와 완화",
        [
            ("tf_oracle", "tf", "TF 학습 + TF 평가 (Oracle 상한)"),
            ("tf_freerun", "fr", "TF 학습 + FR 평가"),
            ("baseline", "fr", "SS 학습 + FR 평가 (기준)"),
        ],
    ),
    "ss_strength": (
        "Scheduled Sampling 강도 — p_end",
        [
            ("ss_low", "fr", "p_end = 0.25"),
            ("baseline", "fr", "p_end = 0.50 (기준)"),
            ("ss_high", "fr", "p_end = 0.75"),
        ],
    ),
}


def load_metrics(metrics_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for fp in sorted(metrics_dir.glob("*.json")):
        with open(fp, encoding="utf-8") as f:
            out[fp.stem] = json.load(f)
    return out


def render_axis(title: str, rows: list[tuple[str, str, str]], data: dict[str, dict],
                with_gap: bool) -> tuple[str, list[str]]:
    """Return (markdown, list_of_missing_variants)."""
    headers = ["변형", "Eval"] + [c[0] for c in METRIC_COLS]
    if with_gap:
        headers.append("TF-FR Gap")
    lines = [f"### {title}", "", "| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    missing: list[str] = []
    for variant, mode, label in rows:
        rec = data.get(variant)
        if rec is None:
            missing.append(variant)
            cells = [label, mode.upper()] + ["—"] * len(METRIC_COLS)
            if with_gap:
                cells.append("—")
            lines.append("| " + " | ".join(cells) + " |")
            continue
        block = rec[mode]
        cells = [label, mode.upper()] + [fmt(block[key]) for _, key, fmt in METRIC_COLS]
        if with_gap:
            cells.append(f"{rec['tf_fr_gap']:+.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines), missing


def write_all_csv(data: dict[str, dict], out_path: Path) -> None:
    metric_keys = [k for _, k, _ in METRIC_COLS]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "split", "n_videos", "n_frames", "mode", *metric_keys, "tf_fr_gap"])
        for variant in sorted(data):
            rec = data[variant]
            for mode in ("fr", "tf"):
                block = rec[mode]
                w.writerow([
                    variant, rec.get("split", ""), rec.get("n_videos", ""), rec.get("n_frames", ""),
                    mode, *[block[k] for k in metric_keys], rec["tf_fr_gap"],
                ])


def _force_utf8_console() -> None:
    """Windows consoles default to cp949/cp1252 and choke on '—'/Korean output."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def main() -> None:
    _force_utf8_console()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-dir", default="reports/metrics")
    p.add_argument("--tables-dir", default="reports/tables")
    args = p.parse_args()

    metrics_dir = ROOT / args.metrics_dir
    tables_dir = ROOT / args.tables_dir
    tables_dir.mkdir(parents=True, exist_ok=True)

    data = load_metrics(metrics_dir)
    if not data:
        print(f"no metric JSONs found in {metrics_dir} — run scripts/evaluate.py first", file=sys.stderr)
        sys.exit(1)
    print(f"loaded metrics for: {', '.join(sorted(data))}")

    all_missing: set[str] = set()
    for axis_key, (title, rows) in AXES.items():
        with_gap = axis_key == "exposure_bias"
        md, missing = render_axis(title, rows, data, with_gap)
        all_missing.update(missing)
        out_path = tables_dir / f"comparison_{axis_key}.md"
        out_path.write_text(md + "\n", encoding="utf-8")
        print("\n" + md)
        print(f"-> {out_path.relative_to(ROOT).as_posix()}")

    csv_path = tables_dir / "all_metrics.csv"
    write_all_csv(data, csv_path)
    print(f"\n-> {csv_path.relative_to(ROOT).as_posix()}")

    if all_missing:
        print(f"\n[warn] missing metric JSONs (rows left blank): {', '.join(sorted(all_missing))}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
