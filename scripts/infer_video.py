"""Inference on a raw test video with the proposed (or any) trained model.

Two input types share one pipeline; --source selects the branch:

  raw video --> extract frames (2fps / short-256) --> eval transform
            --> whole-video Free-Running autoregressive inference --> per-frame phase

  --source youtube  : arbitrary YouTube URL or local mp4. No ground truth -> qualitative
                      outputs only (per-frame predictions + phase timeline).
  --source youcook2 : a held-out YouCook2 video_id. Ground truth is reconstructed
                      from its annotations via prototype embeddings (the same mapping
                      used to build the training labels) -> quantitative metrics too.

Examples
--------
  # arbitrary YouTube (qualitative)
  python scripts/infer_video.py --source youtube --url "https://youtu.be/XXXX" \
      --config experiments/baseline.yaml --checkpoint checkpoints/baseline/best.pt

  # local file (qualitative)
  python scripts/infer_video.py --source youtube --video clip.mp4 \
      --config experiments/baseline.yaml --checkpoint checkpoints/baseline/best.pt

  # held-out YouCook2 (quantitative; downloads via the annotation's video_url)
  python scripts/infer_video.py --source youcook2 --video-id <youcook2_id> \
      --config experiments/baseline.yaml --checkpoint checkpoints/baseline/best.pt

Outputs land in --out-dir (default reports/inference/<name>/):
  predictions.csv  segments.json  timeline.png  [metrics.json for youcook2]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CLASS_NAMES, build_eval_transform
from src.preprocessing.frame_extraction import extract_frames, get_video_fps
from src.evaluation import frame_accuracy, macro_f1
from src.inference import Predictor, build_youcook2_gt, download_video, load_frame_tensors
from src.inference.youcook2_gt import load_entry

DEFAULT_ANNOTATIONS = ROOT / "YouCookII" / "annotations" / "youcookii_annotations_trainval.json"
DEFAULT_PROTOTYPES = ROOT / "configs" / "action_class_prototypes.json"
SELECTED_IDS = ROOT / "configs" / "selected_video_ids.json"


# --------------------------------------------------------------------------- args

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, choices=["youtube", "youcook2"],
                   help="Input type: arbitrary YouTube/local (qualitative) or held-out YouCook2 (quantitative).")
    p.add_argument("--config", required=True, help="Experiment YAML used to build the model.")
    p.add_argument("--checkpoint", required=True, help="Trained weights (e.g. checkpoints/baseline/best.pt).")

    # youtube source
    p.add_argument("--url", help="[youtube] YouTube URL to download.")
    p.add_argument("--video", help="[youtube] Local video file (skips download).")
    p.add_argument("--name", help="[youtube] Output name (default: derived from url/file).")

    # youcook2 source
    p.add_argument("--video-id", help="[youcook2] A single held-out YouCook2 video_id.")
    p.add_argument("--video-list", help="[youcook2] Text file of video_ids (one per line) for batch eval.")
    p.add_argument("--annotations-json", default=str(DEFAULT_ANNOTATIONS))
    p.add_argument("--prototypes", default=str(DEFAULT_PROTOTYPES))
    p.add_argument("--cookies", default=None, help="cookies.txt for yt-dlp (bot-block bypass).")

    # common
    p.add_argument("--out-dir", default=None, help="Output root (default reports/inference/).")
    p.add_argument("--videos-dir", default=str(ROOT / "data" / "videos"),
                   help="Where downloaded mp4s are cached / looked up.")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto).")
    p.add_argument("--image-batch-size", type=int, default=64)
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--resize-short", type=int, default=256)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--keep-frames", action="store_true", help="Do not delete extracted frames afterwards.")
    return p.parse_args()


# ------------------------------------------------------------------ derived outputs

def collapse_segments(records: list[dict]) -> list[dict]:
    """Collapse consecutive equal-label frames into predicted phase segments."""
    segments: list[dict] = []
    for r in records:
        lbl = r["pred_label"]
        if segments and segments[-1]["label"] == lbl:
            seg = segments[-1]
            seg["end_sec"] = r["timestamp_sec"]
            seg["end_frame"] = r["frame_index"]
            seg["n_frames"] += 1
        else:
            segments.append({
                "label": lbl,
                "start_sec": r["timestamp_sec"], "end_sec": r["timestamp_sec"],
                "start_frame": r["frame_index"], "end_frame": r["frame_index"],
                "n_frames": 1,
            })
    return segments


def plot_timeline(records: list[dict], out_path: Path, *, title: str,
                  has_gt: bool) -> bool:
    """Render a class-colored phase strip (pred, and gt if available). Best-effort."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception as exc:  # matplotlib optional at runtime
        print(f"  (timeline.png skipped: {exc})")
        return False

    cmap = plt.get_cmap("tab10")
    color = {name: cmap(i % 10) for i, name in enumerate(CLASS_NAMES)}
    times = [r["timestamp_sec"] for r in records]
    xmax = max(times) if times else 1.0
    # frame band width = median spacing between consecutive frames (= 1/fps)
    diffs = [b - a for a, b in zip(times, times[1:]) if b > a]
    step = (sorted(diffs)[len(diffs) // 2] if diffs else 0.5)

    rows = [("prediction", "pred_label")]
    if has_gt:
        rows.append(("ground truth (pseudo)", "gt_label"))

    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.6 * len(rows) + 0.8), squeeze=False)
    for ax, (row_title, key) in zip(axes[:, 0], rows):
        for r in records:
            lbl = r.get(key)
            if not lbl or (isinstance(lbl, float) and pd.isna(lbl)):
                continue
            ax.axvspan(r["timestamp_sec"], r["timestamp_sec"] + step,
                       color=color.get(lbl, "lightgray"))
        ax.set_xlim(0, xmax + step)
        ax.set_yticks([])
        ax.set_ylabel(row_title, rotation=0, ha="right", va="center", fontsize=9)
    axes[-1, 0].set_xlabel("time (s)")
    handles = [Patch(color=color[n], label=n) for n in CLASS_NAMES]
    fig.legend(handles=handles, loc="upper center", ncol=len(CLASS_NAMES), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.08, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def compute_metrics(merged: pd.DataFrame) -> dict:
    """Frame-level metrics on covered frames (preds joined to pseudo-GT)."""
    from src.data.labels import LABEL_TO_IDX
    preds = [LABEL_TO_IDX[l] for l in merged["pred_label"]]
    gts = [LABEL_TO_IDX[l] for l in merged["gt_label"]]
    weights = merged["gt_weight"].astype(float).tolist()

    n = len(CLASS_NAMES)
    confusion = [[0] * n for _ in range(n)]
    for y, p in zip(gts, preds):
        confusion[y][p] += 1

    return {
        "n_eval_frames": len(merged),
        "frame_accuracy_weighted": frame_accuracy(preds, gts, weights),
        "frame_accuracy": frame_accuracy(preds, gts),
        "macro_f1": macro_f1(preds, gts),
        "class_names": CLASS_NAMES,
        "confusion_matrix": confusion,  # rows = gt, cols = pred
    }


# ------------------------------------------------------------------------- runner

def resolve_video(args, name: str, *, source: str, video_id: str | None,
                  videos_dir: Path, work_dir: Path) -> Path:
    """Return a local mp4 path, downloading if needed."""
    if source == "youtube":
        if args.video:
            path = Path(args.video)
            if not path.exists():
                raise FileNotFoundError(f"--video not found: {path}")
            return path
        if not args.url:
            raise SystemExit("youtube source needs --url or --video.")
        return download_video(args.url, work_dir, video_id=name, cookies=args.cookies)

    # youcook2: prefer a cached mp4, else download from the annotation's video_url
    cached = videos_dir / f"{video_id}.mp4"
    if cached.exists():
        print(f"  using cached video: {cached}")
        return cached
    entry = load_entry(video_id, args.annotations_json)
    if entry is None:
        raise KeyError(f"{video_id} not in {args.annotations_json}")
    url = entry.get("video_url")
    if not url:
        raise ValueError(f"{video_id} has no video_url in annotations")
    return download_video(url, videos_dir, video_id=video_id, cookies=args.cookies)


def run_one(args, predictor: Predictor, transform, *, source: str,
            name: str, video_id: str | None, selected_ids: set) -> None:
    out_dir = Path(args.out_dir or (ROOT / "reports" / "inference")) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    videos_dir = Path(args.videos_dir)

    print(f"\n=== {name} ({source}) ===")
    if source == "youcook2" and video_id in selected_ids:
        print("  ⚠️ WARNING: this video_id is in the selected 300 (TRAINING data). "
              "Results will overstate generalization.")

    # 1) raw video
    video_path = resolve_video(args, name, source=source, video_id=video_id,
                               videos_dir=videos_dir, work_dir=out_dir)
    src_fps = get_video_fps(video_path)
    print(f"  video: {video_path}" + (f" ({src_fps:.1f} fps)" if src_fps else ""))

    # 2) extract frames (identical preprocessing to training)
    ok, result = extract_frames(video_path, frames_dir, fps=args.fps, resize_short=args.resize_short)
    if not ok:
        raise RuntimeError(f"frame extraction failed: {result}")
    print(f"  extracted {result} frames @ {args.fps} fps")

    # 3) load + 4) free-running inference
    loaded = load_frame_tensors(frames_dir, transform)
    print(f"  running Free-Running inference over {loaded['images'].shape[0]} frames ...")
    pred = predictor.run(loaded["images"], image_batch_size=args.image_batch_size)

    records = [
        {"frame_index": fi, "frame_name": fn, "timestamp_sec": round(ts, 4),
         "pred_label": pl, "pred_prob": round(pp, 6)}
        for fi, fn, ts, pl, pp in zip(
            loaded["frame_indices"], loaded["frame_names"], loaded["timestamps"],
            pred["pred_labels"], pred["pred_probs"])
    ]
    pred_df = pd.DataFrame(records)

    # 5) (youcook2) pseudo-GT + metrics
    has_gt = False
    if source == "youcook2":
        frames_meta = pred_df[["frame_index", "frame_name", "timestamp_sec"]]
        gt_df, gt_meta = build_youcook2_gt(
            video_id, frames_meta,
            annotations_json=args.annotations_json, prototypes_path=args.prototypes,
        )
        if len(gt_df):
            pred_df = pred_df.merge(
                gt_df[["frame_index", "gt_label", "gt_quality", "gt_weight", "is_boundary", "segment_id"]],
                on="frame_index", how="left",
            )
            covered = pred_df.dropna(subset=["gt_label"])
            metrics = compute_metrics(covered)
            metrics.update({k: gt_meta[k] for k in ("subset", "recipe_type", "n_segments",
                                                    "n_frames_total", "n_frames_covered")})
            metrics["coverage"] = round(len(covered) / max(len(pred_df), 1), 4)
            metrics["note"] = "GT is auto prototype-embedding pseudo-labels (no human review)."
            (out_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
            has_gt = True
            print(f"  metrics: acc(w)={metrics['frame_accuracy_weighted']:.4f} "
                  f"acc={metrics['frame_accuracy']:.4f} macroF1={metrics['macro_f1']:.4f} "
                  f"(n={metrics['n_eval_frames']}, coverage={metrics['coverage']})")
        else:
            print("  no annotation-covered frames; skipping metrics.")

    # 6) write outputs
    pred_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    (out_dir / "segments.json").write_text(
        json.dumps(collapse_segments(records), indent=2, ensure_ascii=False), encoding="utf-8")
    plot_timeline(pred_df.to_dict("records"), out_dir / "timeline.png",
                  title=f"{name} — phase timeline", has_gt=has_gt)
    print(f"  wrote: {out_dir}/  (predictions.csv, segments.json, timeline.png"
          + (", metrics.json" if has_gt else "") + ")")

    # 7) cleanup
    if not args.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    predictor = Predictor.from_checkpoint(
        args.config, args.checkpoint, device=device, use_amp=not args.no_amp)
    transform = build_eval_transform()

    selected_ids = set()
    if SELECTED_IDS.exists():
        selected_ids = set(json.loads(SELECTED_IDS.read_text(encoding="utf-8"))["video_ids"])

    if args.source == "youtube":
        if args.video:
            name = args.name or Path(args.video).stem
        elif args.url:
            name = args.name or args.url.split("v=")[-1].split("&")[0].split("/")[-1]
        else:
            raise SystemExit("youtube source needs --url or --video.")
        run_one(args, predictor, transform, source="youtube",
                name=name, video_id=None, selected_ids=selected_ids)
    else:  # youcook2
        if args.video_list:
            ids = [l.strip() for l in Path(args.video_list).read_text(encoding="utf-8").splitlines()
                   if l.strip() and not l.startswith("#")]
        elif args.video_id:
            ids = [args.video_id]
        else:
            raise SystemExit("youcook2 source needs --video-id or --video-list.")
        for vid in ids:
            try:
                run_one(args, predictor, transform, source="youcook2",
                        name=vid, video_id=vid, selected_ids=selected_ids)
            except Exception as exc:
                print(f"  ERROR on {vid}: {exc}")


if __name__ == "__main__":
    main()
