import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
PROTOTYPES_PATH = ROOT / "configs" / "action_class_prototypes_v2.json"
ANNOTATIONS_PATH = ROOT / "processed" / "annotation_sentences.csv"
OUTPUT_PATH = ROOT / "reports" / "prototype_classification_demo_v2.csv"
DEFAULT_SAMPLE_SIZE = "200"
DEFAULT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Embed YouCook2 annotation sentences and classify them against "
            "the v2 action-class prototype vectors."
        )
    )
    parser.add_argument(
        "--sample-size",
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of annotation rows to sample, or 'all' for the full input.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used when --sample-size is a number.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="CSV output path for the classification demo.",
    )
    return parser.parse_args()


def load_prototype_config(path):
    with path.open(encoding="utf-8") as f:
        config = json.load(f)

    labels = config["labels"]
    prototype_sentences = config["prototype_sentences"]
    missing_labels = [label for label in labels if label not in prototype_sentences]
    if missing_labels:
        raise ValueError(f"Missing prototype sentences for labels: {missing_labels}")

    return config


def load_annotation_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    required_columns = {"video_id", "segment_id", "sentence"}
    missing_columns = required_columns.difference(rows[0].keys() if rows else [])
    if missing_columns:
        raise ValueError(f"Missing required annotation columns: {sorted(missing_columns)}")

    return [row for row in rows if row.get("sentence")]


def select_rows(rows, sample_size, seed):
    if sample_size.lower() == "all":
        return rows

    try:
        size = int(sample_size)
    except ValueError as exc:
        raise ValueError("--sample-size must be an integer or 'all'.") from exc

    if size < 1:
        raise ValueError("--sample-size must be positive, or use 'all'.")

    rng = random.Random(seed)
    return rng.sample(rows, min(size, len(rows)))


def mean_normalized_vector(vectors):
    matrix = np.asarray(vectors)
    mean_vector = matrix.mean(axis=0)
    norm = np.linalg.norm(mean_vector)
    return mean_vector / norm if norm else mean_vector


def build_class_vectors(model, labels, prototype_sentences):
    class_vectors = {}
    for label in labels:
        # Each prototype sentence is embedded first; the class vector is the
        # normalized average of all prototype embeddings for that class.
        embeddings = model.encode(
            prototype_sentences[label],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        class_vectors[label] = mean_normalized_vector(embeddings)
    return class_vectors


def classify_rows(model, rows, labels, class_vectors):
    sentences = [row["sentence"] for row in rows]
    sentence_embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    class_matrix = np.stack([class_vectors[label] for label in labels])

    output_rows = []
    for row, sentence_embedding in zip(rows, sentence_embeddings):
        # Embeddings are normalized, so dot product is cosine similarity.
        scores = class_matrix @ sentence_embedding
        order = np.argsort(scores)[::-1]
        top_idx = int(order[0])
        second_idx = int(order[1])
        top_score = float(scores[top_idx])
        second_score = float(scores[second_idx])

        output_rows.append(
            {
                "video_id": row["video_id"],
                "segment_id": row["segment_id"],
                "sentence": row["sentence"],
                "predicted_label": labels[top_idx],
                "top1_score": f"{top_score:.6f}",
                "second_label": labels[second_idx],
                "second_score": f"{second_score:.6f}",
                "margin": f"{top_score - second_score:.6f}",
            }
        )

    return output_rows


def write_csv(path, rows):
    fieldnames = [
        "video_id",
        "segment_id",
        "sentence",
        "predicted_label",
        "top1_score",
        "second_label",
        "second_score",
        "margin",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    prototype_config = load_prototype_config(PROTOTYPES_PATH)
    labels = prototype_config["labels"]
    model_name = prototype_config["embedding_model"]
    prototype_sentences = prototype_config["prototype_sentences"]

    annotation_rows = load_annotation_rows(ANNOTATIONS_PATH)
    selected_rows = select_rows(annotation_rows, args.sample_size, args.seed)

    model = SentenceTransformer(model_name)
    class_vectors = build_class_vectors(model, labels, prototype_sentences)
    output_rows = classify_rows(model, selected_rows, labels, class_vectors)
    output_path = Path(args.output)
    write_csv(output_path, output_rows)

    print(f"prototype_file={PROTOTYPES_PATH}")
    print(f"annotation_file={ANNOTATIONS_PATH}")
    print(f"embedding_model={model_name}")
    print(f"class_count={len(labels)}")
    print(f"input_sentence_count={len(annotation_rows)}")
    print(f"output_sentence_count={len(output_rows)}")
    print(f"wrote={output_path}")


if __name__ == "__main__":
    main()
