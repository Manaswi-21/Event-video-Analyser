import argparse
import csv
import math
from pathlib import Path

import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = (
    "persons",
    "laptops",
    "monitors",
    "microphones",
    "chairs",
    "phones",
    "mean_confidence",
)


def load_training_data(csv_path):
    features = []
    labels = []

    with Path(csv_path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {*FEATURE_COLUMNS, "event_type"}
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Training CSV is missing required columns: {missing}")

        for line_number, row in enumerate(reader, start=2):
            try:
                values = [float(row[column]) for column in FEATURE_COLUMNS]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric feature on line {line_number}.") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite feature value on line {line_number}.")

            label = (row["event_type"] or "").strip().lower()
            if not label:
                raise ValueError(f"Missing event_type label on line {line_number}.")
            features.append(values)
            labels.append(label)

    if len(set(labels)) < 2:
        raise ValueError("Training data must contain at least two event_type classes.")
    return features, labels


def train_model(csv_path, output_path):
    features, labels = load_training_data(csv_path)
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=2000,
            random_state=42,
            early_stopping=True,
        ),
    )
    model.fit(features, labels)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(
        description="Train the event MLP from a labeled object-count CSV."
    )
    parser.add_argument("training_csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "event_classifier.joblib",
    )
    args = parser.parse_args()
    model_path = train_model(args.training_csv, args.output)
    print(f"Saved trained event classifier to {model_path}")


if __name__ == "__main__":
    main()
