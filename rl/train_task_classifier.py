"""
Trains an SVM classifier to predict instruction task_type from sentence
embeddings, replacing unreliable nearest-neighbor / raw keyword matching.

Background (see notebook.ipynb, Cell 9): the embedding space has a low
silhouette score (~0.05) and frequent cross-class collisions (e.g. "push"
vs "pull" instructions scoring 0.91 similarity) — nearest-neighbor lookup
is not reliable for task-type inference. A dedicated classifier trained
on the same embeddings hits ~93-94% cross-validated accuracy instead.

Saves:
  rl/spacial fusion/task_classifier.joblib  -- trained SVM
  rl/spacial fusion/task_classifier_labels.json -- label encoding order

Run: PYTHONPATH=. python3 rl/train_task_classifier.py
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import joblib

_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spacial fusion")
_CSV_PATH = os.path.join(_BASE_DIR, "nlp_instructions.csv")
_NPY_PATH = os.path.join(_BASE_DIR, "embeddings.npy")
_MODEL_OUT = os.path.join(_BASE_DIR, "task_classifier.joblib")
_LABELS_OUT = os.path.join(_BASE_DIR, "task_classifier_labels.json")


def main():
    df = pd.read_csv(_CSV_PATH)
    X = np.load(_NPY_PATH).astype(np.float32)
    y = df["type"].tolist()

    assert len(X) == len(y), f"embedding/label count mismatch: {len(X)} vs {len(y)}"

    labels_sorted = sorted(set(y))
    print(f"Training on {len(X)} examples, {len(labels_sorted)} classes: {labels_sorted}")

    # Cross-validated estimate (held-out folds) — sanity check against the
    # notebook's reported ~94.4% before committing to this as production.
    clf_cv = SVC(kernel="linear", probability=True, random_state=42)
    scores = cross_val_score(clf_cv, X, y, cv=5)
    print(f"Cross-val accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Final classifier trained on all available data for production use.
    clf = SVC(kernel="linear", probability=True, random_state=42)
    clf.fit(X, y)

    train_acc = clf.score(X, y)
    print(f"Full-data training accuracy: {train_acc:.4f}")

    joblib.dump(clf, _MODEL_OUT)
    with open(_LABELS_OUT, "w") as f:
        json.dump(labels_sorted, f)

    print(f"Saved classifier to {_MODEL_OUT}")
    print(f"Saved label list to {_LABELS_OUT}")


if __name__ == "__main__":
    main()
