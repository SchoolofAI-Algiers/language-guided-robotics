"""
Generalization test for the SVM task-type classifier (see
rl/train_task_classifier.py). Unlike test_instruction_routing.py, which
tests exact matches against the known 340-instruction dataset, this test
exercises the classifier fallback path directly, using natural/casual
phrasings that do NOT appear in rl/spacial fusion/nlp_instructions.csv.

This is not expected to hit 100% -- the classifier's cross-validated
accuracy on the training data itself is ~94.7% (see notebook.ipynb, Cell
9), so some genuinely ambiguous novel phrasings will be misclassified.
This test documents the current real-world accuracy rather than enforcing
a strict pass/fail threshold, so it doesn't silently regress unnoticed.

Run: PYTHONPATH=. python3 rl/tests/test_classifier_generalization.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np
from sentence_transformers import SentenceTransformer

from rl.reward_shaping import RewardShapingWrapper

# Minimum acceptable accuracy on this held-out, non-CSV test set. Set below
# the current observed rate (10/12 ~= 83%) to leave headroom for a
# reasonable amount of variation, while still catching a real regression.
_MIN_ACCURACY = 0.70

TEST_CASES = [
    ("can you grab the red ball for me", "pick"),
    ("plz pick up the yello box", "pick"),
    ("gently put the yellow box down over there", "lower"),
    ("put the box down slowly", "lower"),
    ("nudge the green cylinder to the side", "push"),
    ("slide the blue box away from you", "push"),
    ("drag the box closer to you", "pull"),
    ("bring the sphere towards yourself", "pull"),
    ("raise the box up a bit", "lift"),
    ("can u lift the ball off the table", "lift"),
    ("go over to the yellow object", "move"),
    ("walk to the left corner", "move"),
]


def test_classifier_generalizes_to_novel_phrasing():
    wrapper = RewardShapingWrapper.__new__(RewardShapingWrapper)
    st = SentenceTransformer("all-MiniLM-L6-v2")

    results = []
    for text, expected in TEST_CASES:
        emb = st.encode([text], normalize_embeddings=True)[0].astype(np.float32)
        wrapper.set_instruction(text, embedding=emb)
        got = wrapper._task_type
        results.append((text, expected, got))

    correct = sum(1 for _, e, g in results if e == g)
    accuracy = correct / len(results)

    print(f"{correct}/{len(results)} correct ({accuracy:.1%}) on novel phrasings:")
    for text, expected, got in results:
        status = "OK  " if expected == got else "MISS"
        print(f"  [{status}] {text!r}: expected={expected!r} got={got!r}")

    assert accuracy >= _MIN_ACCURACY, (
        f"Classifier generalization accuracy {accuracy:.1%} dropped below "
        f"minimum threshold {_MIN_ACCURACY:.0%}"
    )
    print(f"\nPASS: accuracy {accuracy:.1%} >= threshold {_MIN_ACCURACY:.0%}")


if __name__ == "__main__":
    test_classifier_generalizes_to_novel_phrasing()
