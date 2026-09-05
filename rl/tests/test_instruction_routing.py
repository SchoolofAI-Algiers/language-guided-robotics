"""
Unit test for issue #7: all 340 instructions in the NLP dataset must be
classified with the correct task_type by RewardShapingWrapper.set_instruction().

Run: PYTHONPATH=. python3 rl/tests/test_instruction_routing.py
"""
import csv
import os
import sys

sys.path.insert(0, ".")

from rl.reward_shaping import RewardShapingWrapper

_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "spacial fusion", "nlp_instructions.csv"
)


def test_all_instructions_classified_correctly():
    wrapper = RewardShapingWrapper.__new__(RewardShapingWrapper)

    with open(_CSV_PATH) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 340, f"expected 340 instructions, found {len(rows)}"

    mismatches = []
    for row in rows:
        text = row["instruction"]
        expected = row["type"]
        wrapper.set_instruction(text)
        got = wrapper._task_type
        if got != expected:
            mismatches.append((text, expected, got))

    if mismatches:
        detail = "\n".join(f"  {t!r}: expected={e!r} got={g!r}" for t, e, g in mismatches)
        raise AssertionError(f"{len(mismatches)}/{len(rows)} instructions misclassified:\n{detail}")

    print(f"PASS: all {len(rows)} instructions classified correctly.")


if __name__ == "__main__":
    test_all_instructions_classified_correctly()
