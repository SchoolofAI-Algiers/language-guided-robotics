"""
Integration test for issue #7 (Finding 2): confirms rl/pipeline.py's
run_episode() actually calls RewardShapingWrapper.set_instruction(),
so _task_type reflects the real instruction instead of staying stuck
on the default "reach".

This loads the real trained model and background env, so it's slower
than the unit test. Run: PYTHONPATH=. python3 rl/tests/test_pipeline_integration.py
"""
import sys

sys.path.insert(0, ".")

import rl.pipeline as pipeline


def test_run_episode_calls_set_instruction():
    pipeline._load()
    bg_env = pipeline._bg_env

    # Force a wrong task_type first, so we know for certain any change
    # we see afterward came from run_episode() itself, not leftover state.
    bg_env._task_type = "reach"

    pipeline.run_episode("pick up the red cylinder", max_steps=5)

    assert bg_env._task_type == "pick", (
        f"run_episode() did not call set_instruction() correctly — "
        f"expected 'pick', got '{bg_env._task_type}'"
    )
    print("PASS: run_episode() calls set_instruction() and task_type propagates correctly.")


def test_run_episode_updates_task_type_for_different_verbs():
    pipeline._load()
    bg_env = pipeline._bg_env

    bg_env._task_type = "reach"
    pipeline.run_episode("push the blue box", max_steps=5)
    assert bg_env._task_type == "push", (
        f"expected 'push', got '{bg_env._task_type}'"
    )
    print("PASS: task_type correctly updates to 'push' for a different instruction.")


if __name__ == "__main__":
    test_run_episode_calls_set_instruction()
    test_run_episode_updates_task_type_for_different_verbs()
