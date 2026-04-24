import numpy as np
from env.src.environment import KukaEnv

from env.src.config import MAX_EPISODE_STEPS, JOINT_UPPER_LIMITS, JOINT_LOWER_LIMITS

def test_creation():
    env = KukaEnv()
    assert env.action_space.shape == (7,)
    assert env.observation_space.shape == (21,)
    print("[PASS] Environment created with correct spaces")
    env.close()

def test_reset():
    env = KukaEnv()
    obs, info = env.reset()

    assert obs.shape == (21,)
    assert env.observation_space.contains(obs), f"Observation out of bounds: {obs}"

    joint_positions = obs[:7]
    assert np.allclose(joint_positions, 0.0, atol=0.01), (
        f"Joint positions not at home: {joint_positions}"
    )

    ee_pos = obs[14:17]
    assert abs(ee_pos[2] - 1.281) < 0.1, f"EE height unexpected: {ee_pos[2]}"

    print(f"[PASS] Reset: joint_pos~0, ee_pos={ee_pos}")
    env.close()

def test_step():
    env = KukaEnv()
    env.reset()

    action = np.array([0.0, 0.2, 0.0, -0.2, 0.0, 0.2, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (21,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert not terminated
    assert not truncated

    print(f"[PASS] Step returned valid 5-tuple, ee={obs[14:17]}")
    env.close()

def test_multiple_steps(n_steps: int = 10):
    env = KukaEnv()
    env.reset()

    # Normalized action -> real target for error comparison
    normalized = np.array([0.0, 0.2, 0.0, -0.2, 0.0, 0.2, 0.0], dtype=np.float32)
    real_target = (JOINT_UPPER_LIMITS + JOINT_LOWER_LIMITS) / 2.0 + \
                  normalized * (JOINT_UPPER_LIMITS - JOINT_LOWER_LIMITS) / 2.0
    for _ in range(n_steps):
        obs, *_ = env.step(normalized)

    error = np.abs(obs[:7] - real_target)
    assert np.all(error < 0.05), f"Joint position error too large: {error}"

    print(f"[PASS] After {n_steps} steps, max joint error: {error.max():.4f}")
    env.close()

def test_action_clipping():
    """Extreme actions (beyond [-1,1]) get clipped, joints stay in limits."""
    env = KukaEnv()
    env.reset()

    extreme_action = np.ones(7, dtype=np.float32) * 100.0  # way beyond [-1, 1]
    obs, *_ = env.step(extreme_action)

    joint_pos = obs[:7]
    within = (
        np.all(joint_pos <= JOINT_UPPER_LIMITS + 0.1)
        and np.all(joint_pos >= JOINT_LOWER_LIMITS - 0.1)
    )
    assert within, f"Joints exceeded limits: {joint_pos}"

    print("[PASS] Action clipping works, joints within limits")
    env.close()

def test_render_rgb_array():
    env = KukaEnv(render_mode="rgb_array")
    env.reset()

    img = env.render()
    assert img is not None
    assert img.shape == (480, 640, 3)
    assert img.dtype == np.uint8

    print(f"[PASS] rgb_array render: shape={img.shape}")
    env.close()

def test_gymnasium_check():
    from gymnasium.utils.env_checker import check_env

    env = KukaEnv()
    check_env(env, skip_render_check=True)
    print("[PASS] gymnasium check_env passed")
    env.close()

def test_reset_determinism():
    env = KukaEnv()
    obs1, _ = env.reset(seed=42)
    env.close()

    env = KukaEnv()
    obs2, _ = env.reset(seed=42)
    env.close()

    assert np.allclose(obs1, obs2, atol=1e-5), "Reset not deterministic with same seed"
    print("[PASS] Reset is deterministic with same seed")

def test_truncation():
    env = KukaEnv()
    env.reset()

    action = env.action_space.sample()
    truncated = False
    step_count = 0

    while not truncated:
        _, _, _, truncated, _ = env.step(action)
        step_count += 1

    assert step_count == MAX_EPISODE_STEPS, (
        f"Expected truncation at {MAX_EPISODE_STEPS}, got {step_count}"
    )
    print(f"[PASS] Episode truncated at step {step_count}")
    env.close()

if __name__ == "__main__":
    print("=" * 60)
    print("KukaEnv Verification Tests")
    print("=" * 60)

    tests = [
        test_creation,
        test_reset,
        test_step,
        test_multiple_steps,
        test_action_clipping,
        test_render_rgb_array,
        test_gymnasium_check,
        test_reset_determinism,
        test_truncation,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
