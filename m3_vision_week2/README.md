# M3 Vision — Week 2: KukaEnv Integration

**Phase 2 deliverable** — integrate the W1 visual pipeline with the KukaEnv scene
and produce a combined observation tensor ready for the RL team's PPO policy.

---

## What's new this week (Selma's feedback)

| Item | Status |
|---|---|
| Architecture | **ResNet18 (512) → locked** — used for KukaEnv features |
| Physics state | **(9,)** vector: 7 joint angles + EE (x, y) |
| Shortcut-learning mitigation | Random **colour jitter + crop augmentation** on every frame |
| Physics state dropout | **30%** chance per rollout to zero-out physics features |
| Final obs tensor | `512 + 25 + 9 = 546` (ResNet18 path) — *FPN ablation scheduled W3* |

---

## Pipeline overview

```
RGB (224×224×3)  ─────────────────────────────────────────────────────────────────┐
                                                                                  │
  ┌─ Step 1 ─┐   ┌─ Step 2 ─────┐   ┌─ Step 3 ──────────────────┐               │
  │ KukaEnv  │   │ Physics state │   │ ResNet18 + Augmentation   │               │
  │ render   │──▶│ (9,) — 7J+EE │   │ (512,) — anti-shortcut    │               │
  │ RGB/D/Seg│   └───────────────┘   └───────────────────────────┘               │
  └─────┬────┘                                                                    │
        │                                                                         │
  ┌─ Step 4 ─────────────────┐                                                   │
  │ Seg → bounding boxes     │                                                   │
  │ det_vec (25,) — 5 objects │                                                  │
  └──────────────────────────┘                                                   │
                                                                                  │
  ┌─ Step 5 ──────────────────────────────────────────┐                          │
  │ Physics dropout mask (Bernoulli 0.70)             │                          │
  │ Sampled ONCE per rollout; fixed for entire rollout │                         │
  └───────────────────────────────────────────────────┘                          │
                                                                                  │
  ┌─ Step 6 ──────────────────────────────────────────────────────────────────┐  │
  │  obs = cat([ vis_feat(512) | det_vec(25) | physics * mask(9) ]) → (546,)  │  │
  └───────────────────────────────────────────────────────────────────────────┘  │
                                                                                  │
  ──────────────────────────────────────────────────────────────────────────────▶ RL policy
```

---

## Physics dropout — design rationale

If the policy can always rely on exact joint angles + EE position, it has
no incentive to understand the visual scene.

**Strategy:**
- At each **rollout start**, sample `mask ~ Bernoulli(0.7)`:
  - `mask = 1.0` (70% of rollouts) → physics included
  - `mask = 0.0` (30% of rollouts) → physics zeroed out (vision-only)
- The mask is **fixed for the entire rollout** so the policy can adapt its
  strategy, rather than being surprised per-step.
- At **eval time**, `mask = 1.0` always — gives a clean, consistent metric.

---

## Repository layout

```
m3_vision_week2/
├── src/
│   ├── env_render.py       # Step 1 — PyBullet KukaEnv scene rendering
│   ├── physics_state.py    # Step 2 — (9,) physics state vector
│   ├── visual_features.py  # Step 3 — ResNet18 + anti-shortcut augmentation
│   ├── detection.py        # Step 4 — seg-mask → (25,) detection vector
│   └── observation.py      # Steps 5 & 6 — dropout mask + (546,) obs tensor
├── pipeline.py             # High-level convenience wrapper
├── smoke_test.py           # Quick sanity-check (shape assertions)
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
pip install -r requirements.txt
python smoke_test.py
```

---

## Phase 3 ablation plan (W5–W6)

| Ablation | Change | Hypothesis |
|---|---|---|
| **A: FPN** | Swap ResNet18 → Faster R-CNN FPN backbone | Detection-tuned features help? |
| **B: SigLIP** | Swap to SigLIP vision encoder | CLIP-style training helps language grounding? |
| **C: DINOv2** | Swap to DINOv2 ViT-S/14 | Self-supervised features generalise better? |
| **D: No physics** | Permanently zero physics state | How much does proprioception matter? |

All ablations share the same `build_observation()` interface — only the backbone changes.

---

## Observation tensor specification

| Component | Shape | Source | Notes |
|---|---|---|---|
| Visual features (ResNet18) | `(512,)` | RGB → augmented → backbone | L2-normalised |
| Detection vector | `(25,)` | Seg mask → bbox + depth | L2-normalised, 5 slots |
| Physics state (masked) | `(9,)` | Joint angles + EE (x,y) | Zeroed 30% of rollouts |
| **Observation** | **(546,)** | Concatenated | **RL policy input** |
