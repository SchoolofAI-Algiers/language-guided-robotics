# Grasp Gap Diagnosis

**Issue:** #5 — diagnosis only, no fix.
**Fix (out of scope here):** #6 — automatic proximity-triggered grasp, preserving the
7-dim action space *(note: see Finding 3 below — the real trained checkpoint's action
head is actually 8-dim, including an explicit gripper action, which doesn't match this
framing; worth reconciling before scoping #6)*.

## Summary

This diagnosis confirms the bug described in the issue — `info["grasped_object"]` is
never populated, so grasp-dependent success can never be registered — **and** uncovers
a second, more fundamental bug upstream of it: the live serving pipeline never tells
the reward wrapper what task type an instruction is, so every instruction (PICK, LIFT,
PLACE, PUSH, PULL, LOWER, REACH) is currently scored as a plain REACH task, evaluated
purely by a 15cm proximity threshold. The grasp-key bug is real and confirmed, but in
the current deployed system it is masked by the task-type bug: grasp success is never
even checked, because the code path that would check it is never entered.

All findings below were confirmed by running the **actual trained Beta PPO checkpoint**
(`rl/spacial fusion/beta_policy.zip`, loaded via `rl.pipeline.get_model()`) through the
**actual production wrapper stack and code path** used by the live app
(`rl/pipeline.py::run_episode`, which `demo/backend/routes/instruction.py`'s
`/api/instruction` endpoint calls directly) — not a scripted stand-in.

## Finding 1: `RewardShapingWrapper` reads a key `KukaEnv.step()` never sets

`rl/reward_shaping.py`:
```python
grasped_id = info.get("grasped_object")
grasped_target = (grasped_id == self._target_obj_id and grasped_id is not None)
```
`robotics/env/src/environment.py`'s `KukaEnv.step()` only ever sets a key called
`"grasped_object_id"` in its returned info dict — never `"grasped_object"` (that key
only appears in `reset()`'s info, which `step()` never revisits). So
`info.get("grasped_object")` is structurally guaranteed to be `None` on every step,
independent of whether the arm is anywhere near the target or whether the low-level
PyBullet grasp constraint (`_try_grasp`) actually fired.

**Confirmed empirically:** across 4 real trained-policy episodes (`pick up the red
cylinder`, `lift the blue box`, `grab the yellow box`, `lower the green cylinder`),
`info["grasped_object"]` was `None` on every single logged step, no exceptions.

## Finding 2 (new): the live pipeline never calls `set_instruction()`, so every task is scored as REACH

`RewardShapingWrapper._task_type` defaults to `"reach"` in `__init__` and is only ever
changed by an explicit call to `set_instruction(text)`. Searching the codebase:

```
./rl/reward_shaping.py:27:    def set_instruction(self, text: str):
./rl/spacial fusion/pipeline.py:127:            _bg_env.set_instruction(instruction_text)
```

`set_instruction()` is called only from `rl/spacial fusion/pipeline.py` — an older
notebook-adjacent copy that is **not** the one the live app imports. The actual live
pipeline is `rl/pipeline.py`, confirmed by:

```
demo/backend/routes/instruction.py:8:from rl.pipeline import run_episode, find_best_embedding, get_env
demo/backend/routes/stream.py:7:from rl.pipeline import get_env, get_model
```

`rl/pipeline.py::run_episode()` calls `kuka_env.set_target_object()` and
`env.set_embedding()`, but never `set_instruction()`. So `_task_type` never leaves
`"reach"` in production, regardless of the instruction's actual verb.

**Confirmed empirically:** `"pick up the red cylinder"` and `"push the blue box"` both
returned `is_success: True` from the real `/api/instruction` code path, despite the arm
never grasping anything — both succeeded purely because the end-effector passed within
0.15m of the target (final distances 0.129m and 0.052m respectively), which is the
REACH success condition, not PICK's or PUSH's. Every episode's `info["task_type"]`
confirmed `"reach"` regardless of the requested instruction.

**Practical implication:** the impact table in the original issue assumed PLACE/PUSH/PULL
"secretly" succeed via proximity while PICK/LIFT are structurally blocked. Given Finding
2, this needs revision: currently **all** task types succeed or fail via the same 15cm
REACH threshold, including PICK and LIFT. The grasp-key bug in Finding 1 does not
currently affect production behavior at all, because the code path that would check
`grasped_target` (the `"pick"`/`"lift"` branches in `RewardShapingWrapper.step()`) is
never entered.

## Finding 3: the grasp mechanism itself works — when the code path is reachable

To test whether the underlying grasp mechanism is real (not "no mechanism at all," as
the issue's framing suggested), `_task_type` was forced directly on the wrapper
(bypassing the missing `set_instruction()` call) to `"pick"` and `"lift"`, and the real
trained policy was run for the full production step budget (`max_steps=500`).

- In one 500-step forced-`"pick"` trial (`"grab the yellow box"`), the raw env-level
  `grasped_object_id` became `6` — **matching the actual target object** — for 4
  consecutive steps (15–18), with distance as low as 0.145m:
  ```
  ENV-LEVEL GRASP FIRED at 4 step(s):
    step=15 distance_to_target=0.18226853013038635 grasped_object_id=6
    step=16 distance_to_target=0.1452036052942276 grasped_object_id=6
    step=17 distance_to_target=0.14463938772678375 grasped_object_id=6
    step=18 distance_to_target=0.17582060396671295 grasped_object_id=6
  ```
  Had `RewardShapingWrapper` read the correct key, `grasped_target` would have been
  `True` at this exact moment, and the PICK branch would have awarded success
  (`reward += 5.0`, `success = True`) on the spot.
- Two shorter (300-step) `"pick up the red cylinder"` trials with `_task_type` at its
  unmodified default (`"reach"`) also showed the raw grasp firing (`grasped_object_id=3`,
  matching the red cylinder target) at steps 12 and 17, even without task_type being
  forced — confirming the grasp constraint fires opportunistically whenever the
  end-effector and gripper command line up, independent of task type.

**This is the clearest evidence that a real grasp/gripper constraint mechanism exists
and functions correctly at the physics level** — contradicting the issue's framing that
there is "no grasp/gripper/constraint mechanism at all." The mechanism works; its
result is simply invisible to the reward code (Finding 1) and, currently, never even
queried for most instructions (Finding 2).

**However:** the checkpoint's policy network itself has an 8-unit action head
(`action_net: Linear(in_features=64, out_features=8)`), i.e. 7 joints + 1 explicit
gripper action — not the "7-dim, no explicit gripper" design the issue describes as the
target for #6. This is worth reconciling: either the currently-deployed checkpoint
predates a design change, or the #6 framing needs revisiting.

**Also worth noting:** in the majority of forced-task-type trials (5 of 6, including
both full-500-step trials), the policy did **not** get close enough to trigger a grasp
at all (final distances 0.37m–0.96m). Combined with the fact that grasp-related reward
bonuses could never have fired during this policy's training (since the same
`info["grasped_object"]` bug was presumably present then too), it's likely the trained
policy itself never learned reliable pick/lift approach behavior. **Fixing Findings 1
and 2 alone will likely not be sufficient — retraining will probably be needed once the
reward signal is correct.**

## Finding 4: `LOWER` is not recognized by either instruction parser

Two independent parsers exist and neither recognizes "lower":

- `RewardShapingWrapper.set_instruction()` (`rl/reward_shaping.py`) has branches for
  pick/grasp/grab, lift/raise, place/put, push, and pull, falling through to `"reach"`
  otherwise. No `"lower"` branch. (Moot in current production per Finding 2, since this
  method is never called — but would still need fixing once Finding 2 is addressed.)
- `_parse_command()` (`rl/pipeline.py`) only recognizes pick/grab/take/get,
  place/put/drop/set, and push, defaulting to `"reach"` otherwise — no "lift" or
  "lower" branch either. Confirmed this function only feeds the cosmetic
  `command_type` field shown in the frontend UI (`demo/frontend/src/lib/simulator.js`)
  and does not affect scoring — a separate, lower-severity, UI-only issue.

## Impact on the 340-instruction dataset (revised)

| Task type | Approx. count | Current as-run behavior |
|---|---|---|
| REACH | ~100 | Works correctly (this is also, inadvertently, what every other type currently reduces to) |
| PICK | ~80 | Scored as REACH (Finding 2); would additionally hit the grasp-key bug (Finding 1) if task-type routing were fixed |
| LIFT | ~40 | Scored as REACH (Finding 2); would additionally hit the grasp-key bug if task-type routing were fixed |
| PLACE / PUSH / PULL | ~105 | Scored as REACH — "secretly means got within 15cm" is correct, but for a different underlying reason (Finding 2) than originally assumed |
| LOWER | ~15 | Scored as REACH by both parsers (Finding 4) |

## Caveats

- The checkpoint used (`rl/spacial fusion/beta_policy.zip`) was recovered from `main`
  after the version in this branch was found to be a 2-byte Git LFS pointer stub; it
  was confirmed to be a valid, loadable Stable-Baselines3 PPO checkpoint before use.
- All step-level findings come from runs with the real trained policy in
  `deterministic=True` prediction mode, matching production (`rl/pipeline.py`).
- Sample sizes are modest (4–6 episodes per configuration) but sufficient to confirm
  the structural (code-level) claims in Findings 1, 2, and 4, which are deterministic
  properties of the code, not statistical ones. Finding 3's policy-competence
  observation (majority of approach attempts falling short) is more preliminary and
  would benefit from a larger sample if used for anything beyond flagging the risk.

## Scoping the fix (Issue #6)

This issue is diagnosis-only; no fix is applied here. Recommended fix scope for #6,
updated per the findings above:

1. **Call `RewardShapingWrapper.set_instruction()`** somewhere in `rl/pipeline.py::run_episode()`
   (Finding 2) — without this, no other fix here matters, since task_type never leaves `"reach"`.
2. **Fix the `info["grasped_object"]` / `info["grasped_object_id"]` key mismatch**
   (Finding 1) between `KukaEnv.step()` and `RewardShapingWrapper.step()`.
3. **Add a `"lower"` branch** to both parsers (Finding 4).
4. **Reconcile the action-space framing** — confirm whether #6 should target the
   existing 8-dim (7 joint + 1 explicit gripper) action space the current checkpoint
   already uses, or a genuinely new 7-dim/no-explicit-gripper design, since these are
   different designs with different retraining implications (Finding 3).
5. **Budget for retraining**, not just an environment/reward-code fix — the current
   policy likely never received a working grasp reward signal during its own training.

## Acceptance criteria check

- [x] Diagnosis independently confirmed with a logged episode, using the real trained
      Beta PPO policy through the real production code path.
- [x] Write-up committed (this document).
- [x] Fix direction for #6 scoped above, with an explicit note that the scope needs to
      expand beyond the original framing (Findings 2–3) — flagged to the team
      separately before finalizing, since it changes what #6 needs to cover.
- [x] No fix implemented in this issue.
