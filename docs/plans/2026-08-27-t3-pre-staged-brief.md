# T3 Pre-Staged Brief — apply pos-1 fix + run m84

> **STATUS: Not yet run.** This file is a *pre-staged* prompt for a future
> subagent that fires the moment T2 (per-layer pos-1 bisection) lands with
> a clear hypothesis. The actual T2 report will refine the exact fix below;
> most of this brief is reusable scaffolding.

## T2 finding (read this section first when T2 reports)

T2's last transcript before close showed:
- Bisection identified **flash attention output at L9 pos-1 diverges (`cos=0.69`)** despite Q and K matching oracle to 5-6 decimals (`cos > 0.999`)
- Confirmed `attn_scale_one` trait IS set to 1 for gemma4
- But code path `(c->tr.attn_scale_one ? 1.0f : 1.0f / sqrtf((float)HD))` uses **outer `HD`** (256) for the scale denominator — not per-layer `HDl` (512 for full-attn layers)
- The L9 attn output divergence persists even with scale=1.0 set, suggesting the bug is NOT in the scale ternary per se but in the **flash kernel's online-softmax / output assembly** specifically (Q, K inputs are correct but the WARP-LEVEL output computation diverges)

**Most likely fix candidates** (apply the FIRST one that matches T2's precise hypothesis):

1. **Per-layer scale in flash call site**: the `c->tr.attn_scale_one ? 1.0f : 1.0f / sqrtf((float)HD)` line in `kernels/qwen2_cuda.cu` ~line 1140 (where k_flash_gqa is invoked). If gemma4 full-attn layers have `attn_scale_one=1`, the ternary should pick 1.0; if the bug is the scale ISN'T 1.0, then it's `HD` vs `HDl` per layer.

2. **Online-softmax bug in k_flash_gqa itself**: the kernel was unit-tested standalone at 7/7 in `tests/test_flash_multi.cu` and the unit test was for hd=512. But the unit test may not exercise all paths. Inspect the WARP-LEVEL output assembly: `for (int t = t0; t <= pos; ++t) { ... acc = acc * alpha + ex * vp[i]; ... }`. Look for an off-by-one or wrong lane index.

3. **Output writeback / store path**: the `oreg[t]` array may be written in wrong order vs oracle's head-concat convention. Oracle's `build_attn` for gemma4: `ggml_reshape_2d(out, n_embd_head*n_head, n_tokens)`. Engine: `Y[(size_t)row * T + col0 + t] = v;` with `row` from `blockIdx.x * blockDim.y + threadIdx.y` and head ordering implicit. Check if head-major vs token-major ordering is the bug.

## Brief template (paste into agent when firing)

```
You are the T3 agent from docs/plans/2026-08-27-t3-pre-staged-brief.md. Repo:
~/Storage/repos/nnfromscratch. EDIT ONLY `kernels/qwen2_cuda.cu` (NOT other files).

## Stage 1: read the latest bisection report
Read /tmp/hypothesis.txt (T2 wrote the precise fix here). Also read the
most recent subagent transcript at
/tmp/pi-subagents-1000/.../tasks/64f99fbd-4d08-457.output (last 200 lines)
to get the full context of the bisection.

## Stage 2: clean up T2's instrumentation
Before applying your fix, REVERT any TT_DUMP_POS1 instrumentation T2 may
have left in kernels/qwen2_cuda.cu. Verify with `git diff kernels/qwen2_cuda.cu`
that the only changes are YOUR fix (not T2's dumps). If T2's dumps remain
in the tree, this commit will be considered polluted.

## Stage 3: apply the minimal fix
The fix is named in /tmp/hypothesis.txt. Apply it. One file. Minimal
change. The diff should be SMALL (typically 1-5 lines).

## Stage 4: build + regress
make build/dump_logits build/oracle_logits clean
./scripts/verify.sh m61  # must stay green

## Stage 5: run m84
./scripts/verify.sh m84
Expected: at least 5/7 pass (median reduction).

## Stage 6: if not 7/7, iterate
Do NOT chain fixes. If 5/6 PASS, run the engine against the failing
prompts and capture which stage diverges using existing TT_DUMP_LAYER
infra. Apply the next-most-likely fix (or re-dispatch the bisection agent).
Each cycle is one fix.

## Stage 7: commit
M8 fix: <one-line description> — m84 X/7 PASS (was 0/7)
m61 must remain green in the same commit.

## Stage 8: report
Per-prompt table (top-1, median, result). Final commit hash. If fix
wasn't 7/7, what's the remaining gap and your best next-step hypothesis.

## Constraints
- DO NOT touch any other file. Specifically: src/, tests/, scripts/, docs/,
  examples/, Makefile are all owned by other ongoing work.
- DO NOT remove debug printfs not added by T2.
- DO NOT add new functions or refactor — minimal fix only.
- If your fix needs more than 5 lines of change, STOP and re-dispatch a
  bisection agent with your expanded hypothesis.
```
