# Decode FlashAttention-2 Q8_0 Implementation Plan

> **REQUIRED SUB-SKILL:** Use the executing-plans skill to implement this plan task-by-task.

**Goal:** Integrate the FlashAttention-2 tiled Q8_0 split-K decode kernel (`k_fa2_q8_split` + `k_fa2_combine`) into `forward_layers()` in `kernels/qwen2_cuda.cu`. Eliminate long-context decode decay at $N \ge 2\text{k}$ tokens, lifting decode throughput from **$100\text{ tok/s} \to \mathbf{250\text{--}265\text{ tok/s}}$** on RTX 3050 (achieving $\mathbf{90\%+}$ parity with llama.cpp's $277\text{ tok/s}$ across the full context window).

---

## 1. Problem Diagnosis & Why the Prior Attempt Failed

### Previous Failure Points:
1. **Combine NaN / Inactive Slice Handling**: When $S = \text{ceil}(\text{max\_ctx}/32) = 320$ was launched, inactive slices ($s > \text{pos}/32$) wrote $m=-\infty, l=0$. The combine kernel failed to strictly filter out $l=0$ slices, resulting in $0/0 = \text{NaN}$ in softmax normalization and premature generation termination (stopping after 1 token).
2. **Redundant Serial Fallback**: A leftover serial `k_flash_gqa_q8_0` was executed after `k_fa2_combine`, overwriting output, doubling runtime, and corrupting CUDA graph capture replay.
3. **Graph Capture Invariant**: Split-K slice count $S$ must be fixed or clamped per graph capture boundary to ensure deterministic node execution.

### Architectural Solution:
- **Clean Split-K Partitioning**: Dynamic split count $S = \text{clamp}((\text{pos} + 1 + 31) / 32, 1, 16)$ for eager execution, and fixed $S=16$ (or $S=32$) with explicit masking inside the combine kernel for graph capture.
- **Robust Combine Reduction**:
  ```cpp
  float m_global = -1e30f, l_global = 0.0f;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int s = 0; s < S; s++) {
      float m_s = p_m[s * n_heads + h];
      float l_s = p_l[s * n_heads + h];
      if (l_s <= 0.0f || !isfinite(m_s) || m_s <= -1e20f) continue; // skip inactive slices
      float m_new = fmaxf(m_global, m_s);
      float alpha_prev = expf(m_global - m_new);
      float alpha_s    = expf(m_s - m_new);
      l_global = l_global * alpha_prev + l_s * alpha_s;
      acc[0] = acc[0] * alpha_prev + p_acc[s * n_heads * head_dim + h * head_dim + lane * 4 + 0] * alpha_s;
      acc[1] = acc[1] * alpha_prev + p_acc[s * n_heads * head_dim + h * head_dim + lane * 4 + 1] * alpha_s;
      acc[2] = acc[2] * alpha_prev + p_acc[s * n_heads * head_dim + h * head_dim + lane * 4 + 2] * alpha_s;
      acc[3] = acc[3] * alpha_prev + p_acc[s * n_heads * head_dim + h * head_dim + lane * 4 + 3] * alpha_s;
      m_global = m_new;
  }
  float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
  out[h * head_dim + lane * 4 + 0] = acc[0] * inv_l;
  out[h * head_dim + lane * 4 + 1] = acc[1] * inv_l;
  out[h * head_dim + lane * 4 + 2] = acc[2] * inv_l;
  out[h * head_dim + lane * 4 + 3] = acc[3] * inv_l;
  ```
- **Single Source of Truth**: Remove all fallback launches. Output is written directly to `d_att` with zero overwrites.

---

## 2. Implementation Tasks

### Task 1: Microbenchmark & Standalone Validation (`tools/micro_fa2_decode_q8.cu`)
**Goal**: Verify standalone `k_fa2_q8_split` + `k_fa2_combine` against reference `k_flash_gqa_q8_0` across context lengths $N \in \{32, 128, 512, 1024, 2048, 4096, 8192\}$.
- **Files**: Create `tools/micro_fa2_decode_q8.cu`
- **Criteria**:
  - `max_abs_error < 1e-4` across all $N$.
  - Speedup $\ge 5\times$ at $N=2048$ ($<0.02\text{ ms/layer}$ vs $0.14\text{ ms/layer}$ serial).
  - Clean compilation on sm_86 with 0 NaN/Inf.

### Task 2: Engine Workspace Allocation (`kernels/qwen2_cuda.cu`)
**Goal**: Ensure workspace buffers `d_split_pacc`, `d_split_pm`, and `d_split_pl` are allocated for $S_{\max} = 32$ slices ($32 \times 14 \times 128 \times 4\text{ B} \approx 230\text{ KB}$).
- **Files**: Modify `kernels/qwen2_cuda.cu` in `qwen2_engine_create` and `qwen2_engine_free`.

### Task 3: Engine Integration in `forward_layers()`
**Goal**: Wire `k_fa2_q8_split` + `k_fa2_combine` into single-token decode in `forward_layers()`.
- **Files**: Modify `kernels/qwen2_cuda.cu` (replace the serial `k_flash_gqa_q8_0` dispatch in `forward_layers()` when `use_q8_kvcache` is enabled).
- **Checks**:
  - Support sliding window attention (`swa_l`).
  - No fallback serial kernel.
  - Compatible with CUDA graph capture replay.

### Task 4: Verification & Benchmarks
**Goal**: Pass all gates and benchmark decode rate across context lengths.
- **Commands**:
  - `./scripts/ci_local.sh` (GREEN)
  - `./scripts/verify.sh m61` (7/7 logits parity + multiturn chat PASS)
  - `./scripts/verify.sh ple` (3/3 PASS)
  - Benchmark decode at $N \in \{30, 512, 1558, 3081, 7209\}$ using `run_llm_gpu`.
- **Success Target**:
  - Decode at $N=1558$: $\ge 240\text{ tok/s}$ (was $111\text{ tok/s}$).
  - Decode at $N=7209$: $\ge 220\text{ tok/s}$ (was $64\text{ tok/s}$).
  - Short-context decode: $\ge 260\text{ tok/s}$.

---

## 3. Success Metrics

| Context Length ($N$) | Current Decode (tok/s) | Target Decode (tok/s) | llama.cpp Parity Ratio |
|:---:|:---:|:---:|:---:|
| **30** | $269.0$ | **$269.0+$** | **$89\%$** |
| **512** | $120.0$ | **$255.0+$** | **$88\%$** |
| **1,558** | $111.4$ | **$250.0+$** | **$90\%$** |
| **3,081** | $100.0$ | **$245.0+$** | **$88\%$** |
| **7,209** | $64.4$ | **$220.0+$** | **$80\%+$** |
