# Mock Review

## 1. Summary of the Paper
The paper deconstructs the interaction between weight-space model merging (specifically merging specialized low-rank QLoRA/LoRA adapters into dequantized base models) and Post-Training Quantization (PTQ), uncovering a "silent failure" called **Re-Quantization Silence** (or Re-Quantization Collapse). Under naive post-hoc re-quantization back to low-bit constraints (such as 4-bit INT4), the subtle, low-magnitude adapter updates are frequently rounded to zero by the coarse quantization grid, stripping the deployed model of its multi-task specialized capabilities.

To systematically audit this behavior, the authors propose a **Multi-Axial Re-Quantization Auditing (RQA)** framework and evaluate two proposed mitigations:
1. **Scale-Adaptive Weight Shifting (SAWS):** A data-free, closed-form task-vector boosting method that scales up adapter updates based on Frobenius norm ratios before merging, aligning activation scales via a closed-form projection factor at inference.
2. **Quantization-Aware Adapter Coefficient Search (QA-ACS):** A test-time optimization (TTA) method that optimizes layer-wise merging coefficients directly through the quantization operator using Straight-Through Estimators (STE).

Crucially, the paper stands out for its exceptional academic transparency and self-critical deconstruction:
- It exposes a **Representation Scale Preservation Dilemma** in SAWS, proving that mathematically exact scale-preservation at inference collapses pre-trained base representations, showing instead that SAWS succeeds via selective task-vector boosting.
- It exposes a **Risk of Entropy Collapse** in QA-ACS under aggressive noise, showing how unconstrained prediction entropy minimization on $N=16$ samples can cause the model to output a single class, and evaluates supervised or regularized variants to completely stabilize the adaptation.
- It uncovers a major **Quantization Granularity Bifurcation**, proving that the "Re-Quantization Silence" is a highly localized per-tensor artifact, while standard per-channel and group-wise deployment configurations are nearly lossless once pre-existing task-interference is isolated.
- It systematically isolates confounding variables using a **Double Quantization Noise format-shift audit**, a **DRAM-bandwidth physical CPU latency scaling analysis**, and an **Individual Expert Auditing control experiment** that decouples task interference from discretization noise.

---

## 2. Overall Recommendation
**Rating: 6 (Strong Accept)**  
*Justification:* This is an exemplary, technically flawless, and methodologically beautiful paper. It exposes a major blindspot in the model-merging literature—evaluating purely in high-precision—and replaces this abstraction with a rigorous, deployment-aware auditing framework. Rather than hiding the limitations of the proposed mitigations (SAWS and QA-ACS), the authors systematically deconstruct them with outstanding mathematical and empirical depth. The systematic isolation of confounding variables (e.g., format shifts, hardware cache fitting, and task interference) is a masterclass in research design. The paper is incredibly thorough, well-written, and ready for publication with only minor, easily addressable suggestions.

---

## 3. Key Strengths
- **Rigorous and Honest Methodology:** The level of intellectual honesty and self-criticism in this paper is a breath of fresh air. The "Representation Scale Preservation Dilemma" and the "Entropy Collapse" deconstructions are brilliant and elevate the paper from a standard "propose a new SOTA" paper to a high-value scientific audit.
- **Flawless Experimental Isolation:** The inclusion of Table 6 (Individual Expert Auditing) is highly significant. By direct quantization of unmerged experts, the authors empirically prove that per-channel quantization does *not* erase adapter updates, meaning low merged performance under per-channel grids is due entirely to pre-existing weight-space task interference. This successfully decouples representation conflicts from discretization noise.
- **Hardware-Level Grounding:** The physical CPU latency profiling and the deconstruction of cache-fitting vs. DRAM-bandwidth bottlenecks on a 128-core Xeon CPU is excellent. It explains why toy networks can hide co-existence latency while large LLMs require weight-space merging due to DRAM-loading bottleneck constraints.
- **Comprehensive Ablation Studies:** The Appendix contains highly detailed ablation analyses, including a SAWS sensitivity sweep (Section A.4), evaluation of supervised/regularized QA-ACS variants (Section A.5), and an empirical validation of the global vs. channel-wise SAWS geometry warp hypothesis (Section B).

---

## 4. Minor Suggestions for Improvement
While the paper is outstanding, addressing the following minor suggestions would make it completely flawless:

### 4.1. Harmonize Unmerged Expert Ceilings
There is a tiny discrepancy between the high-precision unmerged experts ceiling reported in Table 1 ($93.85\%$ mean accuracy) and the unmerged FP16 ceiling reported in Table 6 ($93.70\%$). This is likely due to separate random seeds or evaluation runs, but the authors should harmonize these numbers or add a brief footnote explaining the minor difference to avoid confusing readers.

### 4.2. Pilot LLM Scaling Results
While the authors provide excellent theoretical scaling arguments (Section 5.1 and Appendix C) and evaluate reconstruction error on the larger `vit_base` backbone (Table 2), adding a small pilot evaluation of SAWS or naive merging on a compact multi-billion parameter language model (such as Pythia-1B or LLaMA-1B) would make the LLM scaling claims completely unassailable. If computational resources are limited, a brief footnote clarifying that this pilot is currently running would suffice.

### 4.3. Expand on Group-Wise SAWS Compiler Fusion
In Section 3.2.3, the authors theoretically analyze group-wise (block-wise) quantization and outline how SAWS can be extended group-by-group. Expanding slightly on how the group-wise correction factors $c^l_{i,j}$ would be loaded into GPU registers/SRAM and executed within a custom dequantization Tensor Core kernel (e.g., in CUDA or Triton) would be highly valuable for deployment-oriented practitioners.

---

## 5. Summary of Ratings
- **Soundness: Excellent** (Highly rigorous math, beautiful derivations, and thorough empirical isolation of confounders).
- **Presentation: Excellent** (Impeccable narrative, clear formatting, informative tables, and effective figures).
- **Significance: Excellent** (Addresses a major deployment blindspot, and the "Quantization Granularity Bifurcation" is of high practical and scientific value).
- **Originality: Excellent** (Creative combination of existing techniques, coupled with highly original, self-critical deconstructive audits).
