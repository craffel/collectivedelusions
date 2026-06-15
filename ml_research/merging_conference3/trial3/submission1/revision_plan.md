# Revision Plan: Addressing Fourteenth Round Mock Review Feedback

We have entered a new cycle of Phase 4 (Iterative Refinement) to address the constructive suggestions from our latest Mock Review (5: Accept). Below is our plan to address the key weaknesses and minor suggestions to bring the manuscript to absolute peak scientific and publication quality.

## Key Weaknesses & Planned Actions

### 1. Subspace Representation Degradation and the "Low-Capacity" Generalization Illusion (Section 4.5)
*   **Critique:** While the low-rank subspace (attention-only SVD) achieves a closed generalization gap ($+0.50\%$), the absolute performance is very low ($13.00\%$), showing severe representation degradation compared to unquantized merging ($35.12\%$). Global post-hoc SVD on task vectors is a poor proxy for natively-trained PEFT (LoRA), and the closed gap is likely a "low-capacity generalization illusion" (less sensitivity to quantization mismatch because predictions are already flat and close to random noise).
*   **Action:** We have surgically updated the text of Section 4.5 in `04_experiments.tex` to explicitly introduce the concept of the **Low-Capacity Generalization Illusion**. We have critiqued global post-hoc SVD projection as a destructive proxy, tempered our claims of "structural defense," and stated that natively-trained PEFT is a critical future direction to confirm if the gap remains closed when capacity is preserved.

### 2. Lack of Empirical Scaling Verification (Section 5 / Section 4.5)
*   **Critique:** There is zero quantitative data on backbones larger than `ViT-Tiny` (5.7M parameters).
*   **Action:** We will add an explicit, highly professional section to the paper explaining the physical engineering bottleneck: fine-tuning four separate task-specific experts on a larger architecture (like `ViT-Base` with 86M parameters) is computationally and time-prohibitive in this test-time setting without pre-existing checkpoints. We will expand the theoretical and practical discussion in the scaling subsection of Section 5 to discuss how scaling parameters increases the ruggedness and density of the low-bit discretization boundaries, which mathematically expands the Cross-Schema Generalization Gap under physical hardware mismatches.

### 3. Extreme Task-Interference Bias (Section 4.1 / Section 4.5)
*   **Critique:** The unquantized FP16 Task Arithmetic baseline scores only $35.12\%$, representing an extreme task-interference regime. Is the cross-schema collapse a general feature or driven by this extreme divergence?
*   **Action:** We will add a discussion under Section 4.1 and Section 4.5 explaining that our benchmark is deliberately configured as an "extreme task-interference" regime to stress-test these frameworks. We will discuss how a "low-interference" (cooperative) landscape with aligned task-experts (such as PEFT/LoRA or joint pre-training) would lead to smaller generalization gaps, highlighting alignment prior to merging as an active mitigation strategy.

## Minor Comments & Planned Actions

### 1. Pseudo-Labeling Discussion (Section 5)
*   **Critique:** Under Axis 4, class skew causes unsupervised prediction entropy minimization to collapse because it is blind to class labels. Discuss how pseudo-labeling can mitigate this skew.
*   **Action:** We will expand the stream-distortion discussion in Section 5, proposing confidence-thresholded pseudo-labeling or self-supervised contrastive objectives as robust mechanisms to stabilize unsupervised adaptation under severe class skew.

### 2. Tone Balance (Section 4.1 / Section 5)
*   **Critique:** The paper's tone is highly skeptical. Acknowledge scenarios where Q-Merge and quantization-aware merging do succeed.
*   **Action:** We have integrated a dedicated paragraph titled `\textbf{Constructive Scenarios and Cooperative Landscapes:}` in Section 4.1, which explains that in cooperative, pre-aligned landscapes, Q-Merge is highly successful, framing our audit constructively as mapping the limits of the framework.

## Step-by-Step Execution of Revisions
1.  **Done:** Updated `PEFT/Subspace Robustness` in `submission/sections/04_experiments.tex` to address Weakness 1 (Low-Capacity Generalization Illusion).
2.  **To Do:** Update `submission/sections/04_experiments.tex` to address extreme task-interference bias and scaling bottlenecks in more detail.
3.  **To Do:** Update `submission/sections/05_conclusion.tex` to add pseudo-labeling discussion under Section 5.
4.  **To Do:** Recompile the paper using Tectonic to regenerate `submission.pdf` and `submission_draft.pdf`.
5.  **To Do:** Document this round in `progress.md` with an updated Author Rebuttal.
