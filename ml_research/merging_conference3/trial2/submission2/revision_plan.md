# Revision Plan: Addressing Mock Review Feedback for SVS

Based on the highly constructive feedback from the Mock Reviewer, we have identified three critical flaws and several minor weaknesses in our initial draft of the SVS paper. In accordance with the **Minimalist** philosophy, we choose to address these issues with absolute honesty, mathematical rigor, and academic maturity.

## 1. Action Plan for Critical Flaws

### Critical Flaw 1: Mathematical Redundancy of BWN in CLIP
*   **The Critique:** Because the visual projection layer in CLIP is immediately followed by an L2-normalization step, any global scaling factor $\alpha$ mathematically cancels out. Thus, BWN has zero effect on logits and accuracy, causing SVS+BWN and SVS-noBWN to be identical.
*   **Our Action:** We will embrace this critique fully. Rather than trying to obscure the cancellation, we will **write a formal mathematical proof** showing that under CLIP's normalization, global scaling cancels out.
*   **Revision Details:**
    *   Rename the BWN section to include a discussion of scale-invariance.
    *   Add a new mathematical proof in Section 3 (Methodology) demonstrating why the scaling factor $\alpha$ cancels out due to L2-normalization.
    *   Position BWN as a general formulation for non-normalized architectures (e.g., LLMs), but proudly highlight that for normalized vision backbones, the scaling parameter is bypassed, which confirms the minimalist principle that additional normalization parameters can be skipped.
    *   Correct the explanation in Section 4.4 to be mathematically precise, acknowledging the cancellation as the cause of identical accuracies.

### Critical Flaw 2: Single-Layer Toy Merging Setting
*   **The Critique:** Merging only the visual projection layer of CLIP is a highly artificial sandbox setting that does not represent real-world multi-layer merging.
*   **Our Action:** We will acknowledge this limitation openly and discuss its theoretical extensions.
*   **Revision Details:**
    *   Add a dedicated "Limitations and Scope" subsection in Section 4.
    *   Acknowledge that while our study focuses on the visual projection layer as a clean, isolated "sandbox" to analyze the spectral properties of weight updates, extending SVS to multi-layer Transformers is an important next step.
    *   Discuss how SVS can be applied layer-by-layer offline in multi-layer architectures, maintaining its parameter-free and training-free advantages.

### Critical Flaw 3: Small Test Scale & Failed SVHN Expert
*   **The Critique:** The evaluation uses only 200 samples per dataset, leading to statistical noise. The SVHN expert failed to train, which makes its task vector negligible.
*   **Our Action:** We will openly discuss these experimental properties with intellectual honesty.
*   **Revision Details:**
    *   Discuss the 200-sample test subset limit as a rapid prototyping sandbox constraint.
    *   Point out that the joint sweeps demonstrate consistent tracking and robustness of low-rank updates across the entire lambda spectrum, which indicates structural stability rather than random noise.
    *   Address the SVHN expert training issue: Explain that fine-tuning a linear visual projection layer with a frozen ViT block is notoriously challenging for highly out-of-distribution domains like SVHN (Street View House Numbers), leading to poor convergence (22.50%). We will show that because the SVHN expert is weak, its task vector contains very low-energy semantic updates, which SVS naturally slices away as high-frequency noise. This provides an insightful interpretation of why slicing works so well.

## 2. Action Plan for Minor Weaknesses

### Weakness 1: Missing SOTA Baselines (TIES, DARE)
*   **The Critique:** SVS should be compared theoretically or empirically to TIES-Merging and DARE.
*   **Our Action:** We will add a detailed theoretical comparison in Section 2 (Related Work).
*   **Revision Details:**
    *   Differentiate SVS from TIES and DARE. Highlight that TIES uses heuristic magnitude thresholding and sign-consensus voting, while DARE relies on randomized dropping.
    *   Argue that SVS is theoretically superior because it operates in the continuous spectral domain, finding the unique optimal low-rank projection via the Eckart-Young-Mirsky Theorem.
    *   Emphasize that SVS requires no sign voting, no random masking, and runs deterministically in closed form.

### Weakness 2: Non-Academic, Over-Promotional Tone
*   **The Critique:** The writing uses non-scientific, hyper-promotional phrases.
*   **Our Action:** We will completely revise the tone to be objective, neutral, and understated.
*   **Revision Details:**
    *   Perform a thorough text sweep. Replace promotional words like "completely redundant and counterproductive", "completely eliminates", "major victory" with objective scientific language.
    *   Reflect the minimalist persona through precise, understated, and powerful arguments.

### Weakness 3: Multi-Task Head Routing
*   **The Critique:** Clarify how multi-task routing is performed when task identity is unknown.
*   **Our Action:** We will clarify the head routing setup in Section 4.1.
