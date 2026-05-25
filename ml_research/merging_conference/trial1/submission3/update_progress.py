import os

progress_content = """# Research Progress Log

## Phase 1: Foundation (Read & Formulate)

### Literature Synthesis
After reviewing the three papers in the `papers/` directory, we identify the following core themes:
1. **SyMerge (Paper 0):** Proposes that model merging should target positive task synergy rather than just avoiding interference. By adapting even a single task-specific layer (Single-Layer Adaptation, SLA) at test-time using self-labeling guided by expert predictions, SyMerge induces positive cross-task synergy.
2. **OrthoMerge (Paper 1):** Focuses on the geometry of model weights. Linear merging in Euclidean space destroys orthogonal weight properties (e.g., hyperspherical energy). OrthoMerge resolves this by decoupling weights into orthogonal components (using the Orthogonal Procrustes problem) and residuals, merging the orthogonal parts on the Riemannian manifold of the orthogonal group (via Lie algebra mapping and Cayley transform).
3. **SAIM (Paper 2):** Proposes Sharpness-Aware Isotropic Merging for continual learning. It uses a Sharpness-Aware Block Coordinate Descent (SA-BCD) optimizer during fine-tuning to find flatter minima, and an adaptive isotropic merging algorithm during merging to balance the singular value spectrum.

### Ten Novel Research Ideas on the Merging Theme

1. **SAM-SLA: Sharpness-Aware Single-Layer Adaptation**
   * **Concept:** Fine-tune task models with Sharpness-Aware Minimization (SAM). Since flat minima have wide low-loss valleys, they should serve as a stabilizer and amplifier for test-time Single-Layer Adaptation (SLA), leading to faster, more data-efficient, and more stable synergy optimization.
   * **Expected Results:** SAM-trained models require 5x fewer adaptation steps during SLA, generalize better, and achieve 2-3% higher multi-task merged accuracy compared to standard SGD.
   * **Expected Impact:** Demonstrates that flat minima drastically improve test-time parameter malleability and synergy in model merging.

2. **Ortho-SLA: Orthogonal Test-Time Adaptation on Lie Algebra**
   * **Concept:** Restrict test-time single-layer adaptation to the manifold of the orthogonal group. Adapt the layer's weights strictly within the Lie algebra so(d) using gradient descent, preserving hyperspherical energy during adaptation.
   * **Expected Results:** Better preservation of representation geometry under test-time shift compared to unconstrained SLA.
   * **Expected Impact:** Bridges Riemannian optimization and test-time model adaptation.

3. **Iso-SLA: Isotropic Single-Layer Adaptation via SVD**
   * **Concept:** Rather than adapting all weights of a layer during test-time synergy, adapt only the singular values (using SVD) of a single critical layer, keeping the singular vectors (representation directions) frozen.
   * **Expected Results:** High data-efficiency and low risk of catastrophic representation collapse or over-fitting during adaptation.
   * **Expected Impact:** Offers a highly parameter-efficient and stable test-time adaptation paradigm.

4. **SA-Ortho: Sharpness-Aware Orthogonal Merging**
   * **Concept:** Investigate the intersection of sharpness-aware optimization and orthogonal geometry. Fine-tune expert models with SAM, and merge them using OrthoMerge. Explore whether flatter minima inherently preserve orthogonal weight structures and reduce Procrustes residual distortion.
   * **Expected Results:** SAM-trained models exhibit lower Procrustes residual norm, and SA-Ortho achieves superior multi-task accuracy by combining flat loss landscapes with manifold-preserving merging.
   * **Expected Impact:** Establishes a fundamental link between loss landscape flatness and weight space orthogonality.

5. **SS-Iso: Self-Supervised Isotropic Merging at Test-Time**
   * **Concept:** Dynamically balance the singular value spectrum of merged models at test-time using self-supervised (unlabeled) objectives, removing the need for labeled validation data.
   * **Expected Results:** Robust, label-free scaling factor estimation that outperforms heuristic linear coefficients under test-time distribution shifts.
   * **Expected Impact:** Makes isotropic model merging viable for real-world, unlabeled deployment environments.

6. **Decoupled Low-Rank Task Synergy**
   * **Concept:** For parameter-efficient fine-tuning (PEFT), merge LoRA adapters by decoupling them into orthogonal and residual components, and adapt the residual component via self-supervised learning at test-time.
   * **Expected Results:** Higher merge quality for PEFT models with extremely low computation/storage overhead.
   * **Expected Impact:** Expands manifold merging and synergy concepts to lightweight parameter-efficient architectures.

7. **Curvature-Guided Single-Layer Selection for SLA**
   * **Concept:** Instead of manually selecting which single layer to adapt during merging, dynamically select the layer by computing the trace of the Fisher Information Matrix (FIM) or Hessian. Adapt the layer with the highest/lowest curvature.
   * **Expected Results:** More robust and consistent synergy across different model architectures and tasks compared to static layer selection.
   * **Expected Impact:** Automates the design choice of SyMerge and provides theoretical insights into layer-wise adaptability.

8. **Momentum-Preserving Manifold Merging for Continual Learning**
   * **Concept:** Merge models sequentially in continual learning by performing moving averages of weight matrices on the orthogonal group manifold, tracking training momentum trajectories in the Lie algebra.
   * **Expected Results:** Reduced catastrophic forgetting and representation drift over long task sequences.
   * **Expected Impact:** Provides a geometric foundation for merging-based continual learning.

9. **Contrastive Self-Labeling for Test-Time Synergy**
   * **Concept:** In SyMerge's test-time adaptation, replace simple pseudo-labeling with a contrastive self-supervised objective on the unlabeled test data to guide the single-layer adaptation, preventing representational collapse.
   * **Expected Results:** More stable and robust adaptation, especially when the expert model's confidence is low or noisy.
   * **Expected Impact:** Enhances the reliability of test-time adaptation in model merging.

10. **Fisher-Weighted Isotropic Merging (F-Iso)**
    * **Concept:** Weigh the singular value spectrum balancing in SAIM using the diagonal Fisher Information of the task-specific models to prioritize task-critical parameter directions.
    * **Expected Results:** Better task-specific accuracy retention by protecting critical representation directions.
    * **Expected Impact:** Infuses statistical parameter importance into geometric singular value balancing.

### Selection and Randomization
Using the Slurm Job ID (22158108) as a pseudo-random seed, we executed `select_idea.py`.
The chosen idea is: **Idea 4: SA-Ortho: Sharpness-Aware Orthogonal Merging**.

### Iteration and Refinement of the Chosen Idea
To improve the novelty, feasibility, and impact of **SA-Ortho**, we refine our hypothesis as follows:
- **Prior Work Context:** SAFT-Merge (2025) and SAIM (2026) show that training models to lie in flat minima improves their linear mergeability. OrthoMerge (2026) shows that standard Euclidean merging destroys the geometric structure (orthogonality and hyperspherical energy) of fine-tuned weight matrices, and proposes merging orthogonal components on the Lie group manifold.
- **Scientific Gap:** How does the flatness of the loss landscape (induced by SAM) interact with the geometric structure of weight matrices (orthogonal vs. residual)? Specifically:
  1. Does sharpness-aware optimization (SAM) during fine-tuning make weight matrices "more orthogonal" or reduce the residual error during Procrustes decoupling?
  2. Does merging SAM-trained models on the orthogonal manifold (OrthoMerge) yield superior linear mode connectivity and multi-task performance compared to both standard merging and standard-trained OrthoMerge?
- **Hypothesis:** Fine-tuning models with SAM restricts their adaptation trajectories to flatter regions which are more structurally coherent and closer to the orthogonal manifold. Consequently, when decoupling these models, the orthogonal components retain more of the essential task knowledge, and merging them on the orthogonal group manifold (SA-Ortho) dramatically reduces representation distortion, leading to state-of-the-art model merging performance.

### Committed Research Hypothesis and Rationale
- **Hypothesis (SA-Ortho):** Combining Sharpness-Aware Minimization (SAM) with Orthogonal Model Merging (OrthoMerge) provides a synergistic effect where flat loss landscapes minimize the structural distortion during orthogonal-residual decoupling, yielding a merged model with preserved feature geometry and superior multi-task performance.
- **Rationale:** Standard SGD/Adam fine-tuning updates parameters along chaotic, sharp trajectories that degrade the geometric properties (like hyperspherical energy and orthogonality) of pretrained weights. SAM constrains these updates to broad, flat valleys. We hypothesize that these flat updates are more geometrically aligned across tasks. When decoupled via the Orthogonal Procrustes problem, SAM-fine-tuned models will exhibit significantly smaller residual distortion, allowing OrthoMerge to capture and merge the task-specific capabilities on the Lie algebra manifold with unprecedented fidelity.
"""

with open("progress.md", "a") as f:
    f.write(progress_content)

print("progress.md updated successfully!")
