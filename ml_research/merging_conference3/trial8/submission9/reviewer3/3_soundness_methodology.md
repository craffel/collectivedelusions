# Intermediate Review Evaluation 3: Soundness and Methodology

## Evaluation of Theoretical Grounding and Clarity
The paper is framed as a "comprehensive, technically rigorous, and honest systems-ML comparative study." However, from a theoretical perspective, the submission is **highly heuristic and lacks formal mathematical rigor**:

1. **Total Absence of Mathematical Proofs and Guarantees:**
   Despite introducing complex on-the-fly routing and online adaptation algorithms, the paper provides **no formal proofs, convergence guarantees, or error bounds**. Specifically:
   - There is no proof of convergence for the online centroid adaptation (EPL-OCA) under non-stationary streams (Eq. 9). The role of the momentum hyperparameter $\beta$ on stability and tracking error is left completely unanalyzed.
   - There are no theoretical conditions or mathematical guarantees under which the minimum prediction entropy routing (EER) is guaranteed to select the correct expert. What are the required margins, classification boundaries, or Lipschitz constants of the expert heads to ensure that $H(p_{k^*}(x_b)) < H(p_j(x_b))$ for all $j \ne k^*$?
   - There are no theoretical bounds on how the ensembling error scales with the number of tasks $K$ or the representation space dimension $D$.

2. **Qualitative Concepts Renamed as "Paradoxes" Without Mathematical Derivation:**
   The paper introduces two terms to explain empirical failures, but treats them as qualitative mysteries rather than deriving them mathematically:
   - **"Representational Sparsity Paradox":** This term describes the phenomenon where running centroids jitter and degrade cosine similarity routing due to class orthogonality within task subspaces. This is actually a straightforward result of high-dimensional geometry that can be easily formalized. For example, if a task contains $C$ orthogonal class prototypes $v_1, \dots, v_C \in \mathbb{R}^d$ of unit norm, and the task centroid is their average $\mu = \frac{1}{C} \sum_{c=1}^C v_c$, then:
     $$\|\mu\|_2 = \sqrt{\frac{1}{C} \sum_{c=1}^C \|v_c\|_2^2} = \sqrt{\frac{C}{C^2}} = \frac{1}{\sqrt{C}}$$
     The cosine similarity between any in-distribution class prototype $v_j$ and the task centroid $\mu$ is:
     $$\text{cos\_sim}(v_j, \mu) = \frac{\langle v_j, \mu \rangle}{\|v_j\|_2 \|\mu\|_2} = \frac{1/C}{1 \cdot (1/\sqrt{C})} = \frac{1}{\sqrt{C}}$$
     As the number of classes $C$ increases, the similarity between any valid sample and its task centroid decays as $O(1/\sqrt{C})$, making centroid-based cosine similarity highly sensitive to noise. The paper completely misses this simple, elegant geometric derivation, choosing instead to present the "Paradox" as an empirical surprise.
   - **"Entropy Calibration Discrepancy":** This term describes the empirical fact that simpler linear classification heads (like MNIST) output overconfident (low entropy) predictions on out-of-distribution (OOD) data. This is a well-known calibration failure of neural networks, typically tied to logit scales, weight norms, and optimization margins. The paper provides no theoretical analysis, mathematical formalization, or structural explanation of why these linear heads fail to calibrate, merely renaming the known issue and presenting a heuristic patch.

3. **Methodological Inconsistency in the "Calibration-Free" Premise:**
   The core thesis of the paper is the elimination of the labeled offline calibration dataset ($|\mathcal{C}_k|=64$) required by SPS-ZCA. However, when transitioning to real-world 512-dimensional ResNet-18 embeddings:
   - The proposed calibration-free methods **completely collapse**: EER falls to **35.38%**, EPL-OCA Hard falls to **27.45%**, and EPL-OCA Soft yields **31.52%** (failing to outperform static Uniform Weight Merging at 31.66%).
   - To resolve this, the authors introduce **Centroid-Gated Entropy Routing (CG-EER)**, which achieves **61.50%** accuracy. But CG-EER is **not calibration-free**—it relies on the *exact same* pre-computed offline task centroids from labeled calibration splits ($|\mathcal{C}_k|=64$) as SOTA (SPS-ZCA).
   - When the authors attempt a fully unsupervised, calibration-free version (UCG-EER), it crashes back down to **28.45%** across all warm-up windows due to self-referential pseudo-label corruption.
   
   This reveals a fatal methodological gap: **the proposed calibration-free paradigms are non-functional on real features.** The only method that works on real features (CG-EER) violates the paper's core premise of being calibration-free.

4. **Ad-Hoc Systems-ML Simplifications:**
   The physical serving complexity analysis (Eq. 10 and 11) is presented as a general mathematical formulation, but relies on highly ad-hoc, model-specific assumptions:
   - Assuming Layers 4 to 12 make up exactly $75\%$ of the network compute ($0.25 + 0.75K$) is highly specific to their 12-layer ViT model and does not generalize.
   - The energy efficiency analysis (DRAM vs SRAM costs of $50\ \text{pJ}$ vs $1\ \text{pJ}$) is a coarse-grained theoretical estimation that ignores specific edge memory hierarchies, instruction pipelining, and data-bus width limitations.

## Summary Rating: Soundness
- **Soundness Rating: Poor / Fair**
- **Justification:** The paper provides zero mathematical proofs or stability guarantees for its online algorithms, renames well-known high-dimensional and calibration phenomena as "paradoxes" without formal derivation, and relies on an offline calibration dataset (via CG-EER) to prevent catastrophic collapse on real-world embeddings, thereby invalidating its core "calibration-free" claim.
