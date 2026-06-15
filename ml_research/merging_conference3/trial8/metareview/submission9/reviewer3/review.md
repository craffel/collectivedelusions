# Peer Review: Zero-Shot Calibration-Free Model Merging

## 1. Summary of the Paper
This paper investigates the problem of dynamic test-time model ensembling of Low-Rank Adaptation (LoRA) experts without requiring offline labeled calibration datasets. The authors propose two main unsupervised paradigms:
- **Zero-Shot Expert Entropy Routing (EER):** A direct routing scheme that runs early backbone layers, executes all expert adapters in parallel, and routes inputs to the expert with the minimum prediction entropy. A scale-invariant Normalized Shannon Entropy is introduced to handle heterogeneous class vocabulary sizes across experts.
- **Entropy-Pseudo-Labeled Online Centroid Adaptation (EPL-OCA):** An ensembling scheme that tracks task centroids on-the-fly using exponential moving averages (EMA) updated with EER's pseudo-labels, followed by single-pass dynamic activation-space blending (SPS).

The paper conducts evaluations in a synthetic 192-dimensional multi-task sandbox (representing orthogonal subspaces with isotropic noise) and on real 512-dimensional ResNet-18 embeddings. While the proposed methods perform well in the idealized synthetic sandbox (e.g., EER achieving 71.38%Joint Mean accuracy), they experience a severe collapse on real features due to out-of-distribution (OOD) expert overconfidence, which the authors term the "Entropy Calibration Discrepancy." To resolve this, they introduce Centroid-Gated Entropy Routing (CG-EER), a hybrid semi-supervised framework that re-introduces pre-computed offline task centroids ($|\mathcal{C}_k|=64$) to gate out overconfident experts. To address physical edge serving overhead under post-Layer-3 activation divergence, the authors propose **Amortized Pseudo-Labeling** ($N_{\text{amortize}} = 10$) and validate its efficiency using CPU wall-clock latency profiling and theoretical energy analyses.

---

## 2. Strengths
1. **Exceptional Intellectual Honesty and Transparency:** The authors display a highly commendable level of academic integrity by prominently documenting the total collapse of their primary unsupervised methods (EER and EPL-OCA) on real embeddings. Rather than obscuring these negative results, they detail the self-referential pseudo-label feedback loops and openly re-classify their only functional real-world method (CG-EER) as a hybrid semi-supervised framework.
2. **Deep Systems-ML Latency and Complexity Analysis:** The paper head-on addresses a critical, often ignored physical serving bottleneck: the post-activation divergence of LoRA adapters. The authors mathematically formulate the FLOP complexity ($0.25+0.75K$ passes) and validate their proposed **Amortized Pseudo-Labeling** mitigation via real single-core CPU latency benchmarks and edge energy-efficiency models, showing high practical utility.
3. **Extensive Experimental Ablations:** The empirical evaluation is incredibly thorough, featuring sensitivity sweeps over the Softmax temperature $\tau$, registry scaling ($K \in \{4, 8, 12\}$), warm-up window sizes $T_{\text{warmup}}$, and vocabulary size configurations.
4. **Normalized Shannon Entropy Formulation:** The introduction of Normalized Shannon Entropy to correct for vocabulary-size bias is mathematically elegant and ensures scale-invariance under heterogeneous expert registries.

---

## 3. Weaknesses

### A. Lack of Formal Mathematical Proofs and Stability Guarantees
The paper proposes a test-time adaptation and dynamic ensembling framework but provides **zero formal proofs or mathematical guarantees**. 
- There is no proof of convergence or stability analysis for the online centroid tracking algorithm (EPL-OCA, Eq. 9) under non-stationary streams. How does the momentum parameter $\beta$ mathematically affect the tracking error under linear or non-linear covariate shift?
- There are no theoretical conditions under which the minimum prediction entropy routing (EER) is guaranteed to select the correct expert. What are the required margins, classification boundaries, or Lipschitz constants of the expert classification heads to ensure correct routing?
- There are no theoretical bounds on how the ensembling error scales with the number of tasks $K$ or the representation space dimension $D$.
The methodology remains heavily dependent on heuristics and empirical trial-and-error.

### B. Qualitative Descriptions of Well-Known Phenomena Instead of Mathematical Derivation
The paper coins new terminology for well-documented geometric and empirical properties, presenting them as qualitative "paradoxes" rather than formalizing them mathematically:
- **"Representational Sparsity Paradox":** The observation that centroids jitter and degrade cosine similarity routing because class prototypes are orthogonal in high-dimensional space is a basic property of high-dimensional geometry that can be easily formalized. For example, if a task contains $C$ orthogonal class prototypes $v_1, \dots, v_C \in \mathbb{R}^d$ of unit norm, and the task centroid is their average $\mu = \frac{1}{C} \sum_{c=1}^C v_c$, then:
  $$\|\mu\|_2 = \sqrt{\frac{1}{C} \sum_{c=1}^C \|v_c\|_2^2} = \sqrt{\frac{C}{C^2}} = \frac{1}{\sqrt{C}}$$
  The cosine similarity between any in-distribution class prototype $v_j$ and the task centroid $\mu$ is:
  $$\text{cos\_sim}(v_j, \mu) = \frac{\langle v_j, \mu \rangle}{\|v_j\|_2 \|\mu\|_2} = \frac{1/C}{1 \cdot (1/\sqrt{C})} = \frac{1}{\sqrt{C}}$$
  As $C$ increases, the similarity between any valid sample and its task centroid decays as $O(1/\sqrt{C})$, rendering centroid-based cosine similarity highly sensitive to noise. By presenting this as an empirical "Paradox" rather than deriving this straightforward geometric result, the paper misses an opportunity to provide rigorous theoretical value.
- **"Entropy Calibration Discrepancy":** The observation that simpler experts exhibit overconfidence on out-of-distribution (OOD) data is a well-established calibration failure in the literature. The paper renames this empirical fact but fails to provide a theoretical analysis, mathematical formalization, or structural explanation of why these linear heads output low entropy on OOD representations.

### C. Methodological Collapse and Contradiction of Core "Calibration-Free" Claim
The paper's core thesis is to achieve dynamic model ensembling completely unsupervised and "calibration-free." However:
- On real-world ResNet-18 embeddings, the proposed calibration-free methods **completely collapse**: EER falls to **35.38%**, and EPL-OCA Soft yields **31.52%**, failing to statistically outperform static Uniform Weight Merging (31.66%). 
- To resolve this collapse, the authors introduce CG-EER, which achieves **61.50%** accuracy. However, CG-EER is **not calibration-free**—it relies on pre-computed offline task centroids from labeled calibration splits ($|\mathcal{C}_k|=64$), the *exact same* requirement that SOTA SPS-ZCA has, which the paper heavily critiques in the introduction.
- Attempting fully unsupervised gating (UCG-EER) crashes back down to **28.45%** across all warm-up windows due to self-referential pseudo-label corruption.
This represents a major methodological gap: **the proposed calibration-free paradigms are non-functional on real features.** The only successful method on real features violates the core "calibration-free" claim of the paper.

### D. Overlapping Namespace Evaluation Bias
In both the synthetic sandbox and the ResNet-18 experiments, all tasks share the exact same class labels $\{0, \dots, 9\}$. Because sample-level accuracy is evaluated via `pred == y`, incorrect routing can still lead to "false correct" predictions. The authors acknowledge this introduces an optimistic background chance bias of $\approx 10\%$, making the reported absolute accuracies mathematically unreliable. This experimental design flaw could have been easily resolved by mapping tasks to disjoint label spaces (e.g., $Y_k \in [10k, 10k+9]$ for task $k$).

### E. Ad-Hoc Systems-ML Assumptions
The systems FLOP complexity equations (Eq. 10 and 11) assume that Layers 4 to 12 make up exactly $75\%$ of the network compute ($0.25+0.75K$ passes). This is a highly specific, model-dependent assumption that holds only for their 12-layer ViT configuration and does not represent a generalizable mathematical formulation of systems complexity.

---

## 4. Questions for the Authors
1. **Mathematical Guarantees:** Can you provide a formal proof of convergence or stability bounds for the online centroid adaptation (EPL-OCA) under non-stationary streams, specifically analyzing the impact of the momentum parameter $\beta$?
2. **Entropy Routing Analysis:** What are the mathematical conditions on the expert classification boundaries (e.g., margins, weight norms, or Lipschitz continuity) that guarantee EER selects the correct expert under a given noise level?
3. **Resolving Overlapping Namespace:** Why did you not map tasks to disjoint label spaces (e.g., $[0..9], [10..19]$, etc.) to eliminate the 10% optimistic evaluation bias and make the absolute accuracy scores mathematically rigorous?
4. **Unsupervised Calibration:** Since the Entropy Calibration Discrepancy causes EER to fail on real features, did you consider exploring theoretical, fully unsupervised test-time calibration techniques (such as temperature scaling based on representation-space density estimation or margin-based scaling) to break the self-referential loop without reverting to offline labeled centroids?

---

## 5. Detailed Ratings

- **Soundness:** **Poor / Fair** (No formal proofs or stability guarantees are provided. The "paradoxes" are qualitative descriptions of standard high-dimensional geometry and calibration phenomena that are left unformalized. The proposed calibration-free paradigms fail completely on real embeddings, and the successful fallback, CG-EER, re-introduces the offline calibration data requirement, invalidating the paper's core thesis.)
- **Presentation:** **Excellent** (The paper is beautifully written, highly structured, and displays an exemplary level of academic transparency and honesty regarding negative results and methodological limitations.)
- **Significance:** **Fair** (The theoretical significance is low due to the heuristic nature of the methods and lack of mathematical rigor. However, the systems-ML profiling, identification of the post-activation divergence bottleneck, and the development of Amortized Pseudo-Labeling have high practical significance for edge-device deep learning deployment.)
- **Originality:** **Fair** (The core routing mechanisms—entropy minimization and online centroid tracking—are standard, incremental heuristics repackaged into a LoRA ensembling framework.)

---

## 6. Overall Recommendation
**Score: 3 (Weak Reject)**

**Justification:** The paper has clear merits, particularly its exceptional writing quality, outstanding academic honesty in presenting negative results, and valuable systems-level profiling and latency mitigation (Amortized Pseudo-Labeling) for edge-device serving. 

However, from a theoretical perspective, the weaknesses overall outweigh the merits. The paper lacks any formal mathematical proofs, convergence guarantees, or error bounds for its online adaptation algorithms. Key high-dimensional geometric properties and calibration failures are described qualitatively with sensationalized terms ("Representational Sparsity Paradox" and "Entropy Calibration Discrepancy") rather than being mathematically analyzed or derived. Crucially, the proposed fully unsupervised, calibration-free methods completely collapse on real-world features, and the only functioning method (CG-EER) relies on offline labeled calibration data, fundamentally contradicting and invalidating the paper's primary "calibration-free" thesis. 

For a top-tier machine learning conference, the submission requires substantial revisions to elevate its mathematical rigor, formalize its qualitative claims, resolve the evaluation bias of overlapping namespaces, and explore theoretical unsupervised calibration techniques before it can be accepted.
