# 3. Soundness and Methodology

## Evaluation of Theoretical Grounding
The primary weakness of this submission is its **lack of rigorous theoretical grounding and formal mathematical guarantees**. While the paper is methodologically systematic and empirical, it relies entirely on intuitive heuristics without establishing any formal proofs or mathematical limits. From a theoretical perspective, several key areas lack rigorous foundation:

### 1. The Orthogonal Noise Hypothesis
The authors hypothesize that low-magnitude updates represent "orthogonal parameter noise" or "uncorrelated background noise," while high-magnitude updates store task-specific specialization. 
- **Lack of Proof**: No formal mathematical modeling is provided to support this. The authors could have treated the parameter updates as high-dimensional random vectors under specific probability distributions (e.g., spherical or Gaussian) to derive mathematical bounds on their cosine similarity and prove that low-magnitude subsets asymptotically approach orthogonality. 
- **Heuristic Nature of Magnitude**: There is no theoretical proof that absolute parameter magnitude is the mathematically optimal metric for identifying non-essential updates. It is possible that small-magnitude updates in highly sensitive coordinate directions (with high curvature or Hessian eigenvalues) are far more critical to model performance than large-magnitude updates in flat directions. This curvature-magnitude trade-off is completely ignored theoretically.

### 2. Qualitative "Mathematical Surrogacy" to Diagonal Fisher Saliency
In Section 4.4 (item 7), the authors discuss a conceptual link between deterministic magnitude-based masking and first-order parameter-saliency baselines, like Fisher-Weighted Averaging:
- **No Formal Derivation**: The claim that "magnitude-based pruning acts as a zero-order surrogate to diagonal Fisher Saliency" is purely qualitative and speculative. To make this mathematically rigorous, the authors should have provided a formal proof showing that under specific assumptions (e.g., a quadratic Taylor approximation of the loss landscape or strongly convex objectives), magnitude-based pruning bounds or approximates the diagonal entries of the Fisher Information matrix or the Hessian of the loss. Without such a derivation, this connection remains a conceptual conjecture rather than a verified mathematical property.

### 3. Optimization Theory for Coordinate Search (Non-Uniform CS)
In Section 3.6.2, the authors relax the uniform hyperparameter assumption and propose a coordinate descent-style Coordinate Search (CS) algorithm over a $2T$-dimensional space.
- **No Convergence Analysis**: The optimization is performed over a non-convex, non-smooth, and discontinuous objective (validation classification accuracy). From an optimization standpoint, coordinate descent on such landscapes has no guarantees of convergence to a global or even local optimum, and it can easily get stuck in coordinate-wise saddles or local traps. No mathematical convergence bounds or theoretical analysis of the optimization error compared to the global joint optimum are provided.
- **Sequence Dependency**: Although the authors empirically show that the reverse-order search results in similar performance, they provide no theoretical analysis of the optimization trajectory or stability guarantees for the sequential update sequence.

### 4. Continuous Landscape Stabilization in Soft Masking
The authors introduce **Sigmoid-Gated Soft Masking (SG-TA-Soft)** in Section 3.7 to address the representational discontinuities of hard binary masking.
- **Lack of Smoothness Analysis**: The authors state that soft-gating "smoothes out weight-space boundaries and potentially stabilizes the multi-task landscape." However, they do not provide any mathematical analysis or proofs demonstrating how the landscape's Lipschitz continuity, gradient variance, or smoothness is affected by the temperature parameter $\beta$. There is no mathematical verification that soft-masking results in a more stable optimization landscape for hyperparameter calibration.

---

## Evaluation of Experimental Methodology and Scope
While the empirical evaluation is exceptionally thorough (comprising 5-seed statistical runs, multi-axial sweeps, validation size sweeps, and competitive baselines), the architectural scope represents a theoretical limitation:
- **Scale and Over-Parameterization**: The evaluation is restricted to a compact Vision Transformer (\texttt{vit\_tiny\_patch16\_224}, 5.7M parameters). In theoretical machine learning, the behavior of compact networks differs fundamentally from over-parameterized foundation models (e.g., LLMs with billions of parameters). In over-parameterized models, the parameter space possesses a vast null space, and fine-tuning updates reside in extremely low-rank manifolds. Therefore, the properties of task vectors (such as orthogonality, representation collapse, and spatial overlap) on a compact ViT-Tiny backbone may not generalize to large-scale foundation models. Evaluating only on a small-scale sandbox limits the theoretical utility of the findings.

---

## Reproducibility
The methodology is exceptionally transparent, and reproducibility is **excellent**:
- The mathematical formulations for all variants (GQ, LQ, Soft, Norm) are clearly defined in Section 3.
- All hyperparameters (learning rates, batch sizes,optimizers, keep-ratios $k$, and scaling factors $\alpha$) and training protocols are explicitly detailed in Section 4.1.
- The authors detail the exact number of model evaluations, validation pool sizes, and calibration runtimes, ensuring that other researchers can easily replicate the results.
