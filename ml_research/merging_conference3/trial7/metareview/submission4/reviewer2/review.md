# Peer Review

## 1. Summary of the Paper
This paper presents a refreshing, training-free approach to sample-wise dynamic model ensembling and merging, guided by Occam’s razor: **Parameter-Free Task-Space Projection (PFSR)**. Rejecting the current trend of introducing complex, over-parameterized routing networks that require specialized calibration datasets and expensive optimization loops, the authors propose a closed-form linear projection. 

PFSR extracts task-specific centroids directly from the static classification heads of pre-trained specialist models using Singular Value Decomposition (SVD)—which successfully bypasses the sum-to-zero cancellation of class prototypes. At runtime, normalized penultimate representations are projected onto these centroids. Gating coefficients are computed via a temperature-scaled Softmax over the absolute projection values, allowing dynamic parameter merging in a single forward pass with zero training or calibration data.

To explore whether orthogonalizing task-space coordinates can eliminate routing cross-talk under active overlap, the authors propose an advanced extension: **L{\"o}wdin-Orthogonalized Task-Space Projection (OTSP)**. Using L{\"o}wdin Symmetric Orthogonalization, OTSP constructs an optimal, order-invariant orthonormal basis offline. 

Through a rigorous 10-seed simulation sandbox and a real-world ImageNet-1K ResNet-18 manifold proof-of-concept, the authors demonstrate that:
1. Under symmetric task correlations, OTSP and PFSR are mathematically equivalent, making orthogonalization redundant.
2. Under asymmetric overlaps and active noise, OTSP systematically underperforms PFSR due to the **Noise Amplification Penalty** (arising from ill-conditioned Gram overlap matrices) and the **Noise Spillover Penalty**.
3. Unconstrained linear routers suffer from "Vectorization Collapse" under vectorized streaming ($B=1$), highlighting the mathematical necessity of the probability-simplex constraint.
4. Disjoint sandboxes suffer from the "Orthogonal Masking Effect," making joint classification accuracy flat and establishing routing accuracy as the primary evaluation metric.
5. Simple zero-initialization of Softmax routing parameters acts as an implicit maximum-entropy prior that shields parametric routers from small-sample inductive overfitting.

---

## 2. Strengths (Conceptual Leaps & Originality)
* **A Bold, Paradigm-Shifting Conceptual Leap**: The paper stands out for its high originality and conceptual ambition. It takes a powerful stand against the growing complexity of dynamic routing. By proving that dynamic ensembling can be executed in closed form with zero trainable parameters and zero calibration data, it opens up a fresh, elegant, and highly efficient path for the model merging community.
* **Rigorous Mathematical Foundation and Proofs**: The theoretical depth of this paper is outstanding. The closed-form derivations and proofs in the appendix verifying the orthonormality of the L{\"o}wdin basis, its symmetric order-invariance, its mathematical equivalence to PFSR under symmetric layout, and the exact Signal-to-Noise Ratio (SNR) equivalence under isotropic noise are exceptionally solid and beautiful.
* **Deep Scientific and Intellectual Honesty**: Proving that the simpler PFSR is systematically more robust and accurate than the more complex OTSP due to the *Noise Amplification Penalty* is an incredibly refreshing result. This level of intellectual honesty—deconstructing the limitations of one's own complex proposed extension in favor of a simpler baseline—is a major, high-signal contribution that deepens our fundamental understanding of representation-space projections.
* **Proactive Non-Parametric Solutions to Edge Cases**: Rather than leaving edge cases as open vulnerabilities, the authors anticipate and resolve multiple complex real-world challenges, such as:
  - *Anisotropic Feature Noise*: Mitigated via offline **Covariance Whitening** (+12.35% routing accuracy gain).
  - *Cardinality Imbalance*: Solved via elegant coordinate scaling/rescaling.
  - *Systems-Level Execution efficiency*: Preserved via **Top-$k$ Sparse Gating**.
  - *Uncertainty Gating*: Controlled via sample-wise self-calibrated temperature scheduling.
* **High-Fidelity Evaluation**: Benchmarking across a 10-seed simulation sandbox and validating generalizability on an actual ImageNet ResNet-18 manifold (~92% routing accuracy) provides extremely robust empirical support for all claims.

---

## 3. Weaknesses (Areas for Improvement)
* **Large-Scale Model Merging Evaluation**: While the ResNet-18 feature manifold is a great real-world proof-of-concept, the framework has not yet been evaluated on actual large-scale model merging benchmarks with fine-tuned specialists or LLM LoRA adapters (e.g., GLUE or MMLU). Although the authors propose a highly promising theoretical pathway (**Data-Free Centroid Representation (DFCR)**) in Section 5.1 to extract virtual centroids from internal transformer layers, showing empirical results on these large-scale benchmarks would significantly elevate the paper's immediate practical impact.
* **Empirical Validation of Activation-Based Centroids**: The authors discuss alternative data-dependent activation-based centroids (Mean/SVD on activations, $K$-Means clustering) for intermediate layers or black-box experts. However, these formulations are not empirically evaluated in the results section. Including even a small toy experiment demonstrating the routing performance of activation-based centroids would strengthen this discussion.
* **Minor Formatting Improvements**: In Section 3.7, the phrase "sign(u'_1 - u'_2) = sign(u_1 - u_2)" is written inline. Using standard LaTeX formatting like `\mathrm{sign}` would improve visual consistency.

---

## 4. Evaluation Ratings

### Soundness: Excellent
The paper is technically flawless and mathematically rigorous. SVD centroid extraction, L{\"o}wdin symmetric orthogonalization, absolute value projection, covariance whitening, and Top-$k$ sparse gating are all mathematically correct and highly appropriate. The proofs are detailed, complete, and correct. Potential technical pitfalls (such as anisotropic noise and cardinality imbalance) are proactively identified and solved.

### Presentation: Excellent
The paper is exceptionally well-structured, clear, and easy to follow. The mathematical notation is clean and consistent. The tables present dense, multi-seed results in an accessible format, and the figure with error bars is highly informative and visually appealing.

### Significance: Excellent
The significance of this work is outstanding. It has the potential to steer the community away from over-engineered, over-parameterized routing networks toward simpler, closed-form linear algebraic solutions. From a systems perspective, the ability to perform zero-parameter dynamic routing enables massive savings in compute, memory, and latency by loading and executing only selected experts.

### Originality: Excellent
The paper offers high originality and deep conceptual novelty. Extracting task centroids via SVD of classification heads for zero-parameter routing is highly creative. The application of L{\"o}wdin orthogonalization and the rigorous closed-form deconstruction of its limits under isotropic and anisotropic noise represents an exceptionally fresh and ambitious contribution to representation learning.

---

## 5. Overall Recommendation

**Rating: 5 (Accept)**

**Justification**: This is an exemplary, high-quality submission that deserves to be widely read. It combines profound theoretical insights with robust empirical results, introducing an elegant, training-free paradigm for dynamic model merging that challenges prevailing over-parameterized trends. The paper’s rigorous mathematical derivations, proactive edge-case engineering (whitening, sparse gating, self-calibrated temperature), and exceptional intellectual honesty make it an outstanding contribution to the machine learning literature. It represents a major step forward for parameter-free representation learning and specialist ensembling.

---

## 6. Questions and Suggestions for the Authors
1. **Large-Scale Models**: Have you experimented with extracting virtual centroids from internal layers of transformer models (using your proposed DFCR pathway) and routing LLM adapters? If so, does the SVD of MLP down-projections or Query-Key-Value projections show similar stability under active representation noise?
2. **Activation-Based Centroids**: Can you provide any empirical routing results for the Mean-on-Activations or $K$-Means centroid formulations? It would be interesting to see how their performance compares to SVD-on-Weights on the ResNet-18 manifold.
3. **Anisotropic Whitening Calibration**: For the covariance whitening step, how sensitive is the whitening matrix $\hat{\Sigma}^{-1/2}$ to the size of the calibration split $N_{\text{cal}}$? Would a tiny split (e.g., 32 samples total) be sufficient to capture the narrow-cone features and restore OTSP's coordinate stability?
