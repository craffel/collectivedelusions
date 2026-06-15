# Experimental Setup and Results Evaluation

This evaluation critical analyzes the experimental setup, datasets, baseline selection, and whether the empirical results substantiate the core claims and theoretical derivations of the paper.

---

## 1. Experimental Design and Sandbox Settings

The empirical evaluation of this paper is divided into two distinct parts: a highly controlled synthetic representation sandbox and a real-world deep neural manifold proof-of-concept. This combination is highly effective:

### A. Calibrated Representation Sandbox
* **Design**: The simulator generates representations in a $D=192$ feature space across $K=4$ tasks (representing MNIST, FashionMNIST, CIFAR-10, and SVHN) with task-specific noise scales ($\sigma \in [0.05, 1.95]$) and subspace-isolated noise.
* **Rigor**: Running experiments over **10 independent random seeds** (seeds 42 to 51) ensures statistical significance and provides robust confidence intervals (mean $\pm$ standard deviation).
* **Appropriateness**: While synthetic, the sandbox is a crucial and appropriate vehicle to study coordinate-level ensembling dynamics, noise scales, and small-sample inductive overfitting in a clean, isolated environment.

### B. Real-World Proof of Concept on the ResNet-18 Manifold
* **Design**: To bridge the real-world evaluation gap, the authors evaluate the methods on a real neural manifold. They extract actual class prototype vectors from the final classification layer of a pre-trained ResNet-18 model on ImageNet-1K.
* **Tasks**: They construct three semantic domains (Dogs, Cats, and Vehicles) and generate 1,250 real-world deep representation vectors by adding realistic feature-space noise ($\sigma = 0.15$) to the pre-trained prototypes.
* **Appropriateness**: This is an excellent, practical setup that validates the generalizability of SVD centroid extraction and closed-form orthonormal projection on a real deep learning manifold.

---

## 2. Selection and Quality of Baselines

The authors compare their methods against a highly comprehensive and fair set of baselines, covering both parameter-free and trained parametric approaches:
1. **Uniform Merging (Task Arithmetic)**: The standard, parameter-free static baseline.
2. **Naive Mean Centroid**: An important baseline ablation showing that a simple average of class prototype weights collapses (25.18% routing accuracy) due to sum-to-zero symmetry, which highlights the necessity of SVD centroid extraction.
3. **Parametric LinearRouter (Unreg)**: A trainable single-layer router trained on the $D$-dimensional representations of a 64-sample calibration split via AdamW.
4. **QWS-Merge (SOTA Parametric)**: An over-parameterized wave-superposition routing layer.
5. **L3-Softmax (Unregularized)**: A trainable Softmax router with random weight initialization.
6. **L3-Softmax Well-Reg (Zero-Init)**: A trainable Softmax router with zero-initialization, demonstrating that zero-init acts as a powerful maximum-entropy prior.

*Evaluation*: The choice of baselines is outstanding. It is particularly commendable that the authors capacity-optimized the parametric baselines, training them directly on $D$-dimensional representations using supervised cross-entropy. This ensures that the parametric baselines represent the strongest possible parameterized routing performance.

---

## 3. Support for Claims and Theoretical Insights

The empirical results perfectly align with and support every single theoretical claim and derivation:

* **SVD Centroid Superiority**: Table 1 shows PFSR and OTSP achieving perfect **100.00%** routing accuracy in the disjoint setup, while Naive Mean Centroid collapses to **25.18%**. This directly supports the claim that SVD successfully captures the principal direction of maximum variance.
* **Vectorization Collapse & Normalization**: Table 1 and Figure 1 show the unconstrained LinearRouter crashing to **55.57%** joint accuracy under sample-wise vectorized streaming ($B=1$), while all simplex-constrained methods remain robustly at **74.46%**. This validates the pedagogical claim that a probability-simplex constraint is mathematically necessary for streaming stability.
* **Orthogonal Masking Effect**: In Table 1, the joint classification accuracy is completely flat at **74.46%** across Uniform, QWS-Merge, L3-Softmax, PFSR, and OTSP. The authors correctly explain this via the Orthogonal Masking Effect in disjoint sandboxes, establishing routing accuracy as the only sensitive metric.
* **Noise Amplification & Spillover Penalties**: Table 2 presents a dense sweep under asymmetric task overlap and varying noise scales. It shows OTSP systematically underperforming PFSR by **0.2% to 1.6%** across all configurations, empirically confirming the Noise Amplification Penalty derived in Section 3.6.
* **Implicit Regularization**: In Table 1, the Well-Reg (Zero-Init) Softmax router achieves **67.22%** routing accuracy, outperforming the unregularized version and supporting the claim that zero-initialization operates as a maximum-entropy prior.
* **Real-World Generalization**: Table 4 shows PFSR and OTSP achieving **92.00%** and **92.08%** routing accuracy on the ResNet-18 manifold, proving that the proposed zero-parameter approach generalizes to actual neural representations. OTSP's +0.08% gain shows that L{\"o}wdin orthogonalization successfully decouples the positive semantic overlap (0.1905) between Dogs and Cats.
* **Anisotropic Noise Mitigation**: In Section 4.5, covariance whitening recovers OTSP's routing accuracy from **77.10%** to **89.45%** under anisotropic noise, proving the effectiveness of the proposed offline spherization.

---

## 4. Scientific Honesty and Balanced Analysis

The paper stands out for its high scientific integrity and balanced discussion:
* **The Gating Penalty Gap**: The authors do not hide that under active overlap, hard-gating PFSR/OTSP can incur a performance gap relative to Uniform Merging. They transparently analyze this "Hard Gating Penalty" and show how softening the temperature $\tau$ (or using self-calibrated temperature scheduling) recovers performance and matches Uniform Merging.
* **The True Value of Dynamic Routing**: The authors honestly deconstruct their "Noise Isolation" hypothesis, showing that classification boundaries are robust to uniform weights, and clarify that the primary advantage of PFSR over Uniform Merging is **systems-level and operational**—enabling massive memory/latency savings by loading onlyselected experts in large-scale registries, rather than a classification accuracy boost.

Overall, the empirical evidence is exceptionally thorough, statistically sound, and thoroughly validates every claim. The authors' rigorous benchmarking and intellectual honesty in analyzing routing trade-offs represent the highest standards of scientific publication.
