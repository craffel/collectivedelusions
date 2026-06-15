# Peer Review - review.md

## Summary of the Paper
The paper presents **Parameter-Free Task-Space Projection (PFSR)**, a training-free and data-free dynamic model ensembling router designed to bypass the complexity of modern over-parameterized learned routing networks. PFSR extracts representative task centroids from the classification heads of frozen specialists using Singular Value Decomposition (SVD), projects online feature representations onto these centroids, and computes ensembling coefficients using a temperature-scaled Softmax gating function. 

To resolve potential task cross-talk under active overlap, the authors also propose and analyze **Löwdin-Orthogonalized Task-Space Projection (OTSP)**, which applies Löwdin Symmetric Orthogonalization to the extracted centroids offline to construct an orthonormal task basis. 

The paper's core contributions are primarily mathematical and theoretical:
1. It proves that under symmetric task correlations, OTSP and PFSR are mathematically equivalent in their routing decisions and share the exact same Signal-to-Noise Ratio (SNR) under isotropic noise.
2. It mathematically deconstructs why OTSP systematically underperforms PFSR in asymmetric environments due to the **Noise Amplification Penalty** and **Noise Spillover Penalty** of the orthogonalized basis.
3. It characterizes ensembling pathologies such as **Vectorization Collapse** (under unconstrained linear routing with $B=1$), the **Orthogonal Masking Effect** (under disjoint sandboxes), and the **Gating Penalty** (under active overlap).
4. It proposes practical mitigations, including Top-$k$ Sparse Gating, Self-Calibrated Temperature Scheduling, and Offline Covariance Whitening for anisotropic features.

---

## Overall Assessment and Recommendation
**Recommendation:** **3: Weak Reject**

**Soundness Rating:** **Fair** (The mathematical proofs are correct and rigorous, but the experimental evaluation is highly limited, relying entirely on synthetic simulations, which compromises the empirical soundness of the claims.)

**Presentation Rating:** **Excellent** (The writing is remarkably polished, mathematically precise, and intellectually honest about its own limitations.)

**Significance Rating:** **Fair** (While the theoretical findings regarding Löwdin orthogonalization and noise amplification are highly elegant, the lack of genuine evaluation on real-world multi-task datasets or modern deep model backbones significantly restricts the immediate practical utility of the proposed method.)

**Originality Rating:** **Good** (Extracting SVD task centroids directly from classifier weights and adapting Löwdin orthogonalization to representation-space task projections is a novel and creative direction.)

---

## Major Strengths
1. **Intellectual Honesty and Transparency:** The authors are highly commended for their self-critical and transparent analysis. They do not attempt to hide negative results; instead, they explicitly highlight, analyze, and mathematically deconstruct the limitations of their own methods (e.g., the gating penalty under task overlap, the noise amplification of OTSP, and the failure of their dynamic routers to outperform Uniform Merging in classification accuracy).
2. **Rigorous Closed-Form Mathematical Analysis:** The theoretical derivations of Symmetric Equivalence (Section 3.7) and SNR Equivalence (Section 3.8) are exceptionally elegant and correct. They provide clear, closed-form explanations of coordinate-level routing dynamics and the trade-offs of orthogonalization under spherical noise.
3. **Relentless Occam's Razor Philosophy:** The paper's conceptual stance is highly refreshing. It challenges the standard paradigm of constructing increasingly complex, over-parameterized routing architectures by showing that a simple, training-free, closed-form linear projection can achieve excellent routing specificity.
4. **Actionable Mitigation Strategies:** The paper proposes practical extensions to bridge theoretical gaps, such as Top-$k$ Sparse Gating for inference efficiency, Self-Calibrated Temperature Scheduling to eliminate manual tuning, and Offline Covariance Whitening to handle anisotropic feature spaces.

---

## Major Weaknesses (Empirical and Methodological Concerns)

### 1. Significant Synthetic-to-Real Evaluation Gap
The most critical weakness of the paper is its **exclusive reliance on synthetic evaluations**. 
- Despite utilizing names like MNIST, FashionMNIST, CIFAR-10, and SVHN, the main "representation sandbox" is entirely simulated in a 192-dimensional space using synthetic Gaussian distributions. No actual images or features from real models trained on these datasets were used.
- The "Real-World Proof of Concept" on a ResNet-18 feature manifold is **also synthetic**. Instead of passing real images of dogs, cats, and vehicles through a pre-trained ResNet-18 to extract real penultimate feature representations, the authors merely took the static weight vectors of the final classifier layer and added Gaussian noise ($\sigma = 0.15$) to generate mock representations. 
Real deep learning feature manifolds are highly complex, anisotropic (narrow-cone), non-Gaussian, and subject to systematic biases. Evaluating solely on synthetic representations with added Gaussian noise fails to validate whether the proposed methods function reliably on real, complex image or text feature manifolds.

### 2. Failure to Outperform Static Uniform Merging in Classification Accuracy
Ultimately, the primary goal of model ensembling/merging is to maximize joint classification performance. However, **the proposed dynamic routers (PFSR and OTSP) fail to outperform the simplest static baseline—Uniform Merging—in joint classification accuracy on any evaluated setup**:
- In the primary primary sandbox (Table 1), joint classification accuracy is completely flat at 74.46% for all simplex-constrained methods.
- In the asymmetric sandbox (Table 3), static Uniform Merging achieves **80.83% $\pm$ 0.51%** classification accuracy, which slightly but consistently *outperforms* both PFSR and OTSP (**80.55% $\pm$ 0.54%**), despite the dynamic methods achieving much higher routing accuracy (70.76% vs. 25.00%).
While the authors argue that Uniform Merging benefits from "prediction-averaging" in overlap regions, this empirical result indicates that dynamic parameter-free routing provides zero actual performance gains (and a slight penalty) over a trivial, static baseline, undermining the practical motivation for deploying a dynamic sample-wise router.

### 3. Artificial Baseline Handicap via Extremely Small Calibration Splits
To evaluate parametric baselines (such as QWS-Merge and L3-Softmax), the authors restrict the calibration split size to an extremely small size of only 64 samples (16 samples per task). In this low-data regime, parametric classifiers are severely handicapped and guaranteed to overfit. 
To ensure a fair and rigorous comparison, the authors should have provided a sweep over calibration split sizes (e.g., from 64 to 2048 samples). Without this, the comparison is highly biased, and we cannot determine at what calibration data volume parametric models overcome overfitting and outperform the zero-parameter PFSR.

### 4. Mathematical Instability of SVD Centroids under Prototype Degeneracy
The authors assume that the top right-singular vector $V_{k,1}$ extracted from the expert weight matrix $W_k$ is unique and stable. However, in maximum-margin classifiers where class prototype vectors are highly symmetric or mutually orthogonal with equal norm (a common configuration in deep classifiers), the Gram matrix $W_k W_k^T$ has degenerate (identical) eigenvalues. 
Under degenerate eigenvalues, the top singular vector is **non-unique**, and the SVD calculation is highly sensitive to minor numerical perturbations. This mathematical edge case is unaddressed, and could lead to extreme centroid instability across different architectures, seeds, or floating-point precisions.

### 5. Lack of Source Code and Reproducibility Concerns
The authors do not release or link to any source code repository. Because the main experimental environments (the 10-seed simulation sandbox and the ResNet-18 simulation) are synthetic and custom-built, replicating the exact data generation process, baseline training protocols, and noise distributions is impossible without the source code. This represents a significant reproducibility concern for an empirical work.

---

## Detailed Questions and Actionable Suggestions for the Authors

1. **Can you evaluate your method on genuine, real-world feature representations?** Instead of generating mock representations by adding Gaussian noise to static classifier prototypes, please pass actual image samples (from ImageNet-1K or VTAB) through a pre-trained ResNet or Vision Transformer backbone, extract the penultimate feature representations, and route them using PFSR and OTSP. This is crucial to show that your SNR guarantees and covariance whitening operate successfully on real-world, anisotropic manifolds.
2. **Can you provide a sweep over the calibration split size for parametric baselines?** Please evaluate the performance of QWS-Merge, L3-Softmax, and LinearRouter as the calibration dataset grows from 64 to 256, 512, and 1024 samples, to find the crossover point where parametric models outperform the parameter-free projection.
3. **How do you address the mathematical non-uniqueness of the top singular vector under degenerate eigenvalues?** If the class prototypes in $W_k$ are perfectly orthogonal with equal norm, how do you guarantee that the SVD centroid is stable across different runs or hardware? Have you considered adding a small diagonal perturbation or utilizing a different centroid formulation in this regime?
4. **Where is the source code?** To ensure reproducibility, please release the implementation source code of your simulation sandboxes and baseline models.
