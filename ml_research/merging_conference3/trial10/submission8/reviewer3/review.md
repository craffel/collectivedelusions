# Peer Review

## Summary of the Paper
The paper addresses the challenge of layer-wise adaptive weight-space ensembling for multi-task expert networks under few-shot calibration budgets. In this setting, optimizing independent layer-wise coefficients is highly susceptible to transductive overfitting. Prior approaches, such as Rademacher-Bounded Polynomial Merging (RBPM), parameterized the layer-wise trajectory using low-degree polynomials, but these suffer from severe boundary runaway (similar to Runge's phenomenon), which degrades low-level feature extraction and final classification performance.

To overcome these issues, this paper introduces **Rademacher-Bounded Fourier Trajectory Merging (RB-FTM)** and its non-periodic counterpart, **Rademacher-Bounded Discrete Cosine Trajectory Merging (RB-DCTM)**. By representing depth-wise ensembling coefficients as a continuous spectral series composed of low-frequency harmonic sinusoids (Fourier) or cosines (DCT), the trajectory is naturally bounded and smooth. The authors derive rigorous empirical Rademacher complexity bounds for both trigonometric classes over the fixed network depth coordinates, establishing a trajectory-space regularization framework that is completely independent of the underlying network's parameter count. To enforce this bound, they formulate a Spectral Lasso ($L_1$) penalty strictly on the harmonic coefficients. Furthermore, they demonstrate that the DCT variant (RB-DCTM) enforces homogeneous Neumann boundary conditions ($h'(0) = h'(1) = 0$), providing a natural stabilizing buffer at the boundary layers while granting boundary value independence. 

The proposed methods are evaluated inside a synthetic "Analytical Coordinate Sandbox" (ACS) and on a small-scale real-world proof-of-concept validation merging CIFAR-10 and CIFAR-100 expert Vision Transformers (ViT-B/16).

---

## Strengths and Weaknesses

### Strengths
1. **Elegant Conceptual Novelty**: The shift from polynomial trajectory parameterizations to harmonic spectral representations (Fourier and DCT bases) is a creative and elegant solution to the boundary runaway pathology. The mathematical realization of this transition is highly satisfying and conceptually clean.
2. **Homogeneous Neumann Boundary regularizer**: Utilizing a half-period cosine basis (DCT) to enforce flat derivatives at the boundary layers ($h'(0) = h'(1) = 0$) is a superb, physics-inspired design. It provides a natural architectural buffer that protects early feature extraction and final classification projections from rapid, destructive weight-space fluctuations during calibration.
3. **Strong Mathematical Rigor**: The theoretical foundation is complete and correct. The derivations of empirical Rademacher complexity bounds for both Fourier and DCT trajectories are flawless, providing exact constants. The authors are highly commended for their transparency in discussing composition-based generalization bounds in deep networks, trajectory-space vs. data-space complexity, and the task-scaling discrepancy.
4. **Smart Optimization Design (Spectral Lasso)**: Restricting the $L_1$ penalty strictly to the harmonic coefficients (excluding the baseline uniform term $a_0$) is a methodologically clever choice. It prevents activation scale collapse while enabling the trajectory to gracefully contract back to the robust Static Uniform baseline as the regularization strength $\gamma$ increases.
5. **Practical Empirical Success**: On actual deep networks (ViT-B/16 on CIFAR-10 and CIFAR-100), the proposed methods show clear improvements. RB-DCTM ($F=2$) achieves a joint average accuracy of **74.90%**, outperforming the Static Uniform baseline (+3.60%), Globally-Scaled Task Arithmetic (+2.40%), and its direct trajectory competitor RBPM (+4.20%). The degradation of RBPM ($70.70\%$) below the Static Uniform baseline ($71.30\%$) empirically validates the boundary runaway pathology of polynomial curves.

### Weaknesses
1. **Limited Scale of Real-World Evaluation**: While the theoretical and mathematical contributions are rich, the empirical validation on actual networks is quite narrow. Evaluating ensembling on only two task experts ($K=2$) on low-resolution toy datasets (CIFAR-10 and CIFAR-100) is insufficient to demonstrate the broad utility of the method. In modern weight-space ensembling, benchmarks routinely incorporate 5 to 10 experts or focus on ensembling Large Language Models (LLMs) across complex reasoning, math, and coding tasks.
2. **Questionable Utility of the Synthetic Sandbox (ACS)**: The majority of the paper's experiments are conducted in the ACS, a stylized linear recurrence system. In this sandbox, the parameter-free, zero-tuning **Static Uniform baseline consistently and significantly outperforms the proposed spectral trajectory methods** across all sweeps (even under coordinate rotation misalignment up to $\eta = 0.6$). While the authors transparently analyze this as the "Static Uniform Dominance Paradox" due to perfect coordinate alignment, it means the sandbox is of limited utility as a comparative benchmark for showing the benefits of adaptive merging.
3. **Under-disclosed Data Footprint**: The authors utilize ZipIt! permutation alignment to align hidden coordinates before optimization. This alignment requires estimating activation covariance matrices, which they disclose requires an unlabeled calibration dataset of 100 samples per task to prevent rank-deficiency. This dual-dataset footprint (10-shot labeled for trajectory optimization + 100-shot unlabeled for alignment) should be more clearly emphasized, as it increases the practical sample footprint of the method compared to unaligned ensembling baselines.
4. **Assumption of Continuous Smoothness**: Parameterizing coefficients as a continuous trajectory across depth assumes adjacent layers behave similarly. While this is valid for highly homogeneous residual blocks, it may fail in networks with heterogeneous structures or where downsampling/pooling layers require abrupt, discrete parameter transitions.

---

## Detailed Evaluation

### Soundness
**Rating: Excellent**  
The technical claims, theoretical proofs, and experimental methodologies are highly sound and mathematically rigorous. The authors' deep reflection on standard composition-based generalization bounds, and their derivation of a formal downstream generalization bound via covering numbers (establishing an explicit $\widetilde{\mathcal{O}}(1/\sqrt{N})$ decay rate), shows outstanding scientific rigor and academic honesty.

### Presentation
**Rating: Excellent**  
The paper is exceptionally well-written, with high clarity and excellent narrative flow. The authors explain complex learning-theoretic and functional analysis concepts beautifully. The scientific transparency regarding the limitations of standard bounds and the exact mechanics of their sandbox is highly commendable.

### Significance
**Rating: Good**  
The paper addresses a highly relevant and important problem (few-shot weight merging). However, its significance is currently limited by the scale of the real-world evaluation. For a method claiming such powerful structural regularization, demonstrating its efficacy on modern LLM or large-scale multi-task vision benchmarks is essential to prove broad practical impact.

### Originality
**Rating: Excellent**  
The paper introduces highly creative and original ideas. The shift to a spectral trajectory parameterization, the enforcement of Neumann boundary conditions via DCT cosine harmonics, and the rigorous joint multi-task complexity analysis represent significant conceptual leaps that advance the theoretical understanding of weight-space merging trajectories.

---

## Overall Recommendation

**Rating: 4 (Weak Accept)**  

**Justification**:  
This is a technically solid, exceptionally well-written paper that introduces a highly creative and mathematically rigorous spectral trajectory framework for model ensembling. By replacing polynomials with Fourier and DCT bases, it elegant resolves the boundary runaway pathology of continuous trajectory models and enforces physics-inspired Neumann boundary regularizers. The theoretical complexity proofs and the downstream prediction generalization bridge are flawless.

The primary limitation that prevents a higher rating is the narrow scale of the real-world validation (limited to CIFAR-10/100 ensembling on a ViT-B/16 checkpoint). To fully establish its significance to the machine learning community, the method needs to be evaluated on larger-scale multi-task benchmarks or modern Large Language Models (LLMs). Nonetheless, the conceptual novelty and mathematical elegance of the spectral trajectory framework are outstanding, and the work provides a solid foundation that others are highly likely to build upon.

---

## Questions and Suggestions for the Authors

1. **Evaluation on Modern LLMs**: Have the authors considered evaluating RB-DCTM on Large Language Models (LLMs)? Modern decoders have 32 to 80 layers, making them highly susceptible to overfitting during layer-wise calibration. Constraining the ensembling coefficients of task-specific LLMs (e.g., merging chat, math, and code experts) to a low-frequency DCT trajectory with $F+1 \approx 3$ parameters would be a highly compelling showcase of the method's scalability and structural regularization.
2. **Heterogeneous Layer Architectures**: How does the continuous trajectory assumption handle network architectures that are less homogeneous than ResNets or standard Transformers? For example, if a CNN contains downsampling bottlenecks or highly heterogeneous blocks, does the continuous trajectory constraint over-constrain the model? Can the method be adapted to handle sub-networks or specific block types independently?
3. **Clarifying the Sample Footprint**: In the main text, please ensure that the 100-sample unlabeled calibration footprint for ZipIt! covariance estimation is clearly declared and compared alongside the 10-shot labeled optimization budget. This ensures a transparent and fair comparison against methods that do not require permutation alignment and can operate strictly on the 10-shot labeled data.
4. **Reframe the ACS Sandbox**: Consider reframing the synthetic coordinate sandbox as an illustrative visualization tool for showing trajectory shapes and boundary oscillations, rather than a primary comparative benchmark, given that the parameter-free Static Uniform baseline dominates the performance throughout all sweeps. This would focus the reader's attention more directly on the successful real-world ViT validation.
