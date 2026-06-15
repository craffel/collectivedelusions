# Peer Review: Rademacher-Bounded Fourier Trajectory Merging for Spectral Regularization

## Summary of the Paper
The paper addresses the challenge of layer-wise adaptive weight-space model merging under extreme data constraints (such as 10-shot calibration). Traditional layer-wise merging optimizes independent coefficients for each task across all layers, creating an overparameterized search space prone to transductive overfitting. To control this capacity, prior work proposed restricting coefficients to low-degree polynomial subspaces (such as Rademacher-Bounded Polynomial Merging or RBPM). However, polynomial curves suffer from boundary runaway (Runge's phenomenon) at the first and last layers of deep backbones, degrading representation propagation.

To resolve these limitations, this paper proposes:
1. **Rademacher-Bounded Fourier Trajectory Merging (RB-FTM)**: Projects discrete ensembling weights onto a low-frequency continuous Fourier series across network depth.
2. **Rademacher-Bounded Discrete Cosine Trajectory Merging (RB-DCTM)**: A non-periodic variant using a half-period cosine basis that eliminates periodic boundary identities while maintaining smoothness and boundary stability.
3. **Spectral Lasso ($L_1$) Regularization**: Restricts trajectory fluctuations by penalizing harmonic coefficients during few-shot calibration.

The authors derive empirical Rademacher complexity bounds for both Fourier and DCT trajectories over network depth coordinates. They evaluate their methods inside a simulated linear **Analytical Coordinate Sandbox (ACS)** across Deep12LayerCNN and CLIP ViT-B/16, and provide a real-world validation merging actual Vision Transformer (ViT-B/16) expert checkpoints fine-tuned on CIFAR-10 and CIFAR-100.

---

## Strengths and Weaknesses

### Strengths
1. **Mathematical Rigor and Formalization**:
   The paper is highly structured and mathematically precise. The derivation of empirical Rademacher complexity bounds for both Fourier (Theorem 3.1) and DCT (Theorem 3.4) trajectory classes is executed with high theoretical quality.
2. **Elegant Boundary Safeguards via DCT**:
   The transition to the **Discrete Cosine Transform (RB-DCTM)** is exceptionally elegant. It resolves the artificial periodic boundary identity of standard Fourier series while naturally enforcing homogeneous Neumann boundary conditions ($h'(0) = h'(1) = 0$). This provides a stable "boundary buffer" that successfully protects sensitive input feature-extraction and final classification projection layers from high-frequency gradient noise.
3. **Commendable Scientific Transparency**:
   The authors are exceptionally transparent about their experimental limitations:
   * They explicitly disclose that the **Analytical Coordinate Sandbox (ACS)** is a highly simplified, purely linear recurrence model.
   * They honestly discuss the **Static Uniform Dominance Paradox** (where the zero-tuning baseline consistently outperforms all tuned methods in the sandbox).
   * They transparently disclose the dual-dataset footprint required for stable ZipIt! coordinate alignment (100 samples per task) in the real-world validation.
4. **Smooth and Interpretable Trajectories**:
   By using low-frequency spectral cutoffs ($F=1$ or $F=2$), the learned trajectories are smooth, stable, and highly interpretable across layers, avoiding the extreme layer-to-layer jumps of unconstrained optimization.

### Weaknesses
1. **Unjustified Complexity of the Mathematical Apparatus (The Over-Engineering Problem)**:
   The core philosophy of the proposed approach is to construct a heavy mathematical and theoretical machinery (Fourier/DCT trajectory mapping, empirical Rademacher complexity bounds, Spectral Lasso regularization, and homogeneous Neumann boundary conditions) to optimize a tiny number of parameters (e.g., 6 parameters across 13 layers). 
   * **The Minimalist Baseline**: In the actual Vision Transformer validation (Table 2), the extremely simple **Globally-Scaled Task Arithmetic ($d=0$)** baseline—which merely tunes a single global scalar per expert (resulting in only 2 parameters total)—achieves an average joint accuracy of **72.50%**.
   * **The Marginal Gain**: The proposed highly complex 6-parameter **RB-DCTM (F=2)** achieves **74.90%**. The absolute performance gain over the global scalar baseline is a modest **+2.40%**.
   * Given the massive increase in mathematical, conceptual, and optimization pipeline complexity, this small gain suggests that the proposed method is heavily over-engineered. A simpler block-wise constant trajectory (e.g., tuning separate weights for early, middle, and late layers) or a simple linear ramp could likely achieve similar performance with far less complexity, but no such baselines are compared.
2. **Contradictory Results in the Primary Evaluation Sandbox (ACS)**:
   In the main simulated sandbox (Table 1), the parameter-free **Static Uniform** baseline (which simply averages experts using $1/K$ weights) consistently acts as an absolute empirical upper bound, achieving **85.10%** categorical accuracy on CNN and **83.75%** on CLIP. 
   * This means that in the authors' primary evaluation sandbox, **the simplest, zero-tuning baseline is actually the best method**, and any parameter adaptation (including our best proposed spectral method) degrades performance due to representation shearing. 
   * This raises serious questions about the choice of the sandbox as a representative environment: if the sandbox is perfectly coordinate-aligned such that uniform merging is mathematically optimal, it fails to demonstrate any practical utility of adaptive ensembling, undermining the motivation of the entire paper.
3. **Complete Absence of Statistical Rigor**:
   Few-shot calibration (10-shot, i.e., 10 samples per task) is highly sensitive to sample noise, resulting in extremely high performance variance across different calibration splits.
   * **The Flaw**: Neither Table 1 nor Table 2 contains standard deviations, error bars, or statistical significance testing (such as p-values or t-tests) over multiple random calibration seeds.
   * **The Risk**: Without showing performance over multiple random runs (e.g., 5 or 10 seeds), the reported +2.40% gain of RB-DCTM over Globally-Scaled could be an artifact of a single lucky calibration split rather than a statistically significant improvement.
4. **Pragmatic Grounding and the Representational Collapse Penalty**:
   * The unmerged specialized expert models achieve accuracies of **89.50% (CIFAR-10)** and **71.20% (CIFAR-100)**, averaging **80.35%**.
   * The best merged model (RB-DCTM, F=2) achieves **74.90%**, which represents a **5.45% absolute accuracy drop** compared to the unmerged experts.
   * In a practical scenario, if a user has the resources to fine-tune expert networks, they are highly likely to favor traditional ensembling (keeping the experts separate) to preserve the full 80.35% average accuracy. Weight-space merging is only justified to avoid running multiple models at inference time, but suffering a 5.45% accuracy penalty is a high price to pay, especially when the alignment and optimization pipeline itself requires a large unlabelled calibration footprint of 100 samples per task.
5. **The Vacuous Generalization Bridge**:
   In Appendix A.4, the authors attempt to bridge trajectory complexity to downstream prediction generalization. However, this theoretical bridge requires calculating a propagated network Lipschitz constant $L_{\text{prop}}$, which scales *exponentially* with depth $L$ for non-contractive networks. Since standard deep networks (CNNs, Transformers) are not contractive, this theoretical generalization bound is practically **vacuous** for deep architectures, leaving a significant gap between the theory and the deep neural networks used in the experiments.

---

## Ratings

### Soundness
* **Rating**: **Fair**
* **Justification**: While the mathematical proofs are correct and elegant, there are significant soundness limitations. The trajectory-space Rademacher bound is evaluated over fixed depth coordinates (treating layers as samples), which does not bound the model's actual predictive generalization error on unseen data. The downstream generalization bridge scales exponentially with depth for non-contractive networks, making it vacuous for realistic deep backbones. Furthermore, the lack of statistical error bars/significance tests in the 10-shot experiments compromises the scientific rigor of the quantitative findings.

### Presentation
* **Rating**: **Excellent**
* **Justification**: The paper is exceptionally well-written, clearly structured, and mathematically precise. The authors are commendably transparent about their assumptions, the limitations of their simulated sandbox, and the dual-dataset footprint required for real-world alignment.

### Significance
* **Rating**: **Fair**
* **Justification**: While the theoretical link between signal processing (spectral trajectories) and statistical learning theory (Rademacher bounds) is academic and elegant, the practical significance is limited. Weight merging is a niche area that suffers from substantial representational collapse compared to keeping experts separate. For edge devices where merging is required, the extremely simple, 2-parameter Globally-Scaled baseline gets within 2.40% of the proposed 6-parameter RB-DCTM, making the practical adoption of such complex, over-engineered trajectory models unlikely.

### Originality
* **Rating**: **Good**
* **Justification**: The trigonometric representation of ensembling trajectories and its corresponding Rademacher complexity bounds are a creative combination. Although the mathematical tools (Fourier series, Massart's Finite Lemma) are standard, their integration to resolve boundary runaway in weight-space ensembling represents a clear and original contribution over prior polynomial-based methods (RBPM).

---

## Overall Recommendation
* **Recommendation**: **3: Weak reject**
* **Justification**: This paper is technically accomplished, mathematically rigorous, and exceptionally well-written. It succeeds in identifying and addressing the boundary runaway (Runge's phenomenon) of prior polynomial ensembling trajectories by elegantly proposing Discrete Cosine Trajectory Merging (RB-DCTM).

However, from a practical and engineering perspective, the paper exhibits a significant "complexity-to-gain" imbalance. It constructs an enormous theoretical apparatus (Fourier series, Rademacher complexity bounds, Spectral Lasso, homogeneous Neumann boundary conditions) to select a handful of layer-wise weights. In the end, this heavy, over-engineered 6-parameter spectral trajectory method achieves only a **+2.40%** accuracy improvement over an extremely simple, 2-parameter **Globally-Scaled Task Arithmetic ($d=0$)** baseline on actual Vision Transformers. Furthermore:
1. In the primary evaluation environment (ACS sandbox), the proposed adaptive method is completely dominated by the zero-parameter, zero-tuning **Static Uniform** baseline.
2. The few-shot experiments completely lack standard deviations or error bars across random seeds, compromising the reliability of the modest quantitative gains.
3. The downstream generalization bounds are practically vacuous for non-contractive deep networks.

Overall, the clear merits of the mathematical derivations and elegant boundary properties are outweighed by the over-engineering of the method, the marginal gains over simpler baselines, and the lack of empirical statistical rigor. To be suitable for acceptance, the authors must:
* Rigorously demonstrate that their method outperforms simpler, intuitive trajectory models (such as block-wise constant weights or linear ramps).
* Provide full statistical significance testing (error bars/standard deviations) over multiple random calibration seeds for their few-shot experiments.
* Ground the theoretical generalization bounds by addressing the non-contractive nature of standard deep neural networks.
