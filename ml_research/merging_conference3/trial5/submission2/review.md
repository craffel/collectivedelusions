# Official Review

## Summary of the Submission
This paper establishes the first rigorous statistical learning-theoretic foundation for adaptive model merging through a novel framework called **Rademacher-Bounded Polynomial Merging (RBPM)**. Post-hoc weight-space merging combines fine-tuned task-specific expert models without retraining, but existing adaptive ensembling methods suffer from severe transductive overfitting. RBPM addresses this by:
1. Constraining the layer-wise ensembling coefficients to follow a smooth, low-degree polynomial trajectory across network depth.
2. Formulating a **Consensus-Pulling Rademacher Penalty** to regularize ensembling parameters back toward their stable, uniform consensus initialization basin, preventing parameter scale distortion.
3. Proving formal empirical Rademacher complexity bounds for the trajectory space, which reduces the complexity of layer transitions by a factor of $\mathcal{O}(\sqrt{L / \log(d)})$, where $L$ is network depth and $d$ is polynomial degree.
4. Bridging trajectory capacity to neural network generalization via spectrally-normalized Rademacher complexity and first-order functional linearization, and deriving fast generalization rates of $\mathcal{O}(1/N_{\text{img}})$ using local Rademacher complexity under Bernstein class conditions.

Extensive empirical evaluations are conducted across two distinct regimes: a highly challenging heterogeneous benchmark (deep CNN on MNIST, FashionMNIST, CIFAR-10, and SVHN) and a modern physical foundation benchmark (CLIP ViT-B/16 on Stanford Cars and Oxford Flowers-102). RBPM achieves state-of-the-art results, outperforming static uniform ensembling, unconstrained few-shot tuning, and prominent coordinate-wise merging heuristics (TIES, DARE, Sparse Task Arithmetic).

---

## Strengths
1. **Outstanding Theoretical-Empirical Synergy:** The paper represents a beautifully balanced contribution where deep statistical learning theory directly informs algorithm design. The trajectory projection and Consensus-Pulling penalty are mathematically motivated and elegantly formulated, rather than being justified ad-hoc.
2. **Exquisite Mathematical Rigor:** The theoretical derivations are complete, precise, and highly sophisticated. The authors prove:
   - Trajectory-space Rademacher bounds using dual norms and sub-Gaussian concentration (Theorem 3.1).
   - Contraction of the sigmoid-parameterized class via the Ledoux-Talagrand Contraction Principle on a shifted sigmoid.
   - Strict Lipschitz continuity of the trajectory via Markov's Theorem for Polynomials.
   - Spectral-norm bounds on the merged network via functional linearization.
   - Fast excess risk rates ($\mathcal{O}(1/N_{\text{img}})$) via local Rademacher complexity theory.
   - Extensions to piecewise cubic B-splines and Neural ODEs.
3. **Exceptional Scientific Controls:** The baselines are remarkably thorough. In particular, the **Regularized Offline Unconstrained Few-Shot Tuning** baseline serves as a perfect control to decouple the geometric trajectory constraint (+4.30% gain) from norm-based capacity control (+1.80% gain). This elevates the empirical analysis to a deep scientific inquiry.
4. **Physical Scalability Validation:** The paper directly addresses the scalability of trajectory-constrained ensembling by evaluating on a real, pre-trained CLIP ViT-B/16 model. RBPM achieves 85.15% average accuracy, retaining 98.6% of the individual expert performance ceiling while outperforming unconstrained tuning (+2.65%) and coordinate pruning methods (up to +4.85%).
5. **Scientific Maturity and Transparency:** The authors explicitly and maturely discuss the core assumptions and limitations of their theory (e.g., treating layers as independent coordinates, first-order functional linearization errors, and Bernstein class assumptions). This transparent and honest reporting is highly commendable.

---

## Weaknesses & Areas for Improvement
1. **Verification of Bernstein Class Conditions:** While the local Rademacher complexity derivation is elegant and mathematically correct, the assumption of Bernstein class conditions for deep neural network ensembling is difficult to verify in practice. The authors could expand their discussion in Section 3.4 or Appendix B to comment on the practical settings under which deep architectures might be expected to satisfy these conditions.
2. **Computational Overhead of Calibration:** While the paper correctly states that RBPM has zero test-time overhead, it lacks an explicit report of the computational time/cost required to perform the offline few-shot calibration. Since calibration only utilizes $M=10$ samples, this training time is likely negligible (seconds), but presenting the exact wall-clock times would strengthen the practical arguments.
3. **Evaluation on Extremely Deep Networks:** The physical validation on CLIP ViT-B/16 ($L=13$) is excellent. However, evaluating on extremely deep modern architectures, such as Large Language Models with $L=32$ or $L=80$ layers, would further emphasize the power of the $\mathcal{O}(\sqrt{L / \log(d)})$ complexity reduction over unconstrained parameters, which suffer from severe parameter explosion.

---

## Questions for the Authors
1. **Calibration Computational Cost:** Could you provide the average wall-clock time required to perform the offline few-shot calibration for RBPM on the CNN and CLIP ViT-B/16 backbones?
2. **Bernstein Conditions:** Are there specific architectural features (e.g., skip connections, layer normalization) or loss properties (e.g., cross-entropy near convergence) that you believe play a crucial role in ensuring the Bernstein class conditions hold for the merged network?
3. **LLM Extension:** Have you considered extending the spline or Neural ODE trajectory formulations to deep decoder-only Large Language Models? How do you expect the knot placement or Lipschitz bounds to behave when scaling to networks with $L \ge 80$ layers?

---

## Detailed Evaluation of Dimensions

### Soundness: Excellent
The submission is technically flawless. Every claim is supported by sound, step-by-step mathematical proofs in the appendix and rigorous empirical validation in the main text. The scientific controls (such as regularizing the unconstrained baseline and comparing varying polynomial degrees) are exceptionally thorough, isolating and proving the distinct benefits of both capacity control and geometric trajectories.

### Presentation: Excellent
The paper is beautifully written, extremely well-structured, and easy to follow. The mathematical notation is formal and clean. The figures and tables are highly informative and visually polished. The appendix contains complete, detailed proofs that provide enough information for an expert reader to reproduce the results.

### Significance: Excellent
The work addresses a major, relevant challenge in post-hoc model merging: transductive overfitting and capacity explosion under data scarcity. By bridging this practical problem with statistical learning theory, it brings mathematical rigor to a field dominated by heuristics. The proposed trajectory projection is architecture-agnostic and highly likely to influence future research on ensembling, foundation models, and adaptive merging.

### Originality: Excellent
The paper introduces highly original concepts, specifically projecting layer-wise ensembling coefficients to a continuous, low-degree polynomial trajectory and formulating the Consensus-Pulling Rademacher Penalty. Centering the penalty around the uniform consensus represents a highly original and critical contribution that avoids representational scale collapse.

---

## Overall Recommendation

**Rating: 6 (Strong Accept)**

**Justification:** This is an outstanding, technically flawless paper that establishes a rigorous statistical learning-theoretic foundation for adaptive model merging. The theoretical framework—combining Rademacher trajectory bounds, Ledoux-Talagrand contraction, and local Rademacher fast rates—is exceptionally deep and mathematically elegant. The empirical evaluation is incredibly thorough, featuring strong baselines, perfect controls, and physical validation on CLIP ViT-B/16. The paper is beautifully written, scientifically transparent, and represents a high-impact contribution that the machine learning community is highly likely to build upon. It deserves a Strong Accept.
