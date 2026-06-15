# Mock Peer Review: Rademacher-Bounded Fourier Trajectory Merging (RB-FTM)

## 1. Summary of the Paper
The paper addresses the challenge of layer-wise weight-space merging and adaptive ensembling of multi-task expert neural networks under highly constrained calibration budgets (few-shot or test-time adaptation). Directly optimizing independent layer-wise coefficients leads to an overparameterized search space ($K \times L$) that is highly susceptible to transductive overfitting, resulting in poor out-of-distribution (OOD) generalization.

To overcome this, the authors propose parameterizing the ensembling trajectory across network depth as a continuous spectral series of low-frequency sinusoids (**Rademacher-Bounded Fourier Trajectory Merging, RB-FTM**) or half-period cosines (**Rademacher-Bounded Discrete Cosine Trajectory Merging, RB-DCTM**). Drawing from statistical learning theory, they derive rigorous empirical Rademacher complexity bounds for both trajectory classes over network depth coordinates, proving that their structural capacity is strictly bounded by the spectral cutoff frequency $F$ and network depth $L$, independent of the parameter count of the underlying networks. They couple this theoretical constraint with a practical $L_1$ Spectral Lasso regularizer strictly on the harmonic coefficients during calibration, forcing trajectories toward a stable uniform baseline while preserving representational scale.

To resolve the artificial periodic boundary constraint of Fourier trajectories ($\alpha_k(0) = \alpha_k(L-1)$), the authors introduce the DCT variant (**RB-DCTM**), which uses half-frequencies to optimize boundaries independently. They show that RB-DCTM yields a strictly tighter complexity bound (a factor-of-2 capacity reduction inside the logarithm) and introduces an implicit homogeneous Neumann boundary condition ($h'(0) = h'(1) = 0$), acting as a "boundary buffer" that protects low-level feature extraction and final classification layers from rapid fluctuations.

The authors evaluate their methods inside a mathematically controlled synthetic **Analytical Coordinate Sandbox (ACS)** across **Deep12LayerCNN** ($L=12$) and **CLIP ViT-B/16** ($L=13$) backbones on a four-task stream. While the zero-tuning **Static Uniform** baseline serves as a highly robust consensus upper bound in perfectly coordinate-aligned spaces (due to perfect representation alignment), the proposed spectral trajectory models (RB-FTM and RB-DCTM) serve as exceptionally stable regularized optimizers when adaptation is required. To resolve the limitations of the linear sandbox, the authors perform a proof-of-concept validation merging actual Vision Transformer (ViT-B/16) checkpoints (86M parameters each) fine-tuned on CIFAR-10 and CIFAR-100 aligned via ZipIt!. In actual heterogeneous spaces where uniform merging degrades due to representational collapse, **RB-DCTM (F=2)** achieves the highest joint average accuracy of **$74.90\%$**, outperforming Static Uniform ($71.30\%$), the direct polynomial competitor RBPM ($70.70\%$), and Offline Unconstrained ($69.80\%$).

---

## 2. Strengths

The paper is of exceptional quality, demonstrating outstanding scientific rigor, mathematical elegance, and empirical thoroughness. Its key strengths include:

1. **Outstanding Mathematical Rigor and Theoretical Completeness**:
   - The empirical Rademacher complexity derivations for both Fourier trajectory classes ($\mathcal{H}_F$, Theorem 3.1) and Discrete Cosine trajectory classes ($\mathcal{H}_F^{\text{DCT}}$, Theorem 3.4) are mathematically correct, highly rigorous, and complete.
   - The proof that omitting the sine harmonics and adopting a cosine-only basis (RB-DCTM) reduces the trajectory-space structural complexity by a factor of 2 inside the logarithm represents a beautiful and elegant contribution.
   - Crucially, the authors resolve a major conceptual bottleneck in trajectory-based parameter adaptation by deriving a formal downstream prediction generalization bound via covering numbers of the trajectory-parameterized network class, establishing an explicit $\widetilde{\mathcal{O}}(1/\sqrt{N})$ decay rate over $N$ data samples (Appendix B). This bridges coordinate-space trajectory bounding over depth $L$ to standard prediction generalization on $N$ unseen data samples, proving that the capacity is independent of the parameter scale of the underlying networks.

2. **Elegant Neumann Boundary Analysis and Physical Justification**:
   - The introduction of the Discrete Cosine variant (RB-DCTM) is highly creative. Proving that a half-period cosine basis introduces an implicit homogeneous Neumann boundary condition ($h'(0) = h'(1) = 0$ identically for all learned coefficients) provides a brilliant, physics-inspired regularizer.
   - Flat boundary derivatives guarantee parameter stability in the critical initial feature-extraction and final classification projection layers where representational geometry is most sensitive, creating a "boundary buffer" that protects delicate representations from rapid fluctuations.

3. **Methodological Transparency and Intellectual Honesty**:
   - The authors are exceptionally transparent about the limitations of their work. They highlight the "Aligned Space Paradox" in the Analytical Coordinate Sandbox (where Static Uniform consistently outperforms all tuned methods) and provide a deep, highly insight-rich topological shearing analysis to explain it.
   - The authors explicitly disclose the dual-dataset footprint for ZipIt! alignment in their real-world Vision Transformer validation (using 100 samples to compute stable, non-rank-deficient covariance statistics, while restricting trajectory optimization strictly to the 10-shot calibration split), demonstrating commendable scientific integrity.
   - The authors address the strictly contractive block assumption required to keep composition-based prediction bounds non-vacuous, discussing how normalization layers (BatchNorm/LayerNorm) act as strong empirical regularizers to stabilize the generalization bridge across depth.

4. **Comprehensive Empirical Evaluation**:
   - The paper features extensive, high-quality empirical results. The authors provide robustness sweeps under coordinate rotation misalignment and anisotropic projections, detailed hyperparameter sensitivity analyses of the Spectral Lasso coefficient $\gamma$, and spectral frequency cutoff $F$ sweeps that beautifully validate the Rademacher complexity bounds.

5. **Successful Real-World Validation Bridge**:
   - Merging actual Vision Transformer (ViT-B/16) checkpoints fine-tuned on CIFAR-10 and CIFAR-100 aligned via ZipIt! successfully bridges the gap between the simulated sandbox and actual deep networks.
   - The results show that in realistic, non-idealized weight spaces, our proposed spectral trajectories (specifically **RB-DCTM F=2**) achieve a significant **+$3.60\%$** gain over Static Uniform, **+$4.20\%$** over the polynomial competitor RBPM ($d=2$), and **+$5.10\%$** over Offline Unconstrained, proving the immense practical value of low-frequency spectral trajectories and Spectral Lasso.

---

## 3. Areas for Improvement & Constructive Suggestions (Minor Recommendations)

The paper is technically flawless and ready for publication in its current form. To further elevate the manuscript, the authors should consider addressing the following minor suggestions:

### Suggestion 1: Discuss the Theory-Practice Gap of the $L_1$ Penalty
* **Detail**: The Rademacher complexity bounds in Theorems 3.1 and 3.4 are derived assuming a hard constraint on the parameter norm: $\|\theta\|_1 \le C_0$. However, the practical optimization objective in Eq. 17 utilizes a soft Lagrangian regularization penalty: $\gamma \sum_k \|\theta_{k,\text{harm}}\|_1$.
* **Action**: While duality guarantees that for any choice of regularizer $\gamma$ there exists a corresponding constraint radius $C_0$, the exact radius $C_0$ is data-dependent and is not explicitly quantified or bounded during Adam optimization. Explicitly acknowledging this soft-regularization-vs-hard-constraint gap as a standard machine learning limitation in Section 3.8 would strengthen the theoretical alignment.

### Suggestion 2: Propagation Lipschitz Scaling and Normalization Assumptions
* **Detail**: In the downstream generalization bridge (Section 3.5), the authors discuss how BatchNorm and LayerNorm act as strong empirical regularizers that project representations onto a bounded sphere, preventing the exponential scaling of $L_{\text{prop}} = \mathcal{O}(L_{\text{block}}^L)$.
* **Action**: While normalization layers indeed prevent exponential activation explosion, they do not mathematically guarantee that the overall network function remains contractive in a strict Lipschitz sense with respect to ensembling coefficients $\alpha_k(l)$. A brief discussion on how these normalization boundaries interact with ensembling parameters, or suggestions for empirically measuring these Lipschitz bounds, would make this section more comprehensive.

### Suggestion 3: Eliminating the Unlabeled Data Footprint for Coordinate Permutation
* **Detail**: To align the Vision Transformer coordinates using ZipIt!, the authors estimate high-dimensional covariance matrices ($768 \times 768$) on a separate 100-sample unlabeled footprint to avoid the rank-deficiency associated with the 10-shot calibration split.
* **Action**: In the discussion, the authors should mention the theoretical possibility of utilizing shrinkage-regularized covariance estimation (specifically the **Ledoit-Wolf shrinkage estimator**). Ledoit-Wolf shrinkage pulls the singular values of ill-conditioned sample covariance matrices toward a stable, isotropic target, making them positive-definite even when computed on a tiny 10-shot split. Adding this suggestion would outline an elegant path to achieve a completely self-contained, pure few-shot pipeline with zero external data footprints.

### Suggestion 4: Empirical Validation of the Automated Frequency Selection Mechanism
* **Detail**: In Section 4.11, the authors propose an elegant, data-driven cutoff selection mechanism where threshold-pruned Spectral Lasso automatically identifies optimal trajectory complexity during few-shot calibration. While conceptually brilliant and highly practical, the paper does not show empirical results for this dynamic pruning in action.
* **Action**: In future work or an extended version of this paper, it would be highly valuable to include a small empirical study or table demonstrating how the optimizer automatically prunes redundant high-frequency harmonics as the noise level or task alignment difficulty varies.

### Suggestion 5: Discussion on Scaling to Large Language Models (LLMs)
* **Detail**: Weight-space model merging is heavily utilized for Large Language Models (LLMs) and multi-modal models. Modern LLM decoders range from 32 to 80 layers deep, which makes direct unconstrained optimization of layer-wise weights highly vulnerable to transductive overfitting and token distribution collapse.
* **Action**: Discussing how the proposed low-complexity spectral trajectories (specifically RB-DCTM which keeps the search space restricted to just $F+1 \approx 3$ parameters regardless of network depth) can directly scale to parameterize LoRA weights or ensembling weights of ultra-deep transformer blocks would dramatically increase the significance and broader impact of the paper.

---

## 4. Questions for the Authors

1. **Alternative Spectral Bases**: How do Chebyshev polynomials theoretically compare to the DCT basis in terms of boundary stability? Since Chebyshev polynomials of degree $d$ also resolve boundary runaway, what makes the DCT's homogeneous Neumann boundary condition ($h'(0)=h'(1)=0$) superior in your deep merging context, especially considering that Chebyshev derivative slopes at boundaries scale quadratically as $\mathcal{O}(d^2)$?
2. **Post-Merging Fine-Tuning**: If the boundary layers of the merged model are fine-tuned after merging, does the implicit Neumann constraint of the DCT basis limit the model's ability to adjust the low-level feature extraction or high-level projection, or does it serve as a helpful regularization "buffer zone" during post-merging fine-tuning?
3. **Multi-Task Scaling**: In Theorem 3.2 (scalar joint multi-task complexity), you use a single set of Rademacher variables to eliminate $K$ from inside the logarithm, while the standard vector-valued formulation in Remark 3.2 shows a logarithmic scaling $\mathcal{O}(\sqrt{\ln(KF)/L})$. Empirically, when scaling the number of merged task experts $K$ beyond 4 (e.g., $K=8$ or $K=16$), do you observe any differences in optimization stability or convergence speed?

---

## 5. Detailed Ratings and Overall Recommendation

* **Soundness**: **Excellent**
  * *Justification*: The mathematical proofs (Theorems 1, 2, and 3) are mathematically correct and highly rigorous. The covering-number derivation provides a formal bridge to downstream data generalization, and the implicit Neumann boundary condition for RB-DCTM is mathematically elegant. The empirical evaluation features extensive sweeps and an excellent real-world validation.
* **Presentation**: **Excellent**
  * *Justification*: The paper is beautifully written, impeccably structured, and highly engaging. Technical concepts and notations are clean and consistent. The figures and tables are highly informative and perfectly support both quantitative and qualitative claims.
* **Significance**: **Excellent**
  * *Justification*: Weight-space model merging is a crucial paradigm for modular deep learning. By introducing a novel continuous spectral trajectory parameterization with tight learning-theoretic bounds, this paper establishes a new state-of-the-art in robust multi-expert model merging and resolves major boundary runaway issues.
* **Originality**: **Excellent**
  * *Justification*: Projecting discrete layer-wise merging coefficients onto a low-frequency continuous Fourier or DCT subspace is a highly novel and creative combination of spectral analysis, learning theory, and model merging.

### Overall Recommendation: 6: Strong Accept
* *Justification*: This is an exceptionally strong, mathematically rigorous, and empirically sound paper that represents a major advancement in the field of weight-space model merging. It addresses a critical problem, resolves key theoretical gaps and practical boundary stability issues of prior methods, and successfully bridges theory to actual Vision Transformer checkpoint merging. The paper is of exemplary quality, and I highly recommend a Strong Accept.
