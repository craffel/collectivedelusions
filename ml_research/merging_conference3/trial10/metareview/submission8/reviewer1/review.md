# Peer Review: Rademacher-Bounded Fourier Trajectory Merging

## Paper Summary
This paper addresses the challenge of layer-wise adaptive weight-space ensembling for deep neural networks on small calibration datasets (few-shot adaptation). Standard unconstrained layer-wise optimization of ensembling coefficients is prone to transductive overfitting and representation shearing. To mitigate this, the authors propose **Rademacher-Bounded Fourier Trajectory Merging (RB-FTM)** and its non-periodic counterpart **Rademacher-Bounded Discrete Cosine Trajectory Merging (RB-DCTM)**. The method projects the ensembling coefficients onto a low-frequency continuous Fourier or Cosine subspace across depth, bounded by a spectral cutoff frequency $F \ll L$. The authors derive tight empirical Rademacher complexity bounds for both trajectory classes over network depth coordinates and bridge them to downstream prediction generalization via a covering-number formulation. To physically enforce these bounds during optimization, they propose a **Spectral Lasso ($L_1$)** penalty strictly on the harmonic coefficients of the trajectories. The proposed methods are evaluated on a synthetic, purely linear "Analytical Coordinate Sandbox (ACS)" across Deep12LayerCNN and CLIP ViT-B/16 backbones, and validated on two actual Vision Transformer (ViT-B/16) checkpoints fine-tuned on CIFAR-10 and CIFAR-100 aligned with ZipIt!.

---

## Strengths and Weaknesses

### Strengths
1. **Mathematical Rigor and Theoretical Completeness**: The paper provides a very solid and complete theoretical foundation. It derives exact, closed-form empirical Rademacher complexity bounds for both Fourier and DCT ensembling classes over network depth coordinates. The step-by-step mathematical proofs are detailed and theoretically sound.
2. **Bridging Trajectory Complexity to Downstream Data Generalization**: Rather than settling for a standard contraction-based bound that exhibits dimensional mismatch (depth $L$ vs. data sample count $N$), the authors provide a formal covering-number derivation. This establishes an explicit $\widetilde{\mathcal{O}}(1/\sqrt{N})$ decay rate over data samples that is completely independent of the underlying network's parameter count, mathematically justifying why spectral trajectory optimization prevents overfitting even for multi-billion parameter networks.
3. **Elegant Physical Insight on Boundary Runaway**: The paper clearly identifies **boundary runaway** (Runge's-like phenomenon) in prior quadratic polynomial ensembling trajectories (RBPM, $d=2$). It offers an elegant solution via half-period cosine functions (RB-DCTM), which enforce an implicit homogeneous Neumann boundary condition ($h'(0) = h'(1) = 0$). This constraint acts as an analytical "boundary buffer" that stabilizes the critical feature-extraction and classification boundaries from destructive gradient propagation.
4. **Principled Optimization Design (Spectral Lasso)**: Applying the $L_1$ penalty strictly to the harmonic (non-DC) coefficients while leaving the uniform baseline coefficient ($a_{k,0}$) unpenalized is a clever and highly effective optimization choice. It preserves the optimal uniform activation scale across layers while constraining trajectory fluctuations and promoting spectral sparsity.
5. **High Transparency and Self-Awareness**: The authors are highly transparent and proactive throughout the manuscript, explicitly discussing and addressing standard machine learning theory-practice gaps, Lipschitz scaling pathologies in deep networks, and representation shearing.

### Weaknesses
Despite its elegant theoretical framework, the paper suffers from severe empirical and logical contradictions, a highly artificial simulation environment, and hidden methodological assumptions that undermine its core claims.

1. **Severe Empirical Contradiction on Cutoff Frequency and Regularization Performance**:
   A major contradiction exists between the main results in Table 1 and the ablation results in Table 2.
   - The authors claim that low-frequency spectral trajectories and the Spectral Lasso regularizer ($\gamma \approx 0.01$) are essential to prevent transductive overfitting and representation shearing.
   - However, in Table 2 (the unregularized sweep where $\gamma=0$), on the CLIP ViT-B/16 backbone, increasing the spectral cutoff frequency $F$ from 1 to 5 actually *improves* categorical accuracy significantly (from 78.50% to 83.20% for RB-FTM, and from 76.75% to 83.05% for RB-DCTM). This directly contradicts the authors' central claim that higher capacity/frequencies lead to overfitting and performance degradation.
   - Even worse, Table 1 reports that the "optimal" regularized RB-FTM (F=2) achieves only 72.70% accuracy, and regularized RB-DCTM (F=2) achieves only 70.35% accuracy. However, Table 2 shows that *without* regularization ($\gamma=0$), the accuracies are 78.00% (Fourier F=2) and 80.50% (DCT F=2). This means that adding the proposed Spectral Lasso regularizer actually *harms* performance substantially—causing a 5.30% drop for Fourier F=2 and a massive 10.15% drop for DCT F=2. Why would a practitioner use a regularizer that severely degrades performance compared to the unregularized baseline? This contradiction severely weakens the claim that Spectral Lasso is a beneficial and necessary regularization component.

2. **Methodological Loophole in Data Budget (Auxiliary Data for Alignment)**:
   The paper claims to operate under a highly sample-efficient "10-shot calibration budget" (10 samples per task).
   - However, in Section 4.5, the authors disclose that because estimating high-dimensional $768 \times 768$ activation covariance matrices on 10 samples per task (total 20 samples) is rank-deficient, they utilized a separate, unlabeled calibration footprint of **100 samples per task** strictly to compute stable permutations for ZipIt! coordinate alignment.
   - This represents a significant data-budget inflation. The overall pipeline actually consumes **110 samples per task** (100 for alignment + 10 for coefficient tuning). If 100 unlabeled samples are available, the motivation for extremely restricted 10-shot trajectory optimization is severely weakened.
   - More importantly, the authors do not evaluate how unconstrained layer-wise ensembling (Offline Unconstrained) or Globally-Scaled ensembling would perform if optimized on this larger pool of 110 samples. If unconstrained optimization were given 110 samples, it would likely be much more stable, potentially eroding the advantage of the highly restricted spectral trajectory class.

3. **Vacuous and Hidden Depth-Scaling in the Generalization Bridge**:
   In Section 3.5, the authors claim that the downstream generalization bound (Eq. 16) is "completely independent of the parameter count of the underlying deep network! The capacity is bounded solely by the trajectory dimension $d$."
   - However, the covering number bound depends on the parameter-Lipschitz constant $L_{\Theta}$, which is the Lipschitz constant of the model's output with respect to the trajectory parameter vector $\Theta$.
   - Because the ensembling coefficients enter the network multiplicatively across all $L$ layers, the final output $f(x; \Theta)$ is a high-degree polynomial of $\Theta$. Consequently, the parameter-Lipschitz constant $L_{\Theta}$ will scale exponentially with network depth $L$ (i.e., $L_{\Theta} = \mathcal{O}(C^L)$) for standard neural network architectures. This means that the exponential scaling of the bound with network depth $L$ is merely hidden inside the variable name $L_{\Theta}$. For deep networks (such as modern LLMs with 32 to 80 layers, which the authors claim their method scales to), $L_{\Theta}$ will be extremely large or infinite, rendering the covering number bound completely vacuous. Presenting this bound as a "rigorous guarantee" is highly misleading.

4. **The Synthetic Sandbox is an Inappropriate Proxy**:
   The primary quantitative evaluation of this paper is conducted in the "Analytical Coordinate Sandbox" (ACS).
   - Inside this sandbox, the parameter-free, zero-tuning **Static Uniform** baseline consistently and significantly outperforms all of the proposed adaptive methods (achieving 85.10% on CNN and 83.75% on CLIP, while the best tuned trajectory method reaches only 70.70% and 72.70%).
   - The authors frame this as the "Static Uniform Dominance Paradox" and explain that in perfectly aligned coordinate spaces, adaptation induces anisotropic shearing that degrades accuracy. However, this raises a fundamental question: if the sandbox is designed such that any parameter adaptation is mathematically guaranteed to be counterproductive, then the sandbox is a highly inappropriate proxy to study and validate adaptive merging methods. The sandbox lacks the non-linearities, scale mismatches, and imperfect alignments that make adaptive merging necessary in the first place, rendering its sweeps and findings of questionable practical utility.

5. **Highly Limited and Small-Scale Real-World Validation**:
   The real-world validation (Section 4.5) is extremely limited, merging only two expert models (CIFAR-10 and CIFAR-100) on a single Vision Transformer architecture (ViT-B/16, 86M parameters). This is a toy setup compared to standard model merging benchmarks in the literature, which typically involve merging 5 to 8 task experts (such as the 8-task CLIP-ViT-L/14 benchmark) or merging multiple massive LLM checkpoints (e.g., LLaMA-2 or Mistral 7B). It does not fully test the claimed scalability of the method to ultra-deep decoder architectures or multi-modal networks.

6. **Missing Competitive Adaptive Baselines**:
   The paper fails to compare its proposed spectral trajectory method against popular layer-wise adaptive ensembling baselines from the recent literature, such as **AdaMerging** (Yang et al., 2024), which optimizes layer-wise weights using prediction entropy minimization during test-time adaptation. While the authors mention AdaMerging in the Related Work, they fail to include it as an experimental baseline.

---

## Form Ratings

### Soundness: Fair
The mathematical derivations and proofs are highly detailed and correct. However, there are major logical contradictions in the empirical evaluation (e.g., the regularized models underperforming unregularized models on CLIP, and higher frequencies performing better without regularization), and a significant methodological loophole regarding the data budget used for coordinate alignment (ZipIt!).

### Presentation: Good
The paper is exceptionally clear, well-structured, and easy to follow. The notation is consistent, and the progression from theory to experiments is logical. The authors are highly transparent and proactive in their remarks, discussing limitations such as Lipschitz scaling and the theory-practice gap, which is commendable.

### Significance: Fair
The concept of continuous, spectral trajectories for model ensembling across depth is highly elegant. However, its current significance is severely limited by: (1) the highly artificial sandbox environment where the proposed method fails to beat a simple static baseline, and (2) the extremely small-scale real-world evaluation (2-task CIFAR merge on ViT-B/16). Without large-scale validation on LLMs or multi-task vision benchmarks, the practical significance remains unproven.

### Originality: Good
The identification of polynomial boundary runaway and the solution via Neumann-constrained DCT ensembling trajectories represent a high level of originality. The formulation of the strict harmonic Spectral Lasso is a clever optimization choice, although the use of spectral curves for trajectory smoothing is mathematically standard.

---

## Overall Recommendation
**Rating: 3: Weak Reject**

### Justification
This paper has clear merits, particularly in its rigorous theoretical derivations (Rademacher complexity bounds, covering-number generalization bridge) and its elegant, physics-inspired solution to polynomial boundary runaway (Neumann-constrained DCT basis).

However, the severe empirical and logical contradictions (proposed Spectral Lasso regularization substantially harming performance compared to unregularized baselines on CLIP, and higher frequencies performing better without regularization), the inflated data-budget loophole for ZipIt! alignment, the highly artificial sandbox environment where the proposed methods fail to beat the static baseline, and the extremely limited small-scale real-world validation outweigh the theoretical merits in the paper's current form.

The paper requires a thorough revision to resolve these empirical contradictions and methodologically validate the claims under a true, uninflated few-shot budget before it can be accepted.

---

## Questions and Constructive Feedback for Authors

1. **Resolution of Regularization Discrepancy**: In Table 2 (unregularized regime, $\gamma=0$), RB-DCTM ($F=2$) achieves **80.50%** accuracy on CLIP. However, in Table 1, regularized RB-DCTM ($F=2, \gamma=0.01$) achieves only **70.35%** accuracy—a massive **10.15% absolute performance drop** caused by adding your regularizer. Furthermore, Table 2 shows that increasing $F$ from 1 to 5 strictly improves accuracy on CLIP (78.50% to 83.20% for Fourier), contradicting your theory that higher frequencies cause overfitting. Please provide a rigorous, mathematically and empirically coherent explanation for this behavior. Why should a practitioner use your Spectral Lasso regularizer if it catastrophically degrades accuracy on Vision Transformers compared to unregularized optimization?
2. **True Few-Shot Validation without Auxiliary Data**: To claim a "10-shot calibration budget" with scientific integrity, please run your real-world Vision Transformer experiments using a regularized covariance estimator (such as the Ledoit-Wolf shrinkage estimator you propose in Appendix Section 5.6) strictly on the 10-shot split for coordinate alignment. Show that the entire pipeline (alignment + trajectory optimization) can execute successfully strictly on the 10-shot budget (10 samples per task) without using the auxiliary 100-sample unlabeled footprint. How does the performance of RB-DCTM hold up under this true sample-efficient setup?
3. **Scaling up Real-World Experiments**: The paper claims that the method is structurally designed to scale to ultra-deep decoder architectures (LLMs with 32 to 80 layers) and multi-modal networks. Please back up these scalability claims with empirical evidence. Specifically, evaluate your method on standard multi-task benchmarks (e.g., merging 5 to 8 vision task vectors on CLIP-ViT-L/14) or merging Parameter-Efficient Fine-Tuning (PEFT) models like LoRAs on LLMs (e.g., LLaMA-3 or Mistral). Merging two CIFAR experts is insufficient to validate your claim of scalability.
4. **Comparison against Competitive Baselines**: Please include popular layer-wise adaptive ensembling baselines from the literature, such as **AdaMerging** (Yang et al., 2024), as experimental baselines. Show that your restricted spectral trajectories outperform these unconstrained or entropy-minimized adaptive alternatives on the real-world benchmarks.
5. **Evaluating unconstrained/globally-scaled baselines on the larger pool**: To ensure a fair comparison, how do the Offline Unconstrained and Globally-Scaled baselines perform if they are optimized on the larger pool of 110 samples per task that you used in your real-world experiments?
6. **Clarifying the Lipschitz Scaling of $L_{\Theta}$**: In Section 3.5, please explicitly address and mathematically detail how the parameter-Lipschitz constant $L_{\Theta}$ scales with network depth $L$. If $L_{\Theta}$ scales exponentially with $L$, please transparently discuss this limitation in the main text and qualify the claim that your generalization bound "completely avoids the curse of deep networks."
7. **Code Release**: Please provide a public code repository link or include anonymous source code in the submission to ensure complete reproducibility of the synthetic sandbox and Vision Transformer experiments.
