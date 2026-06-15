# Peer Review: SpectralMerge

## Summary of the Paper
This paper addresses post-hoc model merging, which seeks to consolidate multiple task-specific expert neural networks (fine-tuned from a shared base initialization) into a single multi-task model without joint retraining. Specifically, the authors focus on **parameterized model merging**, which optimizes layer-wise combining coefficients using a small stream of data or offline validation set. The authors identify a major bottleneck in existing parameterized merging frameworks: they optimize coefficients strictly within the physical, spatial layer coordinate space. Because adjacent layers are functionally coupled, unconstrained spatial optimization exhibits high redundancy and poor conditioning, leading to wild coefficient oscillations that overfit catastrophically under data scarcity (the "Overfitting-Optimizer Paradox").

To resolve this, the paper presents **SpectralMerge**, a framework that maps the layer-wise merging coefficient profile to its spectral representation using the Discrete Cosine Transform (DCT-II). The authors propose two frequency-domain regularizations:
1. **SpectralMerge-LP**: An analytical low-pass filter that restricts trainable parameters to the first $F$ low-frequency DCT coordinates, completely eliminating high-frequency degrees of freedom.
2. **SpectralMerge-Reg**: Optimizes all spectral coordinates but adds a quadratic **Spectral Decay Penalty** ($\lambda_j = \mu \cdot j^2$) to softly penalize high-frequency spatial oscillations.

The framework is evaluated across a continuous Vision Transformer (ViT-B/32) simulation landscape, actual physical Heterogeneous Multi-Layer Perceptrons, and actual pre-trained ResNet-18 checkpoints on CIFAR-10 classification. The results demonstrate that SpectralMerge-Reg and LP achieve state-of-the-art multi-task accuracy, exhibit high resilience to validation selection bias and non-stationary stream noise, and successfully mitigate validation overfitting.

---

## Strengths and Weaknesses

### Strengths
1. **Elegant Conceptual Framing**: Re-parameterizing layer-wise merging coefficients in the frequency domain is highly creative, refreshing, and theoretically elegant. It shifts model-merging optimization from a physical coordinate space to a spectral trajectory space.
2. **Mathematical Rigor**: The paper provides exceptional mathematical justification for its choices. The proof showing that the DCT-II's even symmetric boundary extension mathematically guarantees flat spatial derivatives ($\frac{d\alpha}{dl}=0$) at virtual boundaries is excellent. This explains why DCT-II outperforms other transforms like the Discrete Sine Transform (DST) by preventing boundary gradient spikes.
3. **Perfect Numerical Conditioning**: The paper mathematically and empirically proves that the orthonormal DCT-II basis achieves a condition number of exactly $1.0$ at all scales, bypassing the severe ill-conditioning and optimization instability that plagues polynomial-based spatial smoothing (PolyMerge).
4. **Statistical Rigor in Simulation**: Evaluating all simulation results over **30 independent random seeds** with standard deviations, and multi-axially stress-testing the model under adversarial stream conditions (extreme label shift, bursty streams, small batch noise) is exceptionally thorough and sets a high empirical standard.
5. **Diagnostic Honesty and Insight**: The authors provide an incredibly honest and intellectually deep discussion on the failure of SpectralMerge-LP on the pre-trained ResNet-18 checkpoints. They correctly diagnose this using digital signal processing principles as a **PEFT-Induced Step-Function Discontinuity** (caused by localized updates), explaining why soft spectral regularization (SpectralMerge-Reg) is mathematically required to succeed where hard cutoffs fail.
6. **Value-Added Practical Extensions**: Introducing and evaluating **Block-wise Spectral Merging** (to handle heterogeneous MHA and MLP layer types) and **Adaptive Bandwidth (LP-Adaptive)** provide highly practical guidelines for practitioners.

### Weaknesses
1. **Lack of Statistical Rigor on Physical Networks**: While the simulation benchmark is statistically flawless (using 30 seeds with standard deviations), the actual PyTorch physical experiments (the 12-layer MLP in Section 4.6 and the ResNet-18 checkpoints in Section 4.7) **do not report standard deviations, confidence intervals, or the number of random seeds evaluated**. Reporting single point estimates (e.g., 54.00% accuracy in Table 3) makes it difficult to assess the statistical reliability of the physical deep network experiments.
2. **Small Scale of Physical Datasets**: The physical validation is limited to a synthetic multi-task dataset on MLP and a highly simplified split of CIFAR-10 (two binary classification tasks) on ResNet-18. Evaluating on larger modern models (e.g., standard Vision Transformers on VTAB or LLMs on GLUE/MMLU benchmarks) would provide a far more convincing demonstration of practical scalability.
3. **Modest Absolute Performance on Real-World Tasks**: In the ResNet-18 CIFAR-10 experiment, although SpectralMerge-Reg achieves a massive blowout improvement over spatial and polynomial baselines (54.00% vs. 29.00%), the absolute multi-task accuracy is still far below individual task-specific expert accuracies (86.00% and 65.00%). This indicates that while validation overfitting is resolved, significant task interference and representation clashes remain.
4. **Lack of Hyperparameter Sensitivity Sweep**: The paper introduces critical hyperparameters (the frequency cutoff $F$ in LP, and the soft decay weight $\mu$ in Reg), but does not provide an ablation or sensitivity study demonstrating how performance fluctuates across different values of these hyperparameters.

---

## Detailed Evaluation of Dimensions

### Originality: Excellent
The paper provides a highly original conceptual contribution. While "spectral" methods exist in model merging, they typically involve SVD decomposition on high-dimensional model weight matrices directly. SpectralMerge, by contrast, operates on the *1D sequence of layer-wise task-combining coefficients across network depth*. This low-dimensional re-parameterization is highly creative and computationally negligible ($<0.0001\%$ of a forward pass). Furthermore, the comparison to continuous spatial polynomials (PolyMerge) is beautifully articulated, and the connection between symmetric boundary conditions and gradient stability is highly novel.

### Soundness: Good
The mathematical formulation is exceptionally sound, rigorous, and correct. The proofs regarding boundary derivatives and basis conditioning are mathematically flawless and highly convincing. However, from an empirical perspective, the soundness of the physical PyTorch experiments is slightly limited. Operating validation and optimization in extreme few-shot regimes ($M=10$ or $M=15$ validation samples) is highly sensitive to the specific samples drawn. Reporting physical network results as single point estimates without standard deviations across multiple seeds prevents us from confirming the statistical soundness of the physical deep model evaluations. (If the authors can provide error bars/standard deviations for the physical experiments, this rating would easily rise to Excellent).

### Presentation: Excellent
The paper is outstandingly written, highly polished, and structured with extreme clarity. The narrative flow is engaging, moving logically from the spatial bottleneck to the spectral formulation, analytical proofs, simulation stress-testing, and physical deep network validations. Technical terms are clearly defined, and complex digital signal processing (DSP) concepts (like even boundary symmetry, step-function infinite frequency support, and basis orthogonality) are explained in deep learning terms with exceptional lucidity. The figures and captions are descriptive and self-contained.

### Significance: Good
The paper addresses an important and highly relevant problem in model merging (the Overfitting-Optimizer Paradox). The proposed solution (SpectralMerge) is computationally cheap, elegant, and highly generalizable. The ideas proposed in Future Directions (such as extending the 1D DCT-II to a 2D DCT-II across depth and task experts to handle larger model pools, or using Wavelet-based localized multi-resolution decompositions to handle PEFT step discontinuities) are incredibly visionary and likely to inspire substantial future work. The rating is set to Good because the physical experiments are currently conducted on a smaller scale (MLP and ResNet-18 on CIFAR-10 binary tasks), which slightly limits our ability to verify its immediate impact on large-scale foundation models.

---

## Overall Recommendation

**Rating**: 5: Accept

**Justification**: This is a technically solid, highly innovative, and beautifully written paper that represents a significant contribution to the sub-area of model merging and multi-task parameter consolidation. The concept of re-parameterizing layer-wise combining coefficients in the frequency domain via DCT-II is elegant, mathematically sound, and computationally negligible. The paper successfully refutes the Overfitting-Optimizer Paradox under extreme sample complexity, and the mathematical justifications for basis conditioning and boundary flat derivatives are highly convincing. 

While there are some empirical limitations—most notably the lack of statistical error bars in the physical PyTorch experiments and the smaller scale of the physical datasets (CIFAR-10) compared to full-scale foundation benchmarks—the outstanding rigor of the simulation experiments (30 seeds, standard deviations, multiple severe adversarial streams) and the deep diagnostic honesty surrounding the failure of low-pass filters on localized updates more than justify an acceptance. This paper is highly likely to inspire a new paradigm of frequency-domain parameter management.

---

## Questions and Suggestions for Authors

1. **Statistical Error Bars on Physical Experiments**: Can you run the Heterogeneous MLP (Section 4.6) and the ResNet-18 CIFAR-10 (Section 4.7) experiments over multiple random validation seeds (e.g., 5 or 10 runs) and report the standard deviations in Table 3 and Figure 7? This would provide the necessary empirical backing to confirm the statistical reliability of the physical network findings.
2. **Hyperparameter Sensitivity Sweeps**: It would greatly strengthen the paper to include a brief sensitivity analysis of the core hyperparameters:
   - How does the accuracy of SpectralMerge-LP change as the cutoff frequency $F$ varies (e.g., $F \in \{1, 2, 3, 4, 5, 6\}$)?
   - How sensitive is SpectralMerge-Reg to different values of the global decay weight $\mu$ (e.g., sweeping $\mu \in [0.1, 10.0]$)?
3. **Baseline Tuning**: For the PolyMerge baseline, were learning rates and training iterations swept to ensure a fair comparison? Since the Vandermonde matrix is highly ill-conditioned, optimization can stall or require vastly different hyperparameters than the perfectly conditioned DCT basis. Confirming that PolyMerge was fully tuned would strengthen the fairness of your comparisons.
4. **Hybrid Sparsification Evaluations**: In Section 2.1, you claim that SpectralMerge is conceptually orthogonal to, and synergistic with, sign-and-magnitude-based sparsification heuristics (like TIES-Merging or DARE). Have you attempted to run a hybrid experiment (e.g., applying TIES sign consensus first, and then optimizing scaling coefficients via SpectralMerge-Reg)? Including even a single such hybrid result would empirically validate this highly promising synergy.
