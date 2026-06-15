# Peer Review: Demystifying Test-Time Dynamic Model Merging

## 1. Summary of the Paper
This paper presents a rigorous empirical and conceptual deconstruction of **test-time dynamic model merging** (such as L3 routing) under low-data calibration splits (64 samples). The authors expose a critical methodological and physical vulnerability in current evaluation standards: the **Batch-Average Smoothing Confounder**. Under standard large-batch evaluation ($B=256$), the batch-averaging of predicted layer-wise coefficients acts as an implicit smoothing operator that masks severe overfitting. When evaluated in true, real-time batch-independent or sample-wise vectorized pipelines ($B=1$), this smoothing mask is removed, leading to a catastrophic **Vectorization Collapse** (e.g., standard random-initialized L3-Softmax drops to 41.09% accuracy, nearly 17% below naive Uniform Merging).

To address this, the authors introduce a **Prior-Driven Classical Routing Framework** (specifically **Zero-Initialized Softmax Routing** with $L_2$ weight decay), a low-dimensional unit-state feature projection, and sample-specific vectorized assembly during calibration. Within this framework, they introduce two regularization losses: **Task-Variance Regularization ($\mathcal{L}_{VR}$)** to suppress intra-task routing variance, and **Sequential Smoothness Regularization ($\mathcal{L}_{\text{smooth}}$)** to suppress sequential layer-to-layer routing weight fluctuations. 

Crucially, the authors show that standard L3-Softmax, when properly zero-initialized and regularized with weight decay, completely resolves vectorization collapse and performs identically to VR-Router, proving that simple architectural priors are the true, sufficient drivers of stability, and making explicit training losses redundant. Finally, they formulate the **Dynamic Routing Paradox**: under data scarcity, a dynamic router must be regularized so heavily that its learned coefficients are constrained to stay in an extremely tight, high-entropy neighborhood of the static uniform compromise (MAD of 0.0236 from the 0.25 baseline). This limits the router's functional capacity, yielding a tiny +1.16% joint accuracy gain over training-free Uniform Merging, raising critical questions about whether the massive systems-level memory ($O(B \cdot M)$ expansion) and GPU memory-bandwidth bottlenecks of real-time parameter assembly are worth the marginal performance gains.

---

## 2. Strengths and Weaknesses

### Originality (Excellent)
*   **Strengths**:
    *   **The Dynamic Routing Paradox**: This represents an outstanding, paradigm-clarifying conceptual leap. By demonstrating that a dynamic router must be heavily constrained to its uniform prior to generalize on data-scarce splits, the paper exposes a fundamental limit of the entire test-time dynamic merging literature. This moves the community past minor incremental tweaks and forces a critical re-evaluation of the core paradigm.
    *   **The Batch-Average Confounder & Vectorization Collapse**: Exposing that high-batch evaluations mask severe overfitting is a profound methodological contribution. Defining "Vectorization Collapse" under $B=1$ vectorized streaming provides the community with a vital new evaluation metric and standard.
    *   **Outstanding Intellectual Honesty**: The authors' willingness to show that their own proposed Task-Variance Regularization ($\mathcal{L}_{VR}$) loss is empirically redundant because the simple zero-initialized Softmax prior carries all the regularizing weight is refreshing, transparent, and of high scientific value.
*   **Weaknesses**: None identified. The originality and conceptual depth of this paper are exceptional.

### Significance (Excellent)
*   **Strengths**:
    *   **Paradigm Shifting**: Shints the focus of model merging research away from mathematical over-complexity (e.g., wave cosine activations in QWS-Merge) and towards the critical roles of proper initialization, priors, and systems-level bottlenecks.
    *   **Systems-Level Realism**: The detailed systems complexity analysis (VRAM footprint expansion, memory bandwidth bounds, and reduced GPU arithmetic intensity) coupled with low-rank parameter assembly (Dynamic LoRA) as a solution provides immense practical utility for real-world deployments.
    *   **Advocating for Naive Defaults**: Demonstrating that a well-regularized router barely beats Uniform Merging (+1.16%) establishes static Uniform Merging as an exceptionally strong, cost-free baseline that future dynamic designs must beat.
*   **Weaknesses**: None identified. The paper has the potential to reshape research standards in model merging.

### Soundness (Excellent)
*   **Strengths**:
    *   **Extreme Empirical Rigor**: Every major experiment is evaluated across **10 independent random seeds** (seeds 42 to 51) and reported with means and standard deviations, eliminating any risk of cherry-picking.
    *   **Exhaustive Sweeps**: The paper includes comprehensive sweeps over subspace overlap $\rho \in [0.0, 1.0]$, projection dimension $d \in \{2, 4, 8, 16\}$, calibration data size $|D_{\text{cal}}|$, multi-layer routing depth (MLP), and Dynamic LoRA adapter rank $r$, making the empirical evaluation exceptionally complete.
    *   **Real-World Grounding**: The inclusion of a real-world visual expert merging experiment on actual MNIST and FashionMNIST experts with a shared CNN backbone confirms that the observed phenomena (vectorization collapse, batch-average confounder, prior-layer stabilization) hold perfectly on actual neural network weights.
*   **Weaknesses (Constructive Suggestions)**:
    *   **Layer-Averaging Simplification**: In the coordinate sandbox, the layer-wise routing weights are averaged over the layer dimension ($\bar{\alpha}_k = \frac{1}{L} \sum_{l=1}^L \alpha_k(l)$) to fit the single-layer expert classifiers. While the authors are highly transparent about this simplification and successfully mitigate it by validating the Sequential Smoothness Regularizer ($\mathcal{L}_{\text{smooth}}$), evaluating the sequential routing weight jitter directly on a deep sequential model without layer-averaging is an exciting future direction.

### Presentation (Excellent)
*   **Strengths**:
    *   The paper is beautifully written, logically organized, and highly self-contained.
    *   Mathematical formulations are highly precise and complete, down to details on population vs. sample variance.
    *   Appendix A provides a detailed, comprehensive roadmap for reproducing the findings on real-world CLIP ViT-B/16 checkpoints, ensuring outstanding reproducibility.
*   **Weaknesses**: None. The quality of writing is stellar.

---

## 3. Detailed Ratings

*   **Soundness**: Excellent
*   **Presentation**: Excellent
*   **Significance**: Excellent
*   **Originality**: Excellent

---

## 4. Overall Recommendation

**Recommendation**: 5: Accept (Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations).

**Justification**: 
This paper is an outstanding, highly original, and exceptionally thorough contribution to the field of parameter-space model merging and dynamic routing. Instead of proposing a convoluted and marginal routing variant, this paper takes a step back to *demystify* and *deconstruct* the core assumptions of the entire sub-field. By exposing the **Dynamic Routing Paradox**, the **Batch-Average Smoothing Confounder**, and **Vectorization Collapse**, the authors provide the community with critical conceptual and methodological tools. The empirical evaluation is stellar, utilizing parallel 10-seed sweeps, exhaustive parameter sweeps, systems-level latency analyses, and real-world validation. The exceptional intellectual honesty and transparent framing of the results set a wonderful standard for deep learning research. I strongly recommend this paper for acceptance.

---

## 5. Questions and Constructive Feedback for the Authors

1.  **Scaling to Foundation Models**: While the Systems-Level Complexity section and Dynamic LoRA experiments are mathematically elegant, have you considered executing the CLIP ViT-B/16 validation roadmap outlined in Appendix A? Seeing if the empirical threshold of $|D_{\text{cal}}| \approx 1000$ samples scales identically to billion-parameter transformers in practice would be a highly valuable addition.
2.  **Sequential Routing Jitter**: Under your Sequential Smoothness Regularizer ($\mathcal{L}_{\text{smooth}}$) sweep (Table 8), you show that sequential routing weight jitter is reduced by over 57.5% with zero degradation in classification accuracy. However, due to the sandbox's layer-averaging simplification, routing jitter has no functional impact on accuracy. On a deep sequential network where weights are applied sequentially, what is your hypothesis on how sequential routing jitter affects hidden representation alignment and overall multi-task accuracy?
3.  **Test-Time Adaptation (TTA)**: Your analysis of training-free dynamic merging approaches (such as DAWIN or SE-Merging) shows that they collapse under vectorized $B=1$ streaming due to the lack of test-batch statistics. If we instead integrated self-supervised test-time adaptation (e.g., optimizing entropy minimization directly on individual input representations) to update the linear projections on the fly, do you believe this would successfully resolve the data-scarcity bottleneck of the Dynamic Routing Paradox without requiring a separate calibration phase?
