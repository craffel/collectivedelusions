# Synthesized Peer Review

**Overall Recommendation:** Rating 3 (Weak Reject)
**Soundness:** Fair
**Presentation:** Excellent
**Significance:** Fair
**Originality:** Good

---

## 1. Comprehensive Summary of the Paper
The paper investigates test-time dynamic model merging, which dynamically routes latent inputs to layer-wise merging coefficients. The authors focus on vulnerabilities under data-scarce calibration splits (64 samples) and varying deployment configurations (such as mixed-task streams and small batch sizes). 

Through evaluations on a synthetic 192-dimensional representation sandbox across 10 random seeds, the paper identifies two core phenomena:
1. **Vectorization Collapse**: Dynamic routers trained/evaluated on heterogeneous batches experience a catastrophic performance drop when deployed in sample-wise vectorized streaming pipelines ($B=1$).
2. **Batch-Average Smoothing Confounder**: Batch averaging of predicted coefficients during large-batch training acts as an implicit smoothing operator that hides extreme router overfitting. Under $B=1$ vectorized streaming, this mask is removed, causing accuracy to collapse (e.g., random-initialized L3-Softmax drops to 41.09%).

To resolve this, the paper proposes **Variance-Regularized Classical Routing (VR-Router)**, utilizing a **Task-Variance Regularization ($\mathcal{L}_{VR}$)** penalty to minimize intra-task sample variance. However, through a rigorous baseline audit, the authors demonstrate that **simple zero-initialization of Softmax routing layers combined with weight decay** is the actual, sole driver of stability. A simple zero-initialized softmax baseline (\texttt{L3\_Softmax\_WellReg}) completely resolves vectorization collapse, achieving flatline stability across all batch sizes (59.16% accuracy) and performing identically to VR-Router (59.14%). Finally, the authors formalize the **Dynamic Routing Paradox**: to avoid overfitting under data scarcity, routers must be so heavily regularized that their learned coefficients barely deviate from a static uniform compromise (Mean Absolute Deviation of only 2.36%), yielding a marginal 1.16% gain over naive static Uniform Merging.

---

## 2. Major Strengths
*   **Exemplary Scientific Honesty and Transparency**: The authors are highly commended for their exceptional intellectual honesty. Rather than masking the redundancy of their proposed Task-Variance Regularization ($\mathcal{L}_{VR}$) penalty, they explicitly analyze and discuss it, showing that standard zero-initialization and weight decay drive all the generalization benefits. This is a rare and refreshing level of scientific integrity.
*   **High-Value Diagnostic Insights**: The formalization of the **Batch-Average Smoothing Confounder**, **Vectorization Collapse**, and the **Dynamic Routing Paradox** provides deep conceptual value. It demystifies the actual behavior of dynamic model-merging networks and warns the community against over-engineered routing heads.
*   **Excellent Statistical Rigor**: Evaluating all routing methods across **10 independent random seeds** and reporting full standard deviations provides a solid, highly reliable statistical foundation.
*   **Rigorous Weight Dynamics Deconstruction**: Measuring the Mean Absolute Deviation (MAD) of predicted routing coefficients (2.36%) mathematically validates the Dynamic Routing Paradox, proving that the stable router is heavily constrained to its uniform prior.
*   **Pragmatic and Realistic Outlook**: The paper realistically positions static Uniform Merging as an exceptionally strong, zero-overhead default under small calibration budgets, highlighting the major memory expansion and compute overheads of runtime parameter assembly.

---

## 3. Major Weaknesses & Critical Flaws (Identified: 3)

### Critical Flaw 1: Complete Absence of Real-World Evaluation
The entire paper is evaluated on a synthetic 192-dimensional "Analytical Coordinate Sandbox" with simulated expert classifier accuracies. No actual neural networks (e.g., Vision Transformers, CLIP, or LLMs) are merged, and no real datasets (e.g., MNIST, CIFAR-10, SVHN, or GLUE) are processed.
*   *Why this is critical*: In real-world model merging, parameter-space sign/gradient conflicts, feature transferability, and task geometries are highly non-linear, high-dimensional, and anisotropic. Simulating this via orthogonal task prototypes and additive isotropic Gaussian noise is a severe oversimplification. The paper's core claims regarding Vectorization Collapse, flatline stability, and the Dynamic Routing Paradox must be validated on real, pre-trained weights and real datasets to prove their scientific generalizability and validity.

### Critical Flaw 2: Redundancy and Negligible Performance of the Proposed Method
The paper is titled "Variance-Regularized Classical Routing" and frames "VR-Router" as a primary contribution. However, the authors' own experiments show that the proposed Task-Variance Regularization ($\mathcal{L}_{VR}$) penalty is **empirically useless and redundant**:
*   In Table 1, VR-Router achieves **59.14% ± 1.18%** under Heterogeneous $B=256$, which is slightly **worse** than both QWS-Merge (59.41% ± 1.38%) and the well-regularized standard softmax baseline \texttt{L3\_Softmax\_WellReg} (59.16% ± 1.17%).
*   In Table 4, the ablation study shows that Cross-Entropy only training (\texttt{CE\_only}) gets **59.18% ± 1.25%**, while adding the VR penalty (\texttt{CE\_plus\_VR}) drops performance to **59.16% ± 1.25%**.
*   *Why this is critical*: The proposed method does not improve performance; instead, it is a dead weight that slightly degrades accuracy. Naming the paper after and proposing a redundant, ineffective loss penalty as the primary methodological contribution is highly problematic. The true contribution of the paper is the well-regularized standard softmax baseline and the diagnostic analysis, which is mismatched with the current method-focused title and framing.

### Critical Flaw 3: Infeasible Computational and Memory Scaling of Sample-Wise Parameter Assembly
The paper relies on "true sample-specific parameter assembly during training and deployment" to bypass batch-averaging and heterogeneity collapse:
$$W_{\text{merged}}^{(l)}(b) = W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_{k, b}(l) V_k^{(l)}$$
*   *Why this is critical*: While this mathematically decouples routing from batch size, it is **computationally and memory-wise infeasible** for real-world large-scale models. If we have a batch size of $B=64$ and a backbone with 100M+ parameters, assembling $B$ separate copies of the network's weights to process a single batch scales the model's active parameter footprint by $B$ times. Storing and processing 64 copies of a model in GPU memory destroys any operational efficiency, makes execution heavily memory-bandwidth bound, and causes severe out-of-memory (OOM) errors. The authors must explicitly address and benchmark this severe hardware-level bottleneck instead of briefly dismissing it in footnotes.

---

## 4. Actionable and Constructive Suggestions for Improvement
1.  **Run Experiments on Real Neural Networks and Datasets**: To achieve conference-grade soundness, the authors must validate their findings on real-world models. They should fine-tune a compact backbone (such as ViT-Tiny or ViT-Base) on standard tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) and merge the weights. If they can show that zero-initialization and weight decay resolve vectorization collapse and that the Dynamic Routing Paradox persists on real weight manifolds, the paper's scientific value will be exceptionally high.
2.  **Reframe the Paper and Title Around the Diagnostic Audit**: Given that VR-Router's $\mathcal{L}_{VR}$ penalty is redundant, the authors should rephrase the paper to focus on **deconstructing dynamic model-merging** and **revealing the Batch-Average Confounder**. They should rename the paper (e.g., "Demystifying Dynamic Model Merging: An Empirical Audit of Confounders and the Dynamic Routing Paradox") and present the well-regularized softmax baseline as the primary, minimal solution. VR-Router should be reframed as a candidate loss that they rigorously prove to be redundant, highlighting their diagnostic findings rather than overselling an ineffective method.
3.  **Align Claims with Empirical Data**: The authors must correct claims in the text that contradict their tables. For instance, in Section 2, the text claims VR-Router "significantly and statistically outperforms all other dynamic routers," but Table 1 shows it is outperformed by QWS-Merge and L3-Softmax-WellReg. Additionally, the critique of QWS-Merge's performance should be toned down; a drop to 56.19% under $B=1$ is a mild degradation, not a "catastrophic collapse."
4.  **Provide Hardware and Latency Benchmarks for Sample-wise Assembly**: The authors should conduct a concrete wall-clock time and GPU memory benchmark comparing static Uniform Merging, batch-averaged dynamic merging, and sample-wise vectorized dynamic merging. This will provide practitioners with transparent, hardware-level metrics regarding the true cost-benefit ratio of dynamic routing.

---

## 5. Key Questions for the Authors
1.  Why is the paper titled and framed around "Variance-Regularized Classical Routing" (VR-Router) when your own ablation study shows that the task-variance regularization penalty ($\mathcal{L}_{VR}$) actually reduces accuracy by 0.02% and is entirely redundant?
2.  How does the proposed sample-wise parameter assembly scale to large-scale vision or language models in terms of GPU memory consumption and wall-clock latency? What is the peak memory usage during training when assembling $B$ separate weight copies?
3.  Do your findings regarding the dominance of zero-initialization, the redundancy of $\mathcal{L}_{VR}$, and the 1000-sample resolution threshold hold when merging actual neural networks trained on real-world images?
