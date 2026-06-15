# Peer Review

## 1. Summary of the Paper
This paper addresses the problem of **task interference** in parameter-space **model merging** (e.g., combining task-specific expert neural networks into a single multi-task model without training or data access). The authors argue that a primary cause of interference is **representation scale mismatch** across layers and tasks, which is caused by uncoordinated fine-tuning schedules.

To resolve this, the authors propose a minimalist, training-free approach:
* **Standard-Deviation Scaling (SD-Scale):** Normalizes task vectors layer-wise to unit standard deviation and projects the merged update back using the average of the original standard deviations.
* **Root-Mean-Square Scaling (RMS-Scale):** A mathematically stable, non-translation-invariant counterpart to SD-Scale that avoids potential division-by-zero on small, low-variance bias tensors by using RMS normalization.
* **Parameter-Free Analytical Scale Calibration (PF-RMS):** A completely parameter-free variant that dynamically derives local layer-wise scaling factors ($\lambda^l = 1/\alpha^l$) by inverting the layer-wise alignment ratio to counteract high-dimensional vector shrinkage, with a clipping safeguard $\gamma(K) = C\sqrt{K}$ to prevent over-amplification of extreme conflicts.

The authors prove that element-wise RMS normalization is equivalent to parameter-count-scaled Frobenius-norm normalization. They evaluate their method on a custom SimpleCNN using MNIST, FashionMNIST, and KMNIST, showing that RMS-Scale and SD-Scale perform comparably to complex SVD-based Isotropic Merging. They also evaluate on OpenAI CLIP ViT-B/32 weight layers, demonstrating that RMS-Scale matches SVD Isotropic activation alignment with a 100$\times$ wall-clock speedup.

---

## 2. Strengths
* **Minimalist and Practical Philosophy (Occam's Razor):** The paper makes a compelling case against the escalating computational and algorithmic complexity in recent model-merging methods. Proposing a simple, two-line element-wise PyTorch scaling method that runs in linear time $O(K \cdot N)$ is highly practical and valuable.
* **Elegant Parameter-Free Formulation (PF-RMS):** The analytical derivation of the layer-wise alignment ratio $\alpha^l$ and the dynamic scaling factor $\lambda^l = 1/\alpha^l$ is conceptually beautiful. Showing that the alignment ratio naturally tracks the high-dimensional orthogonal limit ($1/\sqrt{K}$) connects practical model merging directly to high-dimensional geometry.
* **Theoretical Equivalence Proof:** Proving the exact mathematical equivalence between element-wise RMS scaling and Frobenius-norm scaling provides a solid theoretical foundation that connects this lightweight approach to complex manifold alignments.
* **Efficiency Analysis on Real Weights:** Executing a physical wall-clock and activation alignment comparison on real-world OpenAI CLIP ViT-B/32 weight matrices successfully demonstrates that RMS-Scale can achieve identical activation-space alignment as SVD-based merging while bypassing its $O(d^3)$ cubic complexity (yielding a 100$\times$ wall-clock speedup).
* **Excellent Writing and Structure:** The paper is exceptionally clear, precise, and structured. The mathematical derivations are rigorous and easy to follow, and the authors are highly transparent about limitations and the multi-task performance trade-off on FashionMNIST.

---

## 3. Weaknesses (Major Empirical Concerns)

While the paper possesses strong conceptual and theoretical merits, its empirical validation exhibits several major limitations and gaps that must be addressed to support its central claims:

### A. Toy-Scale Primary Evaluation
* **Insignificant Model Scale:** The primary end-to-end evaluation is restricted to a custom `SimpleCNN` containing only 500,000 parameters. Grayscale 28x28 classification (MNIST, FashionMNIST, KMNIST) represents an extremely small, toy-scale setting. 
* **Lack of Real-World Generalization Proof:** Model merging is fundamentally motivated by the prohibitive cost of serving multiple large, deep foundation-scale models (like multi-billion parameter LLMs or deep Vision-Language models). Demonstrating performance on a 500k-parameter CNN does not prove that the method's advantages (such as mitigating scale mismatches or handling parameter conflicts) will generalize to modern, deep architectures with complex multi-head attention and feed-forward layers.

### B. Statistical Insignificance on the Toy Benchmark
* **Overlapping Standard Deviations:** In Table 1, the seed-to-seed variance across the 3 independent random seeds is exceptionally high, which heavily obscures the claimed benefits of the proposed methods.
  * For the validation-tuned setup, Task Arithmetic achieves $72.50 \pm 1.17\%$, while RMS-Scale (Ours) achieves $73.22 \pm 2.15\%$. The difference ($0.72\%$) is substantially smaller than the standard deviations, meaning the confidence intervals heavily overlap.
  * Comparing RMS-Scale ($73.22 \pm 2.15\%$) to SVD Isotropic ($73.13 \pm 2.49\%$), the difference of $0.09\%$ is completely negligible and statistically insignificant.
  * For the un-tuned setup, default Ties-Merging ($71.81 \pm 1.73\%$) and default Task Arithmetic ($71.68 \pm 1.36\%$) have overlapping confidence intervals with PF-RMS ($72.23 \pm 2.25\%$).
* **Weak Empirical Support:** Given these heavily overlapping standard deviations and small mean differences on a toy benchmark, the empirical results do not robustly support the claim that the proposed scaling methods clearly outperform existing training-free baselines (like Task Arithmetic and Ties-Merging).

### C. Artificial "Simulated" Updates in CLIP Evaluation
* **Lack of Real Downstream Experts:** In Section 4.5, the authors scale their evaluation to real-world CLIP ViT-B/32 weight tensors. However, rather than fine-tuning 3 real downstream expert models (e.g., on real downstream datasets), they **simulated** task updates.
* **Isotropic Bias and Orthogonal Convergence:** The paper states: *"we simulate $K=3$ task expert updates... generating severe parameter update scale mismatches with RMS scales of $0.1$, $0.5$, and $2.0$ respectively."* If these simulated updates are generated as isotropic random noise (e.g., random Gaussian matrices), they will naturally be almost perfectly orthogonal in high dimensions. This artificial simulation explains why the alignment ratios $\alpha^l$ in Figure 2(b) converge so perfectly to the theoretical orthogonal limit ($1/\sqrt{3} \approx 0.5774$). Real-world fine-tuning updates are highly structured and lie on low-dimensional manifolds; they are not random isotropic noise. Evaluating on simulated updates might have artificially forced the orthogonal limit to hold, presenting an over-simplistic picture of parameter-free scale calibration.
* **No Downstream Generalization Metrics:** The CLIP evaluation is restricted to Wall-Clock Time and Activation-Space Cosine Alignment on an activation batch of size 100. The authors do **not** report any actual downstream task classification accuracies (e.g., zero-shot or linear probe accuracies on Stanford Cars, DTD, SUN397, etc.) for the merged CLIP model. Activation alignment is a proxy, not a direct measure of model utility. Without end-to-end classification performance, the claim that RMS-Scale and PF-RMS scale successfully to deep foundation models remains empirically unproven.

### D. Lack of Empirical Validation for LoRA Merging
* **Theoretical Discussion only:** Section 3.6 contains a highly detailed and mathematically sound discussion of applying the scaling framework to Low-Rank Adapters (LoRA) (including factorized scaling and post-merging SVD re-factorization).
* **Complete Absence of Experiments:** Despite this detailed treatment, there is **not a single experiment** or empirical result evaluating LoRA merging. Given that LoRA is the dominant paradigm for adapter-based fine-tuning and model merging in foundation models, the complete lack of empirical verification for this specific setup is a notable omission.

### E. Baseline Anomalies and Suspected Tuning Deficits
* **Ties-Merging Tuning Anomaly:** In Table 1, Ties-Merging (Validation-Tuned) achieves $71.77 \pm 2.06\%$, which is actually *worse* than default Ties-Merging ($71.81 \pm 1.73\%$). Since the validation search space includes the default $\lambda=1.0$, the tuned version should mathematically perform equal to or better than the default. This anomaly suggests an issue with the validation-set tuning pipeline (such as severe overfitting to a tiny validation set or a bug in the evaluation script).
* **AdaMerging Collapse:** AdaMerging yields only $62.79 \pm 6.64\%$ average accuracy, representing a massive collapse compared to standard linear averaging. In standard merging benchmarks, AdaMerging is a strong baseline. Its severe collapse on this toy CNN suggests that AdaMerging's hyperparameters (learning rate, entropy steps, batch size) were under-tuned or poorly calibrated.

---

## 4. Questions and Requested Clarifications for the Authors
1. **Details on CLIP Simulated Updates:** How exactly were the $K=3$ simulated task expert updates in Section 4.5 generated? Were they generated as random isotropic Gaussian noise? If so, how can the authors guarantee that this simulation represents the highly structured, low-dimensional parameter manifold of real fine-tuning updates?
2. **End-to-End CLIP Downstream Evaluation:** Can the authors provide actual classification accuracy results (e.g., zero-shot or linear-probe accuracies) on standard downstream visual tasks (such as ImageNet, Stanford Cars, or DTD) for a merged CLIP model fine-tuned on those tasks, rather than relying solely on simulated activation-space alignment?
3. **LoRA Empirical Verification:** Can the authors run a small-scale experiment (even on the SimpleCNN model) fine-tuning experts using LoRA and applying their Reconstructed Weight Merging or factorized scaling to empirically verify the LoRA claims made in Section 3.6?
4. **Ties-Merging Tuning Anomaly:** Can the authors explain why Ties-Merging (Validation-Tuned) performed worse than the default configuration in Table 1? What was the size and split of the validation dataset used for hyperparameter grid searching?
5. **AdaMerging Parameters:** What hyperparameter configurations (learning rate, optimization step count, etc.) were used for AdaMerging? Was a sensitivity study conducted to ensure that its severe performance collapse was not simply due to under-tuning?

---

## 5. Ratings

### Soundness: Fair
* **Justification:** The mathematical derivations, standard deviation instability analysis, and Frobenius-norm equivalence proof are solid and correct. However, the empirical evaluation is fundamentally weak. The primary benchmark is toy-scale, the improvements over baselines are statistically insignificant (overlapping standard deviations), and the CLIP evaluation is based on simulated updates with no downstream classification accuracy metrics.

### Presentation: Excellent
* **Justification:** The paper is exceptionally well-written, clearly structured, and easy to read. The mathematical notation is consistent, and the figures and tables are beautifully presented and informative.

### Significance: Fair
* **Justification:** Resolving scale mismatches is an important problem, and a minimalist, linear-time parameter-free method has substantial practical potential. However, because the empirical success is demonstrated only on a 500k parameter CNN and simulated CLIP weights, the practical significance of the contribution to modern foundation-scale model merging remains unproven.

### Originality: Good
* **Justification:** Normalization and scale calibration are standard concepts, but the analytical derivation of a dynamic parameter-free scaling factor based on the alignment ratio and its convergence to the high-dimensional orthogonal limit are novel, elegant, and highly creative insights.

---

## 6. Overall Recommendation
**3: Weak Reject**

* **Justification:** The paper presents an elegant, mathematically sound, and computationally highly efficient approach to resolving scale mismatches in model merging. Its minimalist philosophy is commendable, and the theoretical proofs and parameter-free derivations are of high quality. However, as an empirical piece of work, the evidence supporting its main claims is currently inadequate. The primary evaluation is on a toy scale, the baseline results exhibit statistical insignificance due to heavily overlapping standard deviations, and the CLIP foundation model evaluation relies entirely on simulated updates without actual downstream task accuracies. Additionally, there are anomalies in the baseline comparisons. If the authors can address these empirical gaps—specifically by evaluating on real (non-simulated) fine-tuned foundation models on downstream tasks, running actual LoRA merging experiments, and resolving baseline anomalies—the paper would be an exceptionally strong contribution. In its current form, however, the weaknesses in the empirical setup outweigh the conceptual merits.
