# 3. Soundness and Methodology

## Clarity of Description and Appropriate Methods
* **Clear Formulation:** The mathematical formulations of SD-Scale, RMS-Scale, and their parameter-free (PF) counterparts are highly clear, structured, and easy to follow.
* **Translation-Invariance and RMS-Scale:** The paper provides a sound mathematical justification for choosing RMS-Scale over SD-Scale. Since standard deviation is translation-invariant (due to mean subtraction), it can become unstable on small, low-variance tensors like bias vectors. RMS-Scale resolves this by incorporating both variance and mean coordinate shift, ensuring absolute stability.
* **Frobenius-Norm Equivalence:** The proof showing that element-wise RMS normalization is equivalent to parameter-count-scaled Frobenius-norm normalization is highly elegant and rigorous, linking element-wise scaling to more complex Riemannian manifold projections.
* **Application to LoRA:** The theoretical analysis of applying this scaling framework to LoRA adapters (Section 3.6) is well-thought-out, explicitly highlighting the mathematical unsoundness of independent factor normalization and proposing sound workarounds (Reconstructed Weight Merging and sequential layer-wise processing to bound memory).

---

## Potential Technical Flaws and Methodological Gaps

### 1. Simulated vs. Real Task Updates in the CLIP ViT-B/32 Evaluation
In Section 4.5, the authors scale their evaluation to OpenAI's CLIP ViT-B/32. However, a close look reveals a significant methodological gap:
* **Artificial Update Vectors:** Rather than fine-tuning 3 independent real downstream expert models (e.g., on different visual domains or tasks), the authors **simulated** task updates.
* **Potential Isotropic Bias:** The paper states: *"we simulate $K=3$ task expert updates fine-tuned under uncoordinated downstream schedules, generating severe parameter update scale mismatches with RMS scales of $0.1$, $0.5$, and $2.0$ respectively."* If these simulated updates are generated as isotropic random noise (e.g., random Gaussian matrices), they will naturally be perfectly orthogonal in high-dimensional space.
* **Artificial Orthogonal Convergence:** This potential isotropic simulation explains why the alignment ratios $\alpha^l$ in Figure 2(b) for CLIP converge so perfectly and stably to the orthogonal limit ($1/\sqrt{3} \approx 0.5774$). Real-world fine-tuning updates are highly structured, localized, and typically lie in low-dimensional manifolds. They are not random isotropic noise. By evaluating on simulated updates, the paper may have artificially forced the orthogonal limit to hold, presenting an over-simplistic picture of parameter-free scaling that may not hold during real downstream merges.

### 2. Lack of Downstream Evaluation on CLIP
* **No Accuracy Metrics:** In Section 4.5, the CLIP evaluation is restricted to **Wall-Clock Time** and **Activation-Space Cosine Alignment**. The authors do **not** report any actual downstream task classification accuracies (e.g., zero-shot or linear probe accuracy on ImageNet, Stanford Cars, DTD, etc.) for the merged CLIP model.
* **Activation Cosine Similarity is a Proxy, Not Performance:** While activation-space cosine alignment is a helpful proxy, it is not a direct substitute for final task accuracy. Without end-to-end downstream accuracy evaluation, the claim that RMS-Scale and PF-RMS scale successfully to deep foundation models remains empirically unproven.

### 3. Lack of Empirical Validation for LoRA Merging
* **No Experiments:** Section 3.6 dedicates significant space to outlining how to apply RMS-Scale and PF-RMS to Low-Rank Adapters (LoRA) via Reconstructed Weight Merging, sequential streaming, and post-merging SVD re-factorization.
* **Empirical Absence:** Despite this detailed theoretical treatment, the paper **does not provide any actual experiments** or empirical results for LoRA merging. Given that LoRA is the dominant paradigm for fine-tuning foundation models, the lack of empirical validation for this specific use case is a notable omission.

---

## Reproducibility
* **PyTorch Snippet:** The paper includes an elegant, self-contained 4-line PyTorch code block for RMS-Scale, which makes the core algorithm exceptionally easy to implement and reproduce.
* **Hyperparameter Details:** The authors clearly document the hyperparameters used for the baselines (e.g., 60% pruning for Ties-Merging, 40% drop for DARE, etc.) and the search ranges for the global scale ($\lambda \in [0.3, 1.5]$), which is highly beneficial for reproducibility.
* **Repository Absence:** However, there is no public repository or official code file provided in the workspace. While the core math is simple, the lack of official data loader, expert fine-tuning, and evaluation scripts makes it difficult to reproduce the exact baseline figures (especially for complex baselines like AdaMerging and SAIM) with high precision.
