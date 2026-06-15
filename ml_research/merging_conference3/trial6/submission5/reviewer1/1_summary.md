# 1. Summary of the Submission

## Main Topic and Approach
This submission focuses on **test-time dynamic model merging**, an emerging paradigm where specialized, task-specific expert neural networks are combined on the fly at inference time. Instead of using a static, fixed set of merging coefficients across all test inputs, test-time dynamic model merging employs a lightweight routing network to map latent representations of individual input samples to layer-wise merging coefficients.

The paper investigates the severe, under-explored vulnerabilities of existing dynamic routers in data-scarce calibration splits (e.g., 64 calibration samples). In doing so, the authors expose a critical failure mode and introduce a remarkably simple, elegant, and highly effective classical regularization alternative.

Specifically, the paper introduces:
1. **Vectorization Collapse**: A phenomenon where standard, unregularized dynamic routers exhibit severe performance degradation when deployed in true, sample-wise vectorized pipelines (batch size $B=1$).
2. **The Batch-Average Smoothing Confounder**: The standard evaluation protocol under large batches ($B=256$) averages predicted coefficients, acting as an implicit smoothing operator that masks the severe overfitting of the router. Removing this mask at $B=1$ exposes the overfitting, causing Vectorization Collapse.
3. **The Dynamic Routing Paradox**: To prevent Vectorization Collapse, dynamic routers must be so heavily regularized (via zero-initialization and weight decay) that their learned coefficients stay in a tight, high-entropy neighborhood of the static uniform compromise (Mean Absolute Deviation of only $2.36\%$). This heavy regularization leaves the router with marginal functional flexibility, yielding only a tiny $+1.16\%$ joint accuracy improvement over naive, training-free, and computationally cost-free **Static Uniform Merging**. This exposes a profound trade-off between the marginal gains of dynamic routing and its massive hardware-level latency/memory overhead.
4. **Prior-Driven Classical Routing Framework**: To resolve these issues, the authors propose an elegant, prior-driven classical routing baseline (Zero-Initialized Softmax Routing with $L_2$ weight decay) and project representations onto a low-dimensional unit sphere. They show that proper architectural initialization and standard regularization completely resolve Vectorization Collapse, matching the performance of complex architectures and explicit loss penalties.
5. **Task-Variance Regularization ($\mathcal{L}_{VR}$)** and **Sequential Smoothness Regularization ($\mathcal{L}_{smooth}$)**: Group-level and sequential-level limit constraints designed to suppress intra-task routing variance and sequential routing jitter, respectively. The authors prove that the zero-initialized Softmax prior naturally and inherently satisfies these constraints without requiring explicit loss tuning.
6. **Low-Rank Parameter Assembly (Dynamic LoRA)**: A system-level mitigation that performs vectorized assembly exclusively on low-rank adapters (e.g., rank $r=10$), completely bypassing the massive VRAM footprint expansion and the $110.06\times$ latency slowdown of naive full-parameter assembly.

---

## Key Findings and Claims (with Evidence)

### 1. Verification of Vectorization Collapse and the Batch-Average Confounder
* **Claim**: Under heterogeneous streams, standard random-initialized dynamic routers overfit to data-scarce splits, which is hidden by batch-average smoothing at $B=256$ but causes catastrophic collapse at $B=1$.
* **Evidence**: In Table 1, standard random-initialized L3-Softmax achieves $59.35\% \pm 1.33\%$ accuracy under batch size $B=256$, but its performance plummets to $41.09\% \pm 3.73\%$ under $B=1$ (a $17\%$ drop below the static Uniform Merging baseline of $58.00\% \pm 1.13\%$). L3-Linear similarly drops from $58.56\%$ to $53.76\%$.

### 2. Elegance and Efficacy of the Well-Regularized Classical Prior
* **Claim**: Simple zero-initialization of Softmax routing layers combined with weight decay completely resolves Vectorization Collapse and matches the performance of complex quantum wave models or explicit variance penalties.
* **Evidence**: 
  * In Table 1, our well-regularized Softmax baseline (`L3_Softmax_WellReg`, which utilizes zero-initialization and weight decay but no explicit variance penalty) and VR-Router achieve stable, flatline joint accuracies of $59.16\% \pm 1.17\%$ and $59.14\% \pm 1.18\%$ respectively across all batch sizes ($B=1$ to $B=512$).
  * The sensitivity frontier sweep in Table 2 shows that VR-Router is completely insensitive to the explicit variance penalty weight $\lambda_{var}$, maintaining a flat $\approx 59.34\%$ accuracy from $\lambda_{var}=0.0$ to $10.0$, proving that the zero-initialized prior is the true driver of stability.
  * The ablation study in Table 4 confirms that optimizing with $\mathcal{L}_{CE} + \mathcal{L}_{reg}$ (Cross-Entropy and $L_2$ weight decay) achieves $59.18\% \pm 1.25\%$ accuracy, and adding the explicit variance loss $\mathcal{L}_{VR}$ yields a statistically identical $59.16\% \pm 1.25\%$.

### 3. Exposing the Dynamic Routing Paradox
* **Claim**: The stable, well-regularized router is constrained to stay extremely close to the static uniform prior, yielding a tiny performance gain over naive Uniform Merging.
* **Evidence**: The Mean Absolute Deviation (MAD) of the learned coefficients from the uniform $0.25$ baseline is only $0.0236$ (or $2.36\%$). This explains why the accuracy improvement of VR-Router ($59.16\%$) over naive Uniform Merging ($58.00\%$) is only $+1.16\%$, highlighting the questionable economics of full-parameter dynamic assembly (which introduces a $110.06\times$ latency slowdown).

### 4. Hardware and Latency Mitigation via Dynamic LoRA
* **Claim**: Restricting dynamic parameter assembly to low-rank adapters (LoRA) completely eliminates the hardware latency bottleneck and VRAM footprint with zero performance degradation.
* **Evidence**: 
  * Physical latency profiling (Table 5) shows that while Dynamic Full-Parameter Assembly slows down execution by $110.06\times$ at $B=512$ (taking $891.76$ ms vs. $8.10$ ms for Static Uniform), Dynamic LoRA ($r=8$) requires only $8.16$ ms (a negligible $1.01\times$ slowdown) and is actually faster than Static Uniform at $B=1$ ($1.89$ ms vs. $3.73$ ms).
  * Table 7 shows that scaling the LoRA rank to $r \ge 10$ achieves $59.26\% \pm 1.45\%$ accuracy, matching the Full-Parameter Baseline ($59.39\% \pm 1.44\%$) with zero capacity loss.

### 5. Real-World Deep CNN Validation
* **Claim**: The Batch-Average Confounder and the robustness of proper priors hold perfectly when merging actual image classification experts.
* **Evidence**: On MNIST + FashionMNIST experts, the unregularized router's accuracy drops from $81.30\%$ at $B=256$ to $82.40\%$ at $B=1$ (exposing the batch-averaging compromise). The well-regularized zero-initialized router (`L3_Softmax_WellReg`) achieves identical high accuracy ($82.40\%$) while guaranteeing convergence and robustness.
