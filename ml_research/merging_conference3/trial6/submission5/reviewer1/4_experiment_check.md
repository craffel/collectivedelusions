# 4. Experimental Evaluation and Baseline Check

## Experimental Setup and Datasets
The experimental evaluation is highly rigorous and dual-faceted:
1. **Synthetic Sandbox Evaluation**: Done on a 192-dimensional **Analytical Coordinate Sandbox** with a 14-layer backbone processing representation vectors from four distinct downstream tasks with calibrated domain noise:
   * **Task 1 (Low Noise)**: $\sigma_{noise} = 0.05$ (separable; oracle ceiling $100.00\%$)
   * **Task 2 (Moderate Noise)**: $\sigma_{noise} = 0.15$ (medium difficulty; oracle ceiling $100.00\%$)
   * **Task 3 (High Noise)**: $\sigma_{noise} = 0.40$ (high difficulty; oracle ceiling $64.40\%$)
   * **Task 4 (Extreme Noise)**: $\sigma_{noise} = 1.20$ (severe domain noise; oracle ceiling $19.20\%$)
   * The joint oracle ceiling is $70.90\%$. The calibration budget is restricted to just **64 calibration samples** (16 per task).
2. **Real-World CNN Evaluation**: Executed on real image classification experts pre-trained on **MNIST** (test accuracy $91.20\%$) and **FashionMNIST** (test accuracy $77.80\%$) using a shared CNN backbone. This setup has a joint oracle ceiling of $84.50\%$ and is calibrated on exactly 64 real images (32 from each task).

---

## Baselines Checked
The paper includes a comprehensive and highly competitive suite of baselines:
* **Static Uniform Merging**: The static average of task vectors (completely training-free, Zero-Overhead).
* **Linear Router**: Unregularized global classical linear routing.
* **QWS-Merge**: State-of-the-art quantum-inspired wave superposition model \cite{qwsmerge2025} using cosine phase activations.
* **L3-Linear**: Unregularized layer-wise classical linear routing \cite{muqeeth2023l3}.
* **L3-Softmax**: Layer-wise routing with random-initialized Softmax activations.
* **L3-Softmax (Well-Reg.)**: Standard layer-wise Softmax routing trained under the same optimized zero-initialized, weight-decayed hyperparameters as VR-Router but without the explicit task-variance penalty ($\mathcal{L}_{VR} = 0$).

---

## Support of Claims by Empirical Results

The empirical results provide overwhelming, statistically robust support for every claim:

### 1. Vectorization Collapse Evidence (Table 1)
Under large heterogeneous batches ($B=256$), unregularized L3-Softmax seems to perform well ($59.35\% \pm 1.33\%$). However, when evaluated at $B=1$ (where batch-average smoothing is removed), its performance plummets to **$41.09\% \pm 3.73\%$**—nearly $17\%$ below naive Uniform Merging ($58.00\% \pm 1.13\%$). This provides clear, empirical proof of Vectorization Collapse and the Batch-Average Confounder.

### 2. Efficacy of Zero-Initialization and Weight Decay (Table 1)
Our well-regularized standard Softmax baseline (`L3_Softmax_WellReg`) completely resolves Vectorization Collapse, achieving a flat, highly-generalizing accuracy of **$59.16\% \pm 1.17\%$** across all batch sizes ($B=1$ to $B=512$). It performs statistically identically to VR-Router ($59.14\% \pm 1.18\%$), proving that the explicit variance penalty is empirically redundant once proper architectural priors are established.

### 3. Flat Regularization Sensitivity Frontier (Table 2)
Sweeping the task-variance penalty weight $\lambda_{var}$ from $0.0$ to $10.0$ across 10 random seeds yields a completely flat joint accuracy curve around $59.34\%$. This indicates that the zero-initialized Softmax prior naturally acts as a powerful implicit variance regularizer, making the explicit loss term insensitive.

### 4. Ablation Study Verification (Table 4)
Optimizing solely with cross-entropy ($\mathcal{L}_{CE}$) on 64 calibration samples yields $59.18\% \pm 1.25\%$ accuracy. Adding $L_2$ weight decay ($\mathcal{L}_{reg}$) maintains this high performance, and adding our explicit variance penalty ($\mathcal{L}_{VR}$) yields a statistically identical $59.16\% \pm 1.25\%$. This confirms that zero-initialization carries the vast majority of the regularizing weight.

### 5. Resolution of Systems-Level Latency (Table 5)
Physical latency profiling shows that naive Dynamic Full-Parameter Assembly slows down execution by **$110.06\times$** at batch size $B=512$ ($891.76$ ms vs. $8.10$ ms for Static Uniform). Low-Rank Parameter Assembly (Dynamic LoRA, $r=8$) resolves this, requiring only $8.16$ ms (a negligible $1.01\times$ slowdown) while maintaining identical multi-task accuracy.

### 6. Real-World Validation (Table 8)
On actual image datasets, both dynamic routers outperform static Uniform Merging ($80.30\%$), achieving up to $82.40\%$ accuracy. Under a large heterogeneous batch ($B=256$), the advantage is masked ($81.20\%$) due to the Batch-Average Confounder, whereas sample-specific vectorized assembly ($B=1$) recovers the full dynamic ensembling capacity ($82.40\%$). This beautifully bridges the gap between the sandbox and real deep visual networks.
