# 3. Technical Soundness and Methodology Evaluation

## Clarity of Description
The methodology of the paper is described with exceptional clarity. The mathematical formulations are complete and standard notation is used:
- Per-channel symmetric quantization is clearly specified.
- The decomposition of task vectors into dense and sparse components is well-defined.
- The optimization problem, loss function, and parameter initialization are documented.
- The hardware deployment complexity and memory/latency implications are thoroughly elaborated in Section 3.4.

## Appropriateness of Methods & Potential Technical Flaws

### 1. Asymmetrical Weight Scaling and Lack of Equivalence
The paper introduces a diagonal scaling matrix $D_l$ applied to the *right* side of the dense task vector updates inside the quantization operator:
$$W_{l, \text{hybrid}}(D_l, \lambda) = Q_b\left(W_{l, \text{base}} + \left( \sum_{t=1}^T \lambda_t \Delta W_{t, l, \text{dense}} \right) D_l \right) + \sum_{t=1}^T \lambda_t \Delta W_{t, l, \text{outlier}}$$

There are two major methodological issues with this formulation:
- **Asymmetric Scaling:** The scaling matrix $D_l$ scales the column features of the *task-vector updates*, but it does **not** scale the base weights $W_{l, \text{base}}$. Since the base weights represent a significant portion of the total weight magnitude and structure, scaling only the updates modifies the relative magnitude and importance of the task-specific features relative to the pre-trained features. This asymmetry is not mathematically or conceptually justified.
- **Lack of Mathematical Equivalence (Function Drift):** In traditional PTQ methods (e.g., SmoothQuant), scaling weights by $D_l$ is accompanied by scaling activations by $D_l^{-1}$ to ensure the unquantized function remains mathematically identical ($Y = (X D_l^{-1}) (D_l W) = X W$). In QP-Merge, the authors permanently apply $D_l$ to the weight updates but do **not** apply any inverse scaling to the activations at inference time. This means they are permanently altering the unquantized model's underlying mapping.
- **The Fine-Tuning Reality:** Because $D_l$ modifies the model's function and is optimized on a calibration set using gradient descent to minimize embedding MSE, this technique is **not** a standard "activation scale calibration" as claimed. It is literally a form of unsupervised parameter fine-tuning on a small dataset (optimizing $d_{\text{in}}$ parameters per layer, plus the merging weights $\lambda$). Calling it "scale calibration" masks the fact that the unquantized FP32 function mapping is being directly distorted and optimized to fit the calibration data.

### 2. The Redundancy and Impracticality of the ORD Path
The primary technical selling point of QP-Merge is its dense-quantized + sparse-high-precision hybrid representation (Outlier-Residual Decoupling, ORD). However, the paper's own ablation study (Table 3, seed 2026) reveals that **ORD is practically redundant**:
- **Tiny Performance Delta:** In INT4 mode, the average accuracy of the Full QP-Merge model is $94.52\%$, while the "No ORD" ablation (which runs QE-Calib but does not decouple outliers, meaning $\gamma = 1.0$) achieves $94.49\%$. The difference is a minuscule **$0.03\%$** average accuracy!
- **Domain-Specific Analysis:** On the more challenging SVHN task, No ORD achieves $89.90\%$ compared to $90.08\%$ for the full model (only $0.18\%$ difference). On the MNIST task, No ORD is actually *better* ($99.08\%$ vs. $98.96\%$).
- **INT8 Redundancy:** In INT8 mode, the difference is even smaller: $95.13\%$ (Full) vs. $95.12\%$ (No ORD), a $0.01\%$ difference.
- **Pragmatic Disconnect:** The authors propose a highly complex deployment architecture: routing outliers to an unquantized FP16 path, storing them in COO/CSR format, running a separate Sparse Matrix-Matrix Multiplication (SpMM) operator alongside the integer GEMM, and suffering a heavy high-level API launch overhead (~$50\ \mu$s in PyTorch). 
- **Methodological Verdict:** Given that removing ORD entirely (No ORD) yields a standard, homogeneous INT4/INT8 quantized model with **virtually identical accuracy (within $0.03\%$ on average)**, the extreme hardware complexity and latency overhead of ORD is entirely unjustified. The performance recovery is almost entirely driven by the QE-Calib optimization, making ORD a redundant and impractical addition.

### 3. Lack of Statistical Significance for Marginal Claims
In Table 4 (Sensitivity Sweep of $\gamma$ in INT4), the authors sweep the outlier threshold $\gamma$. They report:
- $\gamma = 1.0$ (No ORD): $94.49\%$
- $\gamma = 0.995$ (0.5% outliers): $94.74\%$
- $\gamma = 0.99$ (1.0% outliers): $94.71\%$
They claim there is an "optimal sweet spot at 0.5% - 5.0% density". However, these results are reported for a "single representative run (using the default calibration split)". Given that the standard deviation of the full model across 3 seeds is $\pm 0.13\%$ (Table 1), the difference between No ORD ($94.49\%$) and the "sweet spot" ($94.71\%$) is only $0.22\%$, which is within the margin of random variation across seeds. Without reporting means and standard deviations across multiple seeds for these sweeps, the claim that ORD provides a statistically significant "sweet spot" is empirically weak and unconvincing.

## Reproducibility
The paper provides a high degree of reproducibility details:
- Complete fine-tuning hyperparameters for MNISTVal and SVHNVal are documented in Table 5.
- The calibration optimizer (Adam), learning rate ($1\times 10^{-3}$), step count (100), and sample size ($M=128$) are fully specified in Appendix B.
- The mathematical formulation is explicit enough for an experienced ML engineer to reimplement.
- However, no source code repository or anonymized link is provided, which is a minor limitation for instant verification.
