# 5. Presentation, Impact, Strengths, and Areas for Improvement

## Major Strengths
1. **Highly Practical and Relevant Domain:** Bridging the gap between parameter-space model merging and post-training quantization (PTQ) is a critical problem for deploying multi-task foundation models on resource-constrained edge devices.
2. **Exceptionally Strong Presentation and Narrative:** The paper is extremely well-written, logically structured, and easy to follow. The mathematical notation is clean and consistent throughout.
3. **Rigorous and Honest Engineering Analysis:** Section 3.4 and 4.6 provide an incredibly refreshing, honest, and detailed hardware analysis. Discussing the VRAM footprint in COO/CSR formats, measuring wall-clock latencies, and openly reporting the $5.8\times$ PyTorch launch overhead of dynamic dense-sparse execution shows high professional integrity and practical engineering depth.
4. **Thorough Sensitivity and Generalization Tests:** The paper goes beyond standard validation by evaluating:
   - Out-of-distribution (OOD) generalization under noise and contrast shifts (Section 4.7).
   - Extreme calibration bias (100% single-domain MNIST or SVHN calibration in Section 4.8).
   - Calibration sample size sweeps (Section 4.5).
   These experiments prove the robust generalization of the QE-Calib algorithm.

## Areas for Improvement (Critical Weaknesses)

### 1. Inadequate and Non-Representative Empirical Evaluation
Evaluating strictly on **MNIST and SVHN** digit classification tasks using an 86M parameter **ViT-B-32** is an extremely weak setup:
- MNIST and SVHN are practically solved toy digit tasks that do not reflect the complexity of real-world multi-task foundation model merging.
- Standard model merging literature evaluates on 8-task vision benchmarks (ImageNet, CIFAR-100, EuroSAT, STL-10, RESISC45, etc.) or multi-task NLP benchmarks (GLUE, SuperGLUE, MMLU).
- A pre-trained ViT-B-32 has massive representational capacity relative to these digit tasks, making weight updates tiny and trivializing the merging and quantization difficulty.

### 2. Practical Redundancy of the Core Technique (ORD)
The main architectural contribution, Outlier-Residual Decoupling (ORD), is shown to be practically redundant:
- The ablation study (Table 3) shows that removing ORD ("No ORD" which keeps all weights in standard homogeneous INT4 format) only drops average accuracy by a negligible **$0.03\%$** (from $94.52\%$ to $94.49\%$).
- Since ORD introduces massive deployment complexity (storing outliers in sparse format, compilation support for SpMM, and separate kernel launch latencies), a $0.03\%$ accuracy gain (which is well within the $\pm 0.13\%$ margin of random variation) does not justify its inclusion. A standard homogeneous INT4 model with QE-Calib is much more practical.

### 3. Terminological Misdirection ("Activation Scale Calibration")
The proposed "QE-Calib" is described as "activation scale calibration" (analogous to SmoothQuant). However, because $D_l$ is applied permanently to the weight updates and is **not** accompanied by an inverse activation scaling ($D_l^{-1}$) during inference, it permanently alters the unquantized model's mapping:
- This is mathematically **not** an equivalence transformation.
- It is actually a form of unsupervised gradient-based parameter fine-tuning on the calibration set. Calling it "scale calibration" is mathematically misleading.
- Furthermore, scaling only the task updates and not the base weights $W_{\text{base}}$ is asymmetrical and lacks clear theoretical justification.

### 4. Missing Model Merging Baselines
The paper fails to compare QP-Merge against standard advanced merging baselines (e.g., Ties-Merging or DARE) followed by standard quantization. It is possible that these advanced heuristics naturally mitigate outlier scaling by resolving sign and magnitude conflicts, making hybrid decoupling unnecessary.

### 5. Lack of Statistical Rigor in Sweeps
The sensitivity sweeps over $\gamma$ and $M$ are reported for single representative runs. Given the standard deviation across seeds is $\pm 0.13\%$, many of the reported differences in these sweeps (e.g., $94.74\%$ vs. $94.71\%$) are within the margin of noise, and means/standard deviations across multiple seeds should be reported to ensure statistical validity.

## Overall Presentation Quality
**Excellent.** The paper is outstandingly clear, logically organized, and contains beautiful mathematical derivations. The figures and tables are highly polished and informative. The authors should be commended for their writing style and clear communication of complex concepts.

## Potential Impact and Significance
- **Potential (High):** Co-designing merging and quantization has massive potential to unlock real-world multi-task edge deployment.
- **Actualized Significance (Low to Moderate):** Because the empirical evaluation is restricted to toy digit classification tasks (MNIST/SVHN) and the core dense-sparse decoupling technique (ORD) is shown to be practically redundant (providing only a $0.03\%$ accuracy gain over standard homogeneous INT4), the actualized significance of the paper in its current state is limited. It provides a highly complex, hardware-unfriendly solution for a problem that can be solved just as well with simple scale optimization.
