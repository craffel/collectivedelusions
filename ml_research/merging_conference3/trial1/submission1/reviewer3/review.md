# Peer Review

**Title:** QP-Merge: Quantization-Preserving Task Vector Merging  
**Overall Recommendation:** 3: Weak Reject  
**Soundness:** Fair  
**Presentation:** Good  
**Significance:** Fair  
**Originality:** Fair  

---

## 1. Summary of the Paper
The submission addresses the "merging-quantization gap" when deploying merged multi-task models to resource-constrained edge hardware. While model merging successfully combines task-specific models in parameter space, applying naive post-training quantization (PTQ) to merged models causes catastrophic accuracy drops. The authors attribute this to (1) heavy-tailed weight outliers stretching uniform quantization scales and (2) activation scale mismatches across tasks fine-tuned on different domains.

To resolve these issues, the paper proposes **QP-Merge**, which integrates two techniques:
1. **Outlier-Residual Decoupling (ORD):** Isolates the top high-magnitude updates ($\le 1\%$) of each task vector into a sparse high-precision (FP16) tensor, leaving a range-bounded dense remainder that can be quantized to INT4 or INT8 without range stretching.
2. **Quantization-Error Aware Scale Calibration (QE-Calib):** Optimizes layer-wise diagonal weight scaling parameters ($D_l$) and merging coefficients ($\lambda$) over a tiny unlabeled set of 128 domain-balanced calibration samples. This step minimizes the mean-squared error (MSE) of final representations between the unquantized FP32 merged model and the quantized hybrid model.

Evaluating on a pre-trained `ViT-B-32` on dual-task vision classification (MNIST and SVHN), QP-Merge in INT4 mode achieves an average accuracy of 94.70% (within 0.42% of the optimized unquantized FP32 baseline), while in INT8 mode it is virtually lossless (95.14%).

---

## 2. Major Strengths
*   **Pragmatic Problem Statement:** Reconciling parameter-space model merging and post-training quantization is a highly practical and relevant research direction that addresses real-world edge-deployment constraints.
*   **Excellent Parameter Efficiency:** Outlier-Residual Decoupling demonstrates that separating an extremely sparse fraction of weight updates (only 0.5%–1.0%) in high precision is highly effective at stabilizing low-bit quantization grids.
*   **Rigorous Sensitivity Analysis:** The paper provides detailed sweeps over calibration dataset size ($M$), outlier percentile ($\gamma$), out-of-distribution shifts (Gaussian noise and contrast), and biased/imbalanced calibration domains, demonstrating a commendable effort to map out the operational boundaries of the calibration process.
*   **Detailed Scaling Analysis:** The analytical VRAM and DRAM bandwidth transfer modeling at LLM scale (e.g., LLaMA-7B on NVIDIA Jetson Orin Nano) is highly detailed and provides a solid theoretical argument for how weight footprint compression translates to serving speedups on memory-bound workloads.

---

## 3. Major Weaknesses

### A. Severe Latency Overhead (6x Slowdown) and Speculative Acceleration Claims
The abstract states that QP-Merge introduces **"near-zero storage or computational overhead on modern accelerator hardware."** However, Section 4.5's physical GPU latency profiling directly contradicts this, revealing that executing the hybrid representation (dense INT4 GEMM + sparse FP16 SpMM) slows down execution significantly:
*   **FP16 linear projection baseline:** $10.48\ \mu\text{s}$
*   **QP-Merge (0.5% outliers) latency:** $60.92\ \mu\text{s}$ (a **6x physical latency slowdown**).
A 600% increase in wall-clock latency is a critical bottleneck that completely defeats the purpose of low-bit quantization for high-speed edge serving. The authors' defense that this is "PyTorch API overhead" and would disappear under TensorRT or custom Triton compilation is **purely speculative** and unsupported by any actual implementation or hardware measurements. In real-world edge hardware, loading sparse coordinate maps and executing separate GEMM/SpMM passes introduces severe memory-hierarchy bottlenecks that cannot be hand-waved away.

### B. Overstated Novelty and Gaps in Prior Literature
The authors claim to be the **"first framework that co-designs model merging and quantization"** (Section 2.3). This is factually incorrect. Several peer-reviewed papers have recently targeted the exact same space of task vector quantization and compression:
*   **Task Vector Quantization (TVQ) [ICCV 2025]:** Explicitly co-designs model merging and low-precision (2-bit to 4-bit) quantization of task vectors. It decomposes task vectors into a high-precision shared base vector and low-precision task-specific offsets, acting as a direct competitor.
*   **1bit-Merging [2025]:** Integrates ultra-low precision (1-bit) quantization of task vectors with layer-wise importance routing to build highly efficient multi-task LLMs.
*   **Binary Task Switch (T-Switch) [CVF 2024/2025]:** Compresses task vectors down to 1-3% of their original size using binarized activation and sign masks.
The authors' failure to cite, discuss, or compare against these direct competitors is a severe bibliographic gap that falsely inflates the paper's novelty.

### C. Weak, Non-Representative "Toy" Experimental Suite
The entire empirical evaluation is restricted to a dual-task digit classification setup (**MNIST** and **SVHN**) utilizing a pre-trained **ViT-B-32** model.
*   Both datasets represent digit recognition (0-9) and share the exact same label space, making the task alignment trivial.
*   A ViT-B-32 (86M parameters) is massively over-parameterized for basic digits, meaning it is highly robust to quantization noise. MNIST can be solved to >99% accuracy with a tiny 3-layer CNN.
*   Standard model merging literature (e.g., Task Arithmetic, Ties-Merging, DARE) evaluates on an **8-task vision suite** (incorporating diverse domains like SUN397, Cars, DTD, EuroSAT, etc.) or instruction-tuning NLP suites. Drawing broad conclusions about foundation models and general model merging based solely on digit classification is highly premature and non-representative.

### D. Mathematical and Methodological Concerns in Scale Calibration
The formulation of QE-Calib in Equation 11 contains critical methodological inconsistencies:
1.  **Asymmetric Treatment of Weights:** The diagonal scaling matrix $D_l$ is multiplied only to the dense task updates, leaving the pre-trained base weight $W_{l, \text{base}}$ completely unscaled. Since base weights contain the core representation space, scaling only the task vector updates severely warps the relative alignment between base features and task adjustments, violating the assumptions of linear mode connectivity.
2.  **Loss of Mathematical Equivalence:** In traditional quantization scale search (e.g., SmoothQuant), scaling weights channel-wise is mathematically neutralized by applying an inverse scaling matrix ($D_l^{-1}$) to the activations during inference. In QP-Merge, $D_l$ is permanently applied to the weight updates without activation inverse-scaling. 
3.  **Overfitting as Calibration:** Because mathematical equivalence is lost, QE-Calib is not a pure "quantization-error calibration." It is actually acting as a **low-parameter gradient-based weight fine-tuning step**. For a ViT-B-32, optimizing diagonal scaling matrices per layer represents approximately **55,000 learnable parameters**. Optimizing 55,000 parameters over a tiny pool of **128 samples** carries an extreme risk of overfitting. Indeed, in Table 6, when calibrating on 100% MNIST-only data, SVHN accuracy collapses by **4.86%** (dropping from 90.08% to 85.22%). This vulnerability demonstrates that QE-Calib easily overfits to the calibration domain and lacks generalizability, which is a major deployment risk.

---

## 4. Questions and Gaps in Reproducibility

1.  **The Head-Merging Architecture:** MNIST and SVHN represent digits 0-9. Did the fine-tuned task models share a single classification head, or did they have separate linear classification heads? If they had separate heads, how were they merged, evaluated, or routed during the joint inference phase? The paper is silent on this.
2.  **Task Coefficient Bounding:** The paper states that learnable task coefficients are initialized to $\lambda_t = 0.3$ and optimized. Are there any constraints or regularization applied to $\lambda_t$ to prevent them from collapsing to zero or exploding during the 100-step alignment?
3.  **Baseline Soundness:** Why was the framework not compared against standard post-training quantization methods such as AWQ, GPTQ, or AdaRound, which are industry-standard references for Transformer networks?
4.  **Fine-Tuning Details:** The authors provide basic fine-tuning parameters, but omit critical training details such as weight decay, batch size during fine-tuning, learning rate schedules, data augmentation, and the exact train/val/test splits used, making full reproduction of the base checkpoints impossible.

---

## 5. Final Verdict
This paper presents a highly practical research direction and exhibits strong writing clarity, elegant parameter-efficiency in ORD, and a thorough exploration of operational parameters. However, the massive 6x physical latency slowdown directly contradicts the paper's main motivation, the conceptual novelty is overstated due to ignored peer-reviewed competitors (TVQ [ICCV 2025]), the evaluation is limited to a toy digit-recognition setup, and the calibration step exhibits clear signs of domain-overfitting. Major revisions—including citing TVQ/1bit-Merging, implementing a fused custom kernel to prove real-world latency savings, expanding the benchmarks to a standard 8-task suite, and addressing the mathematical non-equivalence—are required before this paper can be accepted.
