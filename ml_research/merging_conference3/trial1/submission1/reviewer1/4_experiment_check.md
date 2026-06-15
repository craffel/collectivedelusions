# 4. Experiment Check

This section provides a critical evaluation of the experimental setup, datasets, baselines, and results to determine whether they empirically support the authors' claims and meet practitioner-level standards.

---

## Evaluation of Experimental Setup & Datasets

### 1. The Core Limitation: Grayscale/Low-Res Toy Tasks (MNIST & SVHN)
The paper's largest experimental weakness is its reliance on **MNIST** (28x28 grayscale handwritten digits) and **SVHN** (32x32 street view house numbers) for multi-task merging. 
- In 2026, both MNIST and SVHN are considered basic toy classification benchmarks. Merging two digit recognition tasks is highly contrived; in real-world scenarios, why would a practitioner merge an MNIST model and an SVHN model when a single tiny model could easily master both with zero effort?
- Real-world model merging is typically applied to large-scale, high-dimensional foundation models, such as:
  - **LLMs (e.g., LLaMA, Mistral, Qwen):** Merging instruction-following, math, coding, and translation capabilities.
  - **Complex Vision Suites (e.g., 8-task suites):** Including ImageNet, CIFAR-10, Stanford Cars, FGVC Aircraft, Flowers102, etc.
- While using a pre-trained `ViT-B-32` is a reasonable step up from ConvNets, evaluating on digit classification fails to prove that the proposed weight-scaling $D_l$ (which lacks mathematical equivalence) scales to complex, high-dimensional manifolds without catastrophic representation drift.

---

## Evaluation of Baselines
The paper includes a robust set of baselines, which is a major strength:
1.  **FP32 Merged Bound (Uniform & Optimized):** Comparing to the Optimized FP32 merged bound (where task coefficients $\lambda_t$ are tuned on the calibration set without quantization) is excellent. It isolates the benefit of coefficient tuning on the 128 samples from the quantization errors, proving that QP-Merge INT8 (+0.02% over Optimized FP32) is virtually lossless.
2.  **Naive Quantization (INT8/INT4):** Establishes the severe failure of standard, quantization-blind merging, where SVHN drops by 6.40% in INT4.
3.  **SmoothQuant Baseline:** A very strong, optimization-based post-hoc PTQ comparison. Outperforming it (94.70% vs. 94.23% in INT4) highlights the value of separating outliers via ORD rather than just applying post-hoc weight scaling.
4.  **Ablations (No QE-Calib, No ORD):** Dissect the individual contributions of each technique, showing that calibration is indispensable (No QE-Calib drops INT4 accuracy to 91.09%).

---

## Do the Results Support the Claims?
- **Within-Task Claims:** Yes. The results clearly show that QP-Merge successfully recovers the performance of `ViT-B-32` on the combined MNIST/SVHN task under 4-bit and 8-bit quantization.
- **Robustness and Generalization Claims:** Supported. The sensitivity sweeps for outlier percentile $\gamma$ (Table 3), calibration size $M$ (Table 4), and the OOD corruptions (Table 5) are highly thorough. 
- **Imbalanced Calibration Resilience:** Fully supported. Table 6 demonstrates impressive resilience: calibrating purely on SVHN images yields 94.92% multi-task accuracy, proving that the scale search converges to stable scaling parameters that do not overfit.
- **Edge Deployment and Latency Claims:** **Not physically supported.** 
  - The authors claim that QP-Merge is "hardware-friendly" and "instantly deployable in production settings." However, the physical GPU profiling shows a **5.8$\times$ slowdown** in PyTorch. 
  - The claim that QP-Merge "translates directly to massive real-world speedups" is only supported via an analytical DRAM-to-SRAM memory transfer model, not a physical wall-clock speedup. While the scaling model is logical, practitioners require a physical demonstration (e.g., via a fused Triton or TensorRT kernel) before accepting "real-world edge execution speed" claims.

---

## Recommendations for Experimental Improvement
1.  **Evaluate on More Complex Multi-Task Suites:** Replace MNIST/SVHN with standard vision-merging suites (such as combining CLIP fine-tuned on CIFAR-100, Stanford Cars, and FGVC Aircraft).
2.  **Demonstrate on Large Language Models (LLMs):** Apply QP-Merge to an instruction/coding/math merge of LLaMA-3-8B to show that the non-equivalent scaling $D_l$ generalizes to complex natural language tasks.
3.  **Provide a Fused Kernel Implementation:** Write a basic Triton or TensorRT fused dense-sparse execution kernel to physically prove that the 5.8$\times$ PyTorch slowdown can be compiled away.
