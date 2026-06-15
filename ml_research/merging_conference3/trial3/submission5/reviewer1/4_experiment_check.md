# 4. Experimental Check and Empirical Evaluation

## Critical Review of the Experimental Setup

### 1. Downscaled Experimental Protocol
The experimental protocol uses a heavily downscaled pipeline:
- **Expert Training:** Fine-tuning on only **512 samples** per dataset across 5 epochs.
- **Calibration Stream:** Only **16 unlabeled samples** for test-time adaptation.
- **Evaluation:** **2000 test samples** per task.

While the authors argue that this setup simulates extreme data scarcity on the edge, it results in **extremely undertrained expert models**:
- In Table 1, the full-precision individual experts achieve an average accuracy of only **79.20 $\pm$ 0.85%** (MNIST: 84.12%, FashionMNIST: 78.85%, CIFAR-10: 82.48%, SVHN: 71.37%).
- In standard academic literature, a ViT-Tiny fine-tuned on the full training sets of these benchmarks typically achieves over **98% (MNIST)** and **90%+ (CIFAR-10)**.

**Empirical Concern:** Fine-tuning on only 512 samples means the experts are severely underfit and their parameter weights have likely not converged to sharp minima. Representation connectivity, gradient dynamics, and quantization noise profiles can be vastly different for underfit models compared to standard, fully converged models. It is highly questionable whether the observed empirical trends and the "Overfitting-Optimizer Paradox" generalize to fully trained, high-accuracy models. The paper lacks a full-scale validation (e.g., using standard training sets) to confirm that the findings are not an artifact of the severely downscaled training setup.

---

## Analysis of Baselines and Claims

### 1. Anomaly 1: Offline Optimization (AdaMerging) Outperforms Direct Quantization-Aware Merging
The authors evaluate a baseline **AdaMerging (Adam)**, which represents optimizing the merging coefficients in full precision (smooth, server-side environment) and subsequently applying post-hoc quantization to the merged weights.

A comparison of the first-order results reveals a significant baseline anomaly:
- **Under 8-Bit PTQ:** AdaMerging (Adam) achieves **62.27 $\pm$ 0.43%**, strictly outperforming the proposed direct quantization-aware Q-PolyMerge (Adam) at **59.76 $\pm$ 1.22%** (a difference of **+2.51%**).
- **Under 4-Bit PTQ:** AdaMerging (Adam) achieves **50.20 $\pm$ 2.21%**, strictly outperforming the proposed Q-PolyMerge (Adam) at **48.87 $\pm$ 1.42%** (a difference of **+1.33%**).

**Analysis:** This shows that direct quantization-aware optimization (navigating the step-cliffs and flat plateaus of the rounded weight landscape via the Straight-Through Estimator) actually **harms** the optimization process compared to simply optimizing in a smooth unquantized space and then quantizing. 
Thus, for any **offline (server-side)** scenario where first-order optimization is possible, the proposed Q-PolyMerge method is **not** the optimal strategy. A practitioner would achieve better results by simply running standard AdaMerging in full-precision and then quantizing the model.

### 2. Anomaly 2: Zero-Order On-Device Adaptation is Harmful or Useless Compared to Doing Nothing
As established in the Soundness evaluation, when running under the **on-device** constraints where backpropagation is unviable:
- **Under 8-Bit PTQ:** The unadapted **M-then-Q** baseline achieves **55.11 $\pm$ 0.22%**, while the proposed **Q-PolyMerge (ES)** achieves only **51.03 $\pm$ 4.35%** (an active degradation of **-4.08%**).
- **Under 4-Bit PTQ:** The unadapted **M-then-Q** baseline achieves **42.92 $\pm$ 2.06%**, while the proposed **Q-PolyMerge (ES)** achieves **43.05 $\pm$ 1.90%** (a statistically insignificant **+0.13%** difference).

**Analysis:** This reveals that the proposed zero-order on-device adaptation fails to provide any practical advantage. In the 8-bit regime, it severely degrades accuracy; in the 4-bit regime, it barely matches the unadapted baseline. 

---

## Summary of the Practical "Utility Dilemma"
Combining Anomaly 1 and Anomaly 2 creates a devastating **utility dilemma** for the proposed method:
1. **If you are offline (server-side):** Do not use Q-PolyMerge. Standard **AdaMerging (Adam) followed by quantization** is superior, achieving **62.27% (8-bit) / 50.20% (4-bit)** vs. Q-PolyMerge's **59.76% / 48.87%**.
2. **If you are on-device (edge):** Do not use Q-PolyMerge (or any adaptation). Bypassing test-time adaptation entirely and deploying a **naive unadapted M-then-Q model** is superior (or equivalent), achieving **55.11% (8-bit) / 42.92% (4-bit)** with zero on-device computational overhead, compared to Q-PolyMerge (ES) which achieves **51.03% / 43.05%** while requiring 100 on-device forward-pass iterations.

There is no practical scenario presented in the paper where Q-PolyMerge is the optimal or even a beneficial choice over existing, simpler baselines. The empirical results completely fail to support the claimed practical significance of the proposed framework.
