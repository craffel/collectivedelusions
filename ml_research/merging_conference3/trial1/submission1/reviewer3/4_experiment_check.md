# Experimental Evaluation Review

This document provides a highly critical assessment of the experimental setup, dataset selection, baseline comparisons, and the validity of the empirical claims in the paper.

## 1. The "Toy" Dataset Bottleneck
The most glaring weakness of the experimental evaluation is its extremely restricted, low-complexity scope.
- **The Toy Selection:** The authors evaluate their framework exclusively on **MNIST** and **SVHN** using a pre-trained **ViT-B-32** backbone. Both datasets are digit-classification tasks that share the exact same 10-class label space (0-9).
- **Over-Parameterization:** A ViT-B-32 architecture (86 million parameters) is massively over-parameterized for basic digit recognition. MNIST can be solved to >99% accuracy with a tiny 3-layer CNN containing under 100,000 parameters. Applying an 86M parameter Transformer to digit recognition makes the task trivial and highly resilient to weight perturbations, masking the true impact of quantization noise.
- **Lack of Task Diversity:** Real-world model merging is valuable because it combines *diverse, non-overlapping* capabilities (e.g., merging a model that classifies cars with a model that classifies flowers, or combining coding and translation capabilities in LLMs). By restricting the tasks to two highly similar digit recognition domains, the authors fail to prove that QP-Merge can scale to diverse task spaces where activation scales and feature representations are truly disjoint and conflicting. Standard model merging literature (such as Task Arithmetic, Ties-Merging, DARE) evaluates on at least an **8-task vision suite** (SUN397, Cars, DTD, EuroSAT, SVHN, GTSRB, MNIST, RESISC45) or massive multi-task NLP instruction-tuning benchmarks. The authors' toy setup is completely non-representative.

---

## 2. Weak and Missing Baselines
The authors set up a convenient comparison by skipping state-of-the-art baselines and using weak, self-constructed references.

### A. Missing SOTA Quantization Baselines
The paper evaluates against a self-constructed "SmoothQuant Baseline" and "Naive Quantization." 
- Why did the authors not compare against standard, widely adopted post-training quantization (PTQ) methods such as **AWQ (Activation-aware Weight Quantization)**, **GPTQ**, or **AdaRound**? These are the true industry-standard baselines for low-bit quantization of Transformer layers. AWQ, for example, is highly effective at protecting salient channels without requiring high-precision outlier matrices.
- The authors' "SmoothQuant baseline" appears to be a customized implementation that performs a post-hoc scale search but does not maintain mathematical equivalence (since it does not scale activations during inference). Calling this a "SmoothQuant" baseline is misleading and does not represent actual SmoothQuant.

### B. Complete Absence of Direct Competitors
The authors completely ignore existing works that address the quantization and compression of task vectors, such as:
1. **Task Vector Quantization (TVQ) [ICCV 2025]**
2. **Binary Task Switch (T-Switch) [CVF 2024/2025]**
3. **1bit-Merging [2025]**
By omitting these directly competing frameworks from their baseline comparison, the authors create an artificial vacuum, making QP-Merge appear highly novel and superior when it has not been compared to its actual peer-reviewed counterparts.

---

## 3. The "Lossless" Illusion and Unfair Comparisons
The authors leverage a misleading comparison to claim that QP-Merge INT8 is "virtually lossless" and even beats the floating-point baseline:
> "QP-Merge INT8 ... actually beats the unquantized uniform baseline by +0.21%... demonstrating that unsupervised joint scaling and merging coefficient calibration directly optimizes parameter fusion in quantized regimes."

This is a classic "apples-to-oranges" comparison:
- **The Catch:** The unquantized uniform FP32 baseline uses a static, unoptimized merging coefficient ($\lambda_t = 0.5$). QP-Merge INT8 uses **optimized merging coefficients** ($\lambda_t$) that have been directly adapted on the target domain calibration set via gradient descent.
- **The Truth:** The performance improvement is entirely driven by this **test-time parameter adaptation** (fine-tuning) on the target domain, which compensates for the quantization loss and happens to slightly exceed the static unquantized reference. When compared to the *true* unquantized optimized baseline (where coefficients are optimized but no quantization is applied), QP-Merge INT8 does not beat it (95.14% vs 95.12%, which is within the statistical noise level of $\pm 0.03\%$). Framing this as a "quantization benefit" is highly misleading.

---

## 4. Evidence of Overfitting and Sensitivity to Calibration Bias
A close look at the authors' "Robustness to Imbalanced Calibration" experiment (Table 6) reveals a severe technical vulnerability that the authors attempt to spin as a strength:
- When calibrating on **100% MNIST-only data**, the SVHN classification accuracy in INT4 drops from **90.08%** (balanced calibration) to **85.22%** (MNIST-only calibration).
- This is a massive **4.86% drop** in SVHN performance.
- The authors describe this as "striking robustness," highlighting that 85.22% still exceeds the naive quantized baseline of 84.32%. However, a method that loses nearly 5% accuracy on a task due to a calibration shift is highly unstable.
- This empirical drop strongly supports our criticism in the Soundness section: because QE-Calib is a non-equivalent weight-tuning step with 55,000 learnable parameters optimized over only 128 samples, it is **highly prone to domain overfitting**. If the calibration set does not perfectly represent all downstream domains, the model's performance on minority domains collapses. This is a severe deployment risk that is glossed over in the text.
