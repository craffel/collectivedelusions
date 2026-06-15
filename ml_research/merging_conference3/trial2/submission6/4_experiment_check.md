# Experimental Design and Results Check: Q-Merge

This report evaluates the empirical rigor, baseline selections, and results of the proposed **Q-Merge** framework, highlighting both exceptional strengths and key limitations.

## 1. Strengths of the Experimental Design
* **Statistical Rigor:** All experiments are executed across **three independent random trials and seeds (42, 100, 2026)**, and both mean and standard deviations are reported. This is excellent practice and ensures results are statistically significant, not cherry-picked.
* **Exemplary Confounding Control:** The author's deconstruction of the optimizer confounding factor is outstanding. By implementing a fully differentiable *AdaMerging (FP16 Optimized with Adam GD)* baseline, they isolate the benefit of the STE from the benefit of the superior first-order optimizer.
* **Rich systems-level evaluations:** The empirical analysis is exceptionally comprehensive and includes:
  - **Advanced PTQ Baselines:** Direct comparison and sequential combination with **AdaRound** (Nagel et al., 2020), showing that Q-Merge's global coordinate alignment is complementary to and superior to standalone post-hoc local rounding.
  - **Latency Benchmarks:** Wall-clock latency on CPU (2.4-4.8s per trial) and GPU (80ms), verifying the lightweight claim.
  - **Calibration Sensitivity Analysis:** Testing calibration sizes $S \in \{8, 16, 64\}$ per task, confirming data efficiency.
  - **Stream Noise/Imbalance Robustness:** Rigorous testing under highly imbalanced streams (Scenario B, where one task is 95% dominant) and proposing a *Confidence-Based FIFO Stratification* heuristic to restore perfect balance.
  - **Fully Integer-Quantized Weight Pipeline:** Quantizing the classification heads to 8-bit to verify that a W8/W4 weight-only pipeline has virtually zero performance loss ($0.00\% - 0.01\%$).

---

## 2. Key Limitations and Weaknesses
While the empirical evaluation is exceptionally thorough, several critical limitations must be highlighted:

### A. Toy-Scale Backbone and Datasets (Scale and Generative Generalizability)
* **Pre-trained Backbone:** The evaluation uses a small **timm ViT-Tiny** backbone with only **5.7M parameters**.
* **Downstream Tasks:** The four classification benchmarks (MNIST, FashionMNIST, CIFAR-10, SVHN) are toy-scale classification tasks.
* **The Scale Gap:** Today's model-merging and quantization interest is heavily focused on **Large Language Models (LLMs)** (e.g., 7B-70B LLaMA, Mistral) and **Large Vision-Language Models (VLMs)** (e.g., CLIP, LLaVA). Classification on low-resolution image datasets does not capture the complexity of autoregressive language generation, text summarization, or reasoning.
* **Emergent Outliers:** Large-scale models, particularly LLMs, exhibit "emergent outlier features" in their activations (highly concentrated high-magnitude coordinates), which significantly complicate low-bit quantization. The behavior of Q-Merge on ViT-Tiny may not directly generalize to these large-scale generative setups.

### B. Low Parameter-Drift Regime (Few-Shot Training)
* **Expert Training:** The task-specific expert models are trained on disjoint subsets of only **512 images** per dataset for 5 epochs.
* **Low Parameter Drift:** This represents a highly localized, few-shot training regime. Consequently, the weights of the expert models remain structurally very close to the pre-trained base model ($\theta_{\text{base}}$).
* **The Drift Challenge:** In real-world enterprise applications, expert models are often fully fine-tuned on massive datasets, resulting in significant parameter drift where weights diverge far from each other and the base checkpoint. In high-drift regimes, linear mode connectivity is weaker, and merging is much more difficult. It is unclear whether Q-Merge (and linear task arithmetic in general) can maintain high-fidelity performance under aggressive low-bit quantization when parameter drift is high.

### C. Low SVHN Expert Accuracy
* **Baseline Expert Accuracy:** Due to the few-shot training, the unmerged unquantized SVHN expert achieves only **41.34%** accuracy.
* **Low Merged Performance:** Consequently, the SVHN performance in merged configurations is quite low (e.g., 32.42% in uniform FP16, and 35.87% in Q-Merge Adam GD). While this is above the 10% random guess, a base expert of 41% accuracy represents a weak starting point and raises questions about the realism of the expert training.

### D. Activation Quantization Omitted
* **Weight-Only Quantization:** The paper only quantizes weights (W8A16 or W4A16), keeping intermediate activations in full floating-point precision (FP32/FP16).
* **Edge NPU Mandate:** Many commodity edge processors, specialized DSPs, and low-power NPUs require **fully integer pipelines** (e.g., W8A8 or W4A4) with integer activations to achieve peak hardware throughput and power efficiency. Activation quantization is notoriously difficult because activation distributions are highly dynamic. Omitting activation quantization limits the edge-deployment claims of the paper.

### E. Lack of Empirical Memory Measurements
* **Theoretical Memory Claims:** The paper discusses backpropagation memory complexity and suggests memory-saving techniques like *Gradient Checkpointing* or *Forward-Mode AD*.
* **Omitted Data:** However, the paper does not report actual empirical peak GPU memory measurements (e.g., in MBs) to support these systems-level claims.

---

## Conclusion on Experiments
The experiments are **good to excellent** in terms of baseline coverage, statistical rigor, and systems-level analysis, representing a highly complete exploration for a conference paper. However, the **toy-scale ViT-Tiny backbone**, **low-drift regime**, and **weight-only quantization (omitting activation quantization)** prevent it from being a flawless, fully deployment-validated study.
