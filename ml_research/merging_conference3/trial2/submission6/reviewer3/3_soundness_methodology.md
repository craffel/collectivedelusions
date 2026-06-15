# 3. Soundness and Methodology

## Clarity and Elegance of Description
The methodology is exceptionally well-written, clear, and mathematically rigorous. From a minimalist standpoint, the paper does not hide behind obscure or unnecessarily complex terminology. Instead, it explains the mathematical formulation of Q-Merge directly and step-by-step:
1. It starts with standard **Task Arithmetic** (Equation 2).
2. It formulates **per-channel uniform post-training quantization** (Equations 3 & 4), detailing how scaling factors $S^l_c$ are calculated dynamically from continuous weights.
3. It describes the **joint prediction entropy loss** (Equation 5) and explains why "class collapse" (a common failure mode in unsupervised test-time adaptation) is naturally prevented by the low-dimensional structural constraints of the layer-wise coefficient bottleneck.
4. It provides a formal **dual-path gradient flow analysis** (Equation 15) to explain how PyTorch Autograd propagates gradients through both the direct coordinates and the dynamic scaling factors.

The description is complete and highly readable, providing the exact mathematical details necessary to fully understand the underlying mechanics of first-order gradient flow through the non-differentiable quantization operator.

---

## Appropriateness of Methods
The proposed methodology is highly appropriate and structurally elegant for the following reasons:
* **Compact Search Space (Structural Regularization):** By parameterizing the merging process via layer-wise coefficients $\Lambda \in [0, 1]^{L \times K}$ rather than optimizing high-dimensional backbone weight coordinates, the search space is limited to just 56 parameters ($14 \text{ layers} \times 4 \text{ tasks}$). This acts as an extremely strong, low-capacity structural bottleneck that prevents high-frequency parameter overfitting and "class collapse" on tiny calibration streams.
* **Unsupervised Test-Time Adaptation:** Minimizing joint prediction entropy over an unlabeled calibration stream of only 64 images is highly practical for edge deployment, where downstream ground-truth labels are generally unavailable, and data privacy must be maintained.
* **Per-Channel Weight Quantization:** The selection of per-channel (channel-wise) weight quantization is crucial. In 4-bit configurations, a single outlier parameter can crush the dynamic range of an entire layer under per-tensor quantization. By dynamically computing scales for each individual output channel, local coordinate representation is preserved, which the authors show is an absolute design mandate to maintain linear mode connectivity in extreme low-bit spaces.
* **Straight-Through Estimator (STE):** Employing STE for first-order gradient flow (Adam GD) is highly effective. The authors' rigorous mathematical derivation (Section 3.4.2) elegantly deconstructs how PyTorch Autograd propagates gradients through dynamic scale factor scaling grids to concurrently optimize weight coordinates and quantization ranges.

---

## Technical Flaws and Critical Assumptions
While the methodology is sound, several critical assumptions and constraints must be noted:

### 1. Shared Pre-trained Checkpoint Dependency
Like all weight-space model-merging techniques, Q-Merge assumes that the task-specific expert models are fine-tuned from the *exact same* pre-trained base model. If the experts were pre-trained or fine-tuned independently from different base checkpoints, linear mode connectivity would break, and Q-Merge would fail to align the models. This is a standard prerequisite in model merging literature, which the authors correctly acknowledge.

### 2. Low-Data Fine-Tuning and Low Parameter Drift
The experimental setup fine-tunes the task-specific experts using disjoint subsets of only **512 images** per task for 5 epochs. This is a low-data regime, meaning that the weights of the fine-tuned experts remain structurally very close to the pre-trained base model (i.e., **low parameter drift**). 
* Under low parameter drift, linear mode connectivity is highly robust, and task vectors are relatively easy to merge without severe parameter-level interference.
* In high-capacity enterprise scenarios where experts are fully fine-tuned on millions of images, the weights drift significantly further from the base checkpoint. In such high parameter drift regimes, weight-space model merging is more severely challenged.
* **Commendable Honesty:** The authors are highly careful and honest about this limitation. In Section 5.2, they explicitly acknowledge that their current experiments operate within a low-parameter-drift regime and advocate for scaling Q-Merge to high-drift and large-scale (e.g., multi-billion parameter LLM) settings as an important future research direction.

### 3. Backpropagation Memory Overhead
Optimizing coefficients using Adam GD with STE requires computing gradients through the model's forward path, which in standard reverse-mode automatic differentiation (AD) requires caching activation maps. For edge deployment on memory-constrained devices, this caching could introduce a memory bottleneck.
* However, the authors address this constraint proactively by outlining three highly compatible mitigation strategies: (1) Gradient Checkpointing to trade compute for memory; (2) Forward-Mode AD (Jacobian-Vector Products) which completely avoids activation caching and is highly suited for Q-Merge's tiny 56-parameter search space; and (3) Zero-Order 1+1 ES, which requires zero activation memory because it treats the model as a black-box oracle. This provides a highly pragmatic, system-aware view of the method's deployment utility.

---

## Reproducibility
The reproducibility of this paper is **excellent**. The authors provide precise details regarding:
* The exact model backbone (`vit_tiny_patch16_224` from the `timm` library).
* The dataset splits (MNIST, FashionMNIST, CIFAR-10, SVHN) and seed selections (42, 100, 2026).
* Training hyperparameters (disjoint training set of 512 images, 5 epochs, Adam optimizer, learning rates of $10^{-5}$ for backbone and $10^{-3}$ for task heads).
* Calibration split details (disjoint split of 16 images per task, 64 total).
* Layer-wise parameter grouping details ($L=14$ layers).
* Quantization configurations (symmetric, uniform RTN at 8-bit and 4-bit per-channel).

Based on these explicit parameters, an expert practitioner could easily recreate the experimental environment and reproduce the reported numbers.
