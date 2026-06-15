# 3. Soundness and Methodology Evaluation

## Clarity of the Description
- **High Quality:** The paper is written with high clarity and precision. The mathematical formulations of the task vector blending (Eq. 2), the five structural granularities (Section 3.2), the unsupervised surrogate loss (Eq. 3), and the regularizers (ESR and TV in Section 3.5) are well-defined and easy to follow.
- **Detailed Descriptions:** Each of the five structural granularities is defined precisely with its exact parameter count under the 4-task, 12-layer ViT setting, which prevents ambiguity.

## Appropriateness of Methods
- **Nested Resolution Hierarchy:** The construct of defining a nested resolution hierarchy (L1 Global to L5 Tensor-wise) is highly logical and appropriate for studying the generalization-granularity trade-off.
- **Dual Optimization Paradigms:** Evaluating both first-order (Adam) and zero-order (1+1 ES) optimization methods is a very clever and appropriate methodological decision. It allows the authors to decouple the capacity of the parameter space from the path taken by the optimizer, showing that optimization trajectory is just as critical as parameter dimension.
- **Regularization Design:** ESR and TV are physically intuitive and appropriate choices to regularize high-resolution weight blending.

## Potential Technical Flaws and Methodological Limitations
1. **Low-Fidelity Expert Regimes (Poorly Converged Experts):** 
   - The expert models are exceptionally weak and underperforming: MNIST is only **61.03%** (typically >99%), FashionMNIST is **62.47%** (typically >92%), CIFAR-10 is **24.93%** (typically >85%), and SVHN is **17.50%** (typically >90%).
   - In this regime, the expert representations are extremely noisy, and the extracted task vectors contain substantial high-frequency parameter noise. Because the underlying representation space is weak and fragile, any local optimization (like test-time entropy minimization) is highly prone to catastrophic representational collapse.
   - This raises a serious concern about the **generalizability of the findings**. The authors' core claim that *unconstrained adaptive merging is inferior to static blending* may be an artifact of these poorly converged, low-fidelity experts. Indeed, in original papers like AdaMerging (Yang et al., 2023), which evaluated high-fidelity converged models on massive datasets, test-time adaptive merging achieved an **11% improvement** over static task arithmetic. Thus, presenting the failure of test-time adaptation as a general law of model merging is a potential over-generalization.
2. **Computational and Latency Overhead of Adaptation:**
   - From a deployment and practitioner's perspective, running 100 steps of 1+1 ES or 60 steps of Adam gradient descent on a calibration batch of 256 samples per task *at test time* is incredibly expensive. 
   - For 100 steps of 1+1 ES across 4 tasks, this requires $100 \times 4 \times 256 = 102,400$ forward passes. On resource-constrained edge devices (where adaptive merging is typically deployed), this massive test-time computational overhead and the resulting high latency are completely impractical.
   - Since the resulting optimized model doesn't even beat the zero-overhead static uniform baseline (30.17% vs 30.41%), the entire test-time adaptation paradigm in this regime has negative utility and massive waste of computational resources. This practical limitation is not sufficiently critiqued in the paper.

## Reproducibility
- **Strong Reproducibility:** The paper provides highly explicit details on the architecture (ViT-Tiny with 12 layers, $d_{\text{model}}=64$, 2 heads), pre-training/fine-tuning samples and epoch counts, optimizer step counts, learning rates, mutation scale parameters, and regularization hyperparameter values. An expert reader would find it straightforward to reproduce these exact results.
