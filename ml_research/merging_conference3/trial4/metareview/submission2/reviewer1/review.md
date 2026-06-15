# Peer Review of OmniMerge

## Summary of the Paper
The paper addresses a highly practical and critical bottleneck in edge-AI deployment: the mismatch between optimized model weights and the diverse, heterogeneous post-training quantization (PTQ) standards of different hardware accelerators and runtime compilers (e.g., Google Edge TPU, Apple Neural Engine, NVIDIA TensorRT). Traditional quantization-aware model merging methods, such as Q-Merge, optimize blending coefficients under a single simulated operator, which induces "Cross-Schema Performance Degradation" when deployed on mismatched target hardware.

To resolve this, the authors propose **OmniMerge**, a training-free, metadata-free, and zero-overhead multi-schema stochastic co-optimization framework for robust model merging. OmniMerge achieves hardware-invariant generalization via two core mechanisms during unsupervised test-time adaptation:
1. **Stochastic Operator Sampling (SOS):** Uniformly sampling an active quantization operator from a discrete pool of hardware-relevant schemas at each optimization step, preventing merging coefficients from overfitting to any single discretization grid.
2. **Scale and Zero-Point Noise Perturbation (SZNP):** Injecting Gaussian noise dynamically into scales and zero-point offsets to smooth out the non-differentiable loss landscape and help the optimizer escape fragile local minima.

Evaluated on a Vision Transformer backbone (`ViT-Tiny`) across 4 diverse image classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN), OmniMerge achieves up to **50.78%** average accuracy and completely closes the cross-schema generalization gap, outperforming Q-Merge, Quantized AdaMerging, and Naive baselines across 5 target post-training quantization schemas (including an unseen Double Quantization schema).

---

## Strengths and Weaknesses

### Strengths
1. **Practical Edge-AI Importance & Scope:** The paper identifies and addresses a critical, real-world bottleneck in edge computing and MLOps: the numerical divergence and performance collapse that occur when deploying a single merged checkpoint across heterogeneous accelerators running incompatible compilers.
2. **Zero-Overhead and Metadata-Free Design:** OmniMerge is training-free and requires no hardware metadata or test-time compute. Once optimized, the merged checkpoint is compiled into standard quantized formats with zero inference-time latency or memory overhead, making it exceptionally easy to integrate into production pipelines.
3. **Rigorous and Clear Methodology:** The mathematical formulations for symmetric and asymmetric quantization, scale/zero-point noise, and double quantization are precise and technically sound. The explicit details regarding Straight-Through Estimator (STE) autograd detachment and continuous-to-discrete noise transitions are commendable.
4. **Comprehensive Evaluation and Generalization Proof:** Evaluating across five target compilers—including an unseen Double Quantization schema—provides convincing empirical proof that the learned coefficients are truly schema-invariant.
5. **Outstanding Transparency:** The authors are highly candid about the limitations of their work, including the under-training of task experts, the statistical modesty of the quantization "denoising" effect, and the compound stochasticity over-regularization.

### Weaknesses
1. **Overlapping Attribution of the Core Vulnerability:** In the Introduction, the authors frame the demonstration of "Cross-Schema Performance Degradation" as a novel finding of this work. However, in the Related Work, they correctly attribute the identification of this single-schema sensitivity to the robustness audit in *qmergeaudit* (2025). The paper would be significantly stronger and more scholastically rigorous if it clearly positioned itself as **providing the first systematic algorithmic solution (OmniMerge) to the vulnerability audited by prior work**, rather than claiming its initial discovery.
2. **Severe Under-Training of Task Experts:** To accommodate a highly restricted local compute budget, the task experts are fine-tuned on only 256 training samples for 3 epochs, leading to extremely low individual validation accuracies (such as 28.91% on SVHN, which is barely above random guessing). While this serves as a valuable low-compute testbed, it remains an open question whether the observed phenomena—specifically the cross-schema performance gap and the quantization "denoising" effect—generalize to highly optimized, fully converged task experts.
3. **Contextualization within Robust Quantization Literature:** While the paper cites standard PTQ techniques and test-time adaptation surveys, it misses a valuable opportunity to situate itself within the broader historical trajectory of robust and hardware-aware post-training quantization frameworks (such as HAQ, HAWQ, SigmaQuant, or SignRoundV2). Discussing or drawing parallels to these works would greatly enrich the paper's scholarly contextualization.

---

## Detailed Evaluation Dimensions

### Soundness: Good
The paper is mathematically rigorous and technically sound. The formulations of symmetric and asymmetric quantization are standard and correct, and the detachment of dynamic min/max tracking from PyTorch's autograd is essential for numerical stability. The Task-Consensus Regularization (TCR) is well-formulated to prevent parameter drift and stabilize multi-task ensembling.

However, the soundness rating is restricted to **Good** (rather than Excellent) due to the severe under-training of the task experts. Because the experts are significantly under-trained, their weight distributions and decision boundaries are fundamentally less stable than converged models. This limits our ability to definitively conclude that OmniMerge's advantages translate seamlessly to standard, fully converged deep learning systems.

### Presentation: Excellent
The paper is exceptionally well-written, clearly structured, and highly readable. The progression from the practical edge bottleneck to the mathematical formulation and empirical results is seamless. The discussion sections show extreme scientific maturity, providing deep reflections on complex phenomena (such as weight denoising as hard-thresholding) while remaining candid about limitations.

### Significance: Good
The paper's significance is **Good**. It provides high practical utility for MLOps and on-device ensembling by enabling a "write-once, deploy-anywhere" model merging pipeline. It also advances model merging theory by showing that test-time adaptation can be made robust to downstream compression and that noise injection acts as an excellent continuous coefficient regularizer.

However, the significance is slightly limited by the restricted evaluation scope (low-compute task experts on small image datasets with a `ViT-Tiny` backbone). Demonstrating this method on larger architectures (e.g., LLaMA, ViT-Base) and more challenging domains would elevate the work's impact to excellent.

### Originality: Good
The originality is **Good**. While model merging, PTQ, and test-time adaptation are established, the integration of Stochastic Operator Sampling (SOS) and Scale/Zero-Point Noise Perturbation (SZNP) specifically to optimize robust, schema-invariant ensembling coefficients is a highly creative and novel combination of existing techniques. The reasoning behind this combination is well-articulated and empirically validated.

---

## Overall Recommendation

**Rating:** 4: Weak Accept

**Justification:**
OmniMerge is a technically solid, beautifully written, and highly practical paper that addresses an important and unaddressed bottleneck in edge-AI deployment. The proposed stochastic co-optimization framework (SOS + SZNP) successfully closes the cross-schema generalization gap and demonstrates strong out-of-pool robustness on Double Quantization with zero runtime overhead. 

The paper is recommended for a **Weak Accept** because, while its methodology and empirical results are highly compelling, there are notable limitations that restrict its immediate impact, primarily the severe under-training of the task-specific experts and the relatively small evaluation splits. These factors introduce a degree of uncertainty regarding whether the observed findings (including the intriguing weight-discretization denoising hypothesis) generalize to standard, fully converged production systems. 

---

## Actionable Feedback and Questions for the Authors

1. **Clarify Scholarly Attribution of Vulnerability:** 
   Please revise the Introduction (Section 1) to clarify that the cross-schema sensitivity of single-schema optimized merged models was first audited and demonstrated by *qmergeaudit* (2025). Frame your core contribution as proposing the first systematic, training-free algorithmic framework (OmniMerge) to solve this audited vulnerability, which will enhance the scholastic integrity and precision of your paper.
2. **Situate within Broad Robust Quantization Literature:**
   To improve the contextualization of your work, please discuss and reference standard hardware-aware and robust post-training quantization literature (e.g., HAQ, HAWQ, SigmaQuant, or SignRoundV2) in Section 3, drawing parallels to how these frameworks handle accelerator heterogeneity.
3. **Address Expert Under-Training Generalizability:**
   Do you have preliminary results or theoretical intuitions indicating whether the cross-schema performance gap and the weight-denoising phenomenon persist when utilizing highly optimized, fully converged task-specific experts? Please discuss this generalizability in the limitations or future work section.
4. **Incorporate Scale Clamping for Numerical Safety:**
   In equation 5, you multiply the scale factor by $(1 + \epsilon_s)$ with a scale perturbation noise $\epsilon_s \sim \mathcal{N}(0, 0.01^2)$. Although highly unlikely, there is a non-zero probability of $1 + \epsilon_s \le 0$ under Gaussian noise, which would collapse the scale factor. Do you employ a clamping mechanism (e.g., $\max(s_{\text{asym}}, \epsilon_{\text{eps}})$) in your implementation to guarantee absolute division safety? If so, please explicitly state it in the methodology.
