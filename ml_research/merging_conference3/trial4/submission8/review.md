# Peer Review: Hessian-Regularized Coefficient Optimization (HessMerge)

**Recommendation:** Reject  
**Soundness:** Poor  
**Presentation:** Fair  
**Significance:** Poor  
**Originality:** Good  

---

## 1. Summary of the Paper
This paper addresses the problem of deploying weight-space model merging frameworks onto resource-constrained edge hardware. Specifically, the authors identify a critical vulnerability in existing test-time adaptation (TTA) merging frameworks (like AdaMerging): **Quantization-Operator Overfitting**. They show that unregularized coefficient optimization on a tiny calibration stream converges to sharp local minima. While these minima achieve high accuracy in floating-point (FP32) space, they trigger catastrophic performance collapse under downstream post-training quantization (PTQ) rounding noise.

To resolve this, the authors propose **Hessian-Regularized Coefficient Optimization (HessMerge)**. By modeling post-training quantization as a parameter-space perturbation, they perform a second-order Taylor expansion to relate the quantization-induced loss gap to the curvature of the loss landscape with respect to the merging coefficients. HessMerge adds an explicit second-order regularizer that penalizes the **trace of the Hessian** of the unsupervised entropy loss with respect to the low-dimensional merging coefficients ($\text{Tr}(\mathcal{H}_{\Theta})$).

To make optimization computationally tractable without double-backpropagation, the authors introduce **Sharpness-Aware Coefficient Minimization (SACM)**, a first-order minimax approximation that perturbs the coefficients in the direction of maximum gradient to find wider local minima.

The method is evaluated using a Vision Transformer backbone (`vit_tiny_patch16_224`) across four visual classification domains (MNIST, FashionMNIST, CIFAR-10, SVHN) under six quantization schemas ranging from FP32 to INT4 Symmetric per-channel.

---

## 2. Main Strengths of the Paper
1. **Interesting and Highly Relevant Problem:** Proposing solutions for the edge deployment of model merging is a highly relevant, practical, and under-explored topic. Highlighting that test-time adaptation suffers from "Quantization-Operator Overfitting" is a high-signal observation.
2. **Creative Coefficient-Space Curvature Projection:** Projecting the curvature of the massive parameter space $W \in \mathbb{R}^D$ into the extremely low-dimensional coefficient space $\Theta \in \mathbb{R}^{L \times K}$ (56 dimensions) to make exact Hessian trace computation tractable is a clever, mathematically elegant design.
3. **Rigorous Evaluation Sweep:** The evaluation systematically sweeps across six different hardware-relevant quantization schemas (including tensor-wise and channel-wise, symmetric and asymmetric, INT8 and INT4), providing a comprehensive set of target deployment environments.
4. **Candid Acknowledgment of PolyMerge:** The authors are to be commended for honestly presenting the superiority of the PolyMerge baseline in Section 4.4 and Section 5, and discussing the important trade-offs between unconstrained test-time adaptation and structured subspace constraints.

---

## 3. Critical Flaws

### Flaw 1: The Subspace vs. Weight-Space Disconnect in Theorem 3.1
The core mathematical foundation of HessMerge relies on Theorem 3.1 (labeled `thm:quant_bound` in the LaTeX), which claims to establish a rigorous upper bound on the quantization-induced loss gap.

However, **there is a fundamental, fatal disconnect in the theorem's formulation and proof**. The authors define $\Delta F = F(\boldsymbol{\lambda}^* + \boldsymbol{\epsilon}) - F(\boldsymbol{\lambda}^*)$ as the loss gap *in the task-vector subspace*. But the actual quantization-induced loss gap evaluated during deployment is $\Delta \mathcal{L} = \mathcal{L}(W_{\text{quant}}) - \mathcal{L}(W_{\text{merged}}(\boldsymbol{\lambda}^*))$, where the quantized weight tensor is $W_{\text{quant}} = W_{\text{merged}}(\boldsymbol{\lambda}^*) + \delta$, and $\delta \in \mathbb{R}^D$ is the full weight-space quantization noise.

The authors claim that because $\boldsymbol{\epsilon} = (V^T V)^{-1} V^T \delta$ is the least-squares projection of $\delta$ onto the coefficient space, they can bound the loss gap. However, $W_{\text{quant}} = W_{\text{merged}}(\boldsymbol{\lambda}^* + \boldsymbol{\epsilon})$ only if $\delta = V \boldsymbol{\epsilon}$ (i.e., if the quantization noise lies *entirely* within the $d$-dimensional column space of $V$).

In reality, the high-dimensional quantization noise vector $\delta$ can be decomposed as:
$$\delta = V \boldsymbol{\epsilon} + \delta_{\perp}$$
where $\delta_{\perp}$ is the component orthogonal to the column space of $V$. Since $D \approx 5.7\text{M}$ (the model weights) and $d \approx 56$ (the merging coefficients), the task-vector subspace is an extremely low-dimensional manifold. Thus, **the vast majority of the quantization noise $\delta$ lies in the orthogonal component $\delta_{\perp}$** (meaning $\|\delta_{\perp}\|_2^2 \approx \|\delta\|_2^2$).

Evaluating the actual weight-space loss gap via a Taylor expansion around $W_{\text{merged}}$ yields:
$$\Delta \mathcal{L} \approx \nabla_W \mathcal{L}^T (V \boldsymbol{\epsilon} + \delta_{\perp}) + \frac{1}{2} (V \boldsymbol{\epsilon} + \delta_{\perp})^T \mathcal{H}_W (V \boldsymbol{\epsilon} + \delta_{\perp})$$
Because HessMerge only penalizes the coefficient-space Hessian trace $\text{Tr}(\mathcal{H}_{\boldsymbol{\lambda}})$, it only flattens the sensitivity to the in-subspace projection $\boldsymbol{\epsilon}$. It does absolutely nothing to control the sensitivity of the model to the orthogonal component $\delta_{\perp}$. Even if $\lambda_{\max}(\mathcal{H}_{\boldsymbol{\lambda}})$ is minimized to exactly zero, the model remains completely vulnerable to the massive out-of-subspace quantization noise $\delta_{\perp}^T \mathcal{H}_W \delta_{\perp}$, which is the primary driver of performance degradation under post-training quantization.

This fundamental theoretical disconnect perfectly explains the empirical collapse of HessMerge: the coefficient-space flatness guarantee does not translate to weight-space quantization robustness, resulting in performance that collapses under INT4 (14.22%, identical to unregularized AdaMerging).

### Flaw 2: The Theoretically Vacuous Unnormalized Bound and Disconnect with Implementation
The theoretical proof uses a well-conditioned normalized task-vector matrix $\hat{V}$ (minimum singular value $\sigma_{\min}(\hat{V}) = 0.80064$ empirically) to bound the gap. However, this normalization creates an irreconcilable disconnect between the theory and the actual implementation:
1. **Vacuous Unnormalized Bound:** If the unnormalized matrix $V$ is used, the bound contains $\frac{1}{\sigma_{\min}^2(V)}$. Empirically, because some layers (especially Layer group 13, final layer norm) have extremely small task-vector $L_2$ norms (average of **$0.009975$**), $V^T V$ is highly ill-conditioned (minimum eigenvalue is **$0.0001607$**, giving $\sigma_{\min}(V) = 0.012675$). Thus, the reciprocal multiplier squared blows up to **$6,224.1$**, making the unnormalized bound completely vacuous.
2. **Normalized Numerical Instability:** To use the well-conditioned normalized formulation, the theory requires weighting the Hessian trace by $\frac{1}{\|\tau_j\|_2^2}$. Since $\|\tau_j\|_2 \approx 0.009975$ for Layer group 13, its coefficients would be penalized with a scaling factor exceeding **$10,000\times$** (compared to $\sim 100\times$ for other layers), leading to severe numerical instability and gradient explosion.
3. **The Implementation Disconnect:** To avoid this instability, the actual code of SACM in `experiments/run_merging.py` (lines 333--347) is completely **unnormalized**, applying uniform coefficient perturbations. This means the code does not minimize the well-conditioned normalized bound of Theorem 3.1. It instead minimizes the unnormalized loss curvature, which is governed by the highly ill-conditioned unnormalized singular value ($\sigma_{\min}(V) = 0.012675$), making the theoretical bound completely vacuous for the actual implementation.

### Flaw 3: Severe Overclaims, Contradictions, and Empirically Insignificant Results
Despite admitting in the experiments (Section 4.4) and conclusion (Section 5) that PolyMerge consistently and significantly outperforms HessMerge (by 3.9% to 8.2% in joint mean accuracy), the paper's conceptual narrative is plagued by a major internal contradiction:
- **The Overclaims:** 
  - Abstract: "*...our proposed Hessian-Regularized Coefficient Optimization (HessMerge)... guarantees PTQ robustness...*"
  - Figure 1 Caption: "*...HessMerge flattens the local loss landscape, delivering robust performance across all deployment quantization schemas.*"
  - Section 1 (Intro): "*By minimizing this trace, HessMerge flattens the local loss landscape, mathematically ensuring that weight-space perturbations from any downstream quantization operator result in a minimal drop in performance.*"
- **The Empirical Reality (Table 1):**
  - **HessMerge is empirically identical to unregularized AdaMerging.** Under INT8 Sym (Tensor), HessMerge gets **48.77%** (identical to AdaMerging). Under INT8 Asym (Channel), HessMerge gets **48.80%** (identical to AdaMerging). Under INT4 Sym (Channel), HessMerge collapses to **14.22%** (exactly identical to AdaMerging's 14.22%).
  - In other settings, HessMerge is actually *worse* than AdaMerging (e.g., INT8 Sym Channel: 49.18% vs 49.20%; INT8 Asym Tensor: 47.90% vs 47.95%).
  - Thus, HessMerge's second-order Hessian regularization provides **almost zero empirical benefit over standard AdaMerging**, and collapses to exactly the same random level under INT4.
  - The claim that HessMerge "guarantees PTQ robustness" and "delivers robust performance across all schemas" is flatly contradicted by their own data. Under INT4, individual task accuracies collapse to: MNIST **18.50%**, FashionMNIST **16.10%**, CIFAR-10 **12.00%**, and SVHN **9.50%** (Joint Mean: **14.22%**), which is essentially the 10.0% random-guessing floor.

---

## 4. Minor Flaws and Suggestions

### 1. Invalid Assumption on Vanishing Weight-Space Gradient
The theoretical derivation in Section 3.2 assumes that the weight-space gradient vanishes ($\nabla_W \mathcal{L}(W_{\text{merged}}) \approx 0$) at the continuous optimized state. This is false. The coefficients $\boldsymbol{\lambda}$ are optimized, NOT the model parameters $W$. Being at a local minimum in the 56-dimensional coefficient space ($\nabla_{\boldsymbol{\lambda}} F(\boldsymbol{\lambda}^*) = 0$) does NOT imply that we are at a local minimum in the massive 5.7M-dimensional weight space ($\nabla_W \mathcal{L} \approx 0$). In fact, since the merged weights are a linear combination of pre-trained and fine-tuned weights, they are highly unlikely to be at a local minimum in the parameter space, meaning $\nabla_W \mathcal{L}$ is non-zero, and the first-order Taylor expansion term does not vanish.

### 2. Empirically Ineffective Regularization (Table 2)
The paper presents an ablation study on the regularization strength $\gamma$ in Table 2 of Section 4.6. However, the reported accuracy values actually undermine the core claims of HessMerge's effectiveness:
- Comparing unregularized AdaMerging ($\gamma = 0.0$) with the "optimal" regularized HessMerge ($\gamma = 0.5$), the joint mean accuracy difference is practically non-existent: a negligible **+0.03%** improvement in FP32 space (49.12% to 49.15%) and exactly **0.00%** difference under INT8 Symmetric Tensor-wise quantization (both at 48.77%).
- As the regularization strength increases beyond $\gamma = 0.5$, the joint mean accuracy collapses steadily, falling to **48.10%** in FP32 and **47.90%** in INT8 at $\gamma = 2.0$.
This demonstrates that the proposed second-order regularizer has virtually zero empirical benefit under realistic post-training quantization, contradicting the theoretical claims that it "guarantees PTQ robustness."

### 3. Extremely Poor Baseline Performance and Practical Utility
The individual task experts are highly accurate (e.g., MNIST expert achievements are **96.30%**), but when merged, the continuous FP32 joint mean accuracy is only **49.15%** under HessMerge and **49.12%** under AdaMerging. Individual task accuracies degrade severely (e.g., MNIST drops to **22.90%** in FP32, and CIFAR-10 drops to **75.70%**). This severe degradation under TTA on prediction entropy suggests that the merging process or calibration setup is highly unstable, limiting the practical utility of the proposed merging framework.

---

## 5. Conclusion
While the idea of projecting weight-space curvature into a low-dimensional merging coefficient space is mathematically elegant and creative, the paper suffers from fatal flaws in its theoretical foundation (the subspace vs. weight-space disconnect, and the disconnect between the normalized theory and the unnormalized code implementation), major empirical overstatements (being consistently and significantly outperformed by the PolyMerge baseline, and collapsing under INT4), and a severely compromised experimental setup (merged models acting as random classifiers under INT4). I must strongly recommend a **Reject** for this submission.
