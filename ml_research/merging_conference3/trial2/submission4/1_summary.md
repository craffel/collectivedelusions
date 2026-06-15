# Paper Summary: EdgeMerge (Forward-Only Adaptive Model Merging)

## 1. Overview and Core Motivation
EdgeMerge addresses a critical bottleneck in state-of-the-art adaptive model merging methods (e.g., AdaMerging, SyMerge, FoldMerge): their reliance on expensive, multi-step gradient-based optimization and test-time backpropagation. For resource-constrained edge systems or fast deployment staging, the associated latency (minutes of optimization), massive memory footprint (for backpropagation gradient graphs), and transductive overfitting risks make gradient-based adaptation highly impractical.

To bridge this gap, **EdgeMerge** is proposed as a **one-shot, training-free, forward-only** adaptive model merging framework. It extracts fine-grained, channel-wise merging coefficients in closed-form using a tiny, unlabeled calibration dataset (e.g., $B = 32$) in a single, near-instant forward pass, completely bypassing backpropagation.

---

## 2. Methodology and Core Techniques
EdgeMerge consists of several key stages localized strictly at a strategic "choke-point" bottleneck layer—the visual projection layer (`model.visual.proj`) of CLIP—while the remaining 99.5%+ of model parameters are merged statically via standard Task Arithmetic:

1. **Forward-Only Activation Sampling (FOAS):** Runs a single forward pass of a tiny, unlabeled calibration dataset through the pre-trained base model and task experts. To minimize memory and latency, EdgeMerge reuses the pre-trained base model's upstream encoder outputs ($X_{base}$) to compute activations for all expert models, keeping the memory footprint restricted to a single model's size (~100 MB).
2. **Scale-Normalized Delta Activation Salience (SNDAS):** Calculates the relative change in internal activations between each expert and the base model ($\Delta H_k = H_k - H_{base}$). It normalizes these delta tensors using the Frobenius norm to prevent tasks with larger activation scales from dominating the merge.
3. **Channel-Wise Softmax Gating (CWSG):** Computes channel-wise salience scores and normalizes them across tasks using a softmax function with a temperature hyperparameter $\tau$. This produces localized, channel-wise merging coefficients ($\alpha_k$) to resolve inter-task parameter conflicts.
4. **Decoupled Scale Routing (DSR):** Overcomes representational dampening caused by softmax normalization. Because softmax-normalized channel-wise gating coefficients average expert updates whereas static layers sum them, a scaling discrepancy arises. DSR decouples the scaling factor of the gated visual projection layer ($\lambda_{proj}$) from the scaling factor of the static layers ($\lambda_{static}$).

The final merged weight matrix $W_{MTL}$ is reconstructed row-by-row (channel-by-channel) as:
$$W_{MTL}[j, :] = W_{base}[j, :] + \lambda_{proj} \sum_{k=1}^K \alpha_k[j] \left(W_k[j, :] - W_{base}[j, :]\right)$$
for the gated projection layer, and using standard Task Arithmetic with $\lambda_{static}$ for the rest of the network's static layers.

---

## 3. Experimental Setup
- **Backbone Model:** Vision-Language CLIP ViT-B/32 (CLIP).
- **Datasets (8 Tasks):** SUN397, Stanford Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, and DTD.
- **Baselines:** Task Arithmetic (TA) (optimized via grid search over global scale $\lambda \in [0.1, 0.8]$), SyMerge (ICML 2026), Decoupled Task Arithmetic (DTA, control), and other static alignment baselines (Git Re-Basin, ZipIt!, TIES-Merging).
- **Metrics:** Multi-task average classification accuracy, merge/adaptation preparation time (seconds), backpropagation requirements, peak GPU memory overhead, and inference latency.

---

## 4. Key Results and Main Findings
- **Accuracy Comparison:**
  - **Individual Experts (Oracle):** 91.02% (upper bound, but requires 8$\times$ parameter storage).
  - **SyMerge (SOTA Gradient-Based):** 89.74% (takes 600s, requires full backpropagation).
  - **Task Arithmetic (Optimal $\lambda = 0.2$):** 68.74% (0.1s merge time).
  - **Standard EdgeMerge (Coupled $\tau = 0.5$, $\lambda = 0.3$):** 68.69% (11.95s calibration time).
  - **Decoupled Task Arithmetic (DTA, $\lambda_{static}=0.25, \lambda_{proj}=0.10$):** 69.45% (completely data-free).
  - **Decoupled EdgeMerge (DSR, $\lambda_{static}=0.25, \lambda_{proj}=0.20, \tau=0.10$):** 69.58% (+0.84% over Task Arithmetic peak).
- **Pragmatic Resource Trade-offs:**
  - **Extreme Efficiency:** EdgeMerge achieves a $50\times$ preparation speedup compared to SyMerge (11.95s vs 10 mins) with negligible training RAM (~100 MB vs high backpropagation memory), enabling swift offline staging.
  - **Hyperparameter Robustness (Plateau Preservation):** Standard Task Arithmetic has a fragile peak at $\lambda = 0.2$ and collapses rapidly outside of it. EdgeMerge's dynamic routing opens up a broad, stable plateau around $\lambda = 0.3$, significantly reducing deployment risks.
  - **Absolute Accuracy Transparency:** Acknowledges a 21.05% performance gap relative to server-grade gradient-based optimization (SyMerge), framing EdgeMerge as an extreme-efficiency exploration for resource-constrained staging rather than a raw accuracy competitor under unconstrained conditions.
  - **Calibration Robustness (Data-Free):** Additional experiments demonstrate that EdgeMerge's salience vectors computed from synthetic inputs (Gaussian noise, zero tensors) have an extremely high cosine similarity (~0.91) with those from real physical data, producing identical test performance and opening up the possibility of completely data-free model merging.
