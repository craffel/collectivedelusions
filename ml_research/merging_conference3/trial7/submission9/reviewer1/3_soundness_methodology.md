# 3. Soundness and Methodology

## Clarity of Description and Mathematical Framework
The mathematical framework of SABLE is presented with exceptional clarity and rigor. The problem formulation of streaming heterogeneity is precisely defined, and the shift from parameter space (where batch averaging causes heterogeneity collapse) to activation space (where ensembling occurs sample-wise) is clearly articulated. 

The paper leverages the distributive property of matrix multiplication:
$$ Y_b = X_b W_{\text{base}} + \sum_{k=1}^K \alpha_{k, b} \cdot \left( (X_b A_k) B_k \right) $$
which is shown to be mathematically identical to parameter-space ensembling on a per-sample basis ($B=1$). The notation is consistent, and the architectural diagrams (e.g., Figure 1) perfectly illustrate the workflow of SABLE Late Adaptation.

## Evaluation of Mathematical Soundness and Methods

### 1. Multi-Layer Equivalence and Non-Linearities
A primary concern in activation-space ensembling across deep architectures is how the distributive property behaves across multiple sequential layers separated by non-linear activations (e.g., ReLU, GeLU, or SwiGLU). 
The paper mathematically demonstrates that by placing activation blending layers directly inside the linear projection interfaces of transformer blocks or feed-forward heads, exact equivalence to parameter-space ensembling is preserved for each layer. 
However, the authors correctly identify a fundamental theoretical boundary: the compounding effect of **cumulative non-linear drift**. Because successive layers are separated by non-linear operators $\sigma$, the relation:
$$ \sigma\left( \sum_k \alpha_k (X W_k) \right) \neq \sum_k \alpha_k \sigma(X W_k) $$
means that representation errors can accumulate across many deep layers.
To mitigate this, SABLE relies on:
- **The Residual Nature of PEFT:** LoRA adapters act as minor directional adjustments (residual corrections) rather than wholesale representational shifts, minimizing activation divergence.
- **Late Adaptation (Mid-Layer Routing):** Leaving early layers unadapted and blending only the final 2-4 layers restricts cumulative drift to the final classification block.
This mathematical defense is highly convincing and supported by representational drift tracking in Table 5, which shows that representational similarity remains extremely high ($>0.83$ cosine similarity) across a 4-layer physical MLP.

### 2. Zero-Data Centroids and Vector Cancellation
The construction of task centroids directly from expert parameters ($c_{\text{zero}, k} = \frac{1}{C}\sum_c W_{\text{expert}, k}[c, :]$) faces a severe theoretical risk of **vector cancellation**. Because discriminatively trained classification heads are optimized to maximize class margins, their weight vectors often point in opposite or orthogonal directions. Simple coordinate-wise averaging can shrink the centroid's norm, making it highly sensitive to high-frequency numerical noise and causing dual-space manifold mismatch.

SABLE resolves this via **Refined Zero-Data Centroids**:
$$ c_{\text{refined}, k} = \frac{1}{C}\sum_c \frac{W_{\text{expert}, 2, k}[c, :]}{\|W_{\text{expert}, 2, k}[c, :]\|_2} $$
Applying weight-space L2-normalization before averaging successfully prevents vector cancellation, preserving the semantic orientation of the task-prior. This is a highly elegant and mathematically sound refinement. The authors are transparent about the remaining 5.80% dual-space gap between completely zero-data and support-split centroids, which is a commendable level of honesty.

### 3. The Representational Blurring Paradox
In Mid-Layer Routing, the first $L_{\text{route}}$ layers are unadapted. This introduces a major theoretical trade-off: **the complete loss of early task-specific expert features**. SABLE is therefore structurally limited to expert pools whose adaptation is concentrated in late-stage layers.
Furthermore, SABLE exposes the **Representational Blurring Paradox**: unadapted base layers act as a feature compression bottleneck that mixes and blurs task representations, making mid-layer features noisier than raw input space features. This explains why Single-Pass Early-Routing (routing directly in input space) dramatically outperforms 2-pass routing by **+12.70%** (65.20% vs 52.50%) on highly separable grayscale pixel inputs.
However, the authors are careful to outline a critical deployability boundary: on complex, high-dimensional natural datasets, inputs lack input-space separability, causing Single-Pass Early-Routing to suffer from extreme routing noise. In such scenarios, practitioners are bound by a hard architectural trade-off between the latency of 2-pass routing (while suffering from representation blurring), external routing backbones, or Late Adaptation (Mid-Layer Routing).

### 4. GPU Systems Analysis
The systems-level analysis is highly detailed and accurate. Rather than assuming theoretical FLOP savings translate directly to wall-clock speedups, SABLE addresses:
- **CUDA Kernel Launch Overhead:** Acknowledging that executing $K$ parallel adapters in naive sequential Python loops dominates wall-clock time, and explaining how multi-tenant serving frameworks (Punica, S-LoRA) resolve this via batched GPU kernels.
- **GPU Memory Bandwidth Constraints:** Explaining how loading massive pools of experts into SRAM limits performance, and how SABLE's Top-$M$ expert pruning bounds memory bandwidth and compute to $O(M)$ instead of $O(K)$.
This demonstrates a deep understanding of hardware-level execution boundaries, making the methodology highly credible.

## Potential Technical Flaws
No major mathematical flaws or errors were identified in the derivations or methodology. The ensembling algebra is mathematically sound, and all identified trade-offs (non-linear cumulative drift, early-feature loss, dual-space mismatch, representational blurring) are explicitly analyzed and resolved with rigorous mathematical or architectural mitigations.

## Reproducibility
The reproducibility of the work is **excellent**:
- The paper details exact hyperparameter settings (temperature $\tau = 0.05$, OOD threshold $\gamma_{\text{OOD}} = 0.2$, adapter rank $r=8$).
- Complete algorithmic details are provided, including an adaptive dynamic thresholding loop (Algorithm 1) and explicit formulations for centroid construction.
- The experiments are conducted across three diverse setups: a synthetic 14-layer Sandbox, a physical CNN trained from scratch on MNIST/FashionMNIST, and standard ResNet-18 foundation features, providing multiple independent avenues of verification.
- The codebase and LaTeX files are fully provided in the workspace, making it straightforward to audit the implementation.
