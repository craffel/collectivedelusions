# Intermediate Evaluation 3: Soundness and Methodology

This document provides a highly rigorous evaluation of the mathematical formulations, proofs, and methodological assumptions presented in the paper, focusing on identifying any potential technical flaws, limitations, or conceptual gaps.

## 1. Clarity of Description
The methodology is exceptionally well-written, structured, and presented with a high degree of mathematical clarity. The equations use standard notation, and the step-by-step formulations of SVS, SVD, and BWN are precise and easy to follow.

---

## 2. Appropriateness of Methods
The choice of Singular Value Decomposition (SVD) for spectral filtering is theoretically flawless. According to the **Eckart-Young-Mirsky Theorem**, truncating the singular value spectrum yields the unique, optimal low-rank matrix approximation under any unitarily invariant norm (such as the Frobenius norm). This provides a rigorous mathematical guarantee for the "energy-preservation" of sliced task vectors.

The grouping strategy for higher-dimensional tensors (e.g., flattening convolutional kernels into 2D matrices by grouping the output channel dimension) is a standard and sensible practice in tensor decomposition, though alternative flattening axes could influence the resulting singular value spectrum.

---

## 3. Deep Mathematical Scrutiny: Potential Gaps & Technical Flaws
While the paper's mathematical framework is impressive, a rigorous "theorist-style" examination reveals several subtle gaps, unstated assumptions, and oversimplifications in the proofs and formulations:

### A. The Bias Term in Scale-Invariance Proofs (Section 3.4.1 & 3.4.2)
In Case 1 (L2-norm) and Case 2 (LayerNorm), the cancellation of the global weight scaling factor $\alpha > 0$ relies on the assumption that the output of the layer scales exactly linearly with $\alpha$, i.e., $\mathbf{h}_{BWN} = \alpha \mathbf{h}$. 
However, the linear projection layer is defined as:
$$\mathbf{h} = X W^T + \mathbf{b}$$
If we scale the weight matrix $W$ by $\alpha$ but **do not** scale the bias vector $\mathbf{b}$ by $\alpha$, the scaled projected activations become:
$$\mathbf{h}_{scaled} = X (\alpha W)^T + \mathbf{b} = \alpha X W^T + \mathbf{b} \ne \alpha (X W^T + \mathbf{b})$$
In Section 3.3 ("Dimensionality and Bias Constraints"), the authors explicitly state that bias vectors $\mathbf{b}$ are merged via standard linear Task Arithmetic and **are omitted from Frobenius norm scale averaging (BWN)**. 
Because the bias $\mathbf{b}$ is non-zero in general and is not scaled by $\alpha$, **the scaling factor $\alpha$ does not factor out exactly in the numerator and denominator, and thus does not mathematically cancel out.** 
* **Theorist Verdict:** The scale-invariance is technically an **approximation** rather than an exact identity whenever a non-zero bias vector is present and not scaled proportionally. The authors should explicitly state this boundary condition.

### B. Crucial Mathematical Flaw in the Residual Block Scaling Assumption (Section 3.4.4)
In Section 3.4.4, the authors analyze the residual block boundary condition:
$$\mathbf{y}_{scaled} = \mathbf{x} + \mathcal{F}_{\alpha}(\text{LN}(\mathbf{x}))$$
They assert: *"Since $\mathcal{F}$ consists of linear projections followed by layer normalization, the output of $\mathcal{F}$ scales linearly: $\mathcal{F}_{\alpha}(\text{LN}(\mathbf{x})) = \alpha \mathcal{F}(\text{LN}(\mathbf{x}))$."*
This assertion is **mathematically incorrect** for standard Transformer block components (MHA and MLP):

1. **In the Multi-Head Attention (MHA) block:**
   The MHA block incorporates a non-linear softmax operation:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
   If we scale the projection weights ($W_q, W_k, W_v, W_o$) by $\alpha > 0$, the Query and Key matrices scale as $Q_{scaled} = \alpha Q$ and $K_{scaled} = \alpha K$. Their product scales quadratically:
   $$\frac{Q_{scaled} K_{scaled}^T}{\sqrt{d_k}} = \alpha^2 \frac{Q K^T}{\sqrt{d_k}}$$
   Because the softmax activation is non-linear, scaling its input by $\alpha^2$ behaves as a **temperature scale**, altering the entropy and shape of the attention distribution. It does *not* scale the output linearly. Indeed:
   $$\text{Attention}_{scaled} = \text{softmax}\left(\alpha^2 \frac{Q K^T}{\sqrt{d_k}}\right) (\alpha V)$$
   Multiplying by the scaled output projection $W_o$ (scaled by $\alpha$):
   $$\mathcal{F}_{\alpha} = \alpha^2 \text{softmax}\left(\alpha^2 \frac{Q K^T}{\sqrt{d_k}}\right) V W_o^T \ne \alpha^2 \mathcal{F} \quad (\text{or } \alpha \mathcal{F})$$
2. **In the MLP block:**
   The activation functions used in modern MLPs (e.g., GELU, Swish/SiLU) are non-linear and **not homogeneous of degree 1**:
   $$\text{GELU}(\alpha z) \ne \alpha \text{GELU}(z)$$
   Consequently, scaling the gating or expansion weights by $\alpha$ does not scale the MLP block output linearly:
   $$\text{MLP}_{scaled}(\mathbf{z}) \ne \alpha \text{MLP}(\mathbf{z})$$
* **Theorist Verdict:** This is a significant conceptual flaw. The block output $\mathcal{F}_{\alpha}$ does not scale linearly as $\alpha \mathcal{F}$ due to softmax temperature effects and non-linear, non-homogeneous activation functions. The scale-invariance fails inside residual blocks not just because the skip connection $\mathbf{x}$ is unscaled, but also because the parameterized blocks themselves do not scale linearly with weight magnitude.

### C. Uniqueness in the Eckart-Young-Mirsky Theorem
The paper states that the sliced task vector $\tilde{T}_t$ is the **unique** optimal low-rank matrix approximation.
Mathematically, the Eckart-Young-Mirsky low-rank approximation is unique **if and only if** the $k$-th singular value is strictly greater than the $(k+1)$-th singular value ($\sigma_k > \sigma_{k+1}$). If $\sigma_k = \sigma_{k+1}$, the optimal rank-$k$ subspace is not unique. While this is a minor boundary case, complete theoretical precision requires stating this condition.

### D. Shannon Spectral Entropy and Multi-Task Aggregation
In Section 3.5, the authors define $k_l$ using the task-specific update matrix $T_t$. However, in a multi-task setting, there are $N$ different task-specific updates $T_1, \dots, T_N$ at each layer, each potentially having a different singular value spectrum and spectral entropy $H(T_t)$. 
The paper does not explicitly detail how these multiple task-specific spectral entropies are aggregated to determine a single layer-wise rank $k_l$ for the merge. (e.g., Are the entropies averaged? Is the maximum taken?).

---

## 4. Reproducibility
Despite the subtle theoretical gaps discussed above, the methodology is highly reproducible. SVS, BWN, and Entropy-SVS are completely deterministic, closed-form, and analytical. They require no iterative optimization or random seed states, ensuring that any researcher can perfectly reproduce the results using standard numerical linear algebra libraries (such as PyTorch or NumPy).
