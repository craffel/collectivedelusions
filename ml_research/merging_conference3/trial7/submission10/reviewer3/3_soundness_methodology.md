# Soundness and Methodology Evaluation: SPS-ZCA

## Clarity of Description
The paper is generally well-structured and explains the proposed SPS-ZCA framework with high clarity. The overall pipeline, from pre-computation to activation blending and robustness calibration, is described step-by-step with accompanying equations. The hardware-aware memory bandwidth and execution cost models are detailed, providing a concrete systems perspective.

## Appropriateness of Methods
From a systems-engineering standpoint, the proposed methods are appropriate for the targeted problem of low-latency on-device multi-expert serving. Using single-pass parallel execution instead of multi-pass batch splitting directly targets the dominant DRAM-to-cache bottleneck of edge CPUs. However, from a **rigorous theoretical perspective**, several of the core methods are heavily reliant on empirical heuristics and lack formal mathematical foundations or guarantees.

## Theoretical Soundness Gaps and Potential Technical Flaws

A theory-minded evaluation reveals several critical mathematical gaps and potential technical flaws in the proposed methodology:

### 1. Linear Blending of Non-Linear Pathways (Lack of Mathematical Stability and Error Bounds)
The core of the Single-Pass Activation Blending (SPS) formulation (Equation 5) assumes that task-specific activations can be blended linearly at each layer $l$:
$$h_b^{(l)} = h_b^{(l-1)} W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_{k, b} \left( h_b^{(l-1)} A_k^{(l)} B_k^{(l)} \right)$$
However, in a deep neural network (such as a Transformer), the output of layer $l$ is passed through highly non-linear operations (including GELU activations, Layer Normalization, and the Softmax function in Multi-Head Attention) before entering layer $l+1$. 
Because these operators are non-linear, a linear combination of adapter outputs at layer $l$ does *not* mathematically correspond to a linear combination of downstream activations at layer $l+1$:
$$\text{Block}^{(l+1)}(h_b^{(l)}) \neq \sum_{k=1}^K \alpha_{k,b} \text{Block}^{(l+1)}\left(h_b^{(l-1)} W_{\text{base}}^{(l)} + h_b^{(l-1)} A_k^{(l)} B_k^{(l)}\right)$$
The paper provides **no theoretical analysis, stability proofs, or mathematical bounds on the error propagation** when blending activations across multiple sequential non-linear layers. If the routing coefficients $\alpha_{k,b}$ are non-binary (e.g., in soft routing or during dynamic temperature relaxation), this linear blending can lead to representation drift, steering intermediate features into out-of-distribution regions of the activation space and degrading downstream performance.

### 2. Lack of Mathematical Separability Guarantees in Early-Stage Routing
The Zero-Shot Centroid Alignment (ZCA) routing relies on the assumption that representations are highly separable in early-stage layers (Layer 3). While the authors compute the empirical Fisher Separability Criterion (FSC) to justify this, they provide no formal proof or mathematical guarantees under which conditions this early separability is guaranteed to hold. 
- Under what assumptions about the pre-trained base model or the data-generating distributions of the task suite is Layer 3 guaranteed to be task-separable?
- How does the separability scale mathematically as the number of experts $K$ increases, or when tasks belong to highly fine-grained or overlapping domains?
The paper acknowledges that spatial overlap causes routing confusion and "activation bleeding," which confirms that the training-free centroid-aligned routing is theoretically brittle and highly sensitive to representational geometry, without providing formal bounds on the maximum acceptable task overlap.

### 3. Overfitting and Estimation Variance in Coordinate GMMs
Fitting a diagonal Gaussian Mixture Model (GMM) with $M=2$ components over similarity coordinates $u_s \in \mathbb{R}^K$ using a tiny calibration split ($|\mathcal{C}_k| = 64$ samples) presents a severe statistical risk of overfitting and high estimation variance.
In multivariate statistics, fitting GMM parameters (means, covariance matrices, and mixture weights) under extreme data scarcity leads to highly unstable density boundaries and singular covariance matrices. Although the authors propose a diagonal ridge regularization term $\gamma I$ ($\gamma = 10^{-4}$), this is a heuristic patch rather than a mathematically derived regularization. The paper lacks a rigorous probabilistic or statistical proof of generalization bounds for this regularized coordinate density estimator under such low-sample regimes.

### 4. Heuristic Calibration Operators
Both **Unit-Norm Calibration (UNC)** and **Intra-Task Dispersion Calibration (IDC)** are presented as novel calibration techniques, but they are purely heuristic:
- **UNC** is mathematically identical to standard cosine similarity. Presenting it as a distinct "calibration technique" is mathematically redundant.
- **IDC** (Equation 9) scales similarity coordinates by dividing them by their expected in-distribution mean. While empirically effective, the authors provide no statistical derivation or proof showing that this division preserves valid probability density properties or corresponds to a rigorous coordinate transformation on the unit hypersphere.

## Reproducibility
The reproducibility of the paper is rated as **good**. The mathematical formulations and parameters (e.g., $L=14$ groups, $D=192$ dimensions, $K=4$ tasks, $r=8$ rank, $\tau=0.001$ temperature, $\gamma = 10^{-4}$ GMM regularization) are explicitly detailed. The authors also outline the detailed hardware-aware formulas and the specific task datasets (MNIST, Fashion-MNIST, CIFAR-10, SVHN). However, because no code repository is attached or referenced, reproducing the exact vectorized C++ compiler-fused loop layout (Appendix A) or the physical PyTorch benchmarking setup would require significant engineering effort from scratch.
