# Evaluation Task 1: Summary of the Submission

## 1. Main Topic and Scope
The submission addresses the problem of post-hoc **model merging**, which aims to combine several task-specific specialized expert neural networks into a single, unified multi-task model without expensive joint retraining. Specifically, the paper focuses on the tension between:
1. **Resource efficiency:** Deploying multiple models on edge devices without scaling active memory linearly with the number of experts ($O(K \times P)$).
2. **Dynamic adaptability:** Customizing model weights sample-by-sample at test-time based on input characteristics rather than using a static average.
3. **Task heterogeneity collapse:** A hardware/framework-level bottleneck where batching heterogeneous task requests averages routing coefficients across the batch, destroying expert specialization.

The scope is bounded within a simulated, multi-task representation sandbox evaluating vision-based classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) using a pre-trained Vision Transformer (ViT-Tiny) backbone.

---

## 2. Methodology and Technical Approach (EHPB)
The authors propose **Endosymbiotic Holographic Parameter Binding (EHPB)**, which draws mathematical inspiration from Vector Symbolic Architectures (VSA) and hyperdimensional computing. The core steps of the EHPB framework are:

1. **Task Vector Extraction:** Compute the parameter difference (task vector) for each expert: $V_k^{(l)} = W_k^{(l)} - W_{\text{base}}^{(l)}$.
2. **Carrier Key Generation:** For each task $k$ and layer $l$, generate a 2D random bipolar carrier key $K_k^{(l)} \in \{-1, 1\}^{R \times C}$ by taking the outer product of two 1D random bipolar sign vectors: $K_k^{(l)} = r_k^{(l)} (c_k^{(l)})^T$.
3. **Holographic Superposition:** Modulate each task vector with its carrier key using element-wise Hadamard multiplication and sum them into a single holographic substrate matrix: $W_{\text{holo}}^{(l)} = \sum_{k} V_k^{(l)} \odot K_k^{(l)}$.
4. **Dynamic Routing:** A lightweight linear routing network trained on a 64-sample multi-task calibration set predicts sample-specific coefficients $\alpha_{k, b}$ for each sample $b$ in a batch.
5. **Holographic Demodulation (Transcription):** Construct a sample-specific unbinding operator $U_b^{(l)} = \sum_k \alpha_{k, b} K_k^{(l)}$ and dynamically transcribe the active weight matrix sample-by-sample: $W_b^{(l)} = W_{\text{base}}^{(l)} + W_{\text{holo}}^{(l)} \odot U_b^{(l)}$. This is parallelized across the batch using PyTorch's vectorized map ($\mathtt{torch.vmap}$).

### Mitigations for Noise Propagation
Because Hadamard binding introduces severe weight reconstruction cross-talk noise ($\Xi_b^{(l)}$), the authors introduce three mitigation strategies:
- **Residual-EHPB:** Isolates the top $p\%$ of critical weight coordinates (by magnitude) in an uncompressed sparse matrix, bypassing holographic superposition to create a "clean path" for propagation. It also includes a hardware-friendly **Structured Row-wise Residual-EHPB** variant.
- **Continuous Cleanup Networks (CCN):** Post-hoc linear or bottleneck MLP blocks placed between layers to denoise intermediate activations and contain noise cascade.
- **ReLU Bias Correction:** Addresses the systematic positive bias shift that occurs when zero-mean reconstruction noise passes through rectifying non-linearities (e.g., ReLU), utilizing running subtraction or learnable scale and shift parameters.

---

## 3. Key Findings
- **Immunity to Heterogeneity Collapse:** Standard dynamic routing models experience performance drops (e.g., -2.7% for QWS SOTA) when ensembling coefficients are averaged across heterogeneous batches to satisfy hardware-batching constraints. EHPB is completely immune (0.0% drop) because it applies sample-specific demodulation operators.
- **The Hadamard Dominance Paradox:** While EHPB is resource-efficient and dynamically adaptable, its joint mean accuracy of **25.4%** in the sandbox is heavily dominated by static Uniform Merging (**52.3%**) and a Direct Sample-wise Router (**51.0%**). The cross-talk noise penalty of element-wise Hadamard binding is exceptionally severe.
- **Scale Invariance of Hadamard Binding (Coordinate Isolation Confounder):** Systematic sweeps across hidden dimensions ($D \in [64, 2048]$) reveal that relative reconstruction error remains constant (around 170%–179%) rather than decaying as $O(1/\sqrt{D})$ like classical VSAs. This is traced to the coordinate-isolated nature of element-wise Hadamard multiplication, which scales signal and noise symmetrically.
- **Circular Convolution Noise-Decay:** In contrast to Hadamard binding, circular convolution-based binding demonstrates the classic $O(1/\sqrt{D})$ noise-decay behavior in clean associative retrieval (cosine similarity), pointing to a critical roadmap for future hyperdimensional model merging.
- **Mitigation Performance:** 
  - *Residual-EHPB:* Rescues joint mean accuracy to 33.7% (unstructured) and 31.0% (structured row-wise) at $p=5\%$.
  - *CCN:* Denoises activations (up to 8.1$\times$ MSE reduction) and significantly rescues MNIST accuracy (+20.0% absolute), but linear post-hoc mapping introduces projection distortions on low-SNR tasks. Bottleneck MLPs partially resolve this.
  - *ReLU Bias Correction:* Learnable scale and shift calibration on a 16-sample calibration set achieves a 31.4% reduction in propagation MSE and increases cosine similarity to 0.9492.

---

## 4. Explicitly Claimed Contributions and Accompanying Evidence
1. **The EHPB Paradigm:** Formalized mathematically in Section 3 and validated empirically in Section 4.
2. **Identification of "Heterogeneity Collapse":** Described conceptually in Section 1 & 2 and verified via an index-shuffled mixed-task deployment audit ($B=256$) in Section 4.3.
3. **Hadamard Boundary and Coordinate Isolation Confounder:** Deconstructed theoretically in Section 4.2 via a dimensional scaling sweep ($D \in [64, 2048]$).
4. **Validation of Hybrid Mitigation Strategies (Residual-EHPB, CCN, ReLU Bias Correction):** Detailed in Section 4.5, 4.6, and 4.7 respectively, with explicit tables mapping MSE reductions and accuracy rescues.
