# Intermediate Evaluation 1: Paper Summary

## 1. Main Topic and Scope
This paper addresses the problem of **post-hoc model merging** (or expert ensembling) of deep neural networks on edge/resource-constrained hardware. Model merging aims to integrate several task-specific specialized expert models (fine-tuned from a common base pre-trained model) into a single model substrate. Traditional static model merging suffers from coordinate conflicts and lacks sample-specific dynamic routing. Traditional dynamic routing networks (like Mixture of Experts or learned routing gates) suffer from **heterogeneity collapse** under streaming mixed-task batching—where deep learning frameworks average ensembling coefficients across the batch dimension to maintain contiguous tensor memory, destroying task-specific specialization. 

To resolve these competing constraints, the paper proposes a novel framework: **Endosymbiotic Holographic Parameter Binding (EHPB)**. EHPB draws from Vector Symbolic Architectures (VSAs) and hyperdimensional computing (HDC) to superimpose multiple task vectors within a single weight matrix and demodulate them dynamically, sample-by-sample, in parallel inside hardware registers.

## 2. Proposed Approach (EHPB)
The EHPB approach operates through the following stages:
1. **Carrier Key Generation:** Random bipolar keys $K_k \in \{-1, 1\}^{R \times C}$ are generated for each expert and layer by taking the sign of the outer product of two 1D random signatures ($K_k = \text{sign}(\epsilon_R) \text{sign}(\epsilon_C)^T$). These keys are pseudo-orthogonal in high-dimensional space and frozen during training and deployment.
2. **Holographic Superposition:** Task-specific parameter offsets (task vectors $V_k = W_k - W_{\text{base}}$) are modulated element-wise via the Hadamard product with their respective carrier keys and summed into a single holographic weight matrix:
   $$W_{\text{holo}} = \sum_k V_k \odot K_k$$
3. **Dynamic Routing:** A lightweight linear router predicts sample-specific task-affinity coefficients $\alpha_{k, b} \in [0, 1]$ based on pooled backbone features.
4. **Holographic Demodulation (Parameter Transcription):** For each sample $b$ in the batch, a sample-specific unbinding operator is constructed: $U_b = \sum_k \alpha_{k, b} K_k$. The dynamic, sample-specific weights are transcribed as:
   $$W_b = W_{\text{base}} + W_{\text{holo}} \odot U_b$$
   This demodulation is executed in parallel across the batch using PyTorch's vectorized map ($\mathtt{torch.vmap}$) or fused Triton kernels.

## 3. Key Findings and Theoretical Frameworks
* **The Post-Hoc Model Ensembling Trilemma:** The authors formalize a theoretical trade-off proving that no existing method can simultaneously achieve **Dynamic Adaptability** (sample-wise gating), **Resource Efficiency** ($O(P)$ active memory footprint independent of the number of experts $K$), and **Weight Integrity** (zero weight reconstruction noise, i.e., $\Xi_b = 0$). Traditional ensembling is constrained to choose at most two axes.
* **The Coordinate Isolation Confounder:** Systematic dimension sweeps reveal that element-wise Hadamard holographic binding yields a scale-invariant weight-reconstruction error ($\sim 170\%$) regardless of the representation dimension $D$. The authors prove that this is due to the coordinate-wise nature of Hadamard multiplication, which prevents the central limit averaging noise-decay ($O(1/\sqrt{D})$) enjoyed by vector symbolic architectures that use circular convolution.
* **Non-Linear Noise Propagation:** The authors mathematically model how zero-mean weight reconstruction noise ($\Xi_b$) degrades deep network activations, deriving:
  1. *Systematic Positive Bias Rectification:* Zero-mean noise passed through ReLUs is rectified into a strictly positive bias offset ($B^{(l)} \geq 0$) that compounds across layers.
  2. *LayerNorm Signal Attenuation:* Weight noise increases pre-activation variance, causing LayerNorm to exponentially scale down the active semantic signal across $L$ layers.
* **Mitigation Roadmaps:** The paper evaluates:
  1. *Residual-EHPB:* Storing the top $p\%$ (e.g., 5%) most critical task-vector coordinates uncompressed, providing a noise-free "clean path."
  2. *Continuous Cleanup Networks (CCN):* Lightweight post-activation bottleneck MLPs that denoise activations.
  3. *Activation-Space Projection Layers (ASPL):* Projecting activations onto task-specific principal subspaces.
  4. *Post-Hoc ReLU Bias Correction:* Subtraction or learning scales/shifts post-activation to absorb rectification bias.

## 4. Explicitly Claimed Contributions (with Evidence in Paper)
1. **Introduction of EHPB:** Design of a hyperdimensional weight-space superposition paradigm (Section 3).
2. **Exposing and Solving Heterogeneity Collapse:** Building a mixed-task streaming benchmark showing that classical routers experience performance degradation (e.g., a drop of -2.7% for SOTA QWS) due to hardware-level ensembling coefficient averaging, while EHPB is perfectly immune (Delta = 0.0%) because unbinding is sample-specific (Section 4.3).
3. **Deconstruction of Hadamard Scale-Invariance:** Proving that element-wise binding suffers from the "Coordinate Isolation Confounder," keeping reconstruction error invariant at ~170% across dimensions $D \in [64, 2048]$ (Section 4.2).
4. **Circular Convolution Weight-Binding Roadmap:** Providing a mathematical proof in Appendix A showing that circular convolution achieves $O(1/\sqrt{D})$ relative reconstruction noise decay, and illustrating this on discrete template lookups where incorrect-template similarity decays as $O(1/\sqrt{D})$ (Section 4.2).
5. **Practical Systems-Level Formulation:** Custom Triton GPU kernel layouts for register-level fused demodulation to achieve true $O(P)$ execution memory and avoid dynamic eager-mode memory bottlenecks (Appendix D & E).
6. **Empirical Evaluation of Hybrid Frameworks:** Validating Residual-EHPB (5% sparsity), row-wise structured residual masking, CCNs (linear and MLP-based), ASPLs, and post-hoc ReLU bias correction to rescue classification performance (Section 4).
