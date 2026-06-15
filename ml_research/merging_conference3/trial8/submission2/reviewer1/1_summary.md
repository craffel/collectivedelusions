# Evaluation Phase 1: Summary of the Paper

## 1. Main Topic and Scope
The paper focuses on the deployment of multi-task foundation models on resource-constrained consumer edge hardware (such as mobile phones, low-power microcontrollers, and IoT nodes). Specifically, it targets the memory, latency, and bandwidth bottlenecks associated with serving multiple concurrent low-rank Parameter-Efficient Fine-Tuning (PEFT) adapters (like LoRA) on-device. The scope covers:
*   On-device multi-expert serving in pure low-precision integer formats (INT8/INT4).
*   Dynamic activation-space ensembling inside a single, parallel forward pass.
*   Training-free, post-hoc quantization-aware calibration.
*   Input routing via early-stage Zero-Shot Centroid Alignment (ZCA) and out-of-distribution (OOD) task detection.
*   Mitigation of latency-heterogeneity trade-offs and routing-blending execution contradictions on edge CPUs.

---

## 2. Technical Approach and Framework
The authors propose **Q-SPS** (Quantized Single-Pass Activation-Space Dynamic Blending) and its execution-gated variant **CG-Q-SPS** (Conditional Gated Q-SPS). The framework is composed of five core pillars:
1.  **Quantized LoRA Experts & Pure Integer Arithmetic:** Expert LoRA weights ($A_k$ and $B_k$) are symmetrically quantized to low bitwidths (INT4/INT8) and executed natively in integer precision to avoid floating-point overhead and DRAM-to-SRAM weight-switching latencies. Intermediate down-projection products are accumulated in 32-bit registers, dynamically scaled via a max-abs heuristic, and re-quantized to 8-bit before up-projection, converting back to floating-point (FP16/FP32) only at the block boundary.
2.  **Quantization-Aware Scale Calibration (QASC):** A post-hoc, training-free calibration protocol that utilizes a small calibration split ($|D_{\text{cal}}| = 64$ samples) to optimize weight scale factors ($s_A$ and $s_B$) by minimizing the Mean Squared Error (MSE) between unquantized and quantized outputs. To prevent $O(N^2)$ joint search complexity, QASC sequentially decouples down-projection and up-projection optimization into highly efficient $O(N)$ line searches.
3.  **Zero-Shot Centroid Alignment (ZCA) Routing with IDC:** Inputs are routed task-agnostically at early Layer 3 based on cosine similarities to pre-computed centroids. The authors introduce **Intra-Task Dispersion Calibration (IDC)** to normalize similarity scales and prevent over-routing. Log-sum-exp stabilization is applied to prevent numerical overflow in FP16 at extremely low routing temperatures ($\tau=0.001$).
4.  **Conditional Gating (CG-Q-SPS):** A dynamic bypass that applies a threshold ($\theta = 0.01$) to the routing coefficients ($\alpha_{k, b}$). If an expert's weight falls below $\theta$, its execution is skipped entirely, solving the "routing-blending contradiction" (where parallel ensembling is computed but a near-one-hot coefficient zeroes out the output).
5.  **Coordinate GMM Safety Shield:** A low-dimensional diagonal Gaussian Mixture Model (GMM) is fitted over the low-dimensional ZCA coordinates of the calibration split. At runtime, if a sample's log-likelihood falls below an empirical threshold ($\eta$), it is flagged as out-of-distribution (OOD), bypassing all adapters to execute strictly via the base backbone.

The authors also mathematically formulate and explore **Gram-Schmidt Cross-Centroid Orthogonalization (GS-CCO)** and **L{\"o}wdin Symmetric Manifold De-Entangling (SMD)** to study basis orthogonalization on non-orthogonal task manifolds.

---

## 3. Key Findings and Quantitative Claims
Based on a high-fidelity, hardware-calibrated analytical simulation of a 12-layer Vision Transformer (ViT-Tiny) backbone across four visual domains (MNIST, Fashion-MNIST, CIFAR-10, SVHN) inside an "Isolating Coordinate Sandbox" (ICS), the paper reports:
*   **Accuracy Preservation:** Under extreme 4-bit quantization, CG-Q-SPS (INT4 + QASC) preserves a simulated joint mean accuracy of **79.40%**, recovering **99.5%** of the unquantized FP32 expert ceiling (**79.80%**), outperforming standard uncalibrated 4-bit merging by **+0.96%** absolute accuracy, and completely avoiding the structural collapse associated with static quantized merging (which drops to a near-random **30.70%**).
*   **Memory Savings:** 4-bit quantization slashes expert adapter memory footprints by **87.5%** (from 2.76 MB down to 0.345 MB), allowing dozens of experts to fit natively inside microcontroller SRAM or L1/L2 caches ($<512$ KB).
*   **Latency Improvements:** CG-Q-SPS delivers a projected **3.97$\times$ physical speedup** (189.1 ms vs 749.8 ms cumulative latency over 1,024 samples) over the state-of-the-art sequential micro-batching baseline (PFSR+MBH) on heterogeneous streams.
*   **Energy Consumption Savings:** CG-Q-SPS (INT4) consumes only **0.46 J** per batch, representing a **56.2% energy savings** over sequential micro-batching (0.90 J) and a **55.2% energy savings** over unquantized parallel ensembling (1.05 J).
*   **Robust OOD Rejection:** The Coordinate GMM safety shield achieves a highly precise **95.2% True Positive Rate (TPR)** at only a **4.3% False Positive Rate (FPR)** for OOD task detection (AUC = 0.98), significantly outperforming uncalibrated global cosine similarity (AUC = 0.72) and deep-learning OOD baselines.
*   **Orthogonalization Detriment:** Explicit orthogonalization (GS-CCO, L{\"o}wdin SMD) is shown to be mathematically redundant and even detrimental under noise compared to unorthogonalized **ZCA-IDC** (which preserves a Routing Accuracy of 94.70% and a low Flicker Rate of 10.34% at $\epsilon=0.8$) due to noise spillover across joint projection spaces.

---

## 4. Explicitly Claimed Contributions
The authors claim the following primary contributions:
1.  **Transparent Analytical Simulation Study:** Framing the ICS sandbox and publishing code, parameters, and hardware assumptions calibrated against Broadcom BCM2711 specifications to isolate systems variables.
2.  **Immunization Against Collapse at Low Precision:** Showing that Q-SPS is completely immune to the heterogeneity collapse of parametric routers and parameter-space merging.
3.  **Resolution of Routing-Blending Contradiction:** Bypassing inactive low-rank expert pathways dynamically with conditional gating ($\theta=0.01$).
4.  **Massive Memory Footprint Savings:** Slashing adapter sizes by 87.5% via 4-bit symmetric integer quantization.
5.  **High-Throughput Serving:** Single-pass parallel execution bypasses the sequential micro-batching bottlenecks of prior edge systems, achieving a flat, predictable latency profile.
6.  **Robust GMM OOD Rejection:** Providing a highly precise diagonal coordinate-space GMM shield to handle OOD prompt rejection.
7.  **Rigorous Evaluation of Orthogonalization Redundancy:** Mathematically formalizing and evaluating GS-CCO and L{\"o}wdin SMD to prove that explicit basis orthogonalization is detrimental under noise.
