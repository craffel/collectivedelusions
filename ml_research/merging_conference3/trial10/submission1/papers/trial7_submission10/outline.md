# Paper Outline: SPS-ZCA

## Title
SPS-ZCA: Single-Pass Sample-Wise Routing with Zero-Shot Centroid Alignment for Ultra-Low Latency and Robust On-Device Model Merging

## Authors & Affiliation
Arthur Vance  
Department of Computer Science, University of Wisconsin-Madison  
`avance@cs.wisc.edu`

---

## 1. Abstract
- **Context:** Modular deep learning via fine-tuning task-specific experts (e.g., LoRA) from a shared pre-trained base model is highly popular for flexible on-device adaptation.
- **Problem:** Dynamic model-merging methods suffer from a severe trade-off. Static weight-merging suffers from "heterogeneity collapse" under mixed-task streams. Micro-Batch Homogenization (MBH) avoids this but introduces up to $4\times$ latency on resource-constrained edge CPUs due to running up to $K$ sequential backbone passes. Furthermore, non-parametric routing based on classification heads (like PFSR) collapses on out-of-distribution (OOD) tasks (e.g., SVHN).
- **Routing Paradox Resolution:** Traditional dynamic routers have a temporal circular dependency: they require late-stage penultimate representations to compute routing coefficients, which means running the backbone twice. SPS-ZCA resolves this by routing using **early-stage embedding-space representations** (right after the Patch Embedding, Layer 0), enabling a true, single forward pass.
- **Proposed Solution:** **SPS-ZCA** (Single-Pass Sample-Wise Routing with Zero-Shot Centroid Alignment).
  - *SPS (Single-Pass Activation-Space Dynamic Blending):* Combines expert adapter activations sample-wise inside a single, parallel forward pass of the shared base model, converting sequential $O(K)$ latency back to constant $O(1)$ latency.
  - *ZCA (Zero-Shot Centroid Alignment):* Projects early representations onto robust task centroids in the embedding space (Layer 0), bypassing noisy classification heads and resolving the routing paradox.
  - *UNC (Unit-Norm Calibration):* Resolves cross-expert representation scale imbalances.
  - *IDC (Intra-Task Dispersion Calibration):* Neutralizes spatial dispersion asymmetries across compact vs. dispersed task manifolds.
- **Key Results:**
  - Recovers **100.0%** of the Expert Ceiling (Joint Mean of **79.80%**), outperforming the prior SOTA (PFSR+MBH) by **+3.66%** absolute accuracy.
  - Slashes inference latency by **3.90x** (199.0 ms vs. 776.4 ms) on highly heterogeneous streams with batch size $B=256$.
  - Delivers a flat latency profile and $>1000$ samples/sec on edge CPU configurations.
  - Provides robust calibration against manifold dispersion differences, and a diagonal GMM-based coordinate density estimator achieves **95.2%** true OOD task rejection.

---

## 2. Introduction
- **Modular Deep Learning:** Shift towards Parameter-Efficient Fine-Tuning (PEFT/LoRA) from massive pre-trained foundation models.
- **The Deployment Challenge:** Serving multiple task-specific adapters on resource-constrained edge devices (smartphones, IoT, smart appliances) with strict latency, memory, and energy budgets.
- **The Battle of Merging Schemes:**
  - *Static Weight Merging:* Fast but suffers from representation collapse when executing mixed-task inputs simultaneously.
  - *Dynamic Merging & MBH:* High task-specificity but sequential execution of micro-batches introduces a linear latency penalty ($O(K)$) which is prohibitive in the wild.
  - *Head-Based Routing (PFSR):* Vulnerable to domain shifts and out-of-distribution (OOD) noise due to reliance on classification heads.
- **The Core Routing Paradox:** Show how computing routing weights from late penultimate features requires running the base model twice, and how routing in early-stage embedding space (Layer 0) resolves it.
- **Our Pragmatic Philosophy:** Dynamic merging must be fast, robust, require zero parameter updates, and have zero additional VRAM or sequential backbone overhead.
- **Our Proposal (SPS-ZCA):**
  - Sample-wise activation-blending inside a single forward pass ($O(1)$).
  - Geometrically grounded early embedding centroid projection (ZCA) for stable, head-independent, paradox-free routing.
  - Robust calibration (UNC and Intra-Task Dispersion Calibration) to tackle real-world representation and manifold spread imbalances.
- **Summary of Contributions:**
  1. Formulating SPS-ZCA to achieve $3.90\times$ speedup and $O(1)$ constant backbone pass scaling on mixed streams.
  2. Introducing embedding-space ZCA to bypass classification head asymmetries and resolve the temporal routing paradox, boosting OOD accuracy by $+3.66\%$ absolute.
  3. Proposing UNC, IDC, and GMM coordinate density estimation to handle expert scale imbalances, manifold variance asymmetries, and OOD queries with up to 95.2% rejection.
  4. Outlining concrete Systems-ML co-design guidelines and hardware-aware memory-bandwidth models (ARM Cortex-A72 / LPDDR4 memory) for edge CPU and microcontroller deployment.

---

## 3. Related Work
- **Parameter-Efficient Fine-Tuning (PEFT):** LoRA, Prefix Tuning, Adapters.
- **Weight-Space Model Merging:** Task Arithmetic, TIES-Merging, DARE, and their limitations.
- **Dynamic Merging & Routing:** MoE (Mixture of Experts) variants, PFSR, and Micro-Batch Homogenization (MBH) and their high computational/latency costs on sequential hardware.
- **Resource-Constrained Deep Learning:** On-device serving, edge CPU constraints, and the gap between theoretical FLOP savings and real-world latency.

---

## 4. Proposed Method: SPS-ZCA
- **Problem Formulation:** Shared frozen backbone $f_\theta$, LoRA experts $\{E_1, \dots, E_K\}$, heterogeneous input batch $X = \{x_1, \dots, x_B\}$.
- **The Routing Paradox & Early Embedding Space Representation:**
  - Prove the temporal circular dependency of penultimate routing.
  - Detail why Layer 0 (Patch Embedding) is mathematically optimal and lightweight.
- **Zero-Shot Centroid Pre-computation:**
  - Extracts early embedding $z^{\text{embed}}_s = \text{Pool}(f_{\theta, \text{embed}}(x_s)) \in \mathbb{R}^D$ from 64-sample calibration splits.
  - Computes robust task centroid $\mu^{\text{embed}}_k = \frac{1}{|\mathcal{C}_k|} \sum_{s \in \mathcal{C}_k} z^{\text{embed}}_s$.
- **Zero-Shot Centroid Alignment (ZCA) Routing:**
  - Normalization and cosine similarity coordinates: $u_{k, b} = \frac{z^{\text{embed}}_b \cdot \mu^{\text{embed}}_k}{\|z^{\text{embed}}_b\|_2 \|\mu^{\text{embed}}_k\|_2}$.
  - Temperature-scaled Softmax: $\alpha_{k, b} = \frac{\exp(u_{k, b} / \tau)}{\sum_j \exp(u_{j, b} / \tau)}$ with $\tau = 0.001$.
- **Single-Pass Activation-Space Dynamic Blending (SPS):**
  - Mathematical formulation of layer-wise dynamic scaling: $h_b^{(l)} = h_b^{(l-1)} W_{base}^{(l)} + \sum_k \alpha_{k, b} (h_b^{(l-1)} A_k^{(l)} B_k^{(l)})$.
  - Show how this avoids batch splitting and runs in a single forward pass.
- **Practical Calibration Enhancements:**
  - *Unit-Norm Calibration (UNC):* Handles cross-expert norm scaling differences.
  - *Intra-Task Dispersion Calibration (IDC):* Normalizes differences in task-manifold spatial spreads.
  - *OOD Rejection via Coordinate GMM:* Fits a diagonal GMM on calibration coordinates $\mathbf{u}$ to reject out-of-distribution queries with high precision.

---

## 5. Experimental Evaluation
- **The Isolating Coordinate Sandbox:** Detailed setup ($L=14$ layers, $D=192$ dims, $K=4$ tasks: MNIST, F-MNIST, CIFAR-10, SVHN).
- **Hardware-Aware Memory Bandwidth & Cost Model:**
  - Formulate physical DRAM memory traffic (ARM Cortex-A72 LPDDR4 bandwidth) and cache line sizes (1MB L2).
  - Justify how SPS-ZCA achieves a **3.67x memory-bandwidth saving** by loading base model weights only once.
- **Main Performance Comparison (Homogeneous Batching, B=256):**
  - Discuss the main table results. Highlight SPS-ZCA achieving Joint Mean 79.80%, recovering 100.0% of Expert Ceiling and outperforming prior SOTA (PFSR+MBH) by +3.66% absolute.
- **Deployment Stream Audit & Latency Scaling:**
  - Discuss the Homogeneous vs. Heterogeneous batch streaming results.
  - Show how SPS-ZCA maintains a flat 199.0 ms latency for $B=256$ in both homogeneous and heterogeneous streams, while PFSR+MBH jumps to 776.4 ms, achieving a **3.90x speedup** on mixed streams.
- **In-depth Ablations:**
  - *Ablation A: Sensitivity to Batch Heterogeneity* (discuss `batch_size_heterogeneity.png`, showing immunity to heterogeneity collapse).
  - *Ablation B: Latency & Throughput Scaling* (discuss `latency_throughput_scaling.png`, highlighting throughput $>1000$ samples/sec).
  - *Ablation C: Unit-Norm Calibration (UNC)* (with/without UNC accuracy comparison: drops to 79.22% without UNC under scale imbalance, fully restored to 79.80% with UNC).
  - *Ablation D: Intra-Task Dispersion Calibration (IDC)* (balanced routing restored from 95.40% misrouting to 47.00%).
  - *Ablation E: OOD Rejection Performance* (discuss `rejection_roc_curve.png`, showing 95.2% true rejection and 4.3% false rejection using diagonal GMM).
  - *Ablation F: Routing Temperature Sensitivity* (discuss `temperature_sensitivity.png`).
- **Systems-ML Co-design & On-Device Deployment Guidelines:**
  - Actionable recommendations for edge CPU/MCU vs. cloud server deployment of dynamic model merging.

---

## 6. Conclusion
- Summary of SPS-ZCA as an extremely fast, zero-parameter, robust alternative for edge-serving.
- Discussion on practical impact and future extensions (LLMs, hardware acceleration).
