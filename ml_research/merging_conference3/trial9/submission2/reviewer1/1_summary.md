# 1. Summary

## Main Topic and Goal
The paper addresses the challenge of deploying multi-task, parameter-efficient fine-tuning (PEFT) models—specifically Low-Rank Adaptation (LoRA) experts—on resource-constrained and power-volatile edge devices. Standard dynamic ensembling approaches (e.g., SABLE, SPS-ZCA) dynamically blend the outputs of multiple expert adapters sample-by-sample, but assume infinite and static hardware resources. Running multiple parallel expert pathways in a single forward pass leads to severe DRAM-to-SRAM weight transfer contention (memory-bus choking) and high latency. To bridge the gap between deep ensembling SOTA and physical deployment constraints, the paper proposes **Resource-Budgeted Top-$M$ Expert Serving (RB-TopM)**, a training-free, hardware-aware dynamic ensembling framework governed by a real-time system resource coefficient $C_{\text{budget}} \in [0, 1]$.

## Proposed Approach
RB-TopM consists of several key architectural components:
1. **Dynamic Compute Budget Control:** A closed-loop control system that dynamically computes:
   - **Dynamic Top-$M$ Cap ($M(C_{\text{budget}})$):** A step function $M(C_{\text{budget}}) = \max(1, \lfloor M_{\max} \cdot C_{\text{budget}} \rfloor)$ that limits the maximum active parallel expert pathways, transitioning from soft-blending to hard single-expert routing as the budget decreases.
   - **Adaptive Gating Threshold ($\theta(C_{\text{budget}})$):** A linear function $\theta(C_{\text{budget}}) = \theta_{\min} + (1 - C_{\text{budget}}) \cdot (\theta_{\max} - \theta_{\min})$ that prunes marginal expert pathways whose ensembling coefficients fall below $\theta$.
2. **Zero-Shot Centroid Alignment (ZCA) with Scale Calibration (IDC):** Computes task similarity coordinates in an early representation space (e.g., Layer 3) via cosine similarity to pre-computed centroids (using a 64-sample calibration split). It applies Intra-Task Dispersion Calibration (IDC) to resolve representation drift and variance imbalance.
3. **Coordinate GMM Out-of-Distribution (OOD) Safety Shield:** Fits a 2-component diagonal Gaussian Mixture Model (GMM) over the similarity coordinates during calibration. It rejects OOD queries (setting routing weights to zero and executing purely the base model) while maintaining a strictly bounded 5% false-positive rate.
4. **Hierarchical Macro-Domain GMM Routing (HMD-GMM):** Mitigates the coordinate-overlap bottleneck of GMMs when scaling to large expert populations ($K \ge 24$) by partitioning tasks into semantic macro-domains.

## Key Findings
- In a **14-layer Analytical Coordinate Sandbox (ICS)** simulating MNIST, Fashion-MNIST, CIFAR-10, and SVHN, RB-TopM matches peak ensembling accuracy (75.37%) while saving **72.4%** of expert computational FLOPs at $C_{\text{budget}} = 1.0$.
- Under extreme resource constraints ($C_{\text{budget}} = 0.0$), it collapses active experts to **0.95**, yielding a **76.2%** FLOP saving while preserving **75.12%** joint accuracy, vastly outperforming static parameter-space merging (65.68%).
- **Non-Monotonic Accuracy trend (The Activation Dilution Paradox):** Decreasing the budget can improve accuracy: $C_{\text{budget}} = 0.4$ achieves the highest overall accuracy of **75.85%** (compared to 75.37% at $C_{\text{budget}} = 1.0$). This is theoretically explained by showing that dynamic pruning acts as an activation-space regularizer that mitigates "activation dilution" caused by noisy secondary expert pathways.
- The GMM safety shield successfully flags and rejects **38.04%** of high-noise OOD queries with a bounded 5% false-positive rate.
- **MobileNetV3-Large TVM Compiler Simulation:** Confirms that reducing DRAM expert weight transfers by 78.5% translates to a **17.5% overall system serving latency reduction** on physical-like edge compiler models, bypassing memory-bound bottlenecks.

## Explicitly Claimed Contributions (with Evidence)
1. **First hardware-aware, resource-budgeted control framework** for dynamic model ensembling, bridging edge hardware monitors to activation blending. (Evidence: Section 3.3, closed-loop mapping equations).
2. **Smooth, monotonic accuracy-latency trade-off frontier** adjustable in microseconds on-the-fly without model retraining or offline calibration. (Evidence: Figure 1, Tables 2 & 3).
3. **Significant FLOP and DRAM savings** (up to 78.4% expert computational savings, 17.5% simulated serving latency reduction) while preserving or improving accuracies. (Evidence: Tables 2 & 3).
4. **Integration of a low-power early-activation Coordinate GMM safety shield** against physical noise and unnecessary adapter execution. (Evidence: Equation 8, Section 4.4, OOD rejection rate of 38.04%).
