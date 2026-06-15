# 4. Experiment Check

## Experimental Setup and Scope
The paper conducts a dual-stage empirical validation:
1. **14-Layer Analytical Coordinate Sandbox (ICS) Simulation:** A mathematically controlled, closed-loop simulation in a 192-dimensional representation space. It models a 14-layer backbone model with 4 specialized LoRA experts (MNIST, Fashion-MNIST, CIFAR-10, SVHN). The experiments are averaged over 10 independent random seeds, providing tight confidence intervals (reported in Table 2).
2. **TVM-Compiled NPU Compiler-Simulation Pilot:** Evaluates a real deep vision backbone (**MobileNetV3-Large**) with fine-tuned LoRA experts on the **DomainNet** dataset (Real, Clipart, Painting, Sketch). The compilation is modeled on the TVM runtime engine v0.15 with static memory planning on a simulated ARM Cortex-M7 core running at 400 MHz.

## Evaluation of Claims
- **Claim: RB-TopM matches peak ensembling SOTA while saving substantial expert FLOPs.** Supported. In the ICS sandbox (Table 2, Part B), at $C_{\text{budget}} = 1.0$, RB-TopM achieves 75.37% joint accuracy (exactly matching SABLE SOTA's 75.77% and SPS-ZCA's 75.44% within statistical error) while executing only 1.11 active experts (saving **72.4%** of expert FLOPs).
- **Claim: Dynamic pruning acts as an activation-space regularizer that improves accuracy.** Supported. At $C_{\text{budget}} = 0.4$, joint accuracy peaks at **75.85%** (higher than 75.37% at $C_{\text{budget}} = 1.0$), while executing only 0.86 active experts (saving **78.4%** of expert FLOPs). This empirically supports the theoretical activation dilution proof in Appendix A.
- **Claim: The framework provides substantial latency and DRAM fetch savings.** Supported. The TVM compiler simulation (Table 3) shows that RB-TopM reduces expert execution latency from 1.450 ms (SABLE) to 0.345 ms (at $C_{\text{budget}} = 0.4$). When factoring in the pre-trained backbone execution (4.85 ms), the overall simulated serving latency drops from 6.30 ms to 5.195 ms, representing a **17.5% full-system serving speedup**.
- **Claim: The GMM safety shield successfully rejects OOD queries.** Supported. The Coordinate GMM shield successfully flags and rejects **38.04%** of high-noise OOD queries with a strictly bounded 5% false-positive rate.

## Appropriateness of Baselines
The baselines are comprehensive, modern, and highly appropriate:
1. **Expert Oracle:** Represents the idealized perfect-routing performance ceiling (1.00 active expert).
2. **Uniform Merging / TIES-Merging / DARE:** Representative static parameter-space merging techniques.
3. **SABLE SOTA:** The state-of-the-art dynamic activation-space ensembling method (uncapped, executing up to $K=4$ parallel experts).
4. **SPS-ZCA:** Single-pass routing with raw, un-calibrated cosine similarity and cold temperature ($\tau=0.001$).
5. **Q-SPS:** Quantized activation blending (INT8 backbone, INT4 experts) using a static pruning threshold.

The inclusion of TIES-Merging and DARE is particularly valuable, demonstrating that static weight merging suffers from severe "heterogeneity collapse" (accuracies of 66.85% and 67.12% respectively) when merging highly diverse visual domains, whereas RB-TopM's dynamic activation-routing outperfroms them by over **8%**.

## SVHN Accuracy Ceiling Justification
In the sandbox, the SVHN Expert Oracle has an exceptionally low accuracy ceiling of **21.68%** due to intentionally modeled high environmental noise and classification bias. The authors explicitly clarify that this is a deliberate design choice simulating extreme sensor degradation (e.g., night-time street-view cameras). Importantly, they demonstrate that this localized noise does not bleed into and corrupt other clean domains (like MNIST and F-MNIST, which maintain near-perfect 100% accuracies), proving the spatial isolation of their centroid routing. To ensure generalizability, they also evaluate a high-performance SVHN expert ceiling (92.40%) in Section 4.3, showing that the ensembling-regularization trade-off persists. This level of self-awareness and detailed reporting is highly commendable.
