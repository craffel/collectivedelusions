# 2. Novelty Check

## Key Novel Aspects
The paper introduces several distinct novel concepts:
1. **Hardware-Governed Dynamic Expert Blending:** The concept of mapping a physical hardware resource budget ($C_{\text{budget}} \in [0, 1]$) directly to the mathematical execution path of an activation-space MoE ensembling framework at runtime.
2. **Theoretical Formulation of Activation Dilution:** Formalizing the empirical degradation in ensembling accuracy under noisy conditions as a "dilution" penalty in the latent representation covariance space (Appendix A), and proving that dynamic gating acts as a hard-thresholding covariance regularizer.
3. **Hierarchical Macro-Domain GMM Routing (HMD-GMM):** A hierarchical routing architecture designed to scale OOD similarity-coordinate GMMs to massive expert populations ($K \ge 24$), addressing coordinate overlap and EM singular-covariance failures on edge hardware.
4. **Integration of Early-Stage GMM Safety Shield with PEFT Serving:** Bypassing downstream specialized adapter pathways completely for noisy, unaligned queries to save memory-bus bandwidth and preserve prediction stability.

## Delta from Prior Work
- **Parameter-Space Model Merging (TIES, DARE, Task Arithmetic):** These methods are fundamentally static and collapse all specialized experts into a single static set of weights offline. While they add zero serving latency, they suffer from "heterogeneity collapse" when merging highly contradictory domains (such as digits, clipart, and sketches simultaneously). RB-TopM preserves task-specialized weights and routes activations sample-by-sample, outperforming static merging by up to 8.7% in accuracy while matching its latency at low budgets.
- **Dynamic Activation-Space Blending (SABLE, SPS-ZCA):** These SOTA methods blend experts dynamically but assume infinite serving resources, executing up to $K$ parallel expert pathways for every single query. This creates unacceptable memory-bus contention (memory-bus choking) on edge hardware. RB-TopM introduces real-time resource-guided pruning to eliminate this scale bottleneck.
- **Quantized Activation Blending (Q-SPS):** Q-SPS applies static, hardcoded thresholds for expert pruning. It cannot adapt to real-time OS-level resource events (e.g., thermal throttling, battery state) in microsecond scales.

## Characterization of Novelty
The novelty of this work is **significant**. Rather than simply proposing another empirical heuristic for model ensembling, the paper establishes a rigorous bridge between hardware systems (dynamic resource monitors, memory-bus bandwidth, and LPDDR5 queue occupancies) and deep learning theory (intrinsic manifold dimensions, representation covariance structures, and hard-thresholding regularization). The theoretical derivation of the "activation dilution" phenomenon is exceptionally elegant and successfully explains why pruning marginal experts can *improve* overall serving accuracy under noisy streams—resolving the Pareto-dominance paradox of low-budget serving.
