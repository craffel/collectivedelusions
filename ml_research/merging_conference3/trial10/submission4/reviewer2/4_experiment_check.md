# 4. Experiment Check

## Critical Evaluation of the Experimental Setup and Datasets
The authors evaluate QA-Merge inside the **Interactive Coordinate Sandbox (ICS)**. 
- **The Strengths of ICS:** From a theoretical perspective, the ICS is an appropriate and valuable tool. By isolating the ensembling dynamics in a controlled latent coordinate-space simulation, the authors are able to abstract away unrelated architectural noise (such as attention-head sparsity, feedforward non-linearities, or random initialization noise) that often complicates and obscures the direct study of ensembling algorithms. This allows them to parameterize and sweep representation-space properties precisely, such as the entanglement parameter $\rho \in [0.0, 0.5]$ and the calibration sample size $N_{\text{cal}} \in [32, 4000]$.
- **The Limitations of ICS:** However, a purely simulated sandbox is a clear experimental limitation for a paper claiming real-world edge serving viability. The ICS abstracts away the actual token-level and layer-level complexities of modern deep architectures (such as Transformer blocks, self-attention mechanisms, and dynamic KV caches). Although the authors address this by providing a PyTorch toy script (`toy_qamerge_lora.py`) demonstrating a dynamic LoRA-mixture layer and a five-step *Real-World Model Porting Protocol*, the paper lacks empirical evaluations on real-world large-scale benchmarks (e.g., merging LLaMA, Mistral, or Vision Transformers on standard downstream datasets like GLUE, GSM8K, or MMLU).

The synthetic datasets represent four classic visual task signatures: MNIST, Fashion-MNIST, CIFAR-10, and SVHN. The inclusion of SVHN as a weak task expert (with a low 22.80% accuracy) is a solid design choice that simulates a "representational distractor" to evaluate routing robustness under domain shifts.

## Evaluation of Baselines
The paper evaluates a comprehensive set of baseline configurations across both full-precision (Float32) and quantized limits:
1. **Expert Oracle (Float32):** The theoretical performance upper-bound (100% routing accuracy).
2. **Uniform Merging (Float32 & Quantized):** Static ensembling baseline (equal blending weights).
3. **SABLE (Float32 & Quantized-Naive):** Distance-based activation ensembling.
4. **ChemMerge (Float32 & Quantized-Naive):** Stateful chemical-kinetics-smoothed routing.
5. **Momentum-Merge (Float32 & Quantized-Naive):** Stateful EMA-smoothed routing.
6. **Parametric Router (Float32 & Quantized-Naive):** Standard gradient-optimized linear routing.

This covers all relevant coordinate ensembling methods, showing that they all suffer from "Quantization Collapse" under naive low-precision settings and are all successfully rescued by QA-Merge.

## Do the Results Support the Claims?
Yes, the empirical results strongly support the authors' claims of complete performance recovery and systems-level efficiency:
- **Quantization Collapse Confirmed:** SABLE (Quantized-Naive) and ChemMerge (Quantized-Naive) drop directly to the performance of static Uniform Merging (65.80% joint accuracy in Table 1, and 65.80% in Table 2). This confirms that naive quantization indeed erases the benefits of dynamic ensembling.
- **Complete Recovery Achieved:** Under INT8 activation and INT4 weight constraints, QA-Merge successfully recovers full-precision ensembling gains. For example:
  - **SABLE (QA-Merge)** reaches 76.10% (at $\rho=0.2$, $N_{\text{cal}}=64$), virtually matching SABLE (Float32) at 76.20% and beating naive quantized SABLE by 2.40% absolute.
  - **Momentum-Merge (QA-Merge)** at $\rho=0.5$ (large-sample Table 2) achieves 90.50%, matching the Float32 ceiling and outperforming static Uniform Merging by 5.30% absolute.
- **Small-Step Quantization Bottleneck Resolved:** Activation Error Feedback (AEF) successfully prevents tiny updates from being rounded to zero, stabilizing the trajectories as shown in the full-precision recovery.
- **Successful Distractor Isolation:** The routing weight allocated to the weak SVHN expert (22.80% accuracy) remains negligible ($\le 0.02$) for non-SVHN queries, confirming that scale-invariant cosine similarity and STE-gated routing successfully isolate noisy pathways without cross-task interference.

## Key Observations and Diagnostic Sweeps
1. **Sample Complexity Sweeps (Figure 2):** Shows that SABLE (QA-Merge) and ChemMerge (QA-Merge) are highly sample-efficient under data scarcity ($N_{\text{cal}} < 128$), whereas standard parametric routers overfit severely, illustrating the advantage of centroid-guided routing.
2. **Sensitivity Sweep of EF-Smooth Decay Factor $\beta$:**
   - At $\beta=1.0$ (perfect feedback), the system achieves 100% recovery but higher jitter ($0.02348$).
   - At $\beta=0.0$ (no feedback), accuracy drops but jitter is zero as coefficients freeze.
   - Setting $\beta=0.8$ balances both, recovering 99.5% accuracy while reducing trajectory jitter by over 40% ($0.01258$).
   - The authors explain a fascinating relationship: $\beta=1.0$ is critical under data-scarcity ($N_{\text{cal}}=64$) to shape and diffuse rounding errors of weak gating logits, while $\beta=0.0$ is optimal under large-sample regimes ($N_{\text{cal}}=4000$) because highly confident, polarized routing probabilities naturally map directly to discrete integer grid boundaries with virtually zero rounding error.
3. **SmoothQuant $\alpha$ Sweep under Heavy Outlier Conditions (Table 4 & Figure 4):**
   - At $\alpha=0.0$ (no scale migration), dynamic activations absorb all difficulty, leading to coarse quantization grids under heavy outliers (Logit MSE of 0.001359, Gating Match Rate of 95.70%).
   - At the sweet spot $\alpha \in [0.1, 0.3]$, migrating scaling difficulty to static ensembling weights allows a finer activation quantization grid, reaching optimal Logit MSE of 0.000869 ($\alpha=0.2$) and peak Gating Match Rate of 97.80% ($\alpha=0.3$).
   - At complete migration ($\alpha=1.0$), low-precision 4-bit weights collapse under dynamic range inflation, leading to high Logit MSE (0.005096) and poor Gating Match Rate (93.20%).
   - This provides an actionable guideline for practitioners deploying models on edge hardware.
4. **Physical Hardware Benchmarks:** STM32H753XI microarchitectural profiling confirms that the integer ensembling loop runs in exactly **0.18 ms** (including the sorting-free, branchless PI-SPA weight apportionment which takes $<0.23$ $\mu$s, or $<0.13\%$ of the budget), representing a **5.2x latency speedup** and a **42% power reduction** compared to the Float32 FPU loop, confirming real-world deployability.
