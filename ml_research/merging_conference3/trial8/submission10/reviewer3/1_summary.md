# Evaluation Step 1: Summary of the Paper

## Main Topic
The paper addresses the challenge of serving multiple specialized Parameter-Efficient Fine-Tuning (PEFT) adapters (e.g., LoRA) simultaneously on resource-constrained edge devices under heterogeneous, noisy, and unpredictable real-world request streams. 

## Proposed Approach
The authors challenge the reductionist assumption that task-specific experts exist in isolated vacuum states. They propose a bio-inspired paradigm shift: treating model ensembling as a self-organizing, dynamic ecosystem in activation space. 
The proposed framework is **Evolutionary Symbiotic Merging via Lotka-Volterra Cooperation (ESM-LVC)**. It features three training-free and parameter-free components:
1. **Lotka-Volterra Activation Dynamics (LVAD):** Models the ensembling coefficients (activation levels) of specialized adapters as interacting biological species populations whose densities evolve over a localized virtual timescale governed by Lotka-Volterra competition-cooperation differential equations inside the forward pass.
2. **Symbiotic Interaction Tensor (SIT):** A pre-computed semantic interaction matrix ($\Gamma$) derived from intermediate centroid similarity alignments that governs task mutualism (cooperative co-activation of similar/complementary tasks) and competitive exclusion (mutual suppression of conflicting/dissimilar tasks) using an automatic off-diagonal threshold heuristic or a Localized Pairwise Threshold Heuristic.
3. **Discrete Euler Symbiosis Solver (DESS):** An ultra-lightweight, projected discrete solver to integrate the differential equations on-the-fly with sub-millisecond latency. It features an Adaptive Step-Size Heuristic to guarantee trajectory boundedness and Decoupled Activation-Inference Sharpening (DAIS) / Exponential Information-Theoretic Adaptive Sharpening (E-ITAS) / Dirichlet-Multinomial Bayesian Self-Calibration (DM-BSC) to separate continuous routing dynamics from final, sharp dilution-free classification decisions.

The entire setup is deployed in a **Paradox-Free Execution Layout** consisting of shared feature extraction layers, an intermediate routing layer, and a specialized expert region where adapter activations are dynamically blended sample-wise using a **Dynamic Scale Alignment (DSA)** operator to maintain norm stability.

## Key Findings & Quantitative Claims
* **Sandbox Simulation (ICS):** In a 192-dimensional synthetic vector space emulating a 14-layer Vision Transformer serving 4 simulated tasks (MNIST, Fashion-MNIST, CIFAR-10, SVHN):
  - ESM-LVC achieves **75.12% Joint Mean accuracy** under standard settings, outperforming recent state-of-the-art training-free methods like SABLE (74.13%) and SPS-ZCA (74.31%).
  - ESM-LVC significantly outperforms the trained **Linear Router (Act)** (64.03%) by **+11.09% absolute** without requiring any parameter updates or backpropagation.
  - Under extreme domain noise (Scale 2.5), ESM-LVC maintains **65.37% accuracy**, outperforming SPS-ZCA by **+2.63% absolute** due to the dynamic, self-regulating noise-filtering properties of Lotka-Volterra competitive exclusion.
  - ESM-LVC exhibits absolute immunity to serving stream heterogeneity and batch size variations (maintaining flatline **75.12%** performance from batch sizes $B=1$ to $B=512$), whereas weight-space ensembling suffer from severe **batch heterogeneity collapse** (decaying to 44.18% at $B=512$).
  - In similarity sweeps ($\rho = 0.40$), ESM-LVC's mutualistic co-activation achieves **75.80% accuracy** (outperforming SPS-ZCA by **+0.58% absolute**), demonstrating the benefits of organic task cooperation over winner-take-all routing.
  - Under a Destructive Interference Penalty ($iw = 0.3$), ESM-LVC experiences a tiny degradation of only -0.32% (maintaining **74.80%** accuracy), outperforming SPS-ZCA by **+0.50% absolute**, verifying its self-sharpening competitive dynamics.
  - Serve-time routing solver latency benchmark on a single CPU core is strictly below **0.6 milliseconds** across all scales (up to $K=16$ experts and batch size $B=256$).
* **Physical Model Verification:** In offline evaluations using CLS activations extracted from Layer 12 of a pre-trained physical ViT-Tiny model across four real-world datasets:
  - **GMC-BSC** (Gaussian Mixture Centroids) successfully breaks the single-centroid attractor bottleneck, boosting routing accuracy to **93.50%** (clean) and **89.75%** (severe noise scale $\sigma=2.0$), outperforming single-centroid zero-shot baselines by **+3.25%** and the Fully-Optimized Linear Router by **+4.75%** absolute under noise.
  - **DM-BSC** (Dirichlet-Multinomial Bayesian Self-Calibration) achieves **28.25%** downstream classification accuracy, outperforming SABLE ($27.25\%$) and matching the Fully-Optimized Linear Router within $0.50\%$ absolute without requiring any trainable parameters, while maintaining exceptionally low routing entropy ($0.0343 \to 0.0748$).
  - The Projected Euler DESS solver is empirically confirmed to have a **0.00% fallback rate** across all real-world test batches, validating its absolute numerical stability under realistic, non-orthogonal manifolds.
