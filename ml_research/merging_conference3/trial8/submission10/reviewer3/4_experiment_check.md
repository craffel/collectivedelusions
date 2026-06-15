# Evaluation Step 4: Experimental Critique & Support for Claims

## Experimental Setup & Benchmarks
The authors perform evaluations across two primary settings:
1. **Isolating Coordinate Sandbox (ICS):** A 192-dimensional synthetic vector space emulating a 14-layer Vision Transformer serving 4 simulated vision tasks (MNIST, Fashion-MNIST, CIFAR-10, SVHN) with calibrated noise coefficients.
2. **Physical Model Verification:** An offline evaluation using CLS token activations extracted from Layer 12 of a pre-trained physical ViT-Tiny model across four real-world datasets (MNIST, Fashion-MNIST, CIFAR-10, SVHN).

The benchmarks include Uniform Merging, SABLE, SPS-ZCA, and trained parametric PyTorch routers (Linear Router Few-Shot and Fully-Optimized), which are highly appropriate.

## Do the Results Support the Claims?
Yes, the experimental results strongly support the paper's core claims:
1. **Dynamic Routing Precision:** Under standard serving, ESM-LVC achieves **75.12% Joint Mean accuracy**, outperforming SABLE (74.13%) and SPS-ZCA (74.31%). More importantly, in the same activation-space context, our parameter-free ESM-LVC significantly outperforms the trained **Linear Router (Act)** (64.03%) by **+11.09% absolute** without requiring any backpropagation or parameter updates.
2. **Domain Noise Resilience:** Under severe domain noise (Scale 2.5), ESM-LVC maintains **65.37% accuracy**, outperforming SOTA SPS-ZCA (62.74%) by **+2.63%** absolute due to the self-regulating competitive exclusion of Lotka-Volterra dynamics.
3. **Immunity to Batch Heterogeneity Collapse:** ESM-LVC exhibits flatline **75.12%** accuracy across batch sizes from $B=1$ to $B=512$, whereas the weight-space Linear Router collapses completely to the uniform ensembling baseline (**44.18%**) due to batch-averaged parameter blending.
4. **Task Mutualism & Cooperation:** Under similarity sweeps ($\rho = 0.40$), ESM-LVC's mutualistic co-activation achieves **75.80% accuracy** (surpassing SPS-ZCA's winner-take-all routing by +0.58% absolute), verifying the value of organic task cooperation.
5. **Resilience to Destructive Interference:** Under a severe Destructive Interference Penalty ($iw = 0.3$), ESM-LVC's self-sharpening dynamics maintain **74.80%** accuracy, outperforming SABLE (73.25%) and SPS-ZCA (74.30%).
6. **Low Computational Latency:** Serve-time routing solver latency on a single CPU core is strictly below **0.6 ms** across all expert and batch scales, validating its practicality for edge deployment.
7. **Breaking the Attractor Bottleneck (GMC-BSC):** In physical model verification, GMC-BSC boosts routing accuracy to **93.50%** (clean) and **89.75%** (severe noise $\sigma=2.0$), outperforming single-centroid zero-shot baselines by **+3.25%** and crushing the Fully-Optimized Linear Router (**85.00%**) by **+4.75%** absolute under severe noise.

## Critical Gaps & Weaknesses in the Experiments
1. **Stylized Destructive Interference Penalty:**
   The Destructive Interference Penalty model (Equation 3) in the simulation is a stylized theoretical surrogate ($P_{k, b} = \sum_{j \neq k} (1 - \rho_{k, j}) \alpha_k \alpha_j$). While its bilinear form is physically motivated by first-order perturbative interactions, the authors did not collect physical empirical data to prove that actual multi-adapter ensembling accuracy drops according to this exact mathematical product. In actual deep networks, interference is highly non-linear, layer-dependent, and subject to multi-expert crosstalk.
2. **No Active Physical Adapter Blending:**
   While CLS token activations were extracted and classified offline to verify the routing solver, no physical PEFT/LoRA adapters were actually trained or served end-to-end to measure physical multi-task accuracy under active activation blending. Thus, the downstream impact of Dynamic Scale Alignment (DSA) and actual physical representation-space interference remains unverified on physical hardware.
3. **Low Downstream Classification probe accuracy:**
   The linear classification probes achieve low absolute classification accuracies (ranging from 20.75% to 28.75%). While theoretically explained by the tiny calibration split (64 samples) and frozen, out-of-domain pre-trained CLS token representations of a tiny ViT-Tiny backbone, the low absolute performance limits the real-world utility of the current proof-of-concept.
