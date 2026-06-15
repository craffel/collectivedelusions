# Evaluation Component 4: Experimental Check

## Evaluation of Experimental Setup and Datasets
The experimental evaluation is divided into three parts:
1. **Task-Conflict Model-Merging Sandbox:** A synthetic representation sandbox representing $K=4$ classification tasks (MNIST, FashionMNIST, CIFAR-10, and SVHN) mapping to a 192-dimensional space. The shared semantic subspace contains class prototypes that are permuted across tasks to create direct weight-space conflicts, and the task-specific style subspace contains domain-specific features.
2. **Physical Sequential Weight-Space Merging:** An empirical setup on 3-layer MLP experts where representations propagate sequentially through physically merged layer weights.
3. **Vision Transformer Pilot Demonstration (Appendix 11):** A PyTorch-level pilot profiling wall-clock CPU latency of BWS-Router on a physical ViT (`vit_tiny_patch16_224`) with simulated task vectors.

### Critiques on Experimental Setup:
* **Reliance on Synthetic Sandbox for Main Results:** The vast majority of the quantitative analyses (including the 1,280 experiment grid sweep, block sweeps, and gating activation sweeps) are conducted within the synthetic representation sandbox. While the sandbox is an elegant, high-throughput proxy to study task conflicts and optimization ruggedness, it is still a *simulated* environment. The feature representations are synthetic, and the classifiers are simple single-layer linear heads.
* **Low Expert Ceiling for SVHN:** The classification accuracy ceiling for SVHN is artificially calibrated to be extremely low ($30.16 \pm 0.89\%$) to simulate a noisy domain. While this provides a stress test, it severely drags down the overall Joint Mean metrics (capping them at around 80%), which makes the absolute joint accuracy numbers appear low and hard to compare with standard multi-task benchmarks.
* **Limitations of the ViT Pilot:** The Vision Transformer pilot in Appendix 11 is highly valuable for demonstrating feasibility and profiling wall-clock latency. However, it does not evaluate *actual classification accuracy* on real downstream multi-task datasets using real fine-tuned expert checkpoints (such as CLIP-ViT fine-tuned on real vision datasets). The task-vectors in the pilot are simulated via parameter perturbations.

## Evaluation of Baselines
The authors compare BWS-Router against an exceptionally comprehensive and appropriate set of baselines:
* **Static Baseline:** Static Uniform merging (direct weight averaging).
* **Global Baselines:** Global Linear (unregularized and regularized).
* **Wave-based Gating:** QWS-Merge.
* **Layer-wise (Unshared) Baselines:** L3-Router with Linear, Tanh, and Softmax activations (both unregularized and regularized).

This represents a very thorough comparison, covering static, global, and layer-wise unshared alternatives, ensuring that the empirical gains of BWS-Router are fully contextualized.

## Do the Results Support the Claims?
In general, yes, the empirical results strongly support the core claims of the paper, though there are a few nuances:

### 1. Verification of Block-wise Sharing and Compression
The sensitivity sweep in Table 2 strongly supports the claim that layer-wise routing specialization is redundant. Sharing weights globally ($M=12$) yields near-optimal sandbox performance ($79.60 \pm 1.15\%$) while reducing parameters by 91.7% compared to the unshared baseline ($M=1$, $79.30 \pm 1.29\%$). This empirical finding is extremely robust and justifies the parameter compression claim.

### 2. Verification of Sequential Stabilization
Table 5 (Physical Sequential Weight-Space Merging on 3-layer MLPs) strongly supports the claim that block-wise weight sharing stabilizes deep sequential propagation. Under the highly challenging task-heterogeneous mixed-batch stream, BWS $M=3$ (ours) achieves **43.20 $\pm$ 22.49%** Joint Mean accuracy, outperforming the unshared $M=1$ baseline (**32.27 $\pm$ 21.28%**) by a substantial $+10.93\%$ absolute.

### 3. Nuance: Over-claiming on QWS-Merge Instability in Sandbox Results
The authors claim that QWS-Merge is "extremely vulnerable to seed variations" and exhibits "frequent training collapse across random seeds" (Section 4.2). However, looking at the quantitative results in Table 1:
* QWS-Merge achieves a Joint Mean of **78.07 $\pm$ 1.06%**.
* BWS M3 Sigmoid Reg achieves **79.56 $\pm$ 1.13%**.
* L3 Linear Unreg achieves **79.14 $\pm$ 0.77%**.

The standard deviation of QWS-Merge across the 5 independent seeds is actually *lower* (1.06%) than that of BWS $M=3$ (1.13%), and its mean performance is within $1.5\%$ of the BWS-Router. The empirical results in Table 1 do *not* reflect "frequent training collapse across random seeds" under the evaluated optimal configurations. The authors should clarify if this collapse only occurs in unregularized or un-tuned hyperparameter regimes, as the reported numbers paint QWS-Merge as a highly competitive and relatively stable baseline.

### 4. High Performance Variance in Physical Sequential Merging
The physical sequential merging experiments (Table 5) exhibit very large standard deviations (e.g., $43.20 \pm 22.49\%$ for Heterogeneous BWS $M=3$, and $32.27 \pm 21.28\%$ for unshared $M=1$). This extremely high variance indicates that sequential deep model merging remains highly sensitive to random seeds and initialization. While the authors propose "sequential smoothing regularization" (Appendix 9) as a powerful remedy (reducing standard deviation to $13.41\%$), the inherent instability of physical sequential weight blending under random seeds remains a major open challenge and limitation of the paradigm that should be highlighted.
