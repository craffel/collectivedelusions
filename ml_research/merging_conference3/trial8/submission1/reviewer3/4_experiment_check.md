# 4. Experimental Evaluation and Consistency Check

A critical and detailed evaluation of the experimental setup, baseline comparisons, and the consistency of the empirical results presented in the paper.

## Critique of the Experimental Setup
The entire evaluation is performed in a custom, synthetic environment called the **14-layer Analytical Coordinate Sandbox** with 192 dimensions. 
* **Lack of Real-World Validation:** There are no actual deep learning models (e.g., LLaMA, Mistral, ViT, ResNet) and no actual datasets (e.g., GLUE, ImageNet, actual MNIST/CIFAR images) used in this work. The "experts" and "datasets" are merely simulated by partitioning the coordinate subspaces.
* **Toy Nature of the Task:** Simulating tasks by partitioning coordinates in a synthetic sandbox heavily simplifies the complex representation shifts, noise, and high-dimensional manifolds found in real-world networks. This raises severe questions about the generalizability of the findings to actual modular deep learning systems.

## Major Numerical Inconsistencies and Contradictions
A close analysis of the experimental results reveals major discrepancies and contradictions between the main results (Table 1) and the ablation studies (Section 4.5 & Table 3):

1. **Conflicting Baseline and Method Performance:**
   In **Table 1** (Main Results), the authors report:
   * **SABLE (Early Routing):** 84.03% $\pm$ 5.15%
   * **SPS-ZCA (Euclidean SOTA):** 83.05% $\pm$ 4.95%
   * **HyperMerge (Ours, default $c=0.1$):** 83.40% $\pm$ 5.15%

   However, in **Section 4.5 (Parametric Sensitivity / Ablation Studies)**, the authors state:
   > "at extremely low curvature ($c = 0.001$), HyperMerge's performance is around **87.65%**. As curvature increases, accuracy rises, peaking at $c = 0.5$ (**91.00%**), where it outpaces all Euclidean baselines (including SABLE at **89.65%** and SPS-ZCA at **88.55%**)."
   
   This introduces a series of major scientific contradictions:
   * **Why do the baselines change?** In Table 1, SABLE gets 84.03% and SPS-ZCA gets 83.05%. But in the ablation section, SABLE is cited as getting 89.65% and SPS-ZCA is cited as getting 88.55%.
   * **Why does "near-Euclidean" HyperMerge change?** At $c=0.001$ (effectively flat space), HyperMerge gets 87.65%. Why is this "near-flat" version scoring higher than SABLE (84.03%) or SPS-ZCA (83.05%) in Table 1?
   * **Selective Reporting / Sub-optimal Default:** If setting $c=0.5$ yields a joint mean accuracy of **91.00%** (outperforming all baselines), why did the authors use a sub-optimal curvature of $c=0.1$ for their main experiments in Table 1, where HyperMerge (83.40%) actually performs **worse** than the Euclidean baseline SABLE (84.03%)? 

   These inconsistencies strongly suggest that the ablation study was run on a completely different configuration, coordinate split, or seed, without any explanation. Comparing these numbers in the same paper represents a major lack of scientific rigor.

2. **SABLE (Late Adaptation) Strawman:**
   In Table 1, `SABLE (Late Adaptation)` is shown to collapse completely to **46.37% $\pm$ 5.95%**. In contrast, `SABLE (Early Routing)` gets **84.03% $\pm$ 5.15%**. The paper provides no explanation for why allowing later layers to adapt routing weights would cause such a catastrophic collapse. This makes the "Late Adaptation" baseline look like a poorly tuned strawman designed to make other methods look better.

## Do the Results Support the Claims?
The authors make several core claims that are directly contradicted or unsupported by their own empirical results:

* **Claim:** HyperMerge resolves representation crowding and cross-talk to achieve superior ensembling under overlapping manifolds.
* **Reality (Table 2):** In the highly crowded "Overlapping Subspace Sandbox Regime," the Euclidean baselines still **outperform** HyperMerge:
  * **SABLE (Early Routing):** **77.98% $\pm$ 2.12%**
  * **SPS-ZCA (Euclidean SOTA):** **77.32% $\pm$ 1.98%**
  * **HyperMerge (Ours, default $c=0.1$):** **76.62% $\pm$ 3.96%**
  * **HyperMerge (Ours, Tuned $c=0.2$):** **76.50% $\pm$ 3.36%**
  
  Even when tuned, HyperMerge fails to outperform the flat Euclidean baselines under the very regime it was designed to solve.

* **Claim:** HyperMerge provides absolute immunity to heterogeneous streams.
* **Reality:** While HyperMerge indeed achieves 0.00% heterogeneity collapse, **so do the Euclidean baselines (SABLE and SPS-ZCA)** in Table 1. Thus, stream immunity is not a unique advantage of hyperbolic geometry; it is a shared property of activation-space ensembling.

## Evaluation of the Baselines
The selection of baselines is reasonably comprehensive (static merging, subspace routing, activation blending, and centroid routing). However, as noted:
* `SPS-ZCA` is evaluated without any citation or bibliographic entry.
* The catastrophic performance of `SABLE (Late Adaptation)` is highly suspicious and lacks diagnostic explanation.
* Referencing a baseline from "Trial 7" in Section 4.2 is unscientific and represents a severe leak of the development process.
