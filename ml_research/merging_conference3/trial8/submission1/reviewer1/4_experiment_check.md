# 4. Experiment Check

## Critical Evaluation of the Experimental Setup
The paper evaluates the proposed HyperMerge method within a simulated **14-layer Analytical Coordinate Sandbox** with 192 dimensions and $K=4$ simulated tasks.
- **Strengths of the Setup:** It simulates both homogeneous and heterogeneous input streams, allowing for a clean study of stream ordering effects. It also includes an "Overlapping Subspace Sandbox Regime" to specifically simulate coordinate crowding near the origin.
- **Major Weakness (Toy Simulation):** The evaluation is entirely synthetic. Features and task spaces are partitioned directly into coordinate subspaces. This is highly idealized and does not reflect the complexity of actual deep learning representations in massive pre-trained models (like LLaMA or ViT) operating on high-dimensional natural datasets. The absence of real-world benchmarks (such as NLP task ensembling or visual domain adaptation) severely limits the generalizability of the findings.

---

## Evaluation of Results and Claims

### 1. No Empirical Superiority over Flat Euclidean Baselines
The authors claim that HyperMerge "shatters geometric limits" and resolves representation crowding. However, the quantitative results in Table 1 and Table 3 directly contradict any claims of a practical advantage:
- **Standard Setup (Table 1):** SABLE (Early Routing), a flat-space Euclidean baseline, achieves **84.03% ± 5.15%**, which is **higher** than HyperMerge (83.40% ± 5.15%). SPS-ZCA (Euclidean SOTA) is virtually identical at 83.05% ± 4.95%.
- **Overlapping/Crowded Setup (Table 3):** Under the highly crowded coordinate subspace regime, SABLE (Early Routing) achieves **77.98% ± 2.12%**, which again **outperforms** HyperMerge (76.62% ± 3.96% at $c=0.1$, and 76.50% ± 3.36% when tuned). SPS-ZCA also outperforms HyperMerge here with 77.32% ± 1.98%.
- **Conclusion:** Across all evaluated scenarios, flat-space Euclidean ensembling (SABLE) is superior or on par. HyperMerge's non-linear hyperbolic transformations and complex Möbius algebra actually lead to a minor performance deficit (likely due to projection distortion). The claims of a "paradigm-shifting approach" are not empirically justified.

### 2. Discrepancies and Inconsistencies in Quantitative Reporting (Crucial Flaw)
There is an unexplained discrepancy between the main results (Table 1) and the parametric ablation study (Table 2) regarding the performance of HyperMerge at $c=0.1$:
- In **Table 1**, HyperMerge is evaluated with standard hyperparameters ($c=0.1$, $\tau=0.05$) and achieves a joint mean accuracy of **83.40% ± 5.15%**.
- In **Table 2** (Ablation on Curvature), for $c=0.1$, the paper reports a joint mean accuracy of **89.30%**.
- This is a significant difference of **5.90%** for the exact same curvature value ($c=0.1$). This discrepancy is not addressed or explained in the text, suggesting that either the ablation was run under a different, unaligned experimental setup (e.g., different calibration splits, routing temperatures, or seeds), or the numbers are inconsistent.
- Furthermore, Table 2 shows that at $c=0.5$ (Optimal), HyperMerge achieves **91.00%** accuracy, which would have substantially outperformed all Euclidean baselines (e.g., SABLE at 84.03%). If this is true, it is highly non-intuitive why the authors chose to report a sub-optimal configuration ($c=0.1$, 83.40%) in the primary comparison of Table 1, rather than using the tuned $c=0.5$ configuration. This raises strong concerns about the reliability, tuning process, or consistency of the ablation study.

### 3. "Heterogeneity Collapse" Immunity is Not Unique
The authors repeatedly present HyperMerge's absolute immunity to heterogeneous stream ordering (0.00% collapse) as a breakthrough. However, Table 1 and Table 3 show that:
- **SABLE (Early Routing)**, **SABLE (Late Adaptation)**, and **SPS-ZCA** all exhibit exactly **0.00% heterogeneity collapse** as well.
- This absolute robustness is not a unique geometric property of HyperMerge. It is a fundamental characteristic of **all sample-wise activation-space ensembling methods** that route and fuse representations dynamically on a single forward pass, completely bypassing batch-level dependencies. The paper overstates this as a specific benefit of its hyperbolic formulation.

### 4. Hyperbolic OOD Rejection (HOR) vs. Euclidean Distance
The authors claim that HOR provides a reliable "hyperbolic geodesic boundary" for out-of-distribution rejection. However, as demonstrated in our theoretical soundness check, because the input activations reside extremely close to the origin ($\|\mathbf{z}\|_2 \ll 1/\sqrt{c}$), the Poincaré geodesic distance is virtually indistinguishable from a scaled Euclidean distance:
$$d_{\mathbb{D}}^c(\mathbf{x}, \mathbf{y}) \approx 2 \|\mathbf{x} - \mathbf{y}\|_2 + \mathcal{O}(c)$$
Thus, HOR is mathematically equivalent to a standard Euclidean distance threshold in this regime. The claim that negative curvature provides a unique security barrier is conceptually exaggerated.
