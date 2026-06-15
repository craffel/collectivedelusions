# 4. Experiment Check

## Critical Evaluation of the Experimental Setup
The experimental setup of the paper is highly comprehensive, multi-dimensional, and statistically rigorous.
- **Statistical Power**: The authors run all simulation configurations across **30 independent random seeds (42 to 71)**, yielding over 700 fully optimized trajectories. This is far superior to standard TTA papers, which often report single-seed runs or average over a small number of seeds.
- **Multi-axis Sweeps**: The sweeps cover multiple complexity axes (polynomial degrees $d \in \{0, 1, 2, 3\}$), optimizer axes (Adam vs. 1+1 ES), and four distinct benchmarks (MNIST, FashionMNIST, CIFAR-10, SVHN).
- **Baselines**: The paper compares PolyMerge against a highly complete set of baselines, including static Task Arithmetic, unconstrained AdaMerging, early-stopped AdaMerging, spatial averaging, Total Variation (TV) regularization, and $L_2$ regularization.

## Do the Results Support the Claims?
Yes, the empirical results overwhelmingly support the paper's core claims, with minor caveats:

### 1. Support for the Overfitting-Optimizer Paradox (Claim 1)
This claim is strongly supported. Under unconstrained Adam adaptation, the average simulated accuracy falls to $82.60\% \pm 12.16\%$, and SVHN accuracy collapses catastrophically to $63.16\% \pm 6.23\%$ (well below the uniform baseline of $73.24\%$). Figure 1 confirms that this is accompanied by extremely jagged coefficient profiles.

### 2. Support for PolyMerge's Efficacy (Claim 2)
This claim is highly supported by both simulative and physical results:
- In Table 1, PolyMerge ($d=2$, Adam) completely stabilizes TTA, reaching $86.57\% \pm 7.48\%$ average simulated accuracy and reducing the SVHN variance compared to unconstrained Adam.
- Formal paired t-tests establish extremely high statistical significance against both unconstrained AdaMerging ($p = 9.53 \times 10^{-13}$) and static Task Arithmetic ($p = 1.70 \times 10^{-28}$).
- On the PyTorch Residual MLP (Table 3), PolyMerge ($d=2$) reduces spatial roughness by $42\times$ while maintaining stable adaptation.

### 3. Support for Black-Box Optimization and Parameter Efficiency (Claim 3)
This claim is exceptionally well-supported. Under zero-order 1+1 ES, PolyMerge ($d=2$, ES) achieves $84.91\%$, significantly outperforming TV-regularized ES ($84.45\%$) and $L_2$-regularized ES ($84.37\%$) with extremely high statistical significance ($p < 10^{-4}$). This confirms that reducing parameters from $L$ to $d+1$ is highly beneficial for derivative-free search algorithms.

### 4. Support for SplineMerge under Layer Heterogeneity (Claim 4)
This claim is supported, but it also exposes the limitations of global polynomials. On real CLIP weights (Table 4):
- Global PolyMerge ($d=2$) underperforms Task Arithmetic (89.00% vs. 94.00%), demonstrating a severe underfitting bottleneck.
- SplineMerge (Piecewise Constant, 3 blocks) successfully resolves this bottleneck, matching the peak accuracy of unconstrained TTA (96.00%) while maintaining a $1.63\times$ reduction in roughness compared to unconstrained adaptation.

## Areas of Concern or Inconsistencies in the Experiments

### 1. Simulative Performance Metrics
As explicitly noted in Table 1, all primary results are simulated. While the authors' transparency is highly commendable, a reader must keep in mind that the impressive numbers (like 86.57%) are generated inside a continuous weight-merging simulator rather than on actual physical checkpoints for those four tasks. This makes the physical PyTorch MLP and CLIP foundation model validations in Sections 4.6 and 4.7 the most critical parts of the empirical evaluation, as they bridge the simulative-to-physical gap.

### 2. Physical Validation Sample Sizes
The physical validations are executed on relatively small evaluation subsets (e.g., 50 images per dataset for the physical CLIP zero-shot validation). While the authors justify this design decision based on CPU hardware constraints and system out-of-memory (OOM) risks on a shared public cluster, evaluating on such small subsets introduces high statistical variance. For example, a single image misclassification represents a 2% accuracy shift on a 50-image dataset. Although the results are highly consistent with the simulative trends, full-scale evaluation on the complete CIFAR-10 and GTSRB test sets would make the physical findings much more robust.

### 3. TV Regularization Strength Underperforming
Under the simulated convex sandbox (Table 1), TV regularization ($\beta=20.0$) achieves $86.58\% \pm 7.23\%$, which is mathematically equivalent to PolyMerge ($d=2$) with no statistically significant difference ($p = 0.80$). This indicates that on simple convex landscapes, standard penalty-based regularization is just as effective as subspace projection. The authors' claim of PolyMerge's superiority rests on:
- Performance in highly realistic coupled non-convex landscapes (Table 2 in Appendix, $p < 0.05$).
- Hyperparameter-free operation (no continuous tuning required online).
- Parameter efficiency in black-box optimization.
While these points are highly valid and convincing, the fact remains that TV regularization is highly competitive if continuous online hyperparameter tuning were somehow possible.
