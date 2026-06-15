# 4. Experimental Check and Empirical Validation

## Strengths of the Experimental Evaluation
The experimental evaluation has several notable strengths:
- **Comprehensive Baselines:** The paper evaluates against a wide and appropriate set of baselines: Static Task Arithmetic, Unconstrained AdaMerging (layer-wise), Regularized AdaMerging (both TV and L2 regularized), and PolyMerge (monomial subspace). This covers the most relevant state-of-the-art approaches.
- **Statistical Rigor:** The simulated experiments (Model I and Model II) are conducted across **30 independent random seeds** (seeds 42 to 71), with mean and standard deviations reported. This is exceptionally rigorous and ensures that the reported improvements are statistically significant and not the result of seed hacking.
- **Aesthetic and Informative Visualizations:** The paper includes clear visualizations of optimization loss trajectories (Figure 1), final coefficient profile reconstructions (Figure 2), and physical optimization trajectories on CLIP (Figure 3), which effectively illustrate the optimization dynamics.
- **Real Physical Validation on CLIP ViT-B/32:** The authors have updated the physical validation to use actual, structured task vectors computed by subtracting pre-trained CLIP vision encoder parameters from task-specific expert checkpoints on MNIST and SVHN, utilizing actual image datasets from `torchvision.datasets`, and zero-shot predictions via pre-trained CLIP projections. Real-world test-set classification accuracies are reported.
- **Intellectual Honesty regarding Limitations:** Section 4.5 (Methodological Limitations) is transparent, clearly listing the sequential topology assumptions, asymmetric sensitivities, and the open challenges of scaling continuous parameterizations to a larger number of tasks.

## Critical Empirical Flaws and Weaknesses
Despite its rigorous theoretical framework and the addition of real-world classification accuracies, the experimental evaluation has two **critical empirical flaws** that severely limit its support for the paper's central claims:

### 1. Almost Complete Reliance on Synthetic Simulators for State-of-the-Art Accuracies
The accuracy results in **Table 1** (Model I) and **Table 2** (Model II)—which claim to show state-of-the-art generalization on MNIST, FashionMNIST, CIFAR-10, and SVHN—are **completely simulated**. 
- The "generalization accuracies" are computed using a synthetic, closed-form Mahalanobis-like distance formula (Equation 13 and Equation 18) rather than being evaluated on actual deep neural networks running on actual test datasets.
- The "transductive noise" is artificially injected using mathematical formulas (Equation 11 and Equation 17), rather than being the natural gradient noise that arises during mini-batch test-time adaptation on real data.
- While the simulator is designed to mimic the layer sensitivity profiles and couplings of deep Vision Transformers, it is still a highly idealized mathematical model. It remains unproven whether the complex, high-dimensional non-convex loss landscape of actual neural networks is accurately represented by a Rastrigin-type formulation.

### 2. Real-World Physical Validation Contradicts the Central Thesis
In the newly introduced physical validation on CLIP ViT-B/32 (Table 3), the results directly contradict the core thesis and claims of the paper:
- **Unconstrained AdaMerging Outperforms Proposed Subspace Methods:** The paper's central claim (derived from the simulator) is that unconstrained AdaMerging collapses under transductive noise, and that continuous subspace restriction (ChebyMerge) acts as a powerful shield to preserve generalization. However, in the real-world physical CLIP experiment:
  - **Unconstrained AdaMerging achieves 78.00% average accuracy**.
  - **ChebyMerge ($d=2$) only achieves 74.00% average accuracy** (4.00% *worse* than unconstrained!).
  - **ChebyMerge-CSD ($d=2$) only achieves 75.50% average accuracy** (2.50% *worse* than unconstrained!).
  This indicates that unconstrained optimization generalizes *better* than the proposed continuous subspace parameterizations, completely undermining the simulated findings where unconstrained Adam collapsed to 78.67% while ChebyMerge reached 85.25%.
- **Test-Time Adaptation Consistently Degrades Accuracy:** All adaptive methods, including ChebyMerge and ChebyMerge-CSD, degrade the final classification accuracy compared to the simple, non-adaptive **Static Task Arithmetic baseline (81.50%)**.
  - AdaMerging (78.00%), ChebyMerge (74.00%), and ChebyMerge-CSD (75.50%) all suffer from accuracy collapse after adaptation.
  - This suggests that minimizing unsupervised prediction entropy on-the-fly is fundamentally misaligned with classification accuracy on real data, making the TTA model-merging pipeline questionable for practical use. The paper lacks a solution to this fundamental alignment issue, only offering a way to mitigate the collapse (CSD) which still fails to beat the static baseline.

## Conclusion on Empirical Evidence
While the mathematical and optimization benefits of ChebyMerge (specifically, the perfect conditioning and rapid convergence) are convincingly proven on both the simulator and real CLIP weights, **the central claim of superior generalization accuracy on real tasks is not supported by real-world empirical evidence**. On real deep neural networks, continuous subspace restriction degrades performance compared to unconstrained optimization, and all test-time adaptation methods perform worse than the simple static uniform baseline.
