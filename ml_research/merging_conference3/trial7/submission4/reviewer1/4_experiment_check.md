# Experimental Design and Results Evaluation - 4_experiment_check.md

## Critical Evaluation of the Experimental Setup
1. **Fully Synthetic Calibration Sandbox:** The main experimental evaluation is conducted in a completely synthetic, simulated representation sandbox. Although the tasks are labeled as MNIST, FashionMNIST, CIFAR-10, and SVHN, no actual images or real feature vectors from models trained on these datasets are utilized. The features are generated via Gaussian distributions in a $D=192$ dimensional space. This highly controlled, simplified environment limits the practical significance of the findings, as real-world deep representations exhibit highly complex, non-Gaussian, anisotropic structures that are not captured here.
2. **Artificial "Real-World" Proof of Concept on ResNet-18:** The paper claims to conduct a "Real-World Proof of Concept" on a ResNet-18 feature manifold. However, a close inspection of Section 4.5 reveals that **the authors did not run ResNet-18 on any real images**. Instead, they extracted the final linear layer's weight rows (prototypes) and *added synthetic Gaussian noise* ($\sigma = 0.15$) to generate 50 sample representations per class. This is essentially another synthetic simulation in a 512-dimensional space, and does not validate the method on actual image representations which feature realistic out-of-distribution drift, class imbalances, or complex intra-class variance.

---

## Evaluation of Baselines
1. **Calibration Split Size Handicap:** The parametric baselines (LinearRouter, QWS-Merge, L3-Softmax) are trained on an extremely small calibration split of only 64 samples (16 samples per task). In such a low-data regime, parametric models are severely handicapped and guaranteed to overfit. To ensure a fair and rigorous comparison, the authors should have evaluated performance across a sweep of calibration split sizes (e.g., 64, 128, 256, 512, 1024 samples) to demonstrate at what volume of calibration data the parametric routers catch up to or exceed the zero-parameter PFSR. 
2. **Uniform Merging Baseline Strength:** The Uniform Merging baseline is exceptionally strong. Due to the "Orthogonal Masking Effect" under perfectly disjoint task spaces, Uniform Merging achieves the exact same ceiling classification accuracy of 74.46% as the proposed PFSR and OTSP. 

---

## Alignment of Claims and Results

### 1. Failure to Outperform Static Uniform Merging in Classification Accuracy
A critical weakness of the empirical results is that **the proposed dynamic routers (PFSR and OTSP) fail to outperform the simplest baseline, static Uniform Merging, in joint classification accuracy on any evaluated setup**.
- In the primary primary sandbox (Table 1), joint classification accuracy is completely flat at 74.46% for all simplex-constrained methods.
- In the asymmetric sandbox (Table 3), static Uniform Merging achieves **80.83% $\pm$ 0.51%** classification accuracy, which slightly but consistently *outperforms* both PFSR and OTSP (**80.55% $\pm$ 0.54%**), despite the dynamic methods achieving much higher routing accuracy (70.76% vs. 25.00%).
- The authors deconstruct this by arguing that Uniform Merging benefits from "prediction-averaging" in overlap regions. However, from a pure model-performance standpoint, this means that dynamic parameter-free routing provides zero performance gains (and actually a slight penalty) over simple uniform merging, undermining the core motivation of designing complex sample-wise dynamic routers.

### 2. SVD Centroid Sensitivity under Perfectly Disjoint Layouts
In Table 1, PFSR and OTSP achieve a perfect 100.00% routing accuracy. However, this is because the non-target coordinates of the expert weight matrices are initialized to exactly zero (perfectly disjoint, uncorrupted setup). In real-world multi-task registries, experts share significant background features and representations, which would cause significant cross-talk. When evaluated under active representation overlap ($\rho = 0.33$), routing accuracy drops, and the hard gating selection collapses classification accuracy to 71.71% (-4.33% compared to Uniform Merging), demonstrating a clear performance trade-off that is obscured by the clean disjoint setup.
