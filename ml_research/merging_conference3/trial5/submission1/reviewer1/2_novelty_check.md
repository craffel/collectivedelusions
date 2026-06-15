# 2. Novelty and Delta Check

## Key Novel Aspects
The paper introduces several novel conceptual and technical elements to the model-merging literature:
1. **The Overfitting-Optimizer Paradox**: While Source-Free Domain Adaptation (SFDA) and Test-Time Adaptation (TTA) are known to suffer from representation collapse under noisy streams, this is the first work to identify, formalize, and analyze this failure mode within the specific low-dimensional coefficient space of adaptive model merging (where optimizing just a few dozen parameters can trigger global representational collapse due to high-frequency spatial oscillations).
2. **Riemannian Curvature-Regularized Spatial TV (RCR-TV)**: Instead of treating the layer-wise coefficient space as a flat Euclidean surface, this work models it as a Riemannian manifold where distance is locally scaled by pre-trained base model sensitivities. Weighting the spatial Total Variation penalty between adjacent layers by the geometric mean of their pre-trained base curvatures ($\sqrt{c_l c_{l-1}}$) is a novel way to establish localized physical barriers.
3. **Gradient Norm Balancing (GNB)**: The unsupervised dynamic initialization of regularization strength ($\beta$) via gradient balancing under a worst-case spectral perturbation (the highest-frequency eigenvector of the 1D graph Laplacian) represents a highly novel and elegant way to resolve the unsupervised hyperparameter selection challenge.
4. **Lightweight Local Charting**: The formulation of a threshold-triggered online curvature re-estimation mechanism offers a novel bridge between short-term static metric approximations and long-term, non-stationary streaming scenarios.

## Delta from Prior Work
The proposed method stands in clear contrast to three main categories of prior work:
- **Static Merging (Task Arithmetic, TIES-Merging, DARE, RegCalMerge)**: These methods rely on uniform scaling coefficients across all layers. RCR-Merge allows layer-wise specialized adaptability but bounds the spatial trajectory to ensure representation smoothness.
- **Adaptive Merging (AdaMerging)**: AdaMerging proposed layer-wise merging coefficients optimized via online test-time entropy minimization but allowed them to optimize freely without constraints. RCR-Merge identifies the catastrophic overfitting collapse in AdaMerging and introduces second-order geometric regularization to solve it.
- **Rigid Subspaces (PolyMerge)**: PolyMerge restricts coefficient trajectories to a rigid low-dimensional quadratic polynomial. RCR-Merge allows full localized adaptability (piecewise-smooth transitions) but bounds the spatial transitions using conformal curvature barriers, proving mathematically and empirically superior on modular landscapes with discrete stage-wise transitions.

## Characterization of Novelty
The novelty of this work is **significant**. 
While the individual components—Total Variation, Fisher Information Matrix (FIM) traces, and gradient balancing—are established concepts in image processing, optimization, and multi-task learning, their integration into a unified, mathematically rigorous second-order geometric framework for online test-time model merging is highly original. 

Rather than proposing a purely empirical heuristic, the authors provide a deep, multi-layered theoretical justification:
- They establish a direct mathematical link between the high-dimensional Fisher manifold and the low-dimensional coefficient space via local pullback metrics.
- They prove that the spatial TV penalty is the natural Riemannian Total Variation on the induced coefficient manifold.
- They prove that the curvature scaling acts as a formal coordinate-level spatial barrier and bounds activation-level adjacent-layer representational drift.
- They analyze the optimization dynamics as a spectral Laplacian low-pass filter that filters out transductive noise.

The "delta" from existing methods is substantial, and the conceptual bridging of loss landscape geometry with online model adaptation represents a highly promising, mathematically rich direction for the community.
