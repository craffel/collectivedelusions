# 2. Novelty and Delta Check

## Novel Aspects
1. **Systematic Mapping of Structural Granularity:** 
   Prior work has operated in isolated pockets of structural resolution: Task Arithmetic at the global model level, AdaMerging at the layer level, and PolyMerge/SplineMerge along the layer depth. This paper is the first to systematically construct a nested hierarchy of 5 granularities (Global, Layer, Block, Component, and Tensor) to map out the complete performance-generalization landscape.
   
2. **First-Order vs. Zero-Order Test-Time Optimization Comparison:**
   While most adaptive merging methods rely exclusively on gradient descent, this work introduces derivative-free zero-order optimization (1+1 ES) as an alternative. It provides a unique comparison of how optimization trajectories (gradient coordinate updates vs. isotropic random walks) interact with transductive overfitting.
   
3. **Deconstruction of Zero-Order Robustness:**
   Instead of simply claiming zero-order methods are "better," the authors provide a rigorous and highly honest evaluation. They formulate the "sluggishness hypothesis"—that zero-order ES appears robust at high dimensions because it fails to optimize in 100 steps, thus acting as an implicit regularizer by remaining close to the high-quality static initialization.

4. **Elastic Spatial Regularization (ESR) and Total Variation (TV) Smoothness:**
   Adapting spatial L2 pull-to-mean penalties and depth-wise TV smoothness constraints specifically to control high-dimensional, fine-grained merging coefficients is a novel combination of established regularization tools for model merging.

---

## The 'Delta' from Prior Work
- **From Task Arithmetic (Ilharco et al., 2023):** Task Arithmetic uses a single global weight (Level 1) manually tuned or swept. GranMerge automates this and expands the search space up to 288 tensor-wise coefficients.
- **From AdaMerging (Yang et al., 2024):** AdaMerging optimizes layer-wise scales (Level 2) using entropy minimization. GranMerge shows that while Level 2 is an improvement over Level 1, further resolution (Levels 3, 4, 5) suffers from severe transductive overfitting, which AdaMerging did not examine.
- **From PolyMerge / SplineMerge (Jung et al., 2025):** PolyMerge restricts layer scales to continuous polynomial curves to prevent overfitting. GranMerge takes the opposite direction by going *finer* (Component and Tensor-wise) and showing that simple soft regularizers (ESR/TV) cannot arrest gradient-based overfitting, which validates the need for hard structural constraints like PolyMerge's splines.

---

## Characterization of Novelty
The novelty of this paper is primarily **empirical, diagnostic, and conceptual** rather than algorithmic.
- **Incremental Algorithmic Novelty:** The individual components (entropy minimization, 1+1 ES, Adam, L2 spatial smoothing, and total variation) are standard techniques in optimization and machine learning.
- **Significant Diagnostic Novelty:** The paper excels in its willingness to report a "negative" result—namely, that no adaptive method beats the static, zero-overhead baseline of 30.41%. By systematically mapping the structural resolution and exposing the transductive overfitting boundary and surrogate loss misalignment, the paper provides a crucial service to the model merging community, serving as a warning against over-parameterizing test-time adaptation.
- **Practitioner's Perspective:** As a practitioner, this diagnostic novelty is highly valuable. It prevents engineers from wasting resources on complex, fine-grained test-time loops that fail to generalize, confirming that simple static blends remain the most robust deployment strategy under compact calibration constraints.
