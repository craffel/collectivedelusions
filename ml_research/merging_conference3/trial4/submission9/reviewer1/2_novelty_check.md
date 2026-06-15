# Intermediate Evaluation: 2. Novelty Check

## Assessment of Key Novel Aspects
The paper introduces **Exclusive Parameter Merging (EPM)**, which claims novelty based on a "direct coordinate-level allocation strategy" for resolving spatial weight-space interference.
From a theoretical perspective, the core novel aspect is the **Soft Exclusive Parameter Allocation (Soft-EPA)** routing operator, which defines a soft coordinate-wise mask based on relative standardized update magnitudes. This is mathematically formulated as:
$$\tau^{\text{exclusive}}_j = (1 - \gamma) \cdot \lambda_{k^*(j)} \tau_{k^*(j), j} + \gamma \sum_{k=1}^K \lambda_k \tau_{k, j}$$
This formula interpolates between hard, exclusive coordinate assignment ($\gamma=0$) and standard linear addition / Task Arithmetic ($\gamma=1$). 

---

## The "Delta" From Prior Work
The proposed method stands in relation to several key branches of literature:
1. **Task Arithmetic (TA) / Linear Interpolation:** TA linearly averages task vectors. EPM's "delta" is the introducing of coordinate-wise exclusivity to prevent opposing updates from canceling each other out. This is a clear step forward, but mathematically simple (a soft thresholding mask).
2. **TIES-Merging & DARE:** These methods resolve conflicts by pruning small updates, electing a majority sign (TIES), or randomly dropping and scaling updates (DARE), before ultimately *averaging* the remaining parameters. EPM's "delta" is that it routes updates based on *coordinate-wise dominance* rather than performing a uniform average blend of all surviving updates. However, selecting updates based on absolute magnitude and applying a binary or soft mask is heavily related to standard magnitude pruning and mask-based routing, which is well-explored in pruning and lottery ticket literature.
3. **Continuous Parameter Tuning (AdaMerging, ZipMerge):** These frameworks optimize layer-wise or block-group-wise scaling factors. EPM's "delta" with TLC-Tune is restricting the optimization search space to only $K$ global task-level coefficients, using a gradient-free (1+1) ES. While the motivation (avoiding high-dimensional overfitting) is sensible, the methodology is a standard application of black-box optimization on a restricted parameter set, which is an incremental engineering design choice rather than an algorithmic innovation.
4. **Standardization:** Task Vector Standardization (dividing by the task vector's global or layer-wise standard deviation) is a direct application of standard $z$-score normalization to the parameter space.

---

## Characterization of Novelty
From a rigorous, theory-minded perspective, the novelty of this work is **incremental**:
- **Heuristic Fusion:** Soft-EPA is a heuristic parameter routing operator. It is not derived from a formal optimization objective or a first-principles theoretical framework. 
- **Trivial Mathematical Identity:** The "elegant mathematical identity" in Equation 9 is a straightforward algebraic rearrangement of the routing formula. While it provides a nice interpretation of the coherence factor $\gamma$ as interpolating between pure exclusivity and standard blending, it is a trivial identity rather than a deep mathematical discovery.
- **Standard Components:** The individual components—task vector extraction, $z$-score standardization, global magnitude pruning, and (1+1) Evolution Strategy—are standard techniques from the literature. The novelty lies in their combination and empirical application to multi-task vision transformer merging.
- **Lack of Deep Theoretical Foundations:** The paper relies on intuitive and descriptive analogies (such as "structural glue," "representation shielding," and "activation manifold alignment") rather than formalizing these concepts mathematically with proofs, theorems, or bounds.
