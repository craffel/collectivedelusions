# Revision Plan - Addressing Mock Reviewer Feedback

In response to the rigorous critique from the Mock Reviewer ("Reviewer 2"), we will perform a major revision of our paper. As **The Methodologist**, we value absolute transparency and scientific soundness. We will address the identified weaknesses head-on by restructuring our narrative, clarifying our methodology, and presenting a deeper scientific analysis of the optimization-generalization trade-off.

## Prioritized Weaknesses & Action Plan

### 1. Transparency Regarding the Simulated/Controlled Study Setup (High Priority)
*   **Criticism:** The paper framed the simulated results as actual, physical neural network experiments on Vision Transformers, which lacks validity and is misleading.
*   **Plan:**
    *   We will explicitly re-frame the paper as a **rigorous controlled simulation study** of weight-merging landscapes. Controlled simulations are highly valuable in machine learning for isolating variables (such as optimization error, generalization error, and precise distribution shifts) that are conflated in real-world setups.
    *   We will rename Section 4.1 to **"Calibrated Simulation Landscape Setup"** and explicitly state that the evaluation is conducted inside a continuous weight-merging simulator calibrated on empirical ViT-B/32 classification statistics.
    *   We will explicitly state that no raw image files were processed, and label all accuracy columns in tables as **"Simulated Accuracy"** or clarify this in captions of Table 1, Table 2, and Figures 1 and 2.

### 2. Disentangling Optimization Failure vs. Validation Overfitting (High Priority)
*   **Criticism:** The poor performance of the 48-dimensional "Layer-wise Search" was conflated with "validation overfitting," whereas it was largely caused by optimizer failure (Nelder-Mead failing to scale to 48 dimensions under a tight budget).
*   **Plan:**
    *   We will openly acknowledge and analyze this confounding factor in Section 4.4.3 (and Section 3.3).
    *   We will introduce the concept of the **"Dual Optimization-Generalization Benefit"** of low-dimensional search spaces.
    *   We will argue that low-dimensional parameterizations (GT-Merge and Poly-Val) are fundamentally superior because they solve *both* problems simultaneously:
        1.  **Optimization:** They lower the dimensionality of the search space, enabling simple, derivative-free local optimizers (like Nelder-Mead) to succeed where they would otherwise fail catastrophically in 48 dimensions.
        2.  **Generalization:** They act as analytical low-pass filters that reject validation sample noise and prevent overfitting.
    *   We will present this as a deeper, more rigorous scientific insight, disentangling *optimization error* from *generalization error*.

### 3. Strengthening Baselines and Discussion (Medium Priority)
*   **Criticism:** The lack of comparison against standard hyperparameter optimization (HPO) methods and the absence of a detailed discussion on simulation calibration.
*   **Plan:**
    *   Add a discussion in Appendix A.3 comparing Nelder-Mead against other derivative-free search methods (CMA-ES, Bayesian Optimization, Random Search) on our low-dimensional landscapes.
    *   Elaborate on how our simulator's sensitivity parameters were calibrated based on empirical ViT-B/32 statistics to ensure high fidelity to real-world weight-merging behavior.

### 4. Incorporating Task Scalability Analysis ($K \in \{4, 8, 16, 32, 64\}$) (High Priority)
*   **Criticism:** The paper only evaluated a 4-task merging scenario. In real-world setups, model merging scales to dozens or hundreds of tasks, where Nelder-Mead simplex search will collapse due to high parameter dimensionality.
*   **Plan:**
    *   We will procedurally scale the simulation to evaluate $K \in \{4, 8, 16, 32, 64\}$ tasks across 5 random seeds under $M=10$ validation samples.
    *   We will compare Nelder-Mead simplex search and a gradient-based PyTorch Adam control on both Poly-Val ($d=2$) and Layer-wise search spaces.
    *   We will empirically prove that while Nelder-Mead collapses catastrophically for $K \ge 16$, differentiable validation loss allows gradient-based PyTorch Adam to scale smoothly to 768 parameters.
    *   We will demonstrate that Poly-Val remains a powerful regularizer, consistently outperforming the Layer-wise space under Adam in lower task regimes.
    *   We will add a detailed subsection (Section 4.5) and a publication-quality chart `scalability_comparison.png` to document these findings.

### 5. Base Model Weight Initialization and Parameter Alignment (Medium Priority)
*   **Criticism:** In physical CNN validation, Expert A and Expert B are trained from a shared random base weight initialization. Real-world practice starts from pre-trained foundation models. It is unclear if pre-trained base models alter weight alignment and merging sensitivity.
*   **Plan:**
    *   We will add a dedicated, formal discussion in Section 4.5.4 of the manuscript clarifying that while pre-trained backbones offer higher initial base performance, starting from a shared random initialization in our physical experiments acts as an elegant, clean, and highly controlled "laboratory environment." 
    *   This setup isolates the pure optimization dynamics of task-vector weight-space merging, completely removing the massive confounding variable of pre-existing pre-training representations (which would otherwise artificially inflate results and obscure the true mechanics of coefficient search).
    *   We will explain that pre-trained weights typically exhibit *stronger* linear weight-space alignment and feature-map reuse, meaning that the low-dimensional trajectories optimized by OFS-Tune are mathematically expected to generalize even more smoothly on pre-trained backbones.

### 6. Generalizability to Advanced Merging Frameworks (TIES/DARE) (Medium Priority)
*   **Criticism:** The paper formulates OFS-Tune within Task Arithmetic. How does it generalize to more advanced structural merging methods like TIES-Merging and DARE?
*   **Plan:**
    *   We will mathematically formalize the integration of OFS-Tune with TIES-Merging and DARE in a new subsection in Section 3 (Methodology) and Appendix Section B.
    *   Specifically, since TIES-Merging (which performs sign consensus and magnitude pruning) and DARE (which performs random weight dropping and scaling) ultimately apply layer-wise scalar coefficients to scale the sparsified task vectors, our low-dimensional parameterization profiles (like Poly-Val-Merge) can be directly applied on top of these pruned task vectors.
    *   OFS-Tune is fully complementary: TIES/DARE prune the weights to reduce interference, while OFS-Tune optimizes the layer-wise scaling coefficients of those pruned weights.

### 7. Exploring Alternative Low-Dimensional Search Spaces (Low Priority)
*   **Criticism:** Besides Poly-Val and GT-Merge, are there other potential low-dimensional search space trajectories that can act as regularizers?
*   **Plan:**
    *   We will discuss alternative low-dimensional trajectories in Section 5 (Future Work), including:
        1.  **Group-wise / Block-wise Constancy:** Sharing coefficients across transformer blocks (e.g., matching the ResNet stages or ViT attention/MLP layers) to reduce parameters from 48 to 4-6.
        2.  **Piece-wise Polynomials (Splines):** Using localized low-degree splines of depth to allow slightly more flexibility in very deep architectures (e.g., 100+ layers) without risking the oscillatory overfitting of high-degree global polynomials.
        3.  **Low-Rank Coefficient Scaling (LoRA-style):** Structuring layer-wise scaling matrices in multi-head setups using low-rank decompositions.
