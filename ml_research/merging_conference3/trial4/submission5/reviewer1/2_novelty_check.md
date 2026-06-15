# Intermediate Review File 2: Novelty Check and Delta from Prior Work

## 1. Key Novel Aspects of the Submission
While the individual concepts of task arithmetic, weight pruning, and hyperparameter calibration are well-established, this paper introduces several novel integrations and systematic insights:
* **The "Sign Consensus is Secondary" Thesis:** The paper challenges the contemporary consensus (e.g., TIES-Merging) that sign-compatibility election is crucial for preventing representational collapse. It demonstrates that a simpler, deterministic, magnitude-based binary mask (SG-TA GQ) applied before merging is sufficient to outperform or match multi-stage sign election.
* **Global Quantile (GQ) vs. Layer-wise Quantile (LQ) Masking Scopes:** The paper formalizes and empirically evaluates the difference between enforcing a strict, homogeneous layer-wise budget (LQ, equivalent to Decoupled Prune-then-Merge) and allowing global budget flexibility (GQ). It demonstrates that global flexibility is essential for allowing task-specialized layers (like attention projections and late MLPs) to retain higher update densities.
* **Sigmoid-Gated Soft Masking (SG-TA-Soft):** It introduces a continuous, differentiable sigmoid-gated mask that interpolates updates near the threshold. This soft-gating is shown to dramatically stabilize calibration across random seeds, halving the standard deviation.
* **Non-Uniform Coordinate Search (CS) Calibration:** It addresses the exponential complexity $\mathcal{O}(P^T)$ of optimizing task-specific parameters ($k_i, \alpha_i$) by proposing a highly scalable, linear-time $\mathcal{O}(T)$ coordinate descent algorithm. It shows that non-uniform optimization rebalances task performances.

## 2. Technical Delta from Prior Work
The proposed SG-TA is compared against, and differentiated from, several key prior works:

* **Delta from Task Arithmetic (TA) & Optimized TA:** 
  * *Standard TA* adds dense, unregularized task vectors, leading to catastrophic representation collapse. 
  * *Optimized TA* optimizes a global scaling factor $\alpha$. 
  * *SG-TA* introduces deterministic magnitude-based binary masks $M_i$ to filter out low-magnitude updates before applying scaling and addition, proving that spatial parameter regularization is the primary driver of performance.
* **Delta from TIES-Merging:** 
  * *TIES* involves three sequential steps: (1) magnitude-based pruning, (2) sign election (voting on dominant signs across tasks), and (3) merging only sign-compatible updates. 
  * *SG-TA GQ* simplifies this by omitting the sign election and sign-compatibility checks entirely. It relies solely on a global magnitude mask and standard linear addition, showing that simple global budget flexibility outpaces complex sign election.
* **Delta from DARE-Merging:** 
  * *DARE* stochastically drops updates (dropout-style) and rescales the remaining weights. 
  * *SG-TA* is completely deterministic and keeps only the highest-magnitude updates, proving that magnitude-based selection is superior to stochastic dropouts.
* **Delta from Decoupled Prune-then-Merge (P-then-M):** 
  * *P-then-M* applies standard magnitude pruning to experts prior to merging, which is mathematically equivalent to *SG-TA LQ* (layer-wise homogeneous thresholding).
  * *SG-TA GQ* relaxes this layer-wise constraint, allowing variable budgets across layers. The $+3.59\%$ absolute improvement of GQ over LQ/P-then-M highlights the substantial value of global budget flexibility.
* **Delta from Fisher-Weighted Averaging:**
  * *Fisher-Weighted Averaging* requires computing first-order gradients on validation splits and storing dense, task-specific diagonal curvature matrices.
  * *SG-TA GQ* is a zero-order, lightweight surrogate that requires no gradient computation or dense curvature storage, achieving nearly double the joint accuracy ($61.40\%$ vs $37.85\%$) while scaling much better.

## 3. Characterization of Novelty
The novelty of this submission is **primarily incremental but highly systematic and rigorous**:
* **Incremental Aspects:** The core operation (magnitude-based binary masking and scaling) is a straightforward combination of magnitude-based pruning and task arithmetic. Both techniques are highly mature. 
* **Significant Aspects:** The value of the paper lies in its *comprehensive scientific dissection* and *calibration innovations*. The rigorous contrast between GQ and LQ, the introduction of continuous soft-gating to smooth and stabilize the loss landscape, and the formulation of the linear-time Coordinate Search algorithm are significant contributions. The paper elevates what would be a simple pruning heuristic into a thoroughly analyzed, stable, and highly scalable model merging paradigm.
