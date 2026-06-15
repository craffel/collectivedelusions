# 2. Novelty and Literature Check

## Main Conceptual Delta from Prior Work
The main novelty of this submission is **deconstructive** rather than constructive. While most recent publications in model merging focus on introducing increasingly complex, multi-stage heuristics (e.g., TIES-Merging's coordinate-wise voting, dominant sign election, and disjoint merging, or DARE's stochastic dropout and rescale factors), this paper applies Occam's razor. It shows that:
1. Coordinate-wise sign-voting and consensus protocols are **entirely redundant**.
2. The apparent performance gap between simple sparse addition and complex methods is caused by a methodological confounder: **update under-scaling** (due to parameter magnitude attenuation after pruning).
3. Once this confounder is corrected by rescaling (R-STA) or tuning ($\lambda$), a stripped-down baseline using only uniform layer-wise magnitude-based pruning and direct linear addition matches or exceeds the performance of TIES-Merging and DARE.

This represents a significant conceptual departure from the current trajectory of the literature, which has been over-engineering solutions to parameter interference.

## Critical Literature Gap: Omission of He et al. (2024)
A major bibliographic omission exists in this paper. The authors present **"Sparse Task Arithmetic" (STA)** as a novel technique that they introduce. However:
* A paper titled **"Localize-and-Stitch: Efficient Model Merging via Sparse Task Arithmetic"** by Yifei He et al. (arXiv:2408.13656) was published in August 2024 and accepted at the peer-reviewed journal *Transactions on Machine Learning Research (TMLR)* in December 2024.
* He et al. (2024) already proposed the general concept of sparsifying task-specific updates (which they also refer to as "sparse task arithmetic") to localize essential parameter regions for different tasks and stitch them back, showing that this reduces interference and improves performance.

### Differentiation and Contextualization
While the general terminology and concept of sparse task vectors overlap, the two works differ in several key ways:
* **Methodology**: He et al. (2024) use a more complex, optimization-based localization process to identify a task-specific mask that preserves performance. This submission proposes a much simpler, training-free, and deployment-friendly uniform layer-wise magnitude-based pruning (retaining the top-$s$\% largest updates).
* **Research Focus**: The main thesis of this submission is the deconstruction of sign-consensus heuristics (showing that sign-voting is redundant) and exposing the under-scaling confounder. This deconstructive critique of TIES-Merging and DARE is entirely absent in He et al. (2024).

The authors **must** cite He et al. (2024), discuss this prior work, and clarify that their specific method is a "Simple Magnitude-based Sparse Task Arithmetic" variant to avoid a direct naming collision and preserve academic integrity.

## Characterization of Novelty
* **Methodological Novelty**: *Incremental to Fair*. The actual math of layer-wise magnitude pruning and rescaling is straightforward and highly related to standard pruning literature.
* **Analytical/Insight Novelty**: *Significant*. The deconstructive analysis—the mathematical proof that sparsity naturally eliminates collisions, the empirical verification of mask overlap, and the exposure of the update under-scaling confounder—is highly insightful and offers a major paradigm shift for model-merging research.
* **Practical Novelty**: *Significant*. From a deployment perspective, showing that a simple 3-line PyTorch script can replace hyperparameter-heavy, complex pipelines is a highly practical contribution. It makes model merging much more accessible and efficient to implement in real-world production systems.
