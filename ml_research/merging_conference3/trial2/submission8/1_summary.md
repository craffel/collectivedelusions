# 1. Summary of the Paper

## Main Topic
The paper addresses the challenge of deploying multiple specialized multi-task expert models on storage- and network-constrained edge and IoT devices. While weight-space merging via Task Arithmetic enables multi-task capabilities by storing task-specific delta vectors, dense weights remain too large for resource-constrained environments. The paper explores post-hoc weight sparsification (compressing these task vectors) to drastically reduce their storage footprint, allowing them to be stored in sparse formats (like Compressed Sparse Row / CSR) and loaded/merged on-the-fly with zero runtime FLOP or latency overhead.

## Proposed Approach
The authors propose **Norm-Preserved Budgeted Task-Vector Pruning (NP-BTVP)**, a post-hoc weight sparsification and merging framework. It introduces two deterministic magnitude-based pruning schemes:
1. **Uniform Pruning (NP-BTVP-U):** Sparsifies task vectors globally, retaining the top $p\%$ of absolute updates and scaling them by $1/p$.
2. **Adaptive Saliency-Based Pruning (NP-BTVP-S):** Allocates parameter budgets layer-wise based on a first-order magnitude-based saliency metric (average $L_1$-norm of updates across tasks normalized by layer size). It evaluates two scaling formulations: Global Rescaling (scaling by $1/p$) and Layer-wise Rescaling (scaling by $1/p_l$).

Crucially, both schemes incorporate **norm-preserving rescaling** (multiplying active elements by the reciprocal of the retention rate) as a deterministic signal-strength preservation heuristic. This prevents update norm shrinkage and provides a signal boost to prevent the base model weights from drowning out the specialized task updates during weight-space fusion.

## Key Findings
1. **Geometric Separation of Flatness and Sparsification:** Contrary to the hypothesis that fine-tuning with Sharpness-Aware Minimization (SAM) provides an additional coordinate-aligned pruning buffer due to flatter loss landscape basins, standard AdamW and SAM experts show nearly identical and extraordinary levels of resilience to post-hoc magnitude pruning under the proposed rescaling framework. Under Uniform Pruning at $p=0.10$ (90% sparsity), AdamW achieves 90.34% and SAM achieves 90.32% average accuracy.
2. **The Saliency Double-Bind:** Adaptive layer-wise budget allocation (NP-BTVP-S) is slightly outperformed by global Uniform Pruning (NP-BTVP-U). The authors identify a fundamental double-bind in layer-wise pruning: Global Scaling ($1/p$) introduces inter-layer scale imbalance (silencing low-saliency layers or over-amplifying high-saliency layers), while Layer-wise Scaling ($1/p_l$) leads to extreme local variance/noise amplification in low-saliency layers with very small budgets. Uniform pruning naturally avoids this by keeping $p_l = p$ everywhere, maintaining perfect scale harmony.
3. **Criticality of Norm-Preserving Rescaling:** Ablating the rescaling factor leads to a catastrophic performance collapse (average accuracy dropping from ~90% to ~80% at $p=0.10$), demonstrating that update norm shrinkage is the primary obstacle and that the $1/p$ scaling heuristic is the critical enabler of sparse task-vector merging.
4. **Competitive Performance against Baselines:** At a tight 90% sparsity budget ($p=0.10$), NP-BTVP-U performs competitively with the advanced stochastic DARE-Merging baseline (which operates at 80% sparsity) and significantly outperforms TIES-Merging by 3.81% while using half the parameter budget.
5. **Practical Storage Gains:** At 90% sparsity, storing task vectors in Compressed Sparse Row (CSR) format leads to an immediate 5x reduction in raw storage (or up to 10x with index compression), compressing a 115 MB CLIP expert to ~23 MB.

## Explicitly Claimed Contributions
1. **NP-BTVP Framework:** Introduction of a training-free weight sparsification and merging framework using norm-preserving rescaling, comparing global (NP-BTVP-U) and layer-wise (NP-BTVP-S) schemes.
2. **Rigorous Empirical Evaluation:** Evaluation across 3 independent random seeds on 4 distinct classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) using a CLIP ViT-B/32 backbone.
3. **Flatness and Sparsification Insight:** Demonstrating that optimizer-driven loss landscape flatness (via SAM) does not provide an additional coordinate-aligned pruning buffer under well-converged regimes compared to standard AdamW.
4. **Analysis of the Saliency Double-Bind:** Providing a detailed analysis of the trade-offs between layer-wise adaptive budget allocation and global uniform pruning under rescaling, showing the pragmatic superiority of simple uniform scale preservation.
