# Paper Summary: Norm-Preserved Budgeted Task-Vector Pruning (NP-BTVP)

## 1. Main Topic and Scope
This paper addresses the challenge of deploying multiple task-specific expert models (e.g., fine-tuned from a shared pre-trained base model like CLIP) on resource-constrained edge and IoT devices. It focuses on post-hoc weight sparsification and merging of these experts via Task Arithmetic. The objective is to heavily compress the "task vectors" (parameter shifts relative to the base model) to reduce storage and transmission footprints, allowing them to be stored in sparse formats (like CSR or COO) and merged on-the-fly without introducing runtime computational latency.

## 2. Proposed Approach
The authors introduce **Norm-Preserved Budgeted Task-Vector Pruning (NP-BTVP)**, a post-hoc weight sparsification and merging framework. It contains the following core components:
1. **Task Vector Extraction:** Computes task vectors as $\tau_k = \theta_k - \theta_{\text{base}}$ for $K$ experts fine-tuned from a shared base model.
2. **Deterministic Magnitude-Based Pruning Schemes:**
   - **Uniform Pruning (NP-BTVP-U):** Prunes each task vector globally to retain the top $p$ fraction of absolute updates.
   - **Adaptive Saliency-Based Pruning (NP-BTVP-S):** Allocates layer-wise budgets $p_l$ based on normalized layer-wise update intensity ($L_1$-norm average), solved via binary search under a global budget $p$.
3. **Norm-Preserving Rescaling Heuristic:** Scaled active updates by $1/p$ (global) or $1/p_l$ (layer-wise). The paper notes that rather than keeping the $L_1$ norm strictly invariant, this reciprocal scaling acts as a deterministic signal-strength boost to prevent the task vectors from being drowned out by the base pre-trained model.
4. **Flatness-Aware Training:** Experts are optionally fine-tuned using Sharpness-Aware Minimization (SAM) to investigate whether loss landscape flatness buffers task vectors against coordinate-wise pruning.

## 3. Key Empirical Findings
- Under the proposed norm-preserving rescaling, both standard AdamW and flatness-aware SAM experts demonstrate an extraordinary and nearly identical level of resilience to heavy sparsification. Under Uniform Pruning with a 10% global budget ($p=0.10$), AdamW achieves 90.34% and SAM achieves 90.32% multi-task average accuracy across 4 datasets, compared to dense unpruned baselines of 90.94% and 91.00% respectively.
- Surprisingly, training-stage flatness (via SAM) does not inherently provide an additional coordinate-aligned pruning buffer under well-converged regimes compared to standard AdamW.
- Uniform pruning (NP-BTVP-U) performs competitively with Saliency-Based pruning (NP-BTVP-S) and is highly statistically indistinguishable, making it preferred due to simplicity.
- Layer-wise rescaling in NP-BTVP-S is subject to the **Saliency Double-Bind**: global scaling introduces severe inter-layer scale imbalance, while layer-wise scaling introduces extreme local noise and variance amplification.
- Combining NP-BTVP-U ($p=0.10$) with post-hoc INT8 quantization yields an overall 40x compression (reducing expert size to 5.74 MB) with a negligible accuracy drop of only 0.12% (90.20% accuracy under SAM).

## 4. Claimed Contributions (with Evidence)
1. **Introduction of the NP-BTVP Framework:** Supported by the mathematical formulation of deterministic pruning and rescaling schemes in Section 3.3.
2. **Rigorous Empirical Evaluation:** Evaluated on a CLIP ViT-B/32 backbone across 4 datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) over 3 random seeds, showing highly competitive performance relative to dense baselines and TIES/DARE merging (Tables 1, 2, 3).
3. **Geometric Separation Insight:** Evidence in Section 4.3 shows that loss landscape flatness (SAM) does not provide a coordinate-aligned pruning buffer compared to standard AdamW under well-converged regimes, highlighting a valuable geometric separation.
4. **Analysis of the Saliency Double-Bind:** Formally analyzed and supported by empirical comparison of global vs. layer-wise rescaling configurations (Section 4.3) and their interaction with quantization (Section 4.6).
