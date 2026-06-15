# Evaluation 1: Paper Summary

## Main Topic
This paper focuses on **test-time adaptive model merging**, a paradigm that combines multiple task-specific expert neural networks (fine-tuned from a shared pre-trained base) into a single multi-task model at test-time without requiring the original joint training datasets. The paper specifically deconstructs **AdaMerging**—the state-of-the-art framework that uses test-time unsupervised entropy minimization on small calibration streams to learn layer-wise merging coefficients.

## Proposed Approach
To systematically address key failure modes of existing adaptive test-time model merging (transductive overfitting and sacrificial task bias), the authors introduce **RegCalMerge** (Calibrated & Regularized Test-Time Model Merging). The framework consists of two main modules:
1. **Calibration Engine (CalMerge)**: Combines:
   - **Class-Capacity Normalization (CCN)**: Scales raw prediction entropy by $1 / \log C_k$ (where $C_k$ is the number of classes) to project entropy values to a uniform $[0, 1]$ scale.
   - **Scale-Normalized Entropy Weighting (SNEW)**: Computes a constant weight $w_k = 1 / \bar{\mathcal{H}}_k(\Lambda_{\text{init}})$ at step 0 (initialization) to balance the optimization gradients, preventing simpler, low-entropy tasks from dominating.
2. **Elastic Spatial Regularization (ESR)**: An optional structural stabilizer that regularizes parameter drift by applying:
   - A **Proximity Penalty ($\beta$)**: Penalizes coefficient deviation from the uniform initialization ($\lambda_{\text{init}} = 0.3$).
   - A **Spatial Deviation Penalty ($\gamma$)**: Penalizes layer-wise coefficient variance around the task-level spatial average ($\bar{\lambda}_k$).

The joint test-time optimization objective is:
$$ \min_{\Lambda} \sum_{k=1}^K w_k \bar{\mathcal{H}}_k(\Lambda) + \mathcal{R}_{\text{spatial}}(\Lambda) $$
where $\mathcal{R}_{\text{spatial}}(\Lambda)$ is the normalized ESR penalty.

## Key Findings and Claims
1. **The Overfitting-Optimizer Paradox (Transductive Overfitting)**: Fine-grained, layer-wise coefficients optimized in standard AdaMerging overfit to the local statistics of tiny test-time calibration batches rather than capturing genuine layer interactions.
   - *Evidence*: A spatial shuffling diagnostic is introduced. Shuffling the optimized layer coefficients across layers maintains nearly 95% of the joint mean performance (60.94% vs. 61.62% for Adam GD; 60.45% vs. 59.77% for 1+1 ES), proving that layer-wise local configuration is largely redundant/overfit parameter-drift.
2. **Sacrificial Task Bias**: Under standard joint entropy minimization, tasks with high baseline entropy or high complexity (e.g., SVHN) are degraded/sacrificed (e.g., SVHN drops to 28.26% under 1+1 ES) to prioritize easier, low-entropy tasks.
   - *Evidence*: Standard AdaMerging (1+1 ES) drops SVHN accuracy from 29.69% (Task Arithmetic) to 28.26%. SNEW and CCN in CalMerge resolve this, raising SVHN to 32.03% and Joint Mean to 61.82%.
3. **The Value of Spatial Degrees of Freedom**: Although layer-wise parameters overfit, completely collapsing them to a single scalar per task (Spatially Averaged AdaMerging) results in standard representation collapse (e.g., CIFAR-10 drops to 76.17%). Even with calibration (Calibrated Spatial Mean), layer-wise CalMerge outperforms it (61.82% vs. 61.13% joint mean, with CIFAR-10 at 85.16% vs. 78.65%).
4. **Generalization-Regularization Trade-off**: Activating ESR introduces a smooth, monotonic trade-off in peak accuracy but prevents unconstrained test-time parameter drift.

## Explicitly Claimed Contributions
- Identification and empirical deconstruction of the **Overfitting-Optimizer Paradox** via a novel spatial shuffling diagnostic.
- Identification of the **Sacrificial Task Bias** in multi-task entropy minimization.
- Introduction of **RegCalMerge** and its key elements: **CCN**, **SNEW**, and **ESR**.
- Extensive empirical evaluations across 3 seeds over 7 baselines and a dense 2D hyperparameter sweep ($\beta \times \gamma$) showing state-of-the-art joint accuracy and a highly predictable generalization surface.
