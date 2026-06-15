# Intermediate Review File 1: Summary of the Paper

## 1. Main Topic and Scope
This paper investigates the training-free consolidation of multiple task-specific neural network experts into a single, unified multi-task model via post-hoc weight-space fusion (model merging). Specifically, it targets the mitigation of "catastrophic representational collision" (destructive parameter interference) that occurs when fine-tuned weights are linearly combined. The evaluation is conducted on a 4-dataset visual classification benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN) using a compact Vision Transformer backbone (`vit_tiny_patch16_224`, 5.7M parameters).

## 2. Proposed Approach: Sparsity-Guided Task Arithmetic (SG-TA)
The authors propose **Sparsity-Guided Task Arithmetic (SG-TA)**, a deterministic weight-space regularization framework that applies magnitude-based binary masking to individual fine-tuned task-specific update vectors (task vectors) prior to merging. This decouples the sparsification (pruning) from other merging steps.

The framework explores several extensions and design choices:
* **Global Quantile (GQ) Masking:** Computes a single magnitude threshold globally across the entire model for each task vector, allowing different layers to retain varying percentages of parameters.
* **Layer-wise Quantile (LQ) Masking:** Computes independent magnitude thresholds for each layer, enforcing a homogeneous parameter keep-ratio across all layers.
* **Sigmoid-Gated Soft Masking (SG-TA-Soft):** Applies continuous, differentiable sigmoid gates to smooth out weight-space boundaries and stabilize hyperparameter optimization.
* **Task Vector Magnitude Normalization (TV-Norm):** Pre-scales task vectors by the inverse of their mean absolute magnitude to resolve magnitude imbalances across tasks.
* **Non-Uniform Hyperparameter Calibration:** Expands calibration from a 2D uniform search ($k, \alpha$) to a $2T$-dimensional search space ($k_i, \alpha_i$) to handle heterogeneous task requirements, proposing **Non-Uniform Coordinate Search (CS)** as a linear-time optimization alternative to exponential grid search or random search.
* **Calibration Protocol:** Employs **Offline Few-Shot Validation Tuning (OFS-Tune)** using $N_{\text{val}} = 10$ samples per task to search for optimal keep-ratios $k$ and scaling factors $\alpha$.

## 3. Key Findings and Quantitative Evidence
1. **Representational Collapse:** Direct linear combination of dense task vectors (Naive Uniform TA) fails dramatically, achieving only **46.32%** Joint Mean Accuracy on the 4-task benchmark compared to the **95.91%** joint expert ceiling.
2. **Efficacy of Spatial Regularization:** Applying magnitude-based masking (SG-TA GQ) significantly recovers performance, achieving **61.40% $\pm$ 1.39%** Joint Mean Accuracy (a $+15.08\%$ absolute improvement over Naive TA and $+2.17\%$ over Optimized TA).
3. **Global Budget Flexibility is Vital:** GQ masking ($61.40\% \pm 1.39\%$) consistently and substantially outperforms LQ masking ($57.81\% \pm 2.52\%$). Sensitivity sweeps at the optimal keep-ratio of $k=0.3$ show GQ at $60.11\%$ vs LQ at $55.06\%$.
4. **Comparison to Complex Baselines:** Simple deterministic GQ masking outperforms or matches more complex methods on this benchmark, such as TIES-Merging ($60.64\% \pm 1.30\%$) and DARE-Merging ($58.44\% \pm 3.02\%$), as well as Fisher-Weighted Averaging ($37.85\% \pm 5.23\%$) and Layer-Group Scaling without pruning ($32.44\% \pm 5.49\%$). However, because of overlapping standard deviations, the improvement over TIES-Merging is not statistically significant.
5. **Absolute Performance Gap:** A massive absolute gap of **34.51%** remains between the best merged model ($61.40\%$) and the joint expert ceiling ($95.91\%$) or Joint MTL ($95.55\%$), which indicates that the consolidated model is not yet deployable for high-stakes applications.
6. **Task Vector Normalization (TV-Norm):** Pre-masking normalization balances task representation, dramatically increasing MNIST accuracy from $36.74\%$ to $53.70\%$ ($+16.96\%$ absolute) under GQ-Norm, though at the expense of SVHN accuracy ($85.35\%$ down to $70.18\%$). Scaling the validation calibration pool size from $N_{\text{val}}=10$ to $20$ reduces the standard deviation of GQ-Norm from $\pm 4.56\%$ to $\pm 1.10\%$.
7. **Landscape Stabilization via Soft Masking:** Continuous sigmoid-gated soft masking (SG-TA GQ-Soft) stabilizes OFS-Tune calibration, halving the standard deviation to **$\pm$ 0.75%** (compared to $\pm 1.39\%$ for hard GQ) while maintaining a high Joint Mean Accuracy of $61.06\%$.
8. **Scalable Calibration:** Non-Uniform Coordinate Search (CS) optimizes task-specific parameters in linear time $\mathcal{O}(T)$ (taking 43.61s for 100 evaluations), rebalancing joint performance by boosting MNIST to $50.38\%$ and CIFAR-10 to $75.04\%$.

## 4. Explicitly Claimed Contributions
1. **Decoupled weight-space sparsification framework (SG-TA):** Proposes a simple, deterministic, magnitude-based binary masking paradigm.
2. **Analysis of masking scopes:** Formulates and contrasts Global Quantile (GQ) and Layer-wise Quantile (LQ) masking, highlighting the importance of global budget flexibility in Transformers.
3. **Magnitude Imbalance Mitigation:** Proposes Task Vector Magnitude Normalization (TV-Norm) and evaluates calibration sensitivity sweeps across validation pool sizes.
4. **Non-Uniform High-Dimensional Calibration:** Proposes and validates Non-Uniform Coordinate Search (CS) as a highly scalable linear-time optimizer.
5. **Continuous Soft Masking:** Proposes and validates Sigmoid-Gated Soft Masking (SG-TA-Soft) to stabilize the calibration landscape.
6. **Rigorous Empirical Characterization:** Conducts exhaustive sweeps across 5 random seeds, reporting standard deviations, cosine similarities of task updates, and a physical Joint Multi-Task Learning (MTL) baseline.
