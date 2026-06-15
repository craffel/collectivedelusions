# 4_experiment_check.md: Experimental Rigor and Evaluation Critique of the Revised Paper

From the perspective of an empirically-driven researcher who believes that true progress comes from exhaustive, scale-controlled validation, large-scale sweeps, and robust statistical analysis, the experimental section of this paper is extremely weak and fails to meet standard academic benchmarks.

## 1. Experimental Setup and Datasets
While the authors select a diverse four-task Vision Transformer benchmark, the training and evaluation protocol is highly compromised:
- **Catastrophic Accuracy Collapse across the board:** The absolute accuracy of all merged models is incredibly low (around 17%-25%), which is barely better than random guessing (10%) on a 10-class dataset, and significantly worse than the specialist expert upper bound of 62.40%. An accuracy of 25.40% mean accuracy is not practically usable, and the claim that it "bridges the performance gap to specialist experts" is false and highly misleading.
- **Low-Data Calibration Constraints:** Calibrating on 16 samples per task is an interesting low-data benchmark, but the optimization parameters (Adam, learning rate of 1e-2, 100 steps) are hardcoded without any hyperparameter sweep or ablation to justify why these specific training parameters were chosen.

## 2. Baseline Selection and Fairness
The authors compare TCPR against a set of 7 baselines (Uniform Merge, Linear Router, BL-Router, BL-Router with L2, BSigmoid-Router, BSigmoid-Router with L2, and QWS-Merge). While the breadth of baselines is visually appealing, their evaluation and tuning are unfair:
- **Under-tuned Baselines:** Baselines are trained with fixed optimization hyperparameters that might have been selected specifically to favor the proposed TCPR method.
- **QWS-Merge Representation:** QWS-Merge is represented as the state-of-the-art wave-inspired method, but there is no proof or literature verification that their custom implementation of QWS-Merge in `run_experiments.py` is optimal or corresponds to the official implementation of QWS-Merge. It seems to have been implemented in a simplified form that might perform poorly on purpose.

## 3. Results vs. Claims Analysis
The empirical results **do not support the claims** made in the paper. The claims are highly inflated and rely on statistically insignificant differences:

### A. The "Zero-Improvement" SOTA of TCPR-Param
In Table 1, the authors report:
- **BSigmoid-Router (Unreg):** MNIST 34.80%, FashionMNIST 26.00%, CIFAR-10 30.00%, SVHN 10.80% — Joint Mean **25.40%**.
- **TCPR-Param (Ours) ($\beta = 10^{-4}$):** MNIST 34.80%, FashionMNIST 26.00%, CIFAR-10 30.00%, SVHN 10.80% — Joint Mean **25.40%**.

Let's calculate the exact number of correct test predictions across 1000 total test samples (250 samples per task $\times$ 4 tasks):
1. **BSigmoid-Router (Unreg):** Correct predictions = $254$ out of $1000$.
2. **TCPR-Param (Ours):** Correct predictions = $254$ out of $1000$.

This is a devastating empirical result:
- **TCPR-Param** achieves **exactly zero improvement** over the simple, unregularized BSigmoid-Router baseline. Every single prediction on the test set is identical down to the decimal point because the regularizer had no effect during training.
- Despite this, the paper lists both as superior, stable, and state-of-the-art variants that "surpass standard L2-regularized baselines and state-of-the-art wave-interference methods by a substantial margin."

### B. Performance Degradation of TCPR-Rep
In Table 1, the authors report:
- **BSigmoid-Router (Unreg):** MNIST 34.80%, FashionMNIST 26.00%, CIFAR-10 30.00%, SVHN 10.80% — Joint Mean **25.40%**.
- **TCPR-Rep (Ours) ($\beta = 10^{-4}$):** MNIST 21.60%, FashionMNIST 20.00%, CIFAR-10 34.00%, SVHN 11.20% — Joint Mean **21.70%**.

This shows that the representation-space prior variant actually **degrades** performance compared to the unregularized baseline by $3.70\%$ in absolute joint accuracy. It even performs worse than the simple isotropic L2-regularized `BSigmoid-Router (Reg)` baseline (24.00%).

### C. Lack of Multi-seed / Variance Analysis
To claim a robust, scale-invariant improvement, a rigorous empirical paper must provide average accuracies and standard deviations computed over multiple independent runs (different calibration splits, different test splits, and different model seeds). Reporting a single run on a 250-sample test split is highly deceptive, as any minor differences are completely within the margin of random noise.
