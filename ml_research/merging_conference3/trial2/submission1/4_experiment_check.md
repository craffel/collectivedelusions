# 4. Experimental Check and Verification

We conduct a thorough assessment of the experimental setup, baselines, and empirical results reported in the paper, identifying several critical limitations and constructive suggestions.

## 1. Small Evaluation Split Size and Sample Noise (Critical Empirical Weakness)
The authors evaluate all model merging methods on extremely restricted test splits consisting of only 2 batches of size 128 (256 images total per dataset):
- **The Issue:**
  MNIST, FashionMNIST, and CIFAR-10 have test sets of 10,000 images, and SVHN has 26,032 images.
  Evaluating on just 256 images per domain represents a tiny fraction (1% to 2.5%) of the standard test sets.
- **The Consequence:**
  With such small test splits, accuracy estimates can be highly sensitive to sample noise. Observed differences of 0.5% or 1% might not reflect true generalization performance on the full test sets.
- **Recommendation:**
  While keeping the calibration batch small (e.g., $N=16$ images) to reflect data-efficient test-time adaptation, the final evaluation of the frozen merged models should be conducted on the **entire** test set of each dataset. This would completely eliminate sample-split variance and provide robust, publication-grade accuracy figures.

---

## 2. Deterministic Seeds vs. Data-Sampling Variance (Limitation of Multi-Seed Setup)
In Table 1, several columns (such as Adam GD, Task Arithmetic, and RegCalMerge) report a standard deviation of $\pm0.00\%$ across three random seeds:
- **The Cause:**
  As the authors correctly identify in Section 4.4.1, first-order gradient descent is completely deterministic when both the calibration split and evaluation split are fixed in memory. Therefore, changing the random seed of PyTorch has no effect on the optimization trajectory.
- **The Limitation:**
  Because the seeds do not change the calibration data, this multi-seed evaluation does not capture **data-sampling variance**—i.e., how robust the method is to *which* 16 calibration images are selected at test-time.
- **Recommendation:**
  To perform a true multi-seed evaluation of test-time adaptation, the authors should sample **different random calibration batches** across different seeds. This would produce realistic standard deviations for all adaptive methods (including Adam GD) and prove that CalMerge and RegCalMerge are robust to the specific samples in the test-time adaptation stream.

---

## 3. Discrepancy in Baseline Execution: "Calibrated Spatial Mean (Cal-Mean)" (Implementational Gap)
The paper introduces and heavily discusses Method 9, **Calibrated Spatial Mean (Cal-Mean)** (yielding 61.13% Joint Mean in Table 1), as a key baseline to show that spatial degrees of freedom are necessary even when calibrated:
- **The Gap:**
  In the codebase, the main script `run_regcalmerge.py` (which produces `results/metrics.json` and the main experimental numbers) **does not actually execute** Calibrated Spatial Mean (Cal-Mean) on the homogeneous setup. It only evaluates uncalibrated Spatial Mean using ES (Method 4).
- **The Heterogeneous Simulation:**
  The Cal-Mean baseline is implemented and executed in the specialized `run_heterogeneous_experiment.py` script for the heterogeneous label space simulation. However, for the main homogeneous table (Table 1), there is no code block running Cal-Mean.
- **Impact:**
  This is a reproducibility discrepancy: the manuscript reports numbers for Method 9 in the main results table, but the main experimental pipeline script does not support or execute this run. The authors should integrate Cal-Mean directly into the main `run_regcalmerge.py` script.

---

## 4. Sensitivity to Calibration Stream Volume
The paper evaluates a single calibration batch size of $N=16$ per dataset.
- Since the core thesis of the paper is that standard adaptive merging overfits to the local calibration statistics, a sweep over the calibration batch size (e.g., $N \in \{8, 16, 32, 64, 128\}$) would be highly valuable. 
- It would demonstrate how the overfitting paradox and the benefit of ESR change as more test-time data becomes available. One would expect that as $N$ increases, the overfitting paradox diminishes and ESR's performance gap to CalMerge narrows. Adding such a sweep would greatly enrich the empirical depth of the paper.
