# Evaluation Step 4: Experimental Evaluation Check

A highly critical review of the empirical validation in Section 4 reveals several significant concerns, unfair comparisons, and hidden weaknesses in the results:

## 1. Unfair Comparison: Supervised vs. Zero-Shot Baselines
- **The Setup:** The paper compares RBPM directly against zero-shot weight-space merging heuristics: Static Uniform, TIES-Merging, DARE-Merging, and Sparse Task Arithmetic.
- **The Issue:** TIES, DARE, and Sparse Task Arithmetic are completely **zero-shot and unsupervised**. They do not use any calibration data or run any optimization. In contrast, RBPM uses a labeled calibration dataset of $M = 10$ samples per task (40 total images) and runs 30 steps of supervised Adam optimization.
- **The Criticism:** Comparing a supervised, calibrated method against zero-shot, data-free heuristics is a classic "apples-to-oranges" comparison. RBPM's performance gain is heavily aided by the information advantage of having labeled calibration samples. The authors must explicitly decouple this data advantage, acknowledging that RBPM is not a data-free ensembling method. The only strictly fair baselines are Offline Unconstrained Few-Shot Tuning and Globally-Scaled Task Arithmetic ($d=0$), which utilize the same calibration data.

## 2. The Illusion of Average Accuracy (Severe Task Dominance)
- **The Setup:** In Table 1 (CNN backbone), the authors report a robust average accuracy of **38.85%** for RBPM vs. **29.05%** for Static Uniform, claiming "superior generalization."
- **The Issue:** A close inspection of the individual task columns in Table 1 reveals an alarming trend:
  - **Task 0 (MNIST):** RBPM achieves **75.20%** (+44.20% over Uniform's 31.00%).
  - **Task 1 (FashionMNIST):** RBPM achieves **48.60%** (**-2.00%** under Uniform's 50.60%).
  - **Task 2 (CIFAR-10):** RBPM achieves **17.20%** (**-2.40%** under Uniform's 19.60%).
  - **Task 3 (SVHN):** RBPM achieves **14.40%** (**-0.60%** under Uniform's 15.00%).
- **The Criticism:** The claimed "+9.80% average accuracy gain" is a complete illusion. In 3 out of 4 tasks, **RBPM actually degrades performance** compared to simple, zero-optimization Static Uniform merging! The average gain is driven entirely by MNIST, which has steep, clean gradients that dominate the joint few-shot optimization. The optimizer heavily overfits the ensembling trajectory to MNIST, destroying the shared representation pathways of the other three tasks.
- **Gradient Surgery (PCGrad):** While the authors implement **RBPM + PCGrad** to mitigate this (improving FashionMNIST to 58.60%), the resulting average accuracy drops to **35.70%**. Furthermore, CIFAR-10 (16.60%) and SVHN (13.20%) remain extremely poor—barely above random guessing (10%) and representing severe degradation compared to their individual expert ceilings (CIFAR-10: 31.00%, SVHN: 21.40%). This demonstrates that weight-space ensembling on highly heterogeneous domains is fundamentally flawed, and RBPM's capacity control does not solve this incompatibility.

## 3. Marginal Performance Gain Over Unsupervised TTA on ViTs
- **The Setup:** In Table 2 (CLIP ViT-B/16), the authors report results on Stanford Cars and Oxford Flowers.
- **The Issue:** Let's compare:
  - **Online PolyMerge ($d=2$, unsupervised online TTA):** **84.30%** average accuracy.
  - **RBPM (Ours, supervised few-shot calibrated):** **85.15%** average accuracy.
- **The Criticism:** The difference is a marginal **+0.85%** absolute. Online PolyMerge is completely unsupervised, requires zero labeled validation data, and runs on-the-fly. The fact that RBPM requires a labeled validation dataset, supervised calibration, and offline optimization only to yield an extra 0.85% gain significantly undermines its practical utility and cost-effectiveness in real-world deployment.

## 4. Unexplained Omission of the CUB-200 Dataset
- **The Setup:** In Section 7.1 ("Experimental Protocol for Scalability on Vision Transformers"), the authors define the ViT benchmark using three datasets: Stanford Cars, Oxford Flowers, and **CUB-200-2011 (Birds)**.
- **The Issue:** Table 2 only presents results for Stanford Cars and Oxford Flowers. **CUB-200-2011 is completely missing from the empirical results.**
- **The Criticism:** Omitting CUB-200-2011 without any explanation raises a major red flag. Was the dataset omitted because RBPM failed to generalize, or did it underperform the baselines? This represents a significant gap in the completeness of the experimental evaluation and suggests potential selective reporting of results.

## 5. Weak Individual Expert Ceilings
- **The Setup:** In Section 4.1, the fine-tuned expert test accuracies are reported as: MNIST: 89.40%, FashionMNIST: 71.80%, CIFAR-10: 31.00%, and SVHN: 21.40%.
- **The Issue:** A test accuracy of **31.00%** on CIFAR-10 and **21.40%** on SVHN is incredibly low for a 12-layer CNN. Even shallow architectures should easily achieve >70% on CIFAR-10 and >80% on SVHN.
- **The Criticism:** The individual task experts are extremely under-trained and weak. Merging models that have barely learned task-specific representations is not representative of practical model merging scenarios (where researchers merge highly converged, state-of-the-art experts). This limits the realism and generalizability of the paper's CNN empirical findings.
