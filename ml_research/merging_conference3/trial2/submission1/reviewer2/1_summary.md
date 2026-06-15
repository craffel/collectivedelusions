# Summary of the Paper

## Main Topic
The paper addresses **test-time model merging**, a paradigm for combining multiple task-specific expert neural networks (fine-tuned from a shared pre-trained initialization) into a single, unified multi-task model without needing original training data or retraining. Specifically, the paper analyzes **AdaMerging**, a state-of-the-art adaptive merging method that optimizes layer-wise merging coefficients via unsupervised test-time entropy minimization on small calibration streams.

## Proposed Approach
The authors deconstruct the adaptive test-time merging framework and expose two severe, under-reported failure modes:
1. **The Overfitting-Optimizer Paradox (Transductive Overfitting):** Fine-grained layer-wise coefficients overfit to the local statistics of small test-time calibration batches, rather than capturing genuine, spatial layer interactions.
2. **Sacrificial Task Bias:** Joint entropy optimization heavily degrades performance on complex, high-entropy tasks (e.g., SVHN) to prioritize easier, low-entropy domains (e.g., MNIST).

To systematically resolve these issues, the paper introduces **RegCalMerge** (Calibrated & Regularized Test-Time Model Merging), consisting of:
- **CalMerge (Calibration Engine):**
  - *Class-Capacity Normalization (CCN):* Scales raw prediction entropy by its theoretical maximum ($\log C_k$) to map entropy from disparate domains onto a uniform $[0, 1]$ interval.
  - *Scale-Normalized Entropy Weighting (SNEW):* Computes task weights as the inverse of the initial uniform task-arithmetic entropy at step $0$, ensuring that complex tasks contribute equally to joint gradients.
- **Elastic Spatial Regularization (ESR):** An optional structural stabilizer with two penalties:
  - *Proximity Penalty ($\beta$):* Penalizes parameter deviation from robust uniform Task Arithmetic initialization.
  - *Spatial Deviation Penalty ($\gamma$):* Penalizes variance of layer-wise coefficients around their task-wise spatial average, smoothing high-frequency optimization noise.

## Key Findings & Claims
1. **Validation of Overfitting-Optimizer Paradox:** Introducing a *spatial shuffling diagnostic* (where optimized coefficients are shuffled across layers) shows that the shuffled coefficients retain nearly 95% of the performance gains over Task Arithmetic, proving that layer-wise optimizations behave primarily as unconstrained parameter-drift mechanisms fitting calibration noise.
2. **Resolution of Sacrificial Task Bias:** The proposed unregularized calibration engine **CalMerge** ($\beta=0, \gamma=0$) completely resolves the task-sacrifice issue on SVHN, elevating its accuracy to a peak of **32.03%** and achieving a state-of-the-art Joint Mean accuracy of **61.82%** across MNIST, FashionMNIST, CIFAR-10, and SVHN on a ViT-B/32 backbone.
3. **Necessity of Layer-wise Flexibility:** Comparing *CalMerge* against a *Calibrated Spatial Mean (Cal-Mean)* baseline (which optimizes only 1 scalar per task with SNEW/CCN calibration) shows CalMerge outperforming Cal-Mean (61.82% vs 61.13%), demonstrating that fine-grained layer-wise flexibility is indeed useful but requires proper calibration.
4. **Generalization-Regularization Trade-off:** Increasing ESR weights ($\beta, \gamma$) leads to a smooth, predictable trade-off in peak local accuracy to guarantee global weight-space stability, acting as a stable parameter-safety dial.
5. **Heterogeneous Label Spaces:** Validates CCN and SNEW under unequal class counts ($C \in \{3, 5, 8, 10\}$) through an additional empirical simulation.

## Explicitly Claimed Contributions
1. Exposing the Overfitting-Optimizer Paradox via a novel spatial shuffling diagnostic.
2. Identifying and analyzing the Sacrificial Task Bias in multi-task entropy objectives.
3. Proposing **RegCalMerge**, featuring a calibration engine (CalMerge) and a regularization stabilizer (ESR) to resolve both vulnerabilities.
4. Providing large-scale, multi-seed empirical grid sweeps (including 7 baselines and a calibrated spatial mean baseline) to map out the generalization surface of test-time model merging.
