# Reviewer Report: 1_summary.md

## 1. Summary of the Paper
The paper, titled **"Sparsity-Guided Task Arithmetic: Decoupled Weight Masking for Interference-Free Model Merging"**, presents a simple and deterministic weight-space regularization framework called **Sparsity-Guided Task Arithmetic (SG-TA)** to mitigate representational collisions in model merging. Standard model merging, which linearly combines task-specific update vectors (task vectors), often suffers from severe interference in weight space, leading to catastrophic performance drops. SG-TA decouples the sparsification phase by applying magnitude-based binary masks to individual task-specific update vectors before merging, filtering out low-magnitude updates that act as orthogonal parameter noise.

The authors evaluate two masking scopes:
1. **Global Quantile (GQ) Masking:** A single magnitude threshold is computed globally across the entire model for each task vector, allowing different layers to retain varying percentages of parameters.
2. **Layer-wise Quantile (LQ) Masking:** Independent magnitude thresholds are computed for each layer, enforcing a homogeneous parameter keep-ratio across all layers.

Additionally, the paper proposes:
* **Task Vector Magnitude Normalization (TV-Norm):** Scaling each task vector by the inverse of its mean absolute magnitude prior to masking to resolve task vector scale imbalance and prevent magnitude-dominant tasks from over-writing weaker ones.
* **Sigmoid-Gated Soft Masking (SG-TA-Soft):** Continuous, differentiable sigmoid-gated task vector updates to avoid representational discontinuities from hard binary masking and stabilize the multi-task landscape.
* **Non-Uniform Calibration Alternatives:** Task-specific Coordinate Search (CS) and Random Search (RS) to explore non-uniform task-specific keep-ratios $k_i$ and scaling factors $\alpha_i$.

For hyperparameter tuning, they employ **Offline Few-Shot Validation Tuning (OFS-Tune)** with a tiny validation split (e.g., 10 samples per task), which avoids the overfitting-optimizer paradox. 

### Experimental Setup
* **Backbone:** Vision Transformer (`vit_tiny_patch16_224`, ~5.7M parameters).
* **Datasets:** A 4-dataset multi-domain visual classification benchmark: MNIST, FashionMNIST, CIFAR-10, and SVHN.
* **Key Baselines Compared:** Naive Uniform TA, Optimized TA, TIES-Merging, DARE-Merging, Decoupled Prune-then-Merge (P-then-M), Layer-Group Scaling (L-Scale), Fisher-Weighted Averaging, and Joint Multi-Task Learning (MTL) upper bound.

### Key Findings and Results
1. **SG-TA (GQ) Superiority:** SG-TA under global quantile masking (GQ) achieves an average joint mean accuracy of **61.40% ± 1.39%**, which is a $+15.08\%$ absolute improvement over Naive Uniform TA ($46.32\%$) and $+2.17\%$ over Optimized TA ($59.23\% \pm 2.08\%$). It outscores TIES-Merging ($60.64\% \pm 1.30\%$) and DARE-Merging ($58.44\% \pm 3.02\%$), though the authors scientifically acknowledge the overlap in standard deviation with TIES-Merging.
2. **Global Budget Flexibility is Crucial:** GQ masking ($61.40\%$) consistently outperforms Layer-wise Quantile (LQ) masking ($57.81\% \pm 2.52\%$), confirming that enforcing homogeneous layer budgets is highly sub-optimal and that allowing task-sensitive layers (e.g., attention projections) to retain more updates globally is essential.
3. **Continuous Gating Stabilizes Learning:** SG-TA (GQ-Soft) achieves **61.06%** joint accuracy and dramatically reduces standard deviation across calibration seeds from $\pm 1.39\%$ (hard) to $\pm 0.75\%$ (soft), confirming that sigmoid gating smooths the landscape and stabilizes hyperparameter selection.
4. **Pre-masking Normalization Balances Tasks:** Incorporating TV-Norm (GQ-Norm) increases MNIST accuracy from $36.74\%$ to $53.70\%$, resolving task vector dominance. Increasing calibration samples from 10 to 20 or 100 cut variance by over 4x and improved performance to **63.90% ± 1.47%**.
5. **Absolute Performance Degradation:** Despite strong relative improvements over unregularized baselines, a massive performance gap remains between the best merged model ($61.40\%$) and the Joint MTL upper bound ($95.55\%$) / Dense Expert Ceiling ($95.91\%$), indicating a severe capacity constraint in compact architectures like ViT-Tiny.
