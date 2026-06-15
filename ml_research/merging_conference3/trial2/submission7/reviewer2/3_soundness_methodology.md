# Evaluation Phase 3: Soundness and Methodology Evaluation

## Clarity of Description and Completeness
The mathematical description of **ThermoMerge** in Section 3 is elaborate and detailed, utilizing extensive equations to map neural network outputs to canonical Boltzmann distributions and Helmholtz Free Energy. However, there are significant gaps in clarity, logical inconsistencies, and potential technical flaws that undermine the soundness of the methodology.

## Potential Technical Flaws and Methodological Concerns

### 1. Degenerate Optimization Trajectory of Trainable Temperature $T_k$
The paper introduces trainable task-wise thermal capacities $\tau_k$ that scale the task-specific temperatures $T_k(t) = \tau_k \cdot T(t)$. This parameter is optimized jointly with the merging coefficients $\boldsymbol{\Lambda}$ by minimizing the Free Energy Discrepancy (temperature-scaled KL divergence).
- **The Degeneracy Flaw:** For any two discrete probability distributions $p$ and $q$ derived from logits $f_1$ and $f_2$ under temperature $T_k$, as $T_k \to \infty$, both distributions flatten out and converge to the uniform distribution. Consequently, the KL divergence $\mathcal{D}_{KL}(p \parallel q)$ converges to **exactly zero**.
- **Optimizer Incentive:** Because the optimization objective is to minimize this KL divergence, the gradient descent optimizer has a trivial, degenerate incentive to drive $\tau_k \to \infty$ (or to its maximum allowed clamp) to flatten the distributions and artificially minimize the loss, completely bypassing the need to actually align the representations of the merged model and the experts.
- **The Hard Clamp Patch:** The authors acknowledge this numerical danger and apply a hard clamp of $\tau_k \in [0.2, 5.0]$. However, they fail to report the final optimized values of $\tau_k$ in their experiments. It is highly probable that the optimized $\tau_k$ simply hits the upper bound of $5.0$ for all tasks, rendering the "optimization" of task-wise temperatures degenerate and meaningless. The paper lacks a rigorous analysis or visualization of the optimized $\tau_k$ trajectories to prove otherwise.

### 2. Serious Inconsistencies in Hyperparameters
There is a direct contradiction in the paper regarding the core hyperparameters of the Thermodynamic Annealing Schedule (TAS):
- **Section 3.4 (Methodology):** *"In our experiments, we set $T_{start} = 5.0$, $T_{end} = 1.0$, and $\beta = 0.05$."*
- **Section 4.5, Table 3 (Appendix), and Section 5.1 (Sensitivity Analysis):** The optimal hyperparameters are listed as $T_{start} = 2.0$ and $\beta = 0.40$. Section 4.5 explicitly references these as the *"optimal hyperparameter configuration ($T_{start}=2.0, \beta=0.40$)"* used to resolve convergence issues on FashionMNIST.
- **Methodological Concern:** It is completely unclear which set of hyperparameters was actually used to generate the main empirical results in Table 1. If $T_{start} = 5.0$ and $\beta = 0.05$ were used, then the main results are suboptimal. If $T_{start} = 2.0$ and $\beta = 0.40$ were used, then the description in Section 3.4 is incorrect and misleading. This inconsistency severely harms the reproducibility of the work.

### 3. Unrealistic "Test-Time Adaptation" Data Scale
The authors employ a "true sequential streaming TTA protocol" where they draw a fresh batch of 128 images per dataset at each optimization step, running for 100 steps.
- **Massive Data Overhead:** This protocol consumes a total of $128 \text{ images/step} \times 100 \text{ steps} = 12,800$ images per task. Across 4 tasks, this requires **51,200 unlabeled images** during inference!
- **Setting Incoherency:** Standard test-time adaptation (TTA) is intended to be a highly lightweight, online adaptation phase using a small calibration set (e.g., 64 to 256 samples total). Consuming 12,800 images per task is closer to a full-blown unsupervised transductive learning or fine-tuning phase than a practical, lightweight test-time adaptation.
- **Dataset Size Violation:** Three of the evaluated datasets (MNIST, FashionMNIST, CIFAR-10) have test sets of exactly **10,000 images**. To stream 12,800 "fresh, unlabeled test images," the authors must be looping over the test set, drawing samples with replacement, or using training set images. This invalidates the claim of a realistic, non-looping streaming TTA setting.

### 4. Severe Lack of Baseline Expert Accuracies
The paper reports the merged model accuracies in Table 1 but **completely omits the original accuracies of the individual task experts** before merging.
- **The Collapse Concern:** Under ResNet-18, ThermoMerge's MNIST accuracy is **20.00%** and FashionMNIST accuracy is **32.60%**. For a standard pre-trained ResNet-18, individual expert models fine-tuned on MNIST and FashionMNIST easily achieve accuracies of **99%+ and 92%+** respectively. 
- **Methodological Lack of Transparency:** A merged model accuracy of 20% on MNIST is a catastrophic collapse of nearly 80% in absolute performance compared to the expert. By hiding the expert baseline accuracies, the authors obscure the absolute failure of all merging methods to preserve grayscale task representations. The "superior multi-task average" of 29.05% is likely just a slightly less catastrophic collapse compared to Task Arithmetic (27.25%), rather than a "robust thermodynamic fusion."

## Reproducibility
While the authors provide the model architecture (Table 2) and some hyperparameters (Table 3), reproducibility is severely compromised by:
1. The hyperparameter contradictions between Sections 3.4 and 4.5.
2. The lack of explanation of how 12,800 "fresh" samples were drawn from 10,000-sample test sets.
3. The omission of the individual expert training details (learning rates, epochs, and final baseline accuracies).
