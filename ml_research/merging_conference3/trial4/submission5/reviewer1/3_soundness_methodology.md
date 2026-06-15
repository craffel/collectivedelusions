# Intermediate Review File 3: Soundness and Methodology Evaluation

## 1. Clarity of Description
The methodology is exceptionally well-written, clear, and mathematically rigorous:
* Every variable, equation, and procedure is formally defined, starting from the task vector definition up to the continuous soft masking.
* The distinction between Global Quantile (GQ) and Layer-wise Quantile (LQ) masking is described with clear mathematical notations.
* The descriptions of the extensions (TV-Norm, Coordinate Search, SG-TA-Soft) are logically structured and consistent with standard literature.

## 2. Appropriateness of Methods
* **Sparsification via Magnitude:** Utilizing absolute magnitude to determine parameter importance is highly appropriate and grounded in deep learning pruning literature.
* **Global vs. Layer-wise Budgets:** Investigating these two scopes is highly appropriate for Transformer architectures, which feature highly heterogeneous components (attention projection matrices vs MLP layers). Enforcing a rigid homogeneous budget (LQ) ignores this heterogeneity.
* **OFS-Tune (Offline Few-Shot Validation Tuning):** This method is highly appropriate for zero-shot merging as it avoids updating model parameters, which bypasses the "Overfitting-Optimizer Paradox" of unsupervised test-time adaptation.
* **Task Vector Magnitude Normalization (TV-Norm):** Addressing the absolute magnitude imbalance across tasks is highly appropriate. Task vectors from simpler tasks (MNIST) are often smaller than those from complex domains (SVHN), making normalization essential for fair merging.
* **Non-Uniform Coordinate Search (CS):** A coordinate descent-style zero-order optimizer is an appropriate, highly scalable $\mathcal{O}(T)$ alternative to exponential grid sweeps or noisy random searches in high dimensions.
* **Sigmoid-Gated Soft Masking (SG-TA-Soft):** Smoothing the non-differentiable hard binary threshold is highly appropriate to stabilize hyperparameter calibration.

## 3. Potential Technical Flaws and Critiques (Empiricist Perspective)
Despite the overall rigor, several aspects of the methodology warrant critical empirical scrutiny:

* **Inadequacy of the Default $N_{\text{val}}=10$ Calibration Size:**
  The default OFS-Tune calibration relies on $N_{\text{val}} = 10$ samples per task. This is extremely small and highly susceptible to small-sample validation noise. The authors' own results in Table 1 show that under $N_{\text{val}}=10$, the standard deviation of SG-TA (GQ-Norm) is a massive **$\pm$ 4.56%** (ranging from highly successful to volatile configurations). While they perform a validation pool size sweep (revealing that $N_{\text{val}}=20$ immediately slashes variance by more than 4x to $\pm 1.10\%$), the main paper still lists the volatile $N_{\text{val}}=10$ as the primary baseline. Empirically, using 10 samples seems unnecessarily restrictive when 20 samples still qualify as "few-shot" and provide a dramatically more stable landscape. The default baseline should be shifted to $20$ or $50$.
* **Unproven "Layer-Starvation" Hypothesis for the GQ-LQ Crossover:**
  In Section 4.3, the authors observe that at larger keep-ratios ($k \ge 0.7$), LQ masking begins to outperform GQ. They hypothesize that because GQ is unconstrained, at large $k$ it may allow certain layers to become completely dense while others are excessively pruned (starved), disrupting global information flow. While this is a plausible explanation, the authors **do not provide any empirical proof** (such as a plot or table of the layer-wise keep-ratios under GQ at $k=0.7$) to back it up. An empiricist would expect to see the actual distribution of layer budgets to confirm this "layer-starvation" hypothesis.
* **Sequence Dependency in Coordinate Search:**
  Section 3.6.2 notes that Coordinate Search is executed in the default order of MNIST $\rightarrow$ FashionMNIST $\rightarrow$ CIFAR-10 $\rightarrow$ SVHN. While the authors run a control sweep in reverse order and report stable performance ($58.28\% \pm 2.45\%$ vs. $58.40\% \pm 2.32\%$), coordinate descent is theoretically sensitive to the initial coordinate and the optimization path. Since there are only 4 tasks, evaluating all $4! = 24$ permutations or a larger randomized sample of sequences would be required to statistically prove sequence insensitivity.
* **The "Oracle Routing" Assumption Simplification:**
  The paper assumes a "test-time task routing oracle" that routes input samples to their corresponding task-specific head. While standard in contemporary model merging literature, this is a major structural simplification. In practical deployments, the task label is unknown. The authors discuss zero-shot routing heuristics, but evaluating the unified backbone under a real routing model (or reporting joint accuracy under a shared head) would be far more convincing and realistic.

## 4. Reproducibility
* **Excellent Algorithmic Description:** The algorithmic steps, mathematical equations, and search spaces are highly detailed.
* **Clear Hyperparameters:** The model backbone (`vit_tiny_patch16_224`), training details (2 epochs, AdamW, learning rate $10^{-3}$, weight decay $0.01$, batch size 256), and grid search ranges are fully specified.
* **Code Availability:** The authors claim to "provide reproducible code," but no code files or repository links are actually provided in the submission workspace. This is a minor limitation for verifying immediate reproducibility.
