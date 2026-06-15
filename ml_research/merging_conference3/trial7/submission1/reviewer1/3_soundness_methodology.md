# Soundness and Methodology Report (`3_soundness_methodology.md`)

## 1. Evaluation of Clarity of Description
The methodology is written in a seemingly rigorous and clear mathematical style. The authors define their symbols clearly, provide detailed step-by-step formulations of their BSigmoid router, and present pseudocode for the SVD diagnostic in the appendix. However, this dense mathematical presentation masks several severe conceptual and mathematical flaws that weaken the overall soundness of the work.

## 2. Potential Technical and Mathematical Flaws

### A. The "Gradient Gating Decoupling" Argument is Mathematically Flawed
In Section 4.4, the authors argue that their Bounded Sigmoid (BSigmoid) router avoids the zero-sum competitive gradient clashing of standard Softmax because it "decouples" the gradient paths during calibration. They state that the independent sigmoids have gradients local to their respective task paths:
$$\frac{\partial \tilde{\alpha}_i}{\partial \alpha_i} = \sigma(\alpha_i)(1 - \sigma(\alpha_i))$$
This is a mathematically deceptive argument. While the *pre-normalized* activation $\tilde{\alpha}_i$ has a local derivative, the actual routing coefficients $\lambda_i$ used to merge the model weights are normalized via a post-gating division (Equation 10):
$$\lambda_{l, k}(x) = \frac{\tilde{\alpha}_{l, k}(x)}{\sum_{j=1}^K \tilde{\alpha}_{l, j}(x) + \epsilon}$$
Because of this division, the derivative of any routing coefficient $\lambda_k$ with respect to *any* input logit $\alpha_i$ is heavily coupled. Specifically, by the quotient rule, for $i \neq k$:
$$\frac{\partial \lambda_k}{\partial \alpha_i} = -\sigma'(\alpha_i) \frac{\tilde{\alpha}_k}{\left(\sum_j \tilde{\alpha}_j\right)^2}$$
And the overall gradient of the loss $\mathcal{L}$ with respect to logit $\alpha_i$ is:
$$\frac{\partial \mathcal{L}}{\partial \alpha_i} = \sum_k \frac{\partial \mathcal{L}}{\partial \lambda_k} \frac{\partial \lambda_k}{\partial \alpha_i}$$
This gradient is mathematically coupled across all task dimensions $k$, exactly like standard Softmax. The normalization re-introduces the competitive zero-sum constraint at both the forward and backward (gradient) levels. The authors' claim that BSigmoid "decouples gradient paths" and "protects individual expert updates from being overridden" is mathematically incorrect. The observed difference in optimization behavior must be due to other factors (such as the slower growth rate of Sigmoids compared to exponentials), not gradient decoupling.

### B. Overstatement of the SVD Collinearity Ratio
The authors define the Collinearity Ratio as:
$$\rho_{collinear} = \frac{\sigma_1}{\sum_{i=1}^{\min(L, K)} \sigma_i}$$
Let us critically evaluate the bounds of this ratio:
- For $K=2$ tasks, the denominator is $\sigma_1 + \sigma_2$. The absolute minimum possible ratio is **0.5** (when $\sigma_1 = \sigma_2$).
- For $K=4$ tasks, the absolute minimum possible ratio is **0.25** (when $\sigma_1 = \sigma_2 = \sigma_3 = \sigma_4$).

In Table 3, the authors report:
- For DeepMLP-12 ($K=2$): $\rho_{collinear} = 0.74 \pm 0.04$ (Low-Conflict) and $0.65 \pm 0.04$ (High-Conflict).
- For TinyCNN-4 ($K=2$): $\rho_{collinear} = 0.66 \pm 0.05$ (Low-Conflict) and $0.64 \pm 0.03$ (High-Conflict).

A ratio of $0.64$ to $0.74$ for $K=2$ is actually extremely high, indicating that the first singular value heavily dominates. In fact, a ratio of $0.65$ means that $\sigma_1$ is nearly **double** the magnitude of $\sigma_2$ ($\sigma_1 \approx 1.86 \sigma_2$). Even for the Cross-Domain suite ($K=4$), where the minimum is $0.25$, the ratios are $0.50$ (MLP) and $0.57$ (CNN)—meaning the first singular value still accounts for more than half of the total singular value energy.
Thus, the routing coefficients still exhibit **strong collinearity**. The authors' assertion that these results "completely deconstruct" the rank-1 collapse theorem is a major overstatement of their empirical data.

### C. Extreme Fragility of the Few-Shot Calibration Setup
The authors' evaluation relies on an incredibly small calibration budget (128 samples per task, 40 optimization steps). Because of this tight limit, the router is highly sensitive to over-parameterization:
- If the projection dimension $d$ is increased beyond 8, the model immediately overfits, and performance collapses.
- If the projection matrix is made learnable, parameters increase by 4,300%, and generalization collapses to $26.12\%$ on TinyCNN-4 (Table 2 performance is $52.52\%$).
This extreme fragility suggests that the proposed method is not robust. It only "works" under highly contrived hyperparameters (like $d=8$ with frozen random weights) and fails to scale to standard representation-learning dimensions.

### D. Questionable Training Setup for Single-Task Experts
The task-specific experts are trained on subsets of only **512 training samples** from MNIST. While the authors claim these experts reach $>99\%$ accuracy, training deep models on such tiny subsets is highly non-standard and prone to high variance. It is highly likely that these experts are severely underfitted or highly specialized in a narrow, unrepresentative region of the data space, which artificially simplifies the weight-space blending problem.

## 3. Reproducibility Concerns
The paper does not include any accompanying source code, configuration scripts, or repository links. To reproduce this paper, a researcher would have to implement:
1. The custom DeepMLP-12 and TinyCNN-4 backbones.
2. The Split-MNIST expert training pipeline (512 samples/task).
3. The custom random Gaussian projection and BSigmoid normalization router.
4. The Adam-based few-shot calibration loop.
5. The SVD decomposition of the batch-averaged matrix.
Given the extreme sensitivity of the results to hyperparameter choices (such as the projection dimension $d=8$ and the learning rate), the lack of public code makes complete physical reproduction highly doubtful and difficult.
