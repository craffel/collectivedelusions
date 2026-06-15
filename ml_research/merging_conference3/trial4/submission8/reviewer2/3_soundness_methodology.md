# Evaluation Component 3: Soundness of Methodology

## Clarity of Description
The methodology is described with outstanding clarity. The mathematical formulation is rigorous, with clean derivations and intuitive explanations. The authors walk through:
- The polynomial parameterization mapping (Equation 2)
- The second-order Taylor expansion decomposing quantization-induced loss (Equation 11)
- The physical layout of deep networks and why it causes a "task-vector norm scale pathology"
- The mathematical formulation of the Clipping-Regularized SACM perturbation (Equation 8 and 9)

## Appropriateness of Methods
The methods chosen are highly appropriate and elegant:
- **Subspace Constraint:** Utilizing a low-degree depth-dependent polynomial constraint is a very logical way to reduce the search space from 56 to 12 parameters.
- **First-Order Minimax Approximation (SACM):** Standard sharpness regularization (using the Hessian trace) is computationally intractable for edge-deployment. Approximating it via a first-order perturbation is highly practical and appropriate.
- **Clipping Threshold:** The clipping threshold $\beta$ is a simple yet effective tool to handle the extreme discrepancy of task-vector norms without triggering numerical instability.

## Potential Technical and Empirical Flaws (Empiricist Critique)

### 1. Regularization Bias and Performance Degradation in High-Precision Schemas
An analysis of Table 1 reveals a major empirical trend: **CR-PolySACM actually performs worse than standard PolyMerge in all non-aggressive precision formats.**
- **FP32 (No Quant):** $57.00\%$ (CR-PolySACM) vs. **$57.40\%$** (PolyMerge) (a drop of $-0.40\%$)
- **INT8 Symmetric Tensor:** $56.62\%$ vs. **$57.62\%$** (a drop of $-1.00\%$)
- **INT8 Symmetric Channel:** $57.23\%$ vs. **$58.15\%$** (a drop of $-0.92\%$)
- **INT8 Asymmetric Tensor:** $56.48\%$ vs. **$56.57\%$** (a drop of $-0.09\%$)
- **INT8 Asymmetric Channel:** $56.93\%$ vs. **$57.43\%$** (a drop of $-0.50\%$)

Only under the aggressive, highly noisy **INT4 Symmetric Channel** format does CR-PolySACM outperform PolyMerge ($19.07\%$ vs. $18.10\%$). 
As an empiricist, we must highlight that in standard deployment regimes (FP32 or INT8 are standard, while INT4 without quantization-aware training is rarely used in production), the proposed method introduces a regularization bias that actively degrades performance. The empirical benefit of CR-PolySACM is demonstrated *only* under severe 4-bit quantization.

### 2. Practical Non-Viability of INT4 Performance
While CR-PolySACM achieves a statistically significant relative improvement of $+0.97\%$ over PolyMerge in the INT4 regime, the absolute joint mean accuracy of **19.07%** is extremely low. Given that the datasets are 10-class classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN), random guessing on each dataset yields a $10.0\%$ accuracy baseline. An absolute performance of $19.07\%$ is barely above random guessing and is completely unusable for any production or real-world application. Therefore, the proposed method only outperforms the baseline in a regime where the model is functionally dead anyway.

### 3. The Expert-to-Merge Capacity Drop
The single-task expert accuracy averages **88.67%** (MNIST: 96.30%, FashionMNIST: 86.90%, CIFAR-10: 90.20%, SVHN: 81.30%). However, the best merged continuous model (PolyMerge) only achieves **57.40%** in FP32—a massive absolute drop of **$-31.27\%$**. 
This catastrophic drop indicates severe representation interference and capacity limitations. While the authors transparently discuss this "domain disconnect limitation," it raises questions about the practical value of merging these specific models. Merging models trained on highly disparate domains with disjoint label spaces is an artificial setup. A more realistic setup would merge models fine-tuned on sub-populations of the same dataset or closely related domains (e.g., DomainNet), where task vectors are highly aligned.

### 4. Sigmoid Gradient Saturation Risk
In Equation 2, the layer-wise coefficients $\lambda_k^l$ are mapped through a logistic sigmoid function $\sigma(\cdot)$ to enforce the hard box constraint $[0, 1]$. The gradient of the loss with respect to the polynomial parameters $\mathbf{p}_k$ is thus scaled by the derivative of the sigmoid: $\sigma'(\cdot) = \lambda_k^l(1 - \lambda_k^l)$. 
If a coefficient converges close to the boundaries $0.0$ or $1.0$, the derivative $\sigma'(\cdot)$ approaches zero, which can freeze the parameter updates. Although the authors state that coefficients stay in the active interior region and that adjacent layer polynomial constraints prevent boundary contact, this remains a potential optimization vulnerability that is not rigorously analyzed mathematically.

### 5. Multi-Task Evaluation with Task-ID Oracle
The evaluation protocol dynamically swaps task-specific classification heads. This assumes that a "task ID oracle" is available at test time to indicate which head should be active. In real-world edge deployment, such an oracle is often not present, or requires a separate task-classifier, which adds complexity and latency.

## Reproducibility
The reproducibility of the paper is **excellent**. The hyperparameters (number of optimization steps $T=40$, learning rate $\eta = 10^{-2}$, calibration size $N=64$, perturbation radius $\rho = 0.05$, clipping threshold $\beta = 0.10$) are clearly specified. The dataset splits and training protocols for the experts are thoroughly documented. The appendix contains extensive details regarding wall-clock times, computational complexity, and alternative subspaces, which further enhances the reproducibility.
