# 3. Soundness and Methodology

## Clarity of the Description
The mathematical and procedural description of OmniMerge is exceptionally clear and structured:
- **Quantization Mathematics:** The paper provides highly explicit mathematical formulations for Symmetric, Asymmetric, and Double Quantization (Section 3.3, Equations 5-14). Rounding and clipping boundaries, scale factors, and zero-point offsets are defined unambiguously.
- **Stochastic Optimization:** The concept of Stochastic Operator Sampling (SOS) is clearly defined as sampling from a discrete pool of hardware-relevant schemas $\mathcal{Q}$ with uniform probability ($P = 0.25$).
- **Objective Function:** The joint objective function, incorporating Prediction Entropy and Task-Consensus Regularization (TCR), is clearly written in Equations 15-16, with explicit parameter values ($\beta = 0.1, \gamma = 0.5$).
- **Gradient Flow:** The paper clearly discusses the gradient flow through the non-differentiable rounding operator using the Straight-Through Estimator (STE) and explains how the scale factors and zero-points are detached from the autograd computation graph to avoid unstable gradients.

## Appropriateness of Methods
- **STE-guided Adaptation:** Using first-order STE-guided optimization of continuous layer-wise coefficients is highly appropriate and computationally efficient compared to zeroth-order evolutionary or random-walk searches (which scale poorly with the number of layer groups).
- **Test-Time Adaptation (TTA):** Using unsupervised prediction entropy minimization over a tiny calibration stream is highly suitable for practical edge scenarios where labeled target data is unavailable.
- **Task-Consensus Regularization:** Including TCR is methodologically sound as it prevents the coefficients of a single dominant task from monopolizing the optimization path, maintaining ensembling stability.
- **Noise Injection (SZNP):** Applying Gaussian perturbation noise to smooth out the non-differentiable loss landscape created by quantization rounding is a standard and effective regularization strategy.

## Potential Technical Flaws and Methodological Inconsistencies

### 1. The Ablation Study Paradox (Table 2)
There is a notable empirical inconsistency in the ablation study:
- The configuration **"Baseline + TCR + SZNP"** (configuration 4) achieves an average accuracy of **50.45%** across the post-training quantization schemas.
- The full **"OmniMerge (SOS + SZNP)"** framework (configuration 5) achieves an average accuracy of **50.33%**.
- This indicates that adding Stochastic Operator Sampling (SOS) to the SZNP baseline actually **degrades average accuracy by 0.12%**. 
- The authors attribute this sub-additive behavior to "compound stochasticity" and high gradient variance within the extremely short 15-step on-device adaptation window. While this explanation is plausible, from an empirical perspective, it means the combination of both techniques is not synergistic on average.
- Crucially, the authors claim that the full OmniMerge framework (combining SOS and SZNP) is necessary to achieve "schema-invariant robustness on completely unseen out-of-pool schemas (such as Double Quantization)." However, Table 2 only reports a single "Average Accuracy" column. To substantiate this claim, the authors **must** provide the full breakdown of the ablation configurations across *all* five target schemas (including Double Quantization). Without this multi-schema breakdown in the ablation study, there is insufficient empirical evidence to prove that combining SOS and SZNP is necessary or beneficial compared to using SZNP alone.

### 2. Optimization Bias and Learning Rate Inconsistency
- During optimization, OmniMerge is evaluated with a learning rate of $\eta = 2 \times 10^{-2}$, while standard AdaMerging and Q-Merge are evaluated with $\eta = 10^{-2}$.
- The authors state that setting $\eta = 2 \times 10^{-2}$ for the baselines causes their sharp local rounding landscapes to oscillate, degrading performance. They argue that SZNP flattens the local loss landscape, permitting a larger learning rate and faster convergence.
- However, from an empirical standpoint, using a larger learning rate for the proposed method could introduce a significant optimization bias, particularly under an extremely tight 15-step optimization budget. A larger learning rate allows the optimizer to make larger updates in fewer steps.
- To eliminate this bias and ensure a fully fair comparison, the authors should evaluate the baselines over a wider variety of step budgets (e.g., 30, 50, or 100 steps) or present learning rate convergence curves. This would verify that the baselines' lower performance is not merely an artifact of the 15-step constraint combined with a restricted learning rate.

## Reproducibility
The methodology is highly reproducible. The authors provide:
- The exact model backbone (`ViT-Tiny` with patch size 16 from `timm`).
- The task datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).
- The training hyperparameters for the task experts (256 training images, 3 epochs, Adam optimizer, learning rate $10^{-4}$, batch size 64).
- The size of the calibration stream ($N_{\text{cal}} = 64$ per task) and evaluation set ($N_{\text{eval}} = 256$ per task).
- The specific noise standard deviations ($\sigma_{\text{scale}} = 0.01$, $\sigma_{\text{zero}} = 0.02$).
- The exact optimization settings (15 steps, learning rates, and coefficient dimensions).
- A detailed selective quantization policy (which layers are quantized and which are preserved in FP16).
This meticulous documentation provides all the necessary details for an independent researcher to replicate the experiments.
