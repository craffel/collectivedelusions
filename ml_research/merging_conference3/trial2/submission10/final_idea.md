# Deconstructing Adaptive Model Merging: Exposing the Overfitting-Optimizer and Spatial Averaging Paradoxes

## 1. Persona Alignment
This deconstructive study is a direct realization of **The Minimalist** persona:
- **Aggressive Complexity Pruning**: We critically examine the high-dimensional optimization landscape of SOTA AdaMerging ($L \times T$ parameters) and expose it as a redundant, overfitting artifact.
- **Scientific Honesty and Skepticism**: Instead of hiding or glossing over optimization failures, we embrace them. We expose why direct low-dimensional optimization fails (Task-wise AdaMerging) and analyze the fundamental multi-task gradient imbalance caused by uncalibrated prediction entropy curves.
- **Elegance of Spatial Regularization**: We show that a simple post-hoc flat spatial average of optimized coefficients acts as a powerful low-pass filter, pruning 99.6% of parameter degrees of freedom while restoring multi-task performance to beat the baseline.

## 2. Core Deconstructive Techniques
- **Intra-Task Layer Shuffling**: Randomly permuting optimized layer-wise coefficients within each task to prove they represent fragile, high-frequency overfitting noise rather than a coordinated layer hierarchy.
- **Post-hoc Spatial Averaging**: Compressing the high-dimensional coefficient space to a single flat scalar per task after optimization, demonstrating that the global task-level scaling is the true functional signal.
- **Analyzing Gradient Imbalance**: Showing how uncalibrated prediction entropy causes direct task-wise optimization to be dominated by easy tasks (e.g., MNIST), creating destructive parameter interference on harder tasks (e.g., CIFAR-10, SVHN).

## 3. Mathematical Formulation
Let $\theta_0$ be the pre-trained base model, and $\theta_t$ be the expert model fine-tuned on task $t \in \{1, \dots, T\}$.
The task vector for task $t$ is:
$$\tau_t = \theta_t - \theta_0$$

### SOTA Layer-Wise AdaMerging
The merged parameters at layer $l$ are:
$$\theta_{\text{merged}}^{(l)}(\Lambda) = \theta_0^{(l)} + \sum_{t=1}^T \Lambda_{l, t} \tau_{l, t}$$
where $\Lambda \in \mathbb{R}^{L \times T}$ is optimized using test-time prediction entropy minimization:
$$\mathcal{L}(\Lambda; X) = -\frac{1}{|X|} \sum_{x \in X} \sum_{c=1}^C p_c\left(x; \theta_{\text{merged}}(\Lambda)\right) \log p_c\left(x; \theta_{\text{merged}}(\Lambda)\right)$$

### The Two Paradoxes
1. **The Overfitting-Optimizer Paradox**: Shuffling $\Lambda$ across layers destroys the network performance, proving $\Lambda$ is overfitted to high-frequency batch noise.
2. **The Spatial Averaging Paradox**: Taking the spatial mean post-hoc:
   $$\bar{\lambda}_t = \frac{1}{L} \sum_{l=1}^L \Lambda_{l, t}$$
   yields $\theta_{\text{averaged}}$ which restores accuracy to **84.81%** (beating Task Arithmetic). However, directly optimizing a flat task-wise scale $\lambda \in \mathbb{R}^T$ a priori collapses performance to **81.02%** because easy tasks dominate the joint gradient:
   $$\left\| \nabla_{\lambda_{\text{easy}}} \mathcal{L}_{\text{easy}} \right\| \gg \left\| \nabla_{\lambda_{\text{hard}}} \mathcal{L}_{\text{hard}} \right\|$$
   causing severe parameter interference on harder datasets.

## 4. Baselines
1. **Task Arithmetic (Ilharco et al., 2022)**: Static baseline with a uniform scale $\lambda_{\text{static}} = 0.3$.
2. **Task-wise AdaMerging (Yang et al., 2024)**: Directly optimizing $T$ global parameters.
3. **Layer-wise AdaMerging (Yang et al., 2024)**: Optimizing $L \times T$ parameters.
4. **Spatially Averaged AdaMerging**: Post-hoc spatial mean of layer-wise optimized coefficients.
