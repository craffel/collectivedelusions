# 1. Summary of the Paper

## Main Topic
The paper addresses the challenge of serving multiple concurrent Parameter-Efficient Fine-Tuning (PEFT) adapters (e.g., Low-Rank Adaptation/LoRA) on resource-constrained edge hardware. In dynamic environments, heterogeneous streams of mixed-task inputs are processed in arbitrary, vectorized batch configurations. This causes "heterogeneity collapse" under static model merging. Standard dynamic routing methods scale sequentially as $O(K)$ (where $K$ is the number of active tasks), introducing severe latency penalties. 

The paper builds on Single-Pass Activation-Space Dynamic Blending (SPS) and Zero-Shot Centroid Alignment (ZCA) which execute the heavy base model exactly once and route inputs sample-wise to task-specific parameters inside a single pass, achieving $O(1)$ latency. The main topic is regularizing and learning the temperature parameters of such routers to prevent overfitting and robustly handle representation fragmentation and heteroscedastic noise.

## Approach: PAC-ZCA
The authors propose **PAC-ZCA**, a learning-theoretic framework for dynamic model-merging routing. The approach includes several key components:
1. **Unsupervised Subspace Energy Projection (SEP):** Reduces high-dimensional representations $z_b \in \mathbb{R}^D$ from an early routing layer $l_{\text{route}}$ to low-dimensional task energy coordinates $e_{k, b} \in \mathbb{R}$ by computing the $L_2$-norm of its projection onto each task's low-dimensional subspace.
   - It is generalized to real distributed manifolds using uncentered Principal Component Analysis (PCA) or SVD on task-specific calibration matrices.
   - To prevent overfitting and test-time norm collapse in the high-dimensional, low-sample regime ($N_c \ll D$), the paper introduces several regularized variants: *Shrinkage-SEP*, *Ridge-SEP*, *LDA-SEP*, and *Unit-Norm PCA Subspace Projection (UN-PCA-SEP)*.
2. **Strictly Temperature-Only Gibbs Routing Policy:** Restricts parameters to task-specific log-temperatures $\mathbf{w} = \ln \boldsymbol{\tau} \in \mathbb{R}^K$. Input queries are routed sample-wise using a Softmax policy over the SEP coordinates.
3. **Parameter-Space PAC-Bayesian Generalization Bound:** Establishes a Gaussian prior and posterior over the log-temperature parameters $\mathbf{w}$ centered at a physically grounded SABLE temperature baseline $\mathbf{w}_0 = \ln(0.05) \cdot \mathbf{1}$. The Kullback-Leibler (KL) divergence serves as a strictly data-independent complexity penalty.
4. **Decoupled Calibration Splits:** Partitions a tiny calibration set $\mathcal{C}_k$ of size 16 into a subspace extraction split ($N_{\text{sub}}=8$) and a temperature optimization split ($N_{\text{opt}}=8$) to satisfy McAllester's theorem's data-independence assumption.
5. **Catoni's Bound Optimization:** Directly minimizes Catoni's PAC-Bayesian bound using Cross-Entropy loss as a smooth, theoretically bounded surrogate risk objective.
6. **Single-Pass Activation Blending:** Evaluates routing coefficients once at $l_{\text{route}}$ and dynamically blends LoRA adapter activations on-the-fly, maintaining constant $O(1)$ backbone latency.

## Key Findings
1. **Analytical Sandbox Evaluation:** Evaluated on a 14-layer, 192-dimensional Coordinate Sandbox with 4 tasks of varying noise profiles (MNIST, Fashion-MNIST, CIFAR-10, SVHN):
   - Under Orthogonal Block features, PAC-ZCA achieves **64.16% $\pm$ 2.23%** joint classification accuracy, matching standard unregularized Empirical Risk Minimization (ERM) (64.16% $\pm$ 2.28%) while slightly reducing ensembling variance.
   - Heuristic, uncalibrated SABLE (SEP-Block) with a static temperature scale ($\tau=0.05$) achieves **66.08% $\pm$ 0.78%** accuracy, outperforming both PAC-ZCA and ERM by nearly $2\%$ while having a fraction of the variance.
2. **Train-Test Feature Mismatch under SVD:** In low-sample regimes ($N_c \ll D$), unsupervised SVD overfits to high-variance calibration noise directions. During offline calibration, SVHN (Task 3) samples have an average norm of **17.29**, which collapses to **5.40** at test time under unseen noise. The router learns high temperatures to balance the calibration norm, resulting in zero SVHN predictions at test time (routing collapse).
3. **Unit-Norm PCA Resolution:** Bounding coordinates between 0 and 1 via UN-PCA-SEP completely eliminates the feature scale mismatch and recovers SVHN predictions, yielding **44.36% $\pm$ 1.30%** orthogonal joint accuracy.
4. **Real-World Servings on Vision Datasets:** On pre-trained ResNet-18 features (MNIST, Fashion-MNIST, CIFAR-10), PAC-ZCA (Isotropic Ours) achieves **70.87% $\pm$ 2.20%** joint accuracy, outperforming standard uncalibrated SABLE (65.67%) and unregularized ERM (69.47% $\pm$ 2.21%) while slightly stabilizing variance.

## Explicitly Claimed Contributions
1. **Theoretical Foundation:** Establishes the first mathematically sound PAC-Bayesian learning framework for dynamic model-merging routing, linking calibration sample complexity to out-of-sample risk.
2. **Parameter-Space Regularization:** Bypasses data-dependency flaws and establishes a formal duality between parameter complexity and routing entropy.
3. **Subspace Energy Projection (SEP):** Formulates SEP, generalizes it to non-orthogonal real-world manifolds using PCA, and provides four regularized subspace extraction extensions.
4. **Empirical Vindication:** Outperforms raw-coordinate SABLE by **+23.70%** (on Block features) inside the Coordinate Sandbox, matches standard ERM in mean performance while reducing ensembling variance, and outperforms standard baselines on a real image ResNet-18 serving benchmark.
