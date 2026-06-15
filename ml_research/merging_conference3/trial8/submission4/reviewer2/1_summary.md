# Intermediate Review Evaluation: 1. Summary of the Paper

## Main Topic
The paper addresses the challenge of **dynamic model-merging routing** on the edge in multi-task deployment scenarios. In these settings, incoming inference requests are heterogeneous and arrive in mixed-task batches. Traditional static model merging (e.g., Task Arithmetic, TIES-Merging, DARE) suffers from *heterogeneity collapse* because a single static weight set cannot adapt sample-wise. Existing dynamic routing methods either scale linearly in latency ($O(K)$) or rely on unregularized heuristics (such as SABLE) that lack generalization guarantees and collapse under heteroscedastic domain noise.

To address these limitations, the paper introduces **PAC-ZCA**, a learning-theoretic, mathematically rigorous framework for dynamic model merging. It reformulates dynamic routing as a strictly temperature-only randomized Gibbs policy (Softmax routing) over unsupervised, task-agnostic Subspace Energy Projection (SEP) features, and optimizes the task-specific log-temperatures by directly minimizing a derived differentiable PAC-Bayesian bound (Catoni's bound) on a tiny offline calibration set (16 samples per task).

---

## Technical Approach (PAC-ZCA)
1. **Unsupervised Coordinate Extraction via SEP:**
   - Pre-trained hidden features at a routing layer ($l_{\text{route}}$) are projected onto task-specific subspaces to yield Subspace Energy Projection (SEP) energy coordinates $\mathbf{e}_b = [e_{1,b}, \dots, e_{K,b}]^T \in \mathbb{R}^K$.
   - Generalizes from orthogonal block features to real distributed, non-orthogonal manifolds via Singular Value Decomposition (SVD)/Principal Component Analysis (PCA) on calibration sets.
   - Proposes four regularized extensions to stabilize projection matrices under low-sample, high-dimensional regimes ($N_c \ll D$): Ledoit-Wolf Shrinkage PCA (Shrinkage-SEP), Ridge-Regularized Subspace Projection (Ridge-SEP), Supervised Linear Discriminant Analysis Projection (LDA-SEP), and Unit-Norm PCA Subspace Projection (UN-PCA-SEP).

2. **Strictly Temperature-Only Gibbs Routing Policy:**
   - Parameterizes the routing probabilities $q_k(\mathbf{e}_b; \boldsymbol{\tau})$ using a Softmax function over SEP coordinates scaled by task-specific temperature parameters $\boldsymbol{\tau}$ ($w_k = \ln \tau_k$).

3. **Parameter-Space PAC-Bayesian Optimization:**
   - Establishes a Gaussian prior $P = \mathcal{N}(\mathbf{w}_0, \sigma_0^2 I_K)$ centered at a physically grounded baseline uncalibrated temperature scale $\mathbf{w}_0 = \ln(0.05)\cdot\mathbf{1}$ and Gaussian posterior $Q = \mathcal{N}(\mathbf{w}, \sigma_0^2 I_K)$.
   - Derives the Kullback-Leibler (KL) divergence as a parameter-space complexity penalty: $\text{KL}(Q \| P) = \frac{\|\mathbf{w} - \mathbf{w}_0\|_2^2}{2 \sigma_0^2}$.
   - Resolves the PCA-induced double data-dependency flaw by enforcing **Decoupled Calibration Splits** where a tiny calibration set of size $N_c = 16$ is partitioned into disjoint splits $\mathcal{C}^{\text{sub}}$ (used to extract subspaces) and $\mathcal{C}^{\text{opt}}$ (used to optimize temperatures).
   - Minimizes Catoni's PAC-Bayesian bound using a smooth Cross-Entropy surrogate, resolving the bounded-loss limitation of McAllester's theorem.

4. **Lipschitz-Entropy Duality:**
   - Proves mathematically that restricting the parameter-space complexity bounds the maximum variation of logits (and the localized Lipschitz constant $L_R \le 0.25 K e^{\sqrt{C}}$ under UN-PCA), which directly enforces a lower bound on the output Shannon routing entropy, preventing deterministic routing collapse.

5. **Single-Pass Activation Blending:**
   - Executes the pre-trained backbone exactly once and blends the low-rank expert adapter activations on-the-fly using the routing coefficients $\alpha_k$, restoring $O(1)$ inference latency. It bounds the theory-practice gap (randomized Gibbs policy vs. continuous activation blending) using subsequent sub-network curvature and manifold divergence.

---

## Key Findings
1. **Resistance to Heterogeneity Collapse:** Weight-space merging (e.g., PFSR) collapses near uniform accuracy (40.56% $\pm$ 0.99%) under mixed streams. In contrast, PAC-ZCA (Block Ours) achieves a robust **64.16% $\pm$ 2.23%** accuracy on both homogeneous and heterogeneous streams.
2. **Empirical Value of Temperature Calibration:** PAC-ZCA learns task-specific temperatures $\boldsymbol{\tau}^* = [0.168, 0.161, 0.141, 0.127]^T$ in the Coordinate Sandbox. It matches standard unregularized Empirical Risk Minimization in mean accuracy (64.16% vs. 64.16% on orthogonal block features) while successfully reducing ensembling variance (reducing standard deviation from 2.28% to 2.23%).
3. **Overfitting in High-Dimensional SVD Projection:** Unsupervised SVD on tiny calibration sets ($N_c = 16$ in a 192-dim space) overfits to sample-specific noise. This leads to a severe train-test feature scale mismatch (e.g., SVHN projected norm collapses from 17.29 to 5.40 at test time), causing the temperature-only policy to permanently neglect the high-noise expert.
4. **Effectiveness of Feature Normalization:** Normalizing features to the unit $L_2$ sphere prior to SVD (UN-PCA-SEP) bounds coordinate values, mathematically eliminating the noise scale mismatch and recovering the neglected SVHN expert (achieving a balanced **44.36% $\pm$ 1.30%** joint accuracy on disjoint splits).
5. **Real-World Viability:** Evaluated on real image datasets (MNIST, Fashion-MNIST, CIFAR-10) using a pre-trained ResNet-18, PAC-ZCA (Isotropic Ours) achieves **70.87% $\pm$ 2.20%** joint accuracy, outperforming uncalibrated SABLE (65.67%) and strictly outperforming unregularized ERM (69.47% $\pm$ 2.21%) while maintaining high stability.

---

## Explicitly Claimed Contributions and Accompanying Evidence
- **Theoretical learning-theoretic foundation:** Established the first mathematically sound PAC-Bayesian framework for dynamic model-merging routing (supported by Gaussian KL formulation, derived Catoni's bound, localized Lipschitz bounds in Lemma 1, and the theory-practice gap proof).
- **Complexity and Entropy Regularization:** Proved that bounding the log-temperature parameter complexity prevents deterministic overfitting by regularizing output routing entropy (supported by Theorem 1, Lemma 1, and the sensitivity analysis of prior variance $\sigma_0^2$ in Table 2).
- **Subspace Energy Projection (SEP):** Formulated SEP and generalized it to distributed pre-trained manifolds using PCA with four regularized variants (Shrinkage-SEP, Ridge-SEP, LDA-SEP, UN-PCA-SEP) (supported by mathematical derivations in Section 3.2, and empirical results for UN-PCA in Table 1 and Table 3).
- **Empirical Vindication on Synthetic and Real Tasks:** Demonstrated that PAC-ZCA achieves high joint accuracies, eliminates heterogeneity collapse, matches or beats ERM while reducing ensembling variance, and resolves the SVD overfitting bottleneck (supported by Table 1 results on synthetic block/PCA features, Table 3 results on real images, and the calibration sample complexity scaling in Table 4).
