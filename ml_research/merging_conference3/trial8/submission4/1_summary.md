# Summary of the Submission

## Core Motivation and Problem Statement
The paper addresses the challenge of **heterogeneity collapse** in serving multiple parameter-efficient fine-tuning (PEFT) adapters (such as LoRA) simultaneously on a shared pre-trained backbone. Standard weight-space merging techniques (e.g., Task Arithmetic, TIES-Merging, DARE) fail when a single vectorized batch contains heterogeneous, mixed-task inputs. To bypass this, existing dynamic routing methods dispatch inputs sample-wise. However, methods that route sequentially scale as $O(K)$ in latency, where $K$ is the number of active tasks, creating severe latency bottlenecks.

Activation-space blending (e.g., SABLE, SPS-ZCA) restores $O(1)$ latency by executing the backbone once and blending adapter activations sample-wise using early-layer routing coordinates. However, the authors argue that existing dynamic routers suffer from three critical limitations:
1. **Lack of Learning-Theoretic Foundations**: Routing temperature parameters are usually hand-tuned or determined via heuristic grid searches, lacking out-of-sample generalization guarantees.
2. **Vulnerability to Heteroscedastic Noise**: Representation spaces across tasks exhibit vastly different spatial variances (e.g., clean MNIST vs. cluttered SVHN), biasing cosine-similarity-based routers and causing task confusion or routing collapse.
3. **Representation Anisotropy and Fragmentation**: High-dimensional representations are often confined to narrow cones and exhibit extreme sparsity, leading to overfitting in low-sample regimes.

---

## Proposed Methodology (PAC-ZCA)
To resolve these bottlenecks, the paper proposes **PAC-ZCA**, a learning-theoretic framework that reformulates dynamic model merging:
- **Subspace Energy Projection (SEP)**: A task-agnostic, unsupervised dimensionality reduction step mapping deep intermediate representations to a $K$-dimensional coordinate vector based on task subspace projection. To handle real-world, non-orthogonal distributed manifolds, SEP is formulated using Principal Component Analysis (PCA).
- **Unit-Norm PCA (UN-PCA-SEP)**: To eliminate heteroscedastic noise spillover bias and prevent norm collapse at test-time, feature representations are normalized to the unit sphere before projection, restricting coordinate magnitudes to $[0, 1]$.
- **Gibbs Routing Policy**: A strictly temperature-only Softmax routing policy over the SEP/UN-PCA coordinates, parameterized by log-temperatures $\mathbf{w} \in \mathbb{R}^K$.
- **Parameter-Space PAC-Bayesian Generalization Bound**: By defining an isotropic Gaussian prior and posterior over the log-temperatures, the authors derive a mathematically valid PAC-Bayesian bound on out-of-sample routing risk. They minimize Catoni's PAC-Bayesian bound to optimize temperatures on a tiny calibration split.
- **Disjoint Calibration Splits**: To satisfy the data-independence assumption of McAllester's theorem, the authors partition a tiny offline calibration set ($N_c = 16$ per task) into a Subspace Split (to compute PCA bases) and an independent Optimization Split (to train temperatures), resolving the double data-dependency flaw.
- **Theoretical Contributions**: 
  - *Lipschitz Constant Boundedness*: They formally prove that the routing risk loss is Lipschitz continuous with localized Lipschitz constant $L_R \le K e^{\sqrt{C}}$ under UN-PCA-SEP (Lemma 3.1).
  - *Lipschitz-Entropy Duality*: Bounding the parameter complexity restricts logit variations, establishing a guaranteed lower bound on Shannon routing entropy and preventing routing collapse (Theorem 3.2).
  - *Theory-Practice Gap*: Bounding the discrepancy between randomized Gibbs routing and continuous activation blending using sub-network curvature and manifold divergence.

---

## Key Experimental Findings
- **Coordinate Sandbox (Analytical 14-Layer Simulation)**: 
  - PAC-ZCA (Block features) achieves **64.16% ± 2.23%** joint classification accuracy on disjoint splits, outperforming raw-coordinate SABLE (+23.70%) and matching unregularized Empirical Risk Minimization (ERM) while successfully reducing ensembling variance.
  - SVD overfitting is identified under uncentered PCA-SEP, and the proposed **UN-PCA-SEP** completely recovers the performance to **44.36% ± 1.30%** (orthogonal) and **45.86% ± 0.76%** (overlapping).
- **Real-World Image Evaluation (ResNet-18)**: 
  - Evaluated on MNIST, Fashion-MNIST, and CIFAR-10, PAC-ZCA (UN-PCA) achieves **70.87% ± 2.20%** joint accuracy, outperforming standard uncalibrated SABLE (65.67%) and strictly outperforming unregularized ERM (69.47% ± 2.21%) while stabilizing ensembling variance.
- **Ablations and Analytical Sweeps**:
  - *Prior Variance Sensitivity*: Sweeping $\sigma_0^2$ from 0.1 to 10.0 illustrates the trade-off between over-regularization and asymptotic convergence to unregularized ERM.
  - *Sample Complexity*: Scaling $N_c$ from 8 to 128 per task demonstrates consistent superiority over uncalibrated SABLE and highlights how the disjoint split penalty vanishes as sample size increases, with PAC-ZCA consistently delivering lower variance than standard ERM under small calibration budgets.
