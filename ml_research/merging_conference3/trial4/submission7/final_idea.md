# SuiteMerge: Deconstructing the Task Suite Bias in Model Merging

## 1. Persona Alignment
This project is deeply rooted in the core tenets of **The Methodologist** persona. Rather than introducing a flashy, complex new architecture or a highly parameterized fusion algorithm, **SuiteMerge** is a rigorous, independent methodological audit of the entire adaptive model-merging literature. 
We identify a major, un-reported **confounding variable and hidden assumption** in existing works (such as AdaMerging, RegCalMerge, PolyMerge, and OFS-Tune): they evaluate their algorithms exclusively on a **single, arbitrary multi-task suite** comprising MNIST, FashionMNIST, CIFAR-10, and SVHN. 
We critically challenge this convention by hypothesizing that the relative ranking and SOTA generalizability claims of these methods are an artifact of this specific 4-task combination. By systematically partitioning these 4 datasets into distinct suites of varying domain distances (homogeneous vs heterogeneous, cross-domain visual digits vs natural objects), we expose the latent **Task Suite Bias** and establish a much-needed, mathematically robust evaluation protocol.

## 2. Core Techniques
*   **Task-Suite Partitioning (Systematic Multi-Suite Evaluation):** We construct 4 distinct task relationships from the available MNIST, FashionMNIST, CIFAR-10, and SVHN datasets:
    *   **Suite A (Highly Homogeneous - Low Conflict):** MNIST + FashionMNIST (grayscale visuals, similar input geometries, low representational overlap friction).
    *   **Suite B (Highly Heterogeneous - High Conflict):** CIFAR-10 + SVHN (natural RGB images vs complex street-view numbers, severe representational clashes in shared subspaces).
    *   **Suite C (Cross-Domain Digits):** MNIST + SVHN (digits with a severe domain shift from clean grayscale to natural street numbers).
    *   **Suite D (Cross-Domain Objects):** FashionMNIST + CIFAR-10 (grayscale clothes vs RGB natural objects).
    *   **Suite E (Full 4-Task Suite - Control):** MNIST + FashionMNIST + CIFAR-10 + SVHN.
*   **Dimensional Constraint Trajectories:** Modeling merging coefficients as continuous polynomials of normalized layer depth:
    $$\alpha_k(l) = \sum_{j=0}^d \theta_{k, j} \left( \frac{l}{L} \right)^j$$
    We contrast unconstrained layer-wise optimization (high-dimensional, $K \times L$ search parameters) against low-degree polynomial parameterizations ($d=1$ or $d=2$; $K \times (d+1)$ parameters).
*   **Multi-Method Optimization Backends:** We evaluate first-order gradient descent (Adam) and derivative-free local search (Nelder-Mead / 1+1 Evolution Strategy) across both online unsupervised test-time adaptation (TTA) and offline few-shot validation tuning (OFS-Tune).

## 3. Mathematical Formulation
Let $S$ represent a specific task suite, where $S \subset \{1, 2, 3, 4\}$ denotes the task indices corresponding to MNIST ($1$), FashionMNIST ($2$), CIFAR-10 ($3$), and SVHN ($4$).
The merging coefficient for task $k \in S$ at layer $l \in \{1, \dots, L\}$ is denoted as $\alpha_k(l)$, which is parameterized by vector $\theta_S$.

### 3.1. Continuous Simulation Calibration (Model II Coupled Sensitivity Landscape)
For a given task $k$ and layer $l$, the local task sensitivity loss $\mathcal{S}_k^{(l)}(\alpha_k(l))$ is defined via a non-convex, coupled Model II landscape:
$$\mathcal{S}_k^{(l)}(\alpha_k(l)) = A_k^{(l)} (\alpha_k(l) - \alpha_{k, opt}^{(l)})^2 + B_k^{(l)} ( \alpha_k(l) - \alpha_{k, opt}^{(l)} )^4 + \text{Interference}_k^{(l)}(\theta_S)$$
where:
*   $\alpha_{k, opt}^{(l)}$ is the optimal task-specific local scaling coefficient at layer $l$.
*   $A_k^{(l)}$ and $B_k^{(l)}$ represent quadratic and quartic curvature parameters calibrated on empirical Vision Transformer (ViT-B/32) classification statistics.
*   The $\text{Interference}_k^{(l)}(\theta_S)$ term represents task-to-task representational clash in shared subspaces:
    $$\text{Interference}_k^{(l)}(\theta_S) = \sum_{k' \in S, k' \neq k} D_{k, k'}^{(l)} (\alpha_k(l) - \alpha_{k'}(l))^2$$
    where $D_{k, k'}^{(l)}$ models the pairwise representational conflict between task $k$ and task $k'$ at layer $l$. Note that $D_{k, k'}^{(l)}$ will scale with the empirical domain distance between tasks (highest for CIFAR-SVHN, lowest for MNIST-FashionMNIST).

### 3.2. Offline Few-Shot Validation Tuning (OFS-Tune) Objective
For a given task suite $S$, using a tiny validation set containing $M=10$ labeled samples per task, the offline supervised objective is:
$$\min_{\theta_S} \sum_{k \in S} \frac{1}{|D_{val}^k|} \sum_{(x, y) \in D_{val}^k} \mathcal{L}_{CE}\left( \hat{f}(x; W_{merged}(\theta_S)), y \right)$$
which is simulated via the joint sum of coupled Model II landscapes across the tasks in suite $S$:
$$\min_{\theta_S} \sum_{k \in S} \left[ \sum_{l=1}^L \mathcal{S}_k^{(l)}(\alpha_k(l; \theta_S)) \right]$$

### 3.3. Online Test-Time Adaptation (AdaMerging) Objective
On an incoming unlabeled test stream, online TTA adapts unconstrained coefficients online by minimizing predicted Shannon entropy on a local batch $B_t$:
$$\min_{\theta_S} \frac{1}{|B_t|} \sum_{x \in B_t} \left[ -\sum_{c} p_c(x; \theta_S) \log p_c(x; \theta_S) \right]$$
which is modeled via our non-convex cosine penalty surrogate to simulate rugged optimization dynamics:
$$\mathcal{L}_{TTA}(\theta_S) = \sum_{k \in S} \left[ \sum_{l=1}^L \mathcal{S}_k^{(l)}(\alpha_k(l)) + \lambda_{rug} \cos(F \cdot \alpha_k(l)) \right]$$
where $F$ is the landscape roughness frequency factor and $\lambda_{rug}$ is the non-convexity weight.

## 4. Architecture Specifications
*   **Calibrated Simulation Architecture:** Calibrated on a 12-layer Vision Transformer (ViT-B/32) backbone containing 86M parameters (12 Transformer layers, hidden dimension 768, 12 attention heads). The output layers feature task-specific linear heads appended to the shared representation.
*   **Physical Evaluation Architecture (DeepCNN):** For physical weight-space validation, we train a 5-layer Convolutional Neural Network (DeepCNN) on CPU, with the following sequence:
    *   `Conv2d(1/3, 16, kernel_size=3, padding=1)` $\rightarrow$ `BatchNorm2d(16)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)`
    *   `Conv2d(16, 32, kernel_size=3, padding=1)` $\rightarrow$ `BatchNorm2d(32)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)`
    *   `Conv2d(32, 64, kernel_size=3, padding=1)` $\rightarrow$ `BatchNorm2d(64)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)`
    *   `Linear(576, 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.1)`
    *   Separate fully-connected task heads for MNIST ($10$ outputs) and FashionMNIST ($10$ outputs) to validate disjoint output manifold behaviors.

## 5. Baselines
We compare and audit the following methods across all 5 evaluation suites:
1.  **Uniform Task Arithmetic (TA) Baseline:** Constant static coefficients across all layers: $\alpha_k(l) = 0.5$ (for 2-task suites) or $\alpha_k(l) = 0.25$ (for the 4-task suite).
2.  **Online AdaMerging (Layer-wise) [Yang et al., 2024]:** High-dimensional, unconstrained online test-time adaptation ($K \times L$ parameters) minimizing local prediction entropy via Adam.
3.  **Online PolyMerge ($d=2$) [PolyMerge, 2025]:** Low-dimensional online test-time adaptation restricting coefficients to quadratic polynomial trajectories.
4.  **Offline Few-Shot Validation Tuning (OFS-Tune, $d=1$) [OFS-Tune, 2026]:** Offline supervised validation tuning using Nelder-Mead to optimize linear ($d=1$) polynomial trajectories on $M=10$ validation samples.

## 6. Step-by-Step Interaction
1.  **Data Partitioning:** Load inputs and labels for MNIST, FashionMNIST, CIFAR-10, and SVHN, partitioning them into the 5 target task suites (Suites A, B, C, D, and E).
2.  **Task Expert Checkpoints & Calibration:** Instantiate individual task-specific model weights. Set up the corresponding simulation landscapes calibrated on task-specific characteristics (such as the task interference coefficient matrix $D_{k, k'}^{(l)}$).
3.  **Coefficient Optimization Loop:**
    *   For **OFS-Tune**, sample $M=10$ labeled validation points per task. Optimize the continuous polynomial coefficients $\theta_S$ offline using Nelder-Mead to minimize the joint validation cross-entropy loss.
    *   For **Online AdaMerging & PolyMerge**, pass unlabeled test batches sequentially. Run gradient descent (Adam) to minimize prediction entropy on the incoming stream.
4.  **Multi-Suite Evaluation:** Test the resulting merged networks on the test partitions of all constituent tasks in each suite.
5.  **Multi-Metric Assessment:** Record task-specific test accuracies, average suite-wide accuracies, and parameter profile smoothness. Compare optimization trajectories and evaluate sensitivity to seed variations (across 30 random seeds, 42 to 71 inclusive).
