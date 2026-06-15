# 1. Summary of the Paper

## Overview and Core Objectives
This paper introduces a novel learning-theoretic framework for layer-wise weight-space model merging, aiming to combine multi-task expert neural networks without the high computational cost of joint training. While layer-wise merging offers fine-grained adaptation of ensembling coefficients across layers, optimizing these coefficients on small calibration datasets is highly susceptible to **transductive overfitting**.

To address this, the authors propose **Rademacher-Bounded Fourier Trajectory Merging (RB-FTM)** and its non-periodic variant, **Rademacher-Bounded Discrete Cosine Trajectory Merging (RB-DCTM)**. Instead of treating ensembling coefficients at each layer as independent parameters, the paper projects these coefficients onto a low-frequency continuous harmonic subspace across the network's depth coordinate $z \in [0,1]$. By combining Fourier/DCT basis functions with an analytical **Spectral Lasso ($L_1$) regularizer**, the authors bound the trajectory capacity, mitigate overfitting on few-shot calibration sets, and resolve the boundary runaway (oscillatory) problems typical of polynomial trajectories.

---

## Core Methodology & Key Mathematical Formulations

### 1. Weight-Space Task Vector Merging
For a pre-trained base model $W_0^{(l)}$ at layer $l \in \{0, \dots, L-1\}$, let $V_k^{(l)} = W_k^{(l)} - W_0^{(l)}$ represent the task-specific weight vectors for $K$ distinct experts. The merged model weights are defined via:
$$W_{\text{merged}}^{(l)}(\Lambda) = W_0^{(l)} + \sum_{k=0}^{K-1} \alpha_k(l) V_k^{(l)}$$
where $\alpha_k(l) \in [0, 1]$ represents the ensembling coefficient for task $k$ at layer $l$.

### 2. Spectral Trajectory Parameterizations
Rather than optimizing each $\alpha_k(l)$ independently, the trajectory of ensembling coefficients across normalized depth $z = l / (L-1)$ is parameterized as a continuous harmonic series.

*   **Fourier Trajectory (RB-FTM):**
    $$\alpha_k(l; \theta_k) = \Pi_{[0,1]} \left( a_{k,0} + \sum_{f=1}^F \left( a_{k,f} \cos(2\pi f z) + b_{k,f} \sin(2\pi f z) \right) \right)$$
    where $\theta_k = [a_{k,0}, a_{k,1}, b_{k,1}, \dots, a_{k,F}, b_{k,F}]^T \in \mathbb{R}^{2F+1}$.
*   **Discrete Cosine Trajectory (RB-DCTM):**
    $$\alpha^{\text{DCT}}_k(l; \theta_k) = \Pi_{[0,1]} \left( a_{k,0} + \sum_{f=1}^F a_{k,f} \cos\left(\pi f z\right) \right)$$
    where $\theta^{\text{DCT}}_k = [a_{k,0}, a_{k,1}, \dots, a_{k,F}]^T \in \mathbb{R}^{F+1}$.

Here, $\Pi_{[0,1]}(x) = \max(0, \min(x, 1))$ is a $1$-Lipschitz projection operator that restricts coefficients to the valid range $[0, 1]$.

### 3. Optimization Objective and Spectral Lasso
To physically restrict trajectory capacity, the optimization objective incorporates an $L_1$ penalty (Spectral Lasso) applied strictly to the harmonic coefficients, excluding the base uniform parameter $a_{k,0}$ (which is initialized to $1/K$):
$$\min_{\Theta} \mathcal{L}(\Theta) = \mathcal{L}_{\text{CE}}(\mathcal{D}_{\text{cal}}; \Theta) + \gamma \sum_{k=0}^{K-1} \|\theta_{k,\text{harm}}\|_1$$

---

## Key Theoretical Results

The paper derives tight empirical Rademacher complexity bounds over the depth coordinates $Z = \{z_1, \dots, z_L\}$ of size $L$, proving that trajectory-space capacity decays with depth $L$ and is strictly bounded by the spectral cutoff frequency $F$, independent of the parameter count of the underlying network.

1.  **Fourier Trajectory Complexity (Theorem 3.1):**
    $$\widehat{\mathcal{R}}_L(\mathcal{H}_F) \le C_0 \sqrt{\frac{2 \ln(4F+2)}{L}}$$
2.  **Discrete Cosine Trajectory Complexity (Theorem 3.4):**
    $$\widehat{\mathcal{R}}_L(\mathcal{H}_F^{\text{DCT}}) \le C_0 \sqrt{\frac{2 \ln(2F+2)}{L}}$$
    *(Note: This bound is strictly tighter due to the cosine-only representation reducing the basis size).*
3.  **Joint Multi-Task Complexity (Theorem 3.3):**
    For a stylized joint multi-task trajectory class, the complexity bound is shown to be independent of the task count $K$. When vector-valued dependencies and independent Rademacher variables are used, the bound scales logarithmically as $\mathcal{O}(\sqrt{\ln(KF)/L})$.
4.  **Downstream Generalization Bridge:**
    The authors bridge trajectory complexity over depth coordinates to downstream generalization on data samples using covering numbers, proving that restricting spectral trajectory complexity mathematically limits downstream prediction generalization error.

---

## Empirical Evaluation

The paper evaluates the proposed methods across two distinct environments:

### 1. Analytical Coordinate Sandbox (ACS)
*   **Setup:** A simulated, highly controlled linear dynamical coordinate recurrence system with depths corresponding to a convolutional network (**Deep12LayerCNN**, $L=12$, $D=128$) and a vision transformer (**CLIP ViT-B/16**, $L=13$, $D=768$).
*   **Benchmarks:** A 4-task visual stream (MNIST, FashionMNIST, CIFAR-10, SVHN) where experts are modeled as directional distributions in a latent coordinate space. Optimization is performed on a tiny 10-shot per task calibration budget (40 samples total).
*   **Key Sandbox Findings:**
    *   **Static Uniform Dominance Paradox:** The zero-tuning Static Uniform baseline (setting $\alpha_k(l) = 0.25$) consistently outperforms all adaptive methods in categorical accuracy ($85.10\%$ on CNN and $83.75\%$ on CLIP). The authors attribute this to the perfect coordinate alignment of the sandbox, where any coefficient adaptation induces *anisotropic representation shearing*.
    *   **Trajectory Bounding Success:** Among the tuned/adaptive methods, RB-FTM and RB-DCTM achieve superior accuracy ($70.70\%$ on CNN and $72.70\%$ on CLIP) compared to unconstrained optimization ($67.05\%$ and $68.45\%$) and polynomial-based merging (RBPM $d=2$, which fails catastrophically on CNN at $39.30\%$ due to boundary runaway).
    *   **Misalignment sweeps:** Sweeping random orthogonal rotations ($\eta \in [0.0, 0.6]$) shows that as misalignment grows, Static Uniform accuracy begins to degrade, and trajectory-based methods provide highly stable adaptation, with RB-DCTM performing particularly well.

### 2. Proof-of-Concept Validation on Actual Vision Transformers
*   **Setup:** Merging two actual ViT-B/16 models fine-tuned on CIFAR-10 and CIFAR-100, initialized from a shared CLIP checkpoint. A dual-dataset protocol is used: 100 unlabeled samples per task are used strictly to compute covariance statistics for coordinate permutation alignment (ZipIt!), while the trajectory parameters themselves are optimized on the tiny 10-shot calibration set.
*   **Key Results:**
    *   Unlike the sandbox, the Static Uniform baseline (ZipIt! Aligned) achieves only $71.30\%$ joint average accuracy due to representational interference in real weight spaces.
    *   The polynomial competitor, RBPM ($d=2$), achieves $70.70\%$, degrading below Static Uniform due to boundary runaway.
    *   The proposed **RB-DCTM (Ours, F=2)** achieves the highest accuracy of **$74.90\%$** ($+3.60\%$ over Static Uniform, $+4.20\%$ over polynomial merging, and $+5.10\%$ over unconstrained optimization), proving the practical efficacy of spectral trajectory regularization in real-world, non-linear weight spaces.
