# Idea Proposal: SRC-DE (Shrinkage-Regularized Coordinate Density Estimation) for Robust, Sample-Efficient OOD Rejection in Dynamic Model Merging

## 1. Persona Alignment
This project is deeply rooted in **The Methodologist** persona, prioritizing rigorous evaluation, baseline deconstruction, and statistical correctness over flashy architectural changes:
*   **Skepticism of "SOTA" Robustness Claims:** SPS-ZCA claims high-precision out-of-distribution (OOD) task rejection using a coordinate-space Gaussian Mixture Model (GMM). However, fitting a diagonal GMM on a tiny split ($|\mathcal{C}_k| = 64$) overfits clean calibration samples. Under real-world covariate shifts (e.g., minor representation perturbations or sensor noise), unregularized coordinate log-likelihoods drop catastrophically, triggering extreme false rejection rates. We directly challenge this robustness claim.
*   **Focus on Under-Tuned Baselines & Hidden Assumptions:** We identify a critical hidden assumption in prior works: that unregularized low-resource GMMs can generalize to noisy out-of-sample representations. We expose this overfitting artifact and propose a mathematically rigorous correction.
*   **Rigorous and Fair Evaluation Protocols:** We design a comprehensive evaluation pipeline that subjects in-distribution samples to systematic representation perturbations and maps OOD performance across sample sizes ($N \in [8, 256]$) under strict data partitioning to guarantee zero coordinate leakage.

---

## 2. Core Techniques
1.  **Low-Resource Diagonal GMM Overfitting Analysis:** We mathematically audit the sample complexity of coordinate-space diagonal GMMs, demonstrating how low sample sizes cause variance under-estimation, particularly in low-similarity inactive coordinate dimensions.
2.  **Ledoit-Wolf-style Covariance Shrinkage:** We replace unregularized maximum likelihood estimation of GMM covariance matrices with Ledoit-Wolf shrinkage. By analytically computing the optimal shrinkage intensity $\alpha_{\text{opt}}$, we shrink the sample covariance towards a stable, spherically structured diagonal target, stabilizing log-likelihood boundaries without requiring hyperparameter tuning.
3.  **Covariate Shift Robustness Protocol:** A rigorous testing protocol that evaluates OOD density estimators under systematic representation drift (adding Gaussian noise of variance $\sigma^2 \in [0.01, 0.2]$) to simulate real-world serving noise.

---

## 3. Mathematical Formulation

### 2.1 Early-Layer Coordinate Extraction (ZCA)
During on-device serving, for each sample $b$, we extract the early-stage activation $h^{(3)}_b \in \mathbb{R}^D$ and compute its similarity coordinate vector $u_b = [u_{1, b}, \dots, u_{K, b}]^T \in \mathbb{R}^K$ against pre-computed early task centroids $\mu^{(3)}_k$:
$$u_{k, b} = \text{cos\_sim}(h^{(3)}_b, \mu^{(3)}_k) = \frac{h^{(3)}_b \cdot \mu^{(3)}_k}{\|h^{(3)}_b\|_2 \|\mu^{(3)}_k\|_2}$$

### 2.2 Unregularized GMM Parameter Estimation
For each task expert $k$, we fit a diagonal Gaussian Mixture Model (GMM) with $M$ components over the calibration coordinates $\{u_s\}_{s \in \mathcal{C}_k}$. For each component $m \in \{1, \dots, M\}$, let $\pi_{k, m}$ be the mixture weight, $\mu_{k, m} \in \mathbb{R}^K$ be the component mean, and $\Sigma_{k, m} = \text{diag}(\sigma^2_{k, m}) \in \mathbb{R}^{K \times K}$ be the estimated diagonal covariance matrix.
The unregularized maximum likelihood estimate of the variance for coordinate $j \in \{1, \dots, K\}$ is:
$$\sigma^2_{k, m, j} = \frac{1}{\sum_{s \in \mathcal{C}_k} \gamma_{s, m}} \sum_{s \in \mathcal{C}_k} \gamma_{s, m} (u_{s, j} - \mu_{k, m, j})^2$$
where $\gamma_{s, m}$ is the posterior responsibility of component $m$ for sample $s$. Under small sample sizes ($N = |\mathcal{C}_k| = 64$), $\sigma^2_{k, m, j}$ under-estimates the true manifold variance (collapsing toward zero), making the log-likelihood calculation highly unstable.

### 2.3 Ledoit-Wolf-style Covariance Shrinkage for GMM
To regularize the estimated diagonal covariance matrices, we define the shrunk diagonal covariance matrix $\Sigma^{*}_{k, m}$ for each mixture component as:
$$\Sigma^{*}_{k, m} = (1 - \alpha) \Sigma_{k, m} + \alpha T$$
where $T = \nu I \in \mathbb{R}^{K \times K}$ is the spherical shrinkage target, $\nu = \frac{1}{K} \text{tr}(\Sigma_{k, m})$ is the average sample variance, and $\alpha \in [0, 1]$ is the shrinkage intensity.
We compute the optimal shrinkage intensity $\alpha_{\text{opt}}$ analytically following the Ledoit-Wolf formulation to balance bias and variance under finite samples:
$$\alpha_{\text{opt}} = \max\left(0, \min\left(1, \frac{\sum_{j=1}^K \text{Var}(\hat{\sigma}^2_{j})}{\sum_{j=1}^K (\sigma^2_j - \nu)^2}\right)\right)$$
where $\hat{\sigma}^2_j$ are the individual sample coordinate variances.

### 2.4 Coordinate Log-Likelihood Evaluation & Fallback
The regularized log-likelihood of a sample's similarity coordinate $u_b$ under task $k$'s GMM is computed as:
$$\log P(u_b \mid \text{GMM}_k) = \log \sum_{m=1}^M \pi_{k, m} \mathcal{N}(u_b \mid \mu_{k, m}, \Sigma^{*}_{k, m})$$
The query is routed if:
$$\max_k \log P(u_b \mid \text{GMM}_k) \ge \eta$$
If below the safety threshold $\eta$, the sample is flagged as OOD, completely bypassing the dynamic adapter paths and triggering the modality-specific fallback flow (raising "OOD / Unknown" label for vision classification, or falling back to the frozen base model language model distribution for sequence generation).

---

## 4. Architecture Specifications
*   **Backbone Network:** Pre-trained, frozen shared Vision Transformer (`vit\_tiny\_patch16\_224`, $L=12$, $D=192$).
*   **Routing Block (ZCA):** Layer 3 CLS token feature extractor mapping to $\mathbb{R}^K$.
*   **Density Estimator Module (SRC-DE):** A diagonal GMM with $M=2$ components per task fitted in coordinate space $\mathbb{R}^K$ during the offline calibration phase using Ledoit-Wolf shrinkage.
*   **Expert Adapters:** LoRA adapters (rank $r=8$) inserted into mid-to-late layers (Blocks 4--12) of the backbone.
*   **Output Blending Layer (SPS):** Single-Pass Scatter-Gather activation blending:
    $$h_b^{(l)} = h_b^{(l-1)} W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_{k, b} \left( h_b^{(l-1)} A_k^{(l)} B_k^{(l)} \right)$$

---

## 5. Baselines
We evaluate SRC-DE against three critical baselines to prove its superiority:
1.  **Raw Cosine Thresholding (No Density Model):** Rejects samples based on the raw maximum cosine similarity $\max_k u_{k,b} < \theta$. This baseline is highly vulnerable to asymmetric task dispersion and scale imbalances.
2.  **Unregularized Diagonal GMM (SPS-ZCA):** Fits the GMM in coordinate space via standard maximum likelihood estimation without covariance regularizations or shrinkage.
3.  **L2-regularized (Ridge) GMM:** Regularizes the GMM by adding a static diagonal ridge term: $\Sigma^{*}_{k, m} \gets \Sigma_{k, m} + \gamma I$ where $\gamma = 10^{-4}$. This baseline is non-adaptive, applying identical regularization regardless of the coordinate dimension $K$ or sample size $N$.

---

## 6. Step-by-Step Interaction

### Phase A: Offline Calibration (One-Time Execution)
1.  **Centroid Extraction:** Extract Layer 3 representation vectors for $N=64$ calibration samples $\mathcal{C}_k$ per task, computing the centroids $\mu^{(3)}_k$.
2.  **Coordinate Mapping:** Map all calibration samples across all tasks to the coordinate space by computing their unit-norm similarity coordinates $u_s \in \mathbb{R}^K$.
3.  **Shrunk GMM Fitting:** Fit diagonal GMMs with $M=2$ components over task-specific coordinates. Apply the **Ledoit-Wolf Covariance Shrinkage** formula to compute regularized component covariance matrices $\Sigma^{*}_{k, m}$.

### Phase B: Online Serving (Per-Sample Execution)
1.  **Early Execution:** Execute the shared, adapter-free early-stage blocks (Layers 1--3) of the shared base model, extracting the CLS token activation $h^{(3)}_b$.
2.  **ZCA Projection:** Project $h^{(3)}_b$ against pre-computed Layer 3 centroids to extract similarity coordinates $u_b \in \mathbb{R}^K$.
3.  **Calibrated Coordinate Scaling:** Calibrate the coordinates using Unit-Norm Calibration (UNC) and Intra-Task Dispersion Calibration (IDC) to standardize significance.
4.  **Density Evaluation:** Compute the regularized log-likelihood $\mathcal{L}_b = \max_k \log P(u_b \mid \text{GMM}_k)$ under each task's regularized GMM.
5.  **Rejection Decision & Routing:**
    *   **If $\mathcal{L}_b \ge \eta$ (In-Distribution):** Identify the predicted task $k^* = \text{argmax}_k \mathcal{L}_b$ and route the query using SPS to blend expert adapter activations, executing Blocks 4--12 in a single forward pass.
    *   **If $\mathcal{L}_b < \eta$ (Out-of-Distribution):** Flag the query as OOD, bypass the expert adapter paths completely, and output the "OOD / Unknown" class label.
