# SpectralMerge: Frequency-Domain Model Merging via Discrete Cosine Transform (DCT)

## 1. Persona Alignment
This project aligns directly with **The Visionary** persona:
- **Radical Departure from Euclidean Norms:** All prior works (such as Task Arithmetic, AdaMerging, and PolyMerge) operate strictly in the spatial weight space. By transitioning the weight-merging coefficients to the **frequency domain**, we completely challenge the flat Euclidean spatial assumption of parameters.
- **Rethinking Fundamental Assumptions:** We hypothesize that layer-wise parameter sensitivity is best modeled as a multi-scale spectral distribution rather than a spatial sequence. High-frequency oscillations represent optimization and generalization noise, while low-frequency components capture the smooth, continuous multi-task pathways across the network's depth.
- **High-Risk, High-Novelty:** Designing spectral-domain optimization with low-pass filters and frequency-dependent decay penalties is an entirely novel approach to deep model merging, introducing concepts of digital signal processing to parameter consolidation.

---

## 2. Core Techniques
1. **1D Discrete Cosine Transform (DCT-II):** We apply the orthonormal Discrete Cosine Transform to map the layer-wise merging coefficient profiles from the physical spatial depth domain to the spectral frequency domain.
2. **Low-Pass Spectral Filtering (SpectralMerge-LP):** A hard-cutoff frequency filter that restricts optimization to the first $F$ low-frequency components, mathematically preventing the fitting of high-frequency local noise.
3. **Soft Spectral Regularization (SpectralMerge-Reg):** An unconstrained spectral optimization framework where all frequency coefficients are optimized but subject to a quadratic **Spectral Decay Penalty** ($\beta_j = \lambda \cdot j^2$), which penalizes high-frequency oscillations.
4. **Offline Few-Shot Validation Tuning (OFS-Tune):** We leverage derivative-free Nelder-Mead simplex search and gradient-based Adam optimizers on tiny validation sets ($M \in [5, 50]$) to optimize the spectral parameters.

---

## 3. Mathematical Formulation
Let $\vec{\alpha}_k = [\alpha_k(1), \dots, \alpha_k(L)]^T \in \mathbb{R}^L$ be the unconstrained layer-wise merging coefficients for expert task $k \in \{1, \dots, K\}$ across network depth $L = 12$.

### Forward Transform (Spatial to Frequency Domain):
We compute the 1D Discrete Cosine Transform (DCT-II) of $\vec{\alpha}_k$ to obtain spectral coordinates $\vec{c}_k = [c_{k,0}, \dots, c_{k,L-1}]^T$:
$$c_{k,j} = \text{DCT}(\vec{\alpha}_k)_j = \sqrt{\frac{2}{L}} \gamma_j \sum_{l=1}^L \alpha_k(l) \cos\left( \frac{\pi j (l - 0.5)}{L} \right)$$
where:
$$\gamma_j = \begin{cases} \frac{1}{\sqrt{2}} & \text{if } j = 0 \\ 1 & \text{if } j > 0 \end{cases}$$

### Inverse Transform (Frequency to Spatial Domain):
We reconstruct the spatial merging coefficients $\vec{\alpha}_k$ using the Inverse Discrete Cosine Transform (IDCT-III):
$$\alpha_k(l) = \text{IDCT}(\vec{c}_k)_l = \sqrt{\frac{2}{L}} \sum_{j=0}^{L-1} \gamma_j c_{k,j} \cos\left( \frac{\pi j (l - 0.5)}{L} \right)$$

### Optimization Objectives:
1. **SpectralMerge-LP ($F$-dimensional constraint):**
   We optimize only the first $F$ low-frequency coefficients, setting $c_{k, j} = 0$ for all $j \ge F$.
   $$\vec{c}_k^* = \arg\min_{\vec{c}_k \in \mathbb{R}^F} \mathcal{L}_{val}\left(\text{IDCT}\left([\vec{c}_k, \vec{0}_{L-F}]^T\right)\right)$$
2. **SpectralMerge-Reg (Regularized unconstrained spectrum):**
   We optimize all $L$ frequency coefficients, subject to the Spectral Decay Penalty:
   $$\vec{c}^* = \arg\min_{\vec{c}} \mathcal{L}_{val}\left(\text{IDCT}(\vec{c})\right) + \sum_{k=1}^K \sum_{j=0}^{L-1} \lambda_j c_{k,j}^2$$
   where $\lambda_j = \mu \cdot j^2$ scales quadratically with frequency $j$, and $\mu > 0$ is the regularization strength.

---

## 4. Architecture Specifications
- **Input representation:** For $K=4$ tasks and $L=12$ layers, the raw unconstrained parameters are represented as a coefficient matrix of size $4 \times 12$.
- **Frequency representation:** In the spectral domain, we optimize a coefficient matrix $\theta_{spec} \in \mathbb{R}^{K \times F}$ (for hard-cutoff) or $\theta_{spec} \in \mathbb{R}^{K \times L}$ (for soft regularization).
- **Activations and Dimensions:** The final outputs are the spatial merging coefficients $\vec{\alpha}_k \in \mathbb{R}^12$, which are mapped via the IDCT from $\theta_{spec}$. These coefficients are then used to scale task vectors during model merging:
  $$W_{merged}^{(l)} = W_{base}^{(l)} + \sum_{k=1}^K \alpha_k(l) V_k^{(l)}$$

---

## 5. Baselines
We evaluate SpectralMerge against:
1. **Uniform (Task Arithmetic):** Serves as the standard unoptimized baseline ($\alpha_k = 0.3$).
2. **Unconstrained Layer-Wise Search (AdaMerging):** Optimizes all $K \times L$ (48) layer parameters independently. This represents the high-frequency overfitted ceiling.
3. **Poly-Val-Merge (PolyMerge):** Uses low-degree polynomials across normalized depth. This is the state-of-the-art spatial-smoothing baseline. Comparing against it verifies whether our orthogonal frequency representation provides better optimization conditioning and generalization.

---

## 6. Step-by-Step Interaction
1. **Spectral Parameter Initialization:** Initialize all frequency coefficients $\vec{c}_k$ by taking the DCT of the uniform baseline $\vec{\alpha}_k = [0.3, \dots, 0.3]^T$. This sets $c_{k,0} = 0.3 \times \sqrt{L} \approx 1.039$ and $c_{k,j} = 0$ for all $j > 0$.
2. **Inverse Transform to Spatial Domain:** Compute the spatial coefficients $\alpha_k(l)$ using the IDCT-III on the current spectral coordinates $\vec{c}_k$.
3. **Weight Merging & Prediction Evaluation:** Blends the task expert vectors using the computed $\alpha_k(l)$, constructs the merged model $W_{merged}(\alpha)$, and runs predictions on the validation set $D_{val}$ (or evaluates the simulated loss).
4. **Objective & Penalty Evaluation:** Evaluates the loss (cross-entropy or simulated accuracy loss). If using `SpectralMerge-Reg`, we compute and add the quadratic Spectral Decay Penalty on the frequency coordinates.
5. **Optimization Step:** Black-box (Nelder-Mead) or gradient-based (Adam) updates are applied to the spectral coordinates $\vec{c}_k$.
6. **Convergence:** Once optimized, the final spectral coordinates $\vec{c}_k^*$ are converted via the IDCT-III to produce static, smooth spatial coefficients $\vec{\alpha}_k^*$, generating the final robust multi-task model.
