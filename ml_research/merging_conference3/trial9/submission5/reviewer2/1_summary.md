# 1. Summary

## Paper Overview
The paper presents a methodological audit and experimental deconstruction of state-of-the-art (SOTA) activation-space dynamic model merging methods—specifically, SABLE (Sample-wise Activation Blending of Low-Rank Experts) and ChemMerge (which uses continuous-time chemical kinetics). The authors investigate the widely held consensus that classical parametric linear routers catastrophically fail in low-data calibration regimes ($N_{\text{cal}} \le 128$). They hypothesize that these reported failures are not due to fundamental representational limitations of linear gating, but are confounding artifacts of weak experimental methodology, specifically random weight initialization and a lack of proper regularization.

To audit these claims, the authors introduce a properly regularized, maximum-entropy zero-initialized classical linear router. They evaluate these gating systems within a 14-layer, 192-dimensional synthetic Analytical Coordinate Sandbox (ICS) under controlled representation anisotropy, and validate their findings on a real pre-trained BERT-Tiny model.

---

## Technical Approach
The proposed classical parametric router maps early network activation features $h_b^{(3)} \in \mathbb{R}^D$ (where $D = 192$) to blending coefficients $\boldsymbol{\alpha}_b \in \mathbb{R}^K$ using a single linear layer $W_g \in \mathbb{R}^{K \times D}$ and bias $b_g \in \mathbb{R}^K$:
1. **Competitive Softmax Router:** $\alpha_{k,b} = \text{Softmax}_k(W_g h_b^{(3)} + b_g)$
2. **Cooperative Sigmoid Router:** $\alpha_{k,b} = \sigma(W_g h_b^{(3)} + b_g)$

To stabilize training and prevent initialization bias, the authors employ **Maximum-Entropy Zero-Initialization**, initializing weights to $W_g = \mathbf{0}, b_g = \mathbf{0}$. Under zero-initialization, the Softmax router outputs uniform weights ($\alpha_{k,b} = 1/K$, which matches static Uniform Merging) and the Sigmoid router outputs $\alpha_{k,b} = 0.5$.

The router parameters are optimized via Empirical Risk Minimization under an $L_2$ weight-decay complexity penalty:
$$\min_{W_g, b_g} \frac{1}{|\mathcal{D}_{\text{cal}}|} \sum_{(x_i, y_i) \in \mathcal{D}_{\text{cal}}} \mathcal{L}\left( f(x_i; W_g, b_g), y_i \right) + \lambda \|W_g\|_F^2$$
where $\lambda$ is swept across $\{0.0, 10^{-4}, 10^{-2}, 1.0\}$.

The paper stress-tests these methods against representation anisotropy by injecting a Toeplitz covariance structure into the synthetic task signatures $v'_k = \Sigma^{1/2} v_k$, where $\Sigma_{i, j} = \rho^{|i - j|}$ and the entanglement coefficient $\rho$ is swept from $0.0$ to $0.5$.

---

## Key Findings
1. **The Small-Sample Bottleneck is an Overfitting Artifact:** In the small-sample regime ($N_{\text{cal}} = 64$), unregularized or poorly regularized classical routers catastrophically degrade, achieving only $67.34\% \pm 0.58\%$ (Softmax) and $63.52\% \pm 0.66\%$ (Sigmoid) accuracy, lagging far behind SABLE ($73.76\% \pm 0.72\%$) and ChemMerge ($76.90\% \pm 0.68\%$). The authors explain that learning $768$ parameters from $64$ samples is an under-determined optimization problem. SABLE and ChemMerge succeed here not because of representational superiority, but because their training-free, cosine-based projections act as highly effective inductive geometric priors requiring zero parameter updates.
2. **Complete Generalization Recovery:** In the large-sample regime ($N_{\text{cal}} = 4000$), classical routers recover spectacularly. The unregularized Softmax router achieves $76.22\% \pm 0.78\%$ accuracy, outperforming SABLE by $+2.46\%$ absolute and closely approaching ChemMerge's performance ceiling ($76.90\% \pm 0.68\%$).
3. **The Bias-Variance Trade-off in Regularization:** The optimal regularization hyperparameter $\lambda$ scales inversely with $N_{\text{cal}}$. In the small-sample regime ($N_{\text{cal}} = 64$), a strong regularizer ($\lambda = 10^{-2}$) is required to restrict the hypothesis space. However, under data abundance ($N_{\text{cal}} = 4000$), strong regularization introduces an unnecessary constraint bias, capping accuracy at $74.10\% \pm 0.85\%$, whereas a weaker regularizer ($\lambda = 10^{-4}$) achieves a near-optimal $75.70\% \pm 0.95\%$.
4. **Deconstruction of Stateful Trajectory Smoothing:** The paper tracks layer-wise intermediate representation quality (cosine similarity to task prototypes). It reveals that ChemMerge's continuous ODE kinetics introduce a severe "representational lag" (reaching $0.912$ at Layer 14 compared to the classical router's $0.992$). However, under a control-theoretic lens, this lag acts as a beneficial closed-loop temporal low-pass filter (stateful inertia) that stabilizes ensembling trajectories under activation noise, explaining its high performance ceiling.
5. **Real-World Foundation Model Validation:** Evaluations on a pre-trained BERT-Tiny model on GLUE tasks (SST-2 vs. QQP) confirm that classical parametric routers outperform SABLE and ChemMerge under a calibration budget of 500 samples, and do not suffer from overfitting in the low-data regime ($N_{\text{cal}} = 32$) due to the natural semantic separability of the task subspaces.

---

## Claimed Contributions and Evidence
- **Methodological Audit of Dynamic Model Merging SOTA:** Exposing that prior reports of classical router failure are artifacts of poor experimental setups. (Evidence: Controlled experiments sweeping $N_{\text{cal}}$, initialization, and regularization in Tables 1, 2, and 3).
- **Maximum-Entropy Zero-Initialization and Regularized Routing:** Proposing standard zero-initialization and proper L2 weight decay as a strong, mathematically sound baseline. (Evidence: Table 1 and 2 demonstrating accuracy improvements and graceful degradation to Uniform Merging under heavy regularization).
- **Control-Theoretic Formulation of Stateful Kinetics:** Reframing ChemMerge's ODE kinetics as a closed-loop feedback controller that acts as a low-pass filter. (Evidence: Figure 1b tracking layer-wise prototype similarity, showing the stabilizing effect of representational lag under noise).
- **Sample-Complexity and Temperature Sensitivity Maps:** Providing transition boundaries and hyperparameter sensitivity profiles. (Evidence: Figures 2a and 2b).
- **Real-World Pre-trained Validation:** Validating generalizability on BERT-Tiny. (Evidence: Table 5 showing a $+2.50\%$ improvement for parametric routers over training-free alternatives under $N_{\text{cal}} = 500$).
