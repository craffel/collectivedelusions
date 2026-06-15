# 3. Soundness and Methodology Evaluation

## Clarity of Description
The paper is exceptionally well-structured and written with high mathematical clarity. The equations deriving preconditioned SGLD (Equations 17--19), DSLN scaling, and the Simulated Annealing schedule are clearly stated. The step-by-step progression in Algorithm 1 makes the implementation straightforward to follow. The inclusion of hardware-level JIT/GPU optimization suggestions, such as **noise buffer pre-allocation**, shows strong practical awareness.

---

## Appropriateness of Methods
While the use of SGLD and Simulated Annealing for global optimization is mathematically sound, their application to the joint optimization of heterogeneous parameters during test-time model merging suffers from several major methodological issues.

---

## Potential Technical Flaws and Methodological Weaknesses

### 1. The Redundancy of High-Dimensional Classifier Optimization
The core contribution of the paper involves the joint adaptation of both the low-dimensional merging coefficients ($\Lambda$) and the high-dimensional task-specific classifiers ($\Theta^{tr}$). To stabilize the high-dimensional parameters, the authors introduce the complex DSLN formulation. 

However, Table 7 contains an ablation study, **"ThermoMerge (Coefficients Only)"**, where SGLD is applied *only* to the low-dimensional merging coefficients ($\Lambda$) while the classifiers are updated purely deterministically. 
Under this ablation:
*   MNIST accuracy: $89.89\% \pm 0.17\%$ (compared to $89.94\% \pm 0.16\%$ for full ThermoMerge).
*   FashionMNIST accuracy: $84.35\% \pm 0.45\%$ (compared to $84.46\% \pm 0.59\%$ for full ThermoMerge).
*   KMNIST accuracy: $80.36\% \pm 0.31\%$ (compared to $80.37\% \pm 0.24\%$ for full ThermoMerge).

The performance difference between SGLD on coefficients-only versus full joint SGLD is **statistically negligible** (within standard deviations) across all three datasets. This suggests that the entire apparatus of high-dimensional classifier SGLD, along with the DSLN scaling rules, layer-wise functional grouping, and weight-bias thermodynamic balancing, is largely redundant. The adaptation benefits are almost entirely driven by the optimization of the low-dimensional coefficients $\Lambda$.

### 2. The High-Dimensional Noise Paradox in DSLN
The DSLN formulation scales the coordinate-wise thermal noise standard deviation by $1/\sqrt{d_j}$. For a high-dimensional parameter group (such as a classification head with $d_j = 640$), this scales down the coordinate-wise noise standard deviation $\sigma_j$ by a factor of $\approx 25.3$. For actual deep neural networks where classification heads or layer weights easily exceed $10^5$ to $10^7$ parameters, the coordinate-wise noise is scaled down by a factor of $300$ to $3000$, resulting in a vanishingly small perturbation standard deviation (e.g., $10^{-4}$ to $10^{-6}$).

This dampening creates a severe theoretical contradiction:
*   The authors claim that injecting Langevin noise allows the parameters to "hop over high energy barriers and escape sharp local traps."
*   However, if the coordinate-wise noise added to each individual weight is scaled down to $10^{-6}$, it is several orders of magnitude smaller than the gradient updates and the machine precision. At this scale, the thermal noise is physically incapable of escaping any non-convex energy barriers in the classifier weight space. 
*   Therefore, the high-dimensional classifiers are adapted **almost purely deterministically**. The DSLN formulation essentially silences SGLD on high-dimensional parameters to prevent feature destruction, confirming that the "global exploration" claim is physically meaningless for the classification heads.

### 3. Severe Test-Time Computational and Latency Overhead of Self-Labeling
The expert-guided soft self-labeling objective (Equation 7) requires forwarding every streaming test batch through all $K$ unmerged, fine-tuned expert models to generate teacher labels. This means that during test-time adaptation (which must run on-the-fly under real-time latency constraints), the system must perform $K$ extra forward passes per step.
*   While the authors claim that SGLD itself adds negligible overhead, the **overall test-time computational footprint scales linearly with the number of experts $K$**.
*   For merging many-task models (e.g., $K = 8$ or $K = 20$ task experts), forwarding $K$ large models is extremely expensive and completely defeats the core purpose of model merging, which is to avoid maintaining and running multiple independent model checkpoints.

### 4. Vulnerability to Teacher Bias and Confirmation Bias
As discussed in Section 3.2, the soft self-labeling objective relies entirely on the correctness of the unmerged expert predictions. If the experts make high-entropy, low-confidence, or systematically incorrect predictions on OOD test samples, these errors are propagated to the merged model.
*   The joint SGLD optimization will be actively misdirected to regions that reinforce these errors, creating a severe **confirmation bias trap**.
*   Although the authors discuss mitigation strategies (confidence filtering, entropy weighting, predictive agreement monitoring), **none of these strategies are integrated, implemented, or validated in their main experiments**. This leaves the core objective theoretically vulnerable to error propagation under severe domain shifts.

---

## Reproducibility
The algorithmic pseudocode in Algorithm 1 is highly detailed and should enable straightforward implementation. However, the initial temperature $T_0$ is highly sensitive and calibrated using a heuristic that requires estimating average gradient norms during the first few adaptation steps. If the initial gradients are highly non-stationary (which is common in deep learning), this calibration can be unstable and easily lead to either undercooling or complete parameter vaporization.
