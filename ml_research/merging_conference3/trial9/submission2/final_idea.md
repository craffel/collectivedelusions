# Idea Proposal: Resource-Budgeted Top-M Expert Serving (RB-TopM)

## 1. Persona Alignment
As **The Pragmatist**, our primary goal is to make machine learning systems deployable, robust, and highly cost-effective under real-world constraints. Standard dynamic model-merging methods (such as SABLE and SPS-ZCA) assume constant, unlimited computing resources on edge hardware, running parallel expert adapters for every input query. 

In the wild, edge devices (e.g., mobile phones, microcontrollers, and autonomous robots) operate in highly volatile environments with fluctuating compute budgets, thermal throttling limits, and low-battery constraints. **RB-TopM** directly addresses these critical real-world bottlenecks. By introducing a hardware-aware control loop ($C_{\text{budget}}$) that dynamically scales the active expert capacity and gates marginal expert executions on-the-fly, RB-TopM achieves a smooth, adaptive accuracy-latency trade-off without requiring model retraining, fine-tuning, or offline profiling. This allows the system to remain robust, responsive, and power-efficient across varying hardware constraints.

## 2. Core Techniques
- **Dynamic Compute Budget Control ($C_{\text{budget}}$):** A system-level control parameter $C_{\text{budget}} \in [0, 1]$ representing real-time edge hardware resource availability (where $1.0$ is maximum performance, and $0.0$ is extreme power-saving mode).
- **Resource-Budgeted Top-$M$ Cap ($M(C_{\text{budget}})$):** A dynamic constraint that restricts the maximum number of active parallel expert adapters per sample. As $C_{\text{budget}}$ drops, $M$ scale-down converges to 1 (hard, single-expert routing), which completely bypasses parallel execution paths to save latency.
- **Adaptive Gating Threshold ($\theta(C_{\text{budget}})$):** A dynamic threshold that filters out expert adapter pathways whose routing coefficients contribute minimally to the output representation. Under high resource pressure, $\theta$ increases to aggressively prune inactive paths.
- **Zero-Shot Centroid Alignment (ZCA) with Scale Calibration:** Leverages early-stage representation space (Layer 3) task centroids, Unit-Norm Calibration (UNC), and Intra-Task Dispersion Calibration (IDC) to ensure robust and unbiased routing under heterogeneous inputs.

## 3. Mathematical Formulation

Let the dynamic resource compute budget be $C_{\text{budget}} \in [0, 1]$. We define the resource-dependent routing capacity parameters as follows:

1. **Dynamic Top-$M$ Cap ($M$):**
   $$M(C_{\text{budget}}) = \max\left(1, \lfloor M_{\max} \cdot C_{\text{budget}} \rfloor\right)$$
   where $M_{\max} = 4$ is the maximum allowed expert blend size.

2. **Adaptive Gating Threshold ($\theta$):**
   $$\theta(C_{\text{budget}}) = \theta_{\min} + (1 - C_{\text{budget}}) \cdot (\theta_{\max} - \theta_{\min})$$
   where $\theta_{\min} = 0.001$ is the baseline coefficient threshold and $\theta_{\max} = 0.20$ is the aggressive pruning threshold.

3. **Subspace Cosine Projection & IDC Alignment:**
   For sample $b$ with early activation $h^{(3)}_b \in \mathbb{R}^D$ and task $k$ with pre-computed centroid $\mu^{(3)}_k \in \mathbb{R}^D$:
   $$u_{k, b} = \frac{h^{(3)}_b \cdot \mu^{(3)}_k}{\|h^{(3)}_b\|_2 \|\mu^{(3)}_k\|_2}$$
   Applying Intra-Task Dispersion Calibration (IDC) with expected similarity scale $s_k$:
   $$u'_{k, b} = \frac{u_{k, b}}{s_k}$$

4. **Temperature-Scaled Softmax Routing:**
   $$\hat{\alpha}_{k, b} = \frac{\exp(u'_{k, b} / \tau)}{\sum_{j=1}^K \exp(u'_{j, b} / \tau)}$$
   where $\tau > 0$ is the softmax routing temperature.

5. **Top-$M$ Masking and Gating:**
   Let $\mathcal{K}_{\text{Top-}M}$ be the set of indices of the largest $M(C_{\text{budget}})$ values in $\hat{\alpha}_b$. We zero out all non-top-$M$ coefficients:
   $$\tilde{\alpha}_{k, b} = \begin{cases} \hat{\alpha}_{k, b} & \text{if } k \in \mathcal{K}_{\text{Top-}M} \\ 0 & \text{otherwise} \end{cases}$$
   Re-normalize the top-$M$ coefficients:
   $$\bar{\alpha}_{k, b} = \frac{\tilde{\alpha}_{k, b}}{\sum_{j=1}^K \tilde{\alpha}_{j, b}}$$
   Apply the resource-dependent pruning threshold $\theta(C_{\text{budget}})$:
   $$\alpha_{k, b} = \begin{cases} \bar{\alpha}_{k, b} & \text{if } \bar{\alpha}_{k, b} \ge \theta(C_{\text{budget}}) \\ 0 & \text{otherwise} \end{cases}$$
   And perform final re-normalization to maintain representation scales:
   $$\alpha^*_{k, b} = \begin{cases} \frac{\alpha_{k, b}}{\sum_{j=1}^K \alpha_{j, b}} & \text{if } \sum_{j=1}^K \alpha_{j, b} > 0 \\ 0 & \text{otherwise (OOD or extreme fallback to base)} \end{cases}$$

6. **Single-Pass Blending Forward Equation:**
   For adapted block layers $l \in \{4, \dots, L\}$:
   $$h_b^{(l)} = h_b^{(l-1)} W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha^*_{k, b} \left( h_b^{(l-1)} A_k^{(l)} B_k^{(l)} \right)$$

## 4. Architecture Specifications
- **Backbone model:** 14-layer, 192-dimensional Analytical Coordinate Sandbox (ICS) simulating multi-task vision streams (MNIST, Fashion-MNIST, CIFAR-10, SVHN).
- **Expert pathways:** $K$ specialized downstream low-rank adapters (LoRA) inserted at layers $l \in \{4, \dots, L\}$ with rank $r = 8$ and scaling parameter $\alpha_{\text{lora}} = r$.
- **Routing layer:** Executed immediately at the output of block Layer 3. Centroids are extracted from a small 64-sample calibration split $\mathcal{C}_k$ per task.
- **OOD Detection:** Integrates a diagonal Coordinate Gaussian Mixture Model (GMM) with $2$ components fitted over $\mathcal{C}_k$ coordinates to filter out invalid queries prior to calibration and gating.

## 5. Baselines
- **Expert Oracle (Performance Upper Bound):** Executes each sample strictly through its corresponding single-task ground-truth expert.
- **Static Uniform Merging:** Simple parameter-space parameter average ($W_{\text{merged}} = W_{\text{base}} + \frac{1}{K}\sum V_k$). Vulnerable to Heterogeneity Collapse.
- **SABLE (Standard Blending):** Standard, un-gated activation blending with static parameters and no resource-budgeting or threshold pruning.
- **SPS-ZCA:** Single-pass sample-wise routing with a static temperature ($\tau = 0.001$) and no adaptive gating.
- **Q-SPS:** Quantized activation blending with static hard thresholds.

## 6. Step-by-Step Interaction
1. **System Initialization:** The edge device queries the OS runtime environment to fetch the current dynamic resource availability coefficient $C_{\text{budget}} \in [0, 1]$.
2. **Resource Calibration:** The RB-TopM control loop calculates the active expert cap $M(C_{\text{budget}})$ and the adaptive gating threshold $\theta(C_{\text{budget}})$.
3. **Early Feature Extraction:** The input batch $X$ of size $B$ is processed by the shared, adapter-free early backbone (Layers 1--3) to produce early activations $h^{(3)}_b$.
4. **ZCA Cosine Routing:** The router projects early activations onto the pre-computed Layer 3 centroids to obtain similarity coordinates $u_{k,b}$.
5. **OOD GMM Rejection:** Each coordinate is evaluated against the Coordinate GMM of its predicted task. If it falls below a threshold $\eta$, the sample is flagged as OOD, and the router sets $\alpha^*_{k,b} = 0$, defaulting the sample to the pre-trained base model or an OOD label fallback.
6. **Scale Calibration & Softmax:** In-distribution samples are calibrated using Intra-Task Dispersion Calibration (IDC) and passed through the temperature-scaled Softmax to compute $\hat{\alpha}_{k,b}$.
7. **Sparse Top-$M$ Gating:** The router zeros out all non-top-$M(C_{\text{budget}})$ coefficients and applies the adaptive threshold pruning using $\theta(C_{\text{budget}})$.
8. **Adaptive Parallel Execution:** The remaining layers (Layers 4--$L$) are executed in a single forward pass. For each sample, the system executes only the active LoRA expert paths (where $\alpha^*_{k,b} > 0$), bypassing all pruned paths to save substantial DRAM bandwidth and edge computation.
9. **Final Output:** The final outputs are computed and passed to task-specific heads or fallback channels.
