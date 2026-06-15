# Evaluation Component 1: Summary of the Paper

## 1. Main Topic and Scope
This paper addresses the problem of **test-time dynamic model merging** in heterogeneous streaming environments. Specifically, when multiple specialized parameter-efficient fine-tuned (PEFT) experts (such as LoRA adapters) are ensembled on-the-fly based on query characteristics, existing parameter-space routing methods (e.g., Parameter-Free Subspace Routing, or PFSR) suffer from **heterogeneity collapse**. In a heterogeneous batch containing a mixture of different tasks, these parameter-space merging methods must average their routing coefficients over the batch dimension to maintain a single merged model for the forward pass, collapsing the merged weights to a static, sub-optimal uniform average. 

The paper proposes **SABLE (Sample-wise Activation Blending of Low-Rank Experts)**, a network-level, activation-space alternative that bypasses weight-space merging and its associated systems-level scheduling mitigations (such as Micro-Batch Homogenization, or MBH) by performing ensembling natively in activation space.

---

## 2. Technical Approach
SABLE operates by shifting the ensembling step from parameter space to activation space, leveraging the distributive property of matrix multiplication. The core mathematical framework consists of:

1. **Subspace Cosine Projection:** Measures the similarity $s_{k, b}$ between the global pooled representation $z_b$ of a sample $b$ and a prototype task centroid $w_k$ derived from pre-trained classification heads:
   $$s_{k, b} = \frac{z_b \cdot w_k}{\|z_b\|_2 \|w_k\|_2}$$
2. **Out-of-Distribution (OOD) Rejection:** Employs a hard threshold $\gamma_{\text{OOD}} > 0$. If $\max_j s_{j, b} < \gamma_{\text{OOD}}$, all expert coefficients are set to zero, defaulting to the pre-trained base model.
3. **Temperature-Scaled Softmax Routing:** Computes dynamic blending coefficients $\alpha_{k, b}$ via Softmax:
   $$\alpha_{k, b} = \frac{\exp(s_{k, b} / \tau)}{\sum_{j=1}^K \exp(s_{j, b} / \tau)}$$
4. **Dynamic Activation Blending:** Computes layer-wise output in activation space:
   $$Y_b = X_b W_{\text{base}} + \sum_{k=1}^K \alpha_{k, b} \cdot \left( (X_b A_k) B_k \right)$$
   where $A_k, B_k$ are low-rank ($r \ll D$) adapter matrices.
5. **Early-Layer vs. Mid-Layer Routing:**
   - **Early-Layer Routing:** Runs only the first layer (Layer 0) of the base model to compute routing coefficients and propagates them in a single, unified pass.
   - **Mid-Layer Routing (Late Adaptation):** Leaves early layers unadapted and executes them strictly through the task-agnostic base network. At a routing depth $L_{\text{route}}$, the intermediate representation is extracted to compute $\alpha_{k, b}$, and SABLE blending is applied strictly at the late-stage layers ($l \ge L_{\text{route}}$).
6. **Scalable Top-$M$ Expert Pruning:** Limits ensembling to the top $M \ll K$ active experts with the highest similarity coefficients, bounding complexity at $O(M)$ instead of $O(K)$.
7. **Task-Agnostic Dynamic Head Blending:** Blends classification heads on-the-fly, handling disjoint output spaces via hard expert selection ($M=1$) at the final head while maintaining soft ensembling ($M \ge 2$) in intermediate layers.

---

## 3. Key Findings and Quantitative Claims
The paper evaluates SABLE in several settings:
* **Analytical Coordinate Sandbox:** A synthetic 14-layer ($L=14$), 192-dimensional ($D=192$) coordinate-space sandbox simulating multi-task streams (MNIST, FashionMNIST, CIFAR-10, SVHN noise profiles).
  * Under homogeneous streams, the full-parameter oracle PFSR achieves 71.70%, while SABLE Late Adaptation ($L_{\text{route}} = 12$) achieves **68.10%** and SABLE Early Routing achieves **66.60%**.
  * Under heterogeneous streams, PFSR collapses to 56.30% (a **15.40% collapse**), whereas SABLE Late Adaptation maintains a flat **68.10%** (with **0.00% collapse**), outperforming the systems-heavy PFSR+MBH pipeline (**67.20%**).
* **Physical CNN Validation:** A 3-layer CNN on MNIST and FashionMNIST.
  * Under heterogeneous streams, SABLE Soft ($r=10, M=2$) with Support-16 centroids achieves **69.30%** (0.00% collapse) and with Completely Zero-Data centroids achieves **63.50%** (0.00% collapse), compared to PFSR weight-merging which collapses from 70.70% to 49.00%.
  * Under domain-confounded blended streams, SABLE Soft ($M=2$) achieves **31.00%** joint recall (at $r=10$), outperforming SABLE Hard ($M=1$) which drops to 14.00% because single-expert routing cannot retrieve both tasks simultaneously.
* **Physical Multi-Layer MLP Validation:** A 4-layer MLP on MNIST and FashionMNIST.
  * Single-Pass Early-Routing SABLE Soft achieves **65.20%** joint accuracy under heterogeneous streams, outperforming Uniform Merging (54.80%) and 2-pass Early-Routing (52.50%).
* **High-Dimensional ResNet-18 Feature Validation:**
  * SABLE Hybrid (with low-rank hidden layers and full-rank output projections) at $r=2$ and Support-16 centroids achieves **62.10%** joint accuracy compared to SABLE Strict's 57.20%.
  * Refined Zero-Data Centroids (using L2-normalized weights before averaging) achieve **57.20%** at $r=2$ compared to 51.30% for Naive Zero-Data Centroids.

---

## 4. Explicitly Claimed Contributions (with Evidence)
1. **Mathematical Equivalence at Single-Sample Boundaries:** The authors claim that SABLE is mathematically equivalent to parameter-space ensembling with sample-specific merged weights under single-sample batches ($B=1$).
   * *Evidence:* Derived algebraically from the distributive property of matrix multiplication ($X_b(W_{\text{base}} + \sum_k \alpha_{k, b} A_k B_k) = X_b W_{\text{base}} + \sum_k \alpha_{k, b} (X_b A_k) B_k$).
2. **Absolute Heterogeneity Robustness:** SABLE is claimed to completely eliminate heterogeneity collapse across all evaluation tasks.
   * *Evidence:* SABLE exhibits flatline performance (0.00% collapse) in the Analytical Coordinate Sandbox, CNN, MLP, and ResNet-18 experiments under fully mixed heterogeneous streams.
3. **Stateless serving with Zero Latency Penalty:** Bypasses systems-level dynamic buffering, temporal queues, sorting, and stream partitioning, returning serving to a stateless deep learning forward pass.
   * *Evidence:* average wall-clock latency of 12.4 ms for SABLE vs. 84.6 ms for PFSR+MBH on an NVIDIA A100 GPU (a **6.8$\times$ latency reduction**), with **36.4% peak memory savings** (412 MB vs. 648 MB).
4. **Data-Free Centroid Construction:** Introduces a completely zero-data task centroid construction method derived directly from pre-trained expert classification weights.
   * *Evidence:* Refined Zero-Data Centroids (L2-normalizing classification weights before row-averaging) achieve **61.60%** at $r=16$ on ResNet-18 features, outperforming Naive Zero-Data (58.20%) and closely matching Support-16 centroids (69.30%).
5. **Mitigation of Cumulative Representation Drift:** Claims that activation-space ensembling across multiple sequential hidden layers does not cause catastrophic activation divergence because PEFT updates act as minor localized residual corrections.
   * *Evidence:* Quantitatively tracks layer-by-layer cosine similarity between SABLE and Oracle Expert activations, showing high similarity ($>0.83$) across all intermediate hidden layers and logits in a 4-layer MLP.
