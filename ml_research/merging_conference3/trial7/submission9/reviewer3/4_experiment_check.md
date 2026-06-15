# Evaluation Component 4: Experimental Evaluation and Claims Check

## 1. Critique of the Experimental Setup and Datasets
The authors evaluate SABLE across three primary environments:
1. **The Analytical Coordinate Sandbox:** A synthetic 14-layer ($L=14$), 192-dimensional ($D=192$) coordinate-space model.
2. **Physical CNN and MLP Models (from scratch):** Tested on MNIST and FashionMNIST ($28\times 28$ grayscale images).
3. **Pre-trained ResNet-18 (frozen feature extractor + 2-layer MLP head):** Tested on MNIST and FashionMNIST.

### Critique of Dataset and Network Scale:
* **Idealized Sandbox:** The 14-layer sandbox is highly synthetic and clean. While it is useful for analyzing isolated subspace projections under controlled conditions, its orthogonal subspaces do not reflect the complex, highly non-linear, and cluttered representation manifolds found in standard deep neural networks trained on natural data.
* **Toy-Scale Physical Evaluations:** The physical CNN and MLP experiments on MNIST and FashionMNIST represent extremely simplified, toy-scale benchmarks. Modern deep learning operates in highly complex, high-dimensional spaces (e.g., ImageNet, large-scale multi-modal corpora). Grayscale $28\times 28$ images are highly separable and easy to classify, meaning SABLE's strong performance on these benchmarks may not generalize to real-world tasks.
* **Lack of Full-Scale Multi-Layer Network Evaluations:** SABLE claims to scale gracefully to deep architectures, but there are no actual end-to-end evaluations on full, standard deep neural networks (e.g., full ResNet-50, ViT-B, or LLaMA) where all layers are actively ensembled. The "Real-World Validation Blueprint" in Section 4.2 and the discussion in Section 5.2 are purely conceptual outlines. Consequently, SABLE's performance in full-scale, deep, multi-layer natural settings remains unproven.

---

## 2. Baseline Comparisons and Omissions
The paper compares SABLE against appropriate model-merging baselines, including Uniform Merging, PFSR, and PFSR+MBH. 

However, there is a notable **lack of direct quantitative comparisons against established PEFT-specific ensembling frameworks** (such as LoraHub or MoE-Adapters). While the authors justify this omission qualitatively by pointing out that these methods require training/calibration splits or are static at test-time, omitting them from the experimental tables prevents a rigorous benchmark against the state-of-the-art in parameter-efficient ensembling.

---

## 3. Analysis of Claims vs. Experimental Evidence

### A. The "0.00% Collapse" Claim
* *Claim:* SABLE is completely immune to heterogeneity collapse under heterogeneous streams.
* *Evidence:* Supported. Across all experiments (Sandbox, CNN, MLP, and ResNet-18 features), SABLE's performance remains completely flat when transitioning from homogeneous to fully mixed heterogeneous streams, whereas parameter-space PFSR suffers massive accuracy drops (e.g., falling by 15.40% in the Sandbox and 21.70% in the CNN).

### B. The Multi-Layer Ensembling Claim
* *Claim:* SABLE can scale to multi-layer deep networks without catastrophic representational drift.
* *Evidence:* Supported *only under highly specific and restrictive routing configurations*, and actually **refuted under standard multi-layer setups**:
  * In the 4-layer MLP experiment (Section 4.4), when SABLE is run under standard multi-layer execution, SABLE Soft achieves **52.50%** joint mean accuracy, and SABLE Hard achieves **50.10%**.
  * Crucially, **static, non-adaptive Uniform Merging achieves 54.80% in this setup**, outperforming both SABLE variants!
  * SABLE only outperforms Uniform Merging when using **Single-Pass Early-Routing** (achieving **65.20%**).
  * However, as the authors admit, Single-Pass Early-Routing requires routing strictly in input space, which is only feasible when the raw input features are highly separable (e.g., grayscale MNIST pixels vs. FashionMNIST pixels). For high-dimensional, complex, or noisy real-world inputs, raw input-space routing is not feasible and will suffer from severe routing noise.
  * In those complex settings, practitioners face a severe theoretical bottleneck: they must either accept the $2\times$ latency of 2-pass routing (which degrades accuracy to **52.50%** due to "Representational Blurring" in the base model), introduce an external routing backbone (violating the minimalist, single-model claim), or restrict ensembling to late-stage layers (Late Adaptation).
  * This is a critical experimental finding: SABLE's multi-layer activation blending is highly sensitive and can easily perform *worse* than static Uniform Merging unless highly domain-specific, input-dependent routing conditions are met.

### C. The "Low-Rank Regularization Paradox" Claim
* *Claim:* Constraining the hidden layer adapter to $r=2$ acts as a powerful regularizer that improves accuracy compared to $r=4$ (achieving 62.10% vs. 58.90% with Support-16 on ResNet-18 features).
* *Evidence:* Weak. While the empirical difference is clear, the explanation is entirely speculative and post-hoc. The paper does not provide any theoretical proof, spectral analysis of the adapter activations, or quantitative metrics demonstrating that $r=2$ actually filters out "high-frequency representation noise." An alternative explanation could simply be optimization noise or overfitting of the $r=4$ expert during fine-tuning.

### D. The "Destructive Representational Interference" Claim
* *Claim:* High-capacity experts ($r=8$) suffer from destructive manifold interference under confounded streams, making hard routing ($M=1$) superior to soft blending ($M=2$), whereas low-capacity experts ($r=2$) benefit from soft blending.
* *Evidence:* Partially supported empirically, but lacks mathematical grounding. The paper shows that SABLE Hybrid Soft ($r=2, M=2$) achieves 26.00% joint recall on confounded inputs compared to Hard's 24.00%, while at $r=8$ Soft drops to 15.00% and Hard remains at 16.00%. However, this is a very narrow margin (1-2% absolute recall difference), and the underlying mechanics are described only via qualitative analogies ("manifolds collide", "representation scrambling") rather than a formal algebraic or geometric proof.
