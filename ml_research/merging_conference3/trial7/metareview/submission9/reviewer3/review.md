# Peer Review

## 1. Summary of the Paper
This paper proposes **SABLE (Sample-wise Activation Blending of Low-Rank Experts)**, a network-level, activation-space framework for test-time dynamic model merging. When deploying multiple specialized PEFT experts (such as LoRA adapters) in heterogeneous streaming environments, existing parameter-space model merging methods (e.g., Parameter-Free Subspace Routing, or PFSR) suffer from **heterogeneity collapse**—where averaging routing coefficients over a batch dimension collapses task specialization back to a uniform static merge. SABLE resolves this by shifting ensembling from parameter space to activation space during the forward pass.

SABLE utilizes Subspace Cosine Projection against expert classification heads to compute sample-specific coefficients on-the-fly, blends the outputs of parallel low-rank ($r=8$) adapters, and applies Top-$M$ expert pruning to restrict inference overhead to $O(M)$ complexity. The paper evaluates SABLE in a synthetic 14-layer coordinate-space sandbox and on physical CNN, MLP, and pre-trained ResNet-18 benchmarks. SABLE configured with its default Late Adaptation (Mid-Layer Routing) achieves a joint accuracy of **68.10%** under both homogeneous and heterogeneous streams, exhibiting **0.00% collapse** and outperforming the systems-heavy PFSR+MBH pipeline (**67.20%**) while executing in a single, stateless forward pass with a **6.8$\times$ wall-clock latency reduction** and **36.4% peak memory savings** on an NVIDIA A100 GPU.

---

## 2. Strengths and Weaknesses

### Strengths:
1. **Elegant Mathematical Shift:** Recognizes that shifting ensembling from parameter space to activation space leveraging the distributive property of matrix multiplication ($X(W_{\text{base}} + \sum_k \alpha_k A_k B_k) = X W_{\text{base}} + \sum_k \alpha_k (X A_k) B_k$) preserves exact mathematical equivalence under single-sample boundaries ($B=1$), enabling sample-wise ensembling in heterogeneous batches without breaking parallel computation.
2. **Exemplary Scientific Self-Awareness and Transparency:** The authors are exceptionally thorough and refreshingly honest about their framework's limitations. They identify and scientifically analyze critical boundaries such as the **Representational Blurring Paradox**, the **Early-Feature Loss Trade-Off**, the **Theoretical Limitations of Zero-Data Centroids**, and **Cumulative Non-Linear Drift**. This high level of rigor and scientific self-awareness is rare and commendable.
3. **Profound Practical Systems and Serving Benefits:** SABLE is completely stateless and network-level, stripping away temporal queues, buffering, sorting, and stream partitioning. The physical A100 GPU benchmarks show a dramatic **6.8$\times$ latency reduction** (12.4 ms vs. 84.6 ms for PFSR+MBH) and **36.4% peak memory savings** (412 MB vs. 648 MB), demonstrating highly practical serving scalability.
4. **Comprehensive Empirical Ablation and Validation:** The authors provide thorough sweeps over adapter rank $r$, routing temperature $\tau$, OOD threshold $\gamma_{\text{OOD}}$, routing depth $L_{\text{route}}$, and centroid strategies, alongside layer-by-layer activation similarity tracking ($>0.83$ in a 4-layer MLP) to quantitatively track representational drift.

### Weaknesses:
1. **Lack of Deep Theoretical Foundations and Stability Proofs:** While the core algebraic observation (the distributive property of matrix multiplication) is correct, the paper lacks formal mathematical proofs or stability bounds for multi-layer activation ensembling. 
   * Specifically, because successive layers in deep networks are separated by non-linear activations ($\sigma$), the distributive property does not strictly hold across multiple layers:
     $$\sigma\left( \sum_k \alpha_k (X W_k) \right) \neq \sum_k \alpha_k \sigma(X W_k)$$
     The paper does not provide a formal theoretical derivation or bound on this **cumulative non-linear drift** as a function of network depth, Lipschitz constants of the activations, or spectral properties of the adapter updates.
2. **Dual-Space Mismatch of Zero-Data Centroids:** The "Zero-Data Centroids" construction method computes similarity by projecting activation feature representations $z_b$ onto classification weights $w_k$.
   * From a theoretical standpoint, this constitutes a **dual-space mismatch**. Weights lie on a manifold optimized to maximize class margins in a parameter space; activations lie on feature manifolds in representational space. Measuring their cosine similarity directly has no formal theoretical guarantees.
   * Additionally, taking the mean of discriminative class weight vectors can trigger **vector cancellation**, shrinking the centroid's norm. While weight L2-normalization (Refined Zero-Data Centroids) empirically mitigates this, it remains a geometric heuristic lacking a mathematically rigorous probabilistic proof of convergence to the true activation-space task centroids.
3. **Extreme Sensitivity of Multi-Layer Ensembling:** In the physical 4-layer MLP experiments (Section 4.4), under standard multi-layer execution, SABLE Soft and Hard achieve **52.50%** and **50.10%** joint accuracy, respectively.
   * Crucially, **static Uniform Merging outperforms both SABLE variants (achieving 54.80%)**.
   * SABLE only outperforms Uniform Merging when using **Single-Pass Early-Routing** (65.20%), which requires routing strictly in input space. This input-space routing assumption completely breaks down for complex, high-dimensional, or noisy real-world inputs where raw feature spaces lack semantic task separability. In those complex settings, SABLE's multi-layer ensembling can easily perform *worse* than static, non-adaptive Uniform Merging unless late adaptation is enforced.
4. **Speculative and Qualitative Explanations of Anomalies:** Several highly interesting empirical phenomena are explained via qualitative metaphors rather than rigorous mathematical analysis:
   * **The Low-Rank Regularization Paradox:** Explaining why $r=2$ outperforms $r=4$ under SABLE Hybrid as "pruning high-frequency representation noise" is speculative and lacks a mathematical definition or formal proof of this regularizing filter.
   * **Destructive Representational Interference:** Explaining why Soft blending ($M=2$) underperforms Hard routing ($M=1$) at $r=8$ under confounded streams as "colliding in intermediate layers, causing severe mutual cancellation" is purely qualitative and lacks algebraic or geometric grounding.
5. **Toy-Scale Physical Evaluations:** The physical experiments are restricted to a 3-layer CNN, a 4-layer MLP, and a frozen ResNet-18 feature extractor on MNIST and FashionMNIST ($28\times 28$ grayscale images). While they include a "Real-World Validation Blueprint" (Section 4.2), they do not provide actual end-to-end evaluations on standard full-scale deep architectures (e.g., full ViT-B or LLaMA-3) on standard visual or text datasets, leaving SABLE's scalability to full deep models unverified.

---

## 3. Soundness
* **Rating:** Good
* **Justification:** SABLE's underlying linear algebraic identity is correct, and the sample-wise activation blending is technically sound. The physical serve-time latency and memory tracking on the A100 GPU are highly rigorous, and the authors are exceptionally honest about their framework's limitations (e.g., detail-rich analysis of the Representational Blurring Paradox). However, the technical soundness is limited from a theoretical perspective due to:
  1. The dual-space mismatch and vector cancellation heuristics used in zero-data routing.
  2. The lack of formal mathematical proofs/bounds on cumulative non-linear representation drift across multiple layers.
  3. The fact that standard multi-layer SABLE without input-separable features can perform worse than simple static Uniform Merging.

---

## 4. Presentation
* **Rating:** Excellent
* **Justification:** The paper is exceptionally well-written, structured, and easy to follow. The mathematical notations are precise and consistent, and Figure 1 provides a very clear architectural schematic of SABLE's Late Adaptation block. More importantly, the level of thoroughness and transparency in discussing limitations, trade-offs, and future work is exemplary and represents a model of high-quality scientific reporting.

---

## 5. Significance
* **Rating:** Good
* **Justification:** The paper addresses a highly important serving bottleneck—heterogeneity collapse in test-time dynamic model merging under heterogeneous streams. By shifting ensembling to activation space, SABLE provides a highly scalable, stateless alternative that completely bypasses stateful systems pipelines like MBH, cutting wall-clock serve latency by 6.8$\times$. This systems impact is highly significant. However, the theoretical significance is modest due to the lack of formal convergence proofs and the reliance on heuristic parameter-based centroid construction.

---

## 6. Originality
* **Rating:** Good
* **Justification:** The shift from parameter-space ensembling to activation-space ensembling to resolve batch-level heterogeneity collapse is a clever and original idea. The proposed "Refined Zero-Data Centroids" and "Layer-Dependent Hybrid-Rank Selection Protocol" are novel practical designs. However, the theoretical originality is incremental, as the underlying algebra relies on the basic distributive property of matrix multiplication, and the routing mechanics are direct analogues of existing sparse MoE gating layers.

---

## 7. Overall Recommendation
* **Rating:** 4: Weak Accept
* **Justification:** SABLE is a highly practical, clever, and elegantly engineered solution to a severe systems bottleneck (heterogeneity collapse) in real-time model ensembling. Its stateless, network-level single-pass execution delivers massive physical latency and memory savings, returning serving to a clean, stateless paradigm. The writing quality is outstanding, and the authors' rigorous self-awareness and thoroughness in documenting limitations are highly commendable.

However, the paper is limited by a lack of deep theoretical grounding. It relies heavily on heuristic centroid constructions (introducing a dual-space mismatch and vector cancellation), lacks formal mathematical proofs/stability bounds on cumulative non-linear representational drift, and underperforms static Uniform Merging under standard multi-layer setups unless late adaptation or highly specific input-separable routing is used. Furthermore, physical evaluations are restricted to toy-scale grayscale datasets (MNIST/FashionMNIST). 

Overall, SABLE represents a solid contribution that others in the parameter-efficient ensembling and multi-tenant serving community are highly likely to build on. I recommend a **Weak Accept**, and I strongly encourage the authors to address the theoretical gaps (by deriving formal bounds on non-linear drift and proving the geometric properties of zero-data centroids) and scale up physical evaluations in their final version to unlock full standard-setting status.
