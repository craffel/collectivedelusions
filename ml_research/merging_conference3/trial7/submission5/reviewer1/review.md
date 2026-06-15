# Peer Review Report

## 1. Summary of the Paper
This paper addresses the challenge of serving heterogeneous (mixed-task) inference streams in a multi-task learning environment. Deploying multiple specialized Parameter-Efficient Fine-Tuning (PEFT) experts in production introduces severe memory overhead, whereas weight-space model merging forces static compromises that degrade individual performance due to parameter-level interference. Standard dynamic weight-merging routers suffer from "heterogeneity collapse" when processing mixed-task batches because batch-level average pooling flattens the routing coefficients, scrambling intermediate representations. Prior systems-level solutions, such as Micro-Batch Homogenization (MBH), partition heterogeneous streams into homogeneous sub-batches and dispatch them sequentially, resulting in latency scaling linearly with task diversity ($O(G)$ sequential passes).

To resolve this systems-ML complexity dilemma, the paper proposes **Parameter-Free Activation Blending (PFAB)**, a non-parametric framework that shifts the blending of specialized expert adapters from parameter-space to activation-space. Since activations are indexed sample-wise, activation blending processes heterogeneous batches concurrently in a single, parallelized forward pass of the base model backbone, achieving flat, constant-time execution latency ($O(1)$ backbone passes) while completely avoiding heterogeneity collapse. 

To achieve this, the paper introduces:
1. **Unit-Norm Calibration (UNC):** Projects penultimate activations and classifier weights onto the unit hypersphere to resolve cross-expert scale imbalances.
2. **Non-Parametric Task Coordinates:** Computes sample-wise task routing coefficients by projecting activations onto normalized classification heads using maximum cosine similarity, corrected for class-cardinality bias via an extreme-value statistical divisor $\sqrt{2\log C'_k / D}$.
3. **Activation-Space Adapter Blending (ASAB):** Vectorizes parallel adapter evaluation using batched matrix multiplications (`torch.bmm`) and blends features sample-wise in activation-space.

The paper proposes two execution pathways: **PFAB-BOP** (a two-pass pathway where a base prototyping pass computes routing coefficients from penultimate activations, followed by an activated second pass) and **PFAB-ELC** (a single-pass pathway that resolves task identity at Layer 0 using pre-computed offline task centroids). When cross-task subspaces are entangled, a joint **SVD orthogonalization** of adapter parameters prior to serving is proposed to restore accuracy. When evaluated on the Isolating Coordinate Sandbox, PFAB-BOP matches the prior SOTA (PFSR+MBH) perfectly at **81.50% Joint Mean accuracy** (matching the expert ceiling) but does so with a **2.52$\times$ wall-clock speedup** ($B=64$, $G=4$). The authors also present organic pilot validations of PFAB on a pre-trained ViT-B/16 over DomainNet and map their formulation theoretically to generative LLMs using Task-Specific Vocabulary-Head Anchoring (TSVHA) with a Dynamic Gate Reset (DGR) safeguard.

---

## 2. Strengths
- **Elegant Systems-ML Co-design Concept:** Shifting the model synthesis operation from weight-space (batch-bound) to activation-space (sample-bound) is an elegant conceptual shift. It completely prunes serving-layer database partitioning, compilation dependencies, and scatter-gather re-sorting.
- **Training-Free, Non-Parametric Routing:** The proposed gating strategy projects activations onto existing, frozen classification weights to derive sample-wise routing coefficients. This completely eliminates active parameter training or labeled calibration data, providing a elegant, minimalist alternative to learned parametric gates.
- **Intellectual Honesty and Transparency regarding Limitations:** The authors dedicate substantial sections of the paper to discussing critical physical and semantic limitations, such as the pipeline causality dilemma, base-representation sufficiency, early-layer abstraction gaps, intermediate scale drift, and LLM physical routing lag.
- **Pure PyTorch Portability:** Unlike low-level CUDA serving engines (Punica/SGMV), PFAB operates on pure PyTorch tensor broadcasting. This democratizes zero-overhead multi-task serving on non-NVIDIA hardware (AMD, TPUs, edge CPUs) without complex compilation dependencies.
- **Strong Empirical Latency Scaling:** Both PFAB pathways exhibit completely flat, constant wall-clock execution latency profiles with respect to task diversity, avoiding the linear sequential latency penalty of MBH.

---

## 3. Weaknesses

While the systems-ML conceptual design is elegant, the paper exhibits significant theoretical gaps, unverified mathematical assumptions, and incomplete formulations that must be addressed:

### A. Critical Mathematical Incompleteness of SVD Orthogonalization
To resolve cross-task subspace entanglement, the authors propose a joint Singular Value Decomposition (SVD) on stacked parameter updates to project task-specific adapters onto mutually orthogonal subspaces, showing that it restores accuracy from 51.30% to 80.50% under extreme leakage ($\epsilon = 0.5$). 
However, **the paper completely omits any mathematical formulations, equations, or algorithms** for this SVD-based projection. 
Let $W_k^{(l)} = B_k^{(l)} A_k^{(l)}$ represent low-rank adapter updates ($B_k^{(l)} \in \mathbb{R}^{D \times r}$, $A_k^{(l)} \in \mathbb{R}^{r \times D}$). Orthogonally projecting $W_k^{(l)}$ in the $D \times D$ parameter space will generally increase the rank of the projected matrix, violating the low-rank PEFT constraint. If singular values are truncated to maintain rank $r$, does this truncation degrade the expert's specialized capabilities? How does this projection affect the fine-tuned representations and original task accuracy?
Without formal equations or proofs of representation preservation, this key component is a mathematically incomplete "black box" that cannot be theoretically verified or reproduced.

### B. Unverified Base Representation Sufficiency and Fragility of PFAB-ELC
- **The Prototyping Pass (BOP):** The two-pass PFAB-BOP pathway relies on the assumption that the pre-trained base model $\mathcal{M}_{base}$ contains sufficient semantic signal within its representation space to identify the task domain *before* specialized expert adapters are active. If a task is highly specialized and requires the adapters to expose task-indicative features in the first place, the base representations will remain collapsed or uninformative, yielding uniform routing coefficients ($\alpha \approx 1/K$) and causing dynamic routing to fail. The paper lacks a theoretical analysis or bound on when this representation sufficiency is guaranteed to hold.
- **Early-Layer Centroid Collapse (ELC):** In PFAB-ELC, the paper projects early-layer representations onto pre-computed task centroids. The authors acknowledge a severe semantic representation mismatch because early layers extract low-level style features. In the organic DomainNet pilot (Table 6), **PFAB-ELC's accuracy collapses catastrophically to 42.50%** (a **36.30% absolute drop** below the expert ceiling of 78.80%). This collapse confirms that early-layer centroid gating is highly fragile and lacks theoretical robustness under organic covariate and style shifts, making the single-pass pathway practically unusable.

### C. Heuristic, Loose Normalization Bounds (LAS and Class-Size Scaling)
- **Layer-Wise Adapter Scaling (LAS):** To neutralize physical scale imbalances across experts, the paper scales intermediate activations by the Frobenius norm of low-rank weights: $s_k^{(l)} = \|B_k^{(l)} A_k^{(l)}\|_F$. However, there is **no mathematical guarantee** that intermediate activation norms scale proportionally with Frobenius weight norms. By the submultiplicative property:
  $$\|H B_k^{(l)} A_k^{(l)}\|_F \le \|H\|_F \|B_k^{(l)} A_k^{(l)}\|_F$$
  This bound is mathematically loose and depends on the alignment of the input activations with the row space of the weights. Under out-of-distribution or adversarial inputs, weight-based scaling can fail to prevent scale dominance, representing a loose, unrigorous heuristic.
- **Class-Size Scaling Calibration:** The divisor $\sqrt{2\log C'_k / D}$ is derived assuming class prototype weights behave as independent, random projections on the unit hypersphere. The authors acknowledge that real trained neural network weight vectors violate this assumption due to correlation and hierarchical structure, meaning this divisor is a calibrated heuristic rather than a mathematically rigorous identity.

### D. Missing Error Propagation Analysis for LLM One-Token Lag
For autoregressive LLMs, the authors propose Task-Specific Vocabulary-Head Anchoring (TSVHA). Due to causal generation, routing coefficients for token $t$ must be derived from penultimate representations of the previous step $t-1$, introducing a **one-token physical routing lag**.
At transition boundaries, the first token of a new domain is guaranteed to be executed with the wrong adapter weights. Because errors in autoregressive sequence decoding propagate exponentially, executing the wrong adapter on a transition token can introduce localized representation perturbations that degrade downstream generation quality. The paper reports 100% gating synchrony on a toy sequence simulation, but **fails to provide a rigorous mathematical analysis or error bounds** for autoregressive error propagation under physical routing lag.

### E. Lack of Organic LLM Downstream Validation
Modern parameter-efficient expert serving research is primarily targeted at generative Large Language Models. While the authors present a solid pilot validation on a Vision Transformer (DomainNet), their generative LLM results are restricted to a simplified, synthetic token simulation of only $T=50$ tokens. The lack of organic, downstream LLM evaluations (e.g., perplexity, ROUGE, or GSM8K accuracy) on standard autoregressive tasks fails to validate the practical viability of TSVHA and the DGR safeguard under real linguistic noise and vocabulary overlaps.

---

## 4. Detailed Feedback and Questions for Authors

To improve the theoretical rigor and completeness of the manuscript, the authors must address the following points:

1. **Provide the Formal Mathematical Formulation for SVD Orthogonalization:** 
   Please provide the precise equations, projection matrices, and algorithmic pseudocode for the joint SVD orthogonalization of adapter weights. Specifically, explain:
   - How the low-rank PEFT constraint of rank $r$ is maintained during orthogonal projection.
   - Whether the projection involves singular value truncation, and if so, how the reconstruction error is bounded.
   - Prove or analytically demonstrate that this projection does not degrade the original, specialized task-specific expert capabilities when evaluated in isolation.
2. **Derive Mathematical Error Bounds for the One-Token Physical Routing Lag:**
   Please provide a formal error-propagation analysis under autoregressive decoding. Let $H^{(l)}_t$ be the hidden state at step $t$ under the routing lag, and let $H^{(l)*}_t$ be the true, lag-free hidden state. Derive a bound on the representation drift:
   $$\|H^{(l)}_t - H^{(l)*}_t\|_F$$
   as a function of sequence length and the Lipschitz constant of the network, showing that the localized perturbation of the one-token lag does not lead to catastrophic decoding drift.
3. **Analyze the Robustness of LAS under Covariate Shifts:**
   Provide an analytical sensitivity analysis of the Layer-Wise Adapter Scaling (LAS) mechanism under out-of-distribution (OOD) activations. Show how the approximation error of weight-based Frobenius norm scaling behaves when input activations are misaligned with the expert's weight manifold.
4. **Conduct Organic LLM Downstream Validation:**
   To validate your theoretical generative proposals (TSVHA and DGR), please replace the toy 50-token sequence simulation with a real-world pilot validation on an organic pre-trained LLM (such as LLaMA-3-8B or Mistral-7B). Report downstream task accuracy or perplexity on a mixed-task document stream.
5. **Mitigate the Collapse of PFAB-ELC:**
   Address the severe collapse of early-layer centroid gating (42.50% on DomainNet) by exploring intermediate-layer centroid gating. Provide a theoretical analysis of how the semantic abstraction gap behaves across depths, and suggest an optimal depth (e.g., Layer 4 instead of Layer 0) that balances semantic robustness with single-pass latency benefits.

---

## 5. Ratings and Overall Recommendation

- **Soundness:** **Fair**  
  The core activation-blending and non-parametric gating equations are mathematically sound, and Proposition 1 is correct. However, key components of the methodology—specifically SVD-based parameter orthogonalization—are completely missing their mathematical formulations and equations. Additionally, the single-pass pathway (ELC) collapses on organic data due to a lack of semantic robustness, and intermediate layer scaling (LAS) and class-size scaling rely on loose, unproven heuristics.
- **Presentation:** **Good**  
  The writing is clear, structurally organized, and articulate, with an exceptionally honest discussion of limitations. However, omitting the mathematical formulation for SVD orthogonalization represents a major presentation and completeness gap.
- **Significance:** **Good**  
  The conceptual shift from serving-layer batch partitioning (MBH) to pure PyTorch activation-space blending is highly significant for systems execution, offering flat constant latency and excellent portability across AMD/TPU/edge devices. However, its overall scientific impact is currently limited by the theoretical gaps and the lack of organic LLM downstream validation.
- **Originality:** **Good**  
  Shifting dynamic synthesis to activation space and deriving sample-wise routing coefficients non-parametrically from existing classification heads are clever and elegant concepts. However, the physical execution layer is algebraically identical to standard LoRA-MoE serving.

- **Overall Recommendation:** **3 (Weak Reject)**  
  This paper has clear merits and proposes a highly elegant, portably efficient systems-ML co-design. However, from a rigorous theoretical perspective, the weaknesses currently outweigh the strengths. The complete mathematical omission of SVD-based parameter orthogonalization renders a core part of the methodology non-reproducible and unverified. This, combined with the lack of organic LLM downstream validation, the exponential error risks of the one-token routing lag, and the severe empirical collapse of the single-pass ELC pathway on real-world data, indicates that the manuscript is mathematically and empirically incomplete. The paper requires a thorough revision—specifically, completing the SVD mathematics, deriving error propagation bounds for routing lag, and executing an organic LLM pilot—before it can be accepted and built upon by the scientific community.
