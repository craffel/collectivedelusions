# Paper Summary: SABLE (Sample-wise Activation Blending of Low-Rank Experts)

## 1. Main Topic and Problem Formulation
The paper addresses the challenge of **test-time dynamic model merging** in realistic, highly heterogeneous streaming environments. In such environments, incoming inference batches contain mixtures of different tasks. Traditional parameter-space model-merging methods (such as Parameter-Free Subspace Routing, PFSR) suffer from **heterogeneity collapse** because they are forced to average routing coefficients across the batch dimension to maintain a single set of merged weights. This destroys individual task specialization, degrading accuracy to static uniform-merge levels. Prior solutions (e.g., Micro-Batch Homogenization, MBH) resolve this by wrapping models in complex, stateful systems pipelines that buffer, sort, and partition streams. SABLE aims to solve this at the network level, bypassing systems-centric wrappers entirely.

## 2. Proposed Approach (SABLE)
SABLE shifts the ensembling step from parameter space to activation space, leveraging the distributive property of matrix multiplication. 
- **Subspace Cosine Projection:** Derived sample-specific routing coefficients ($\alpha_{k, b}$) by calculating the cosine similarity between intermediate features ($z_b$) and frozen task classification weights/centroids ($w_k$).
- **OOD Rejection and Routing:** Implements a hard threshold ($\gamma_{\text{OOD}}$) to reject out-of-distribution queries and a temperature-scaled Softmax ($\tau$) to produce dynamic blending coefficients.
- **Activation Blending Layer:** Runs a single pass through a shared pre-trained base model, alongside parallel, low-rank ($r=8$) expert adapter passes (LoRA), blending activations on-the-fly:
  $$Y_b = X_b W_{\text{base}} + \sum_{k=1}^K \alpha_{k, b} \cdot \left( (X_b A_k) B_k \right)$$
- **Mid-Layer Routing & Late Adaptation:** Resolves representational alignment issues by leaving early layers unadapted and computing routing coefficients from unadapted mid-layer activations.
- **Top-$M$ Expert Pruning & Head Blending:** Limits active experts and classification heads to the top $M \ll K$ high-confidence experts, bounding computation and memory bandwidth overhead to $O(M)$.

## 3. Key Findings
- SABLE achieves flatline joint mean accuracy across both homogeneous and heterogeneous streams, completely avoiding heterogeneity collapse (**0.00% collapse**).
- SABLE's Late Adaptation configuration in a synthetic 14-layer sandbox achieves **68.10%** joint mean accuracy, outperforming the systems-heavy PFSR+MBH pipeline (**67.20%**).
- In physical real-world CNN experiments on MNIST and FashionMNIST, SABLE Soft Blending ($r=10, M=2$) achieves **69.30%** (with 16 support samples) and **63.50%** (with completely zero-data centroids), significantly outperforming collapsing PFSR (49.00% under heterogeneous streams).
- In multi-layer Deep MLP experiments, SABLE Soft Single-Pass Early-Routing achieves **65.20%**, outperforming standard 2-pass ensembling and proving sequential single-pass execution.
- Physical tracking shows high activation cosine similarity ($>0.83$) between SABLE and oracle experts, confirming minimal cumulative activation divergence.

## 4. Explicitly Claimed Contributions and Associated Evidence
1. **Absolute Robustness to Stream Heterogeneity:** Supported by flatline performance (0.00% collapse) across synthetic sandbox, physical CNN, deep MLP, and ResNet-18 evaluations (Tables 1, 3, 5).
2. **Stateless Network-Level Alternative to Complex Schedulers:** Supported by wall-clock latency benchmarks on an NVIDIA A100 GPU showing SABLE achieves **12.4 ms** latency (vs. **84.6 ms** for MBH) and **412 MB** peak memory (vs. **648 MB** for MBH), representing a 6.8$\times$ latency reduction and 36.4% memory saving.
3. **Completely Zero-Data Centroid Construction:** Proposes constructing centroids directly from expert classification weights, achieving **63.50%** accuracy on CNN (Table 1) and **61.60%** with Refined Zero-Data Centroids on ResNet-18 (Table 3), matching support-split performance without any calibration data.
4. **Layer-Dependent Hybrid-Rank Protocol:** Recommends keeping final classification heads full-rank while aggressively compressing hidden layers. Evidence in Table 3 shows SABLE Hybrid at $r=2$ outperforms SABLE Strict at $r=2$ by +4.90% to +5.90% absolute accuracy.
5. **Empirical Validation of Soft Blending on Confounded Streams:** Proves that soft blending ($M=2$) significantly outperforms hard routing ($M=1$) on 50-50 blended inputs under a rigorous joint Top-2 retrieval metric (Tables 2, 4), resolving the methodological contradiction.
