# 1. Summary of the Paper

## Main Topic and Objective
The paper addresses the challenge of **test-time dynamic model merging** in realistic, highly heterogeneous streaming environments. Traditional parameter-space merging methods suffer from **heterogeneity collapse** under mixed-task batches because they are forced to average routing coefficients over the batch dimension to maintain a single set of merged weights. Prior state-of-the-art solutions like Micro-Batch Homogenization (MBH) mitigate this by wrapping the model in complex, stateful systems scheduling pipelines that buffer, sort, and partition streams. 

The objective of this paper is to introduce a minimalist, network-level, stateless alternative called **SABLE (Sample-wise Activation Blending of Low-Rank Experts)**. SABLE completely bypasses parameter averaging and systems-level queuing by shifting ensembling from weight space to activation space, allowing mathematically exact ensembling on a per-sample basis during a single forward pass.

## Core Approach: SABLE
SABLE utilizes the distributive property of matrix multiplication ($X(W_1 + W_2) = X W_1 + X W_2$) to move ensembling from parameter space into activation space. Its core mechanisms include:
1. **Subspace Cosine Projection:** Computes non-parametric routing coefficients on-the-fly by projecting intermediate activation representations onto frozen expert classification heads (or centroids) via cosine similarity.
2. **Out-of-Distribution (OOD) Rejection:** Employs a hard threshold $\gamma_{\text{OOD}}$ to route noisy/OOD inputs solely through the pre-trained base model.
3. **Temperature-Scaled Softmax Routing:** Normalizes cosine similarity scores into dynamic blending coefficients $\alpha_{k, b}$ via Softmax.
4. **Dynamic Activation Blending:** Executes a shared forward pass through the pre-trained base model alongside parallel, low-rank ($r=8$) adapter passes, scaling the adapter activations by the sample-specific coefficients on-the-fly.
5. **Early-Layer & Mid-Layer Routing (Late Adaptation):** Bypasses the $2\times$ forward pass latency paradox (or activation caching memory) by either routing from Layer 0 or restricting adaptation and routing to late-stage layers ($L_{\text{route}} = 12$).
6. **Top-$M$ Expert Pruning & Dynamic Head Blending:** Bounds execution complexity to $O(M)$ (where $M \ll K$) by only computing and blending the top $M$ experts' activations and classification heads.

## Key Findings
- **Perfect Heterogeneity Robustness:** SABLE exhibits flatline performance under both homogeneous and fully heterogeneous streams (0.00% collapse), outperforming PFSR (which suffers a 15.40% collapse to uniform merging).
- **Superior Serving Efficiency:** SABLE Late Adaptation achieves **68.10%** joint mean accuracy in the Analytical Coordinate Sandbox, outperforming the complex, stateful PFSR+MBH systems pipeline (**67.20%**) while completely stripping away systems-level scheduling and latency.
- **Physical Validation Success:** 
  - On a physical CNN trained on MNIST/FashionMNIST, SABLE with completely zero-data centroids achieves **63.50%** accuracy at $r=10$, outperforming collapsing weight-space routers (49.00%) and static uniform merging (49.40%).
  - On standard ResNet-18 foundation features, SABLE Hybrid (with a layer-dependent hybrid-rank protocol) achieves **69.30%** accuracy with Support-16 centroids, which is within 0.10% of the full-parameter weight-space oracle PFSR (**69.40%**) while avoiding collapse.
  - In a 4-layer physical MLP, SABLE's multi-layer ensembling is validated, with Single-Pass Early-Routing achieving **65.20%** joint accuracy. Representational drift tracking shows high cosine similarity ($>0.83$) across all intermediate hidden layers.
- **Significant Systems Advantages:** End-to-end wall-clock serving benchmarks on an NVIDIA A100 GPU demonstrate that SABLE reduces serving latency by **6.8$\times$** (12.4 ms vs 84.6 ms) and saves **36.4%** VRAM memory (412 MB vs 648 MB) compared to PFSR+MBH.

## Explicitly Claimed Contributions (with Evidence)
1. **Mathematical Relocation of Ensembling:** Shifting merging from parameter space to activation space via the distributive property, mathematically guaranteeing sample-wise exact equivalence to parameter-space ensembling for single-sample batches. (Supported by mathematical formulations in Sections 3.1 & 3.5 and empirical verification on mixed-task streams).
2. **Non-parametric, Zero-Calibration Subspace Cosine Projection:** Routing inputs using pre-trained expert classification heads/centroids, avoiding trainable parameters or calibration splits. (Supported by the Zero-Data centroid construction method, which achieves highly competitive accuracies of up to 63.50% in Table 1).
3. **Single-Pass Early-Layer & Mid-Layer Routing Architectures:** Designing a sequential forward-pass pipeline that resolves the representational alignment paradox without 2-pass execution latency or activation caching memory. (Supported by ablations of routing depths in Table 6, and physical single-pass MLP results in Section 4.4 showing 65.20% accuracy).
4. **Complexity Bounding via Top-$M$ Pruning & Dynamic Head Blending:** Bounding computational complexity to $O(M)$ for both hidden layers and final classification heads, allowing scalable serving for massive expert pools. (Supported by Top-$M$ expert sweeps in Section 4.8 and Top-2 Joint Retrieval metrics under domain-confounded blended streams in Table 2 and Table 4).
5. **Layer-Dependent Hybrid-Rank Selection Protocol:** Combating capacity bottlenecks in low-dimensional output projection layers by keeping them full-precision while maintaining aggressive low-rank hidden layers. (Supported by ResNet-18 feature experiments in Table 4, showing a +4.90% to +5.90% accuracy surge when using the hybrid protocol at $r=2$).
6. **Physical Systems Benchmarking:** Proving wall-clock serving and memory advantages on an NVIDIA A100 GPU. (Supported by empirical latency/memory statistics in Section 4.8 showing a 6.8$\times$ latency reduction).
