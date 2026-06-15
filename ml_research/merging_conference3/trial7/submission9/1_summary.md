# Paper Summary

## Context & Motivation
In modern machine learning, adapting deep foundation models to downstream tasks is typically handled via Parameter-Efficient Fine-Tuning (PEFT), particularly Low-Rank Adaptation (LoRA). As the pool of specialized task-specific experts grows, the ability to ensemble these experts at test-time without additional training or calibration data becomes crucial. Traditional parameter-space model merging methods (e.g., Task Arithmetic, TIES-Merging, RegMean, and Fisher Merging) perform static merges. More recently, Parameter-Free Subspace Router (PFSR) introduced dynamic test-time model merging by routing queries based on penultimate layer representations using cosine similarity projections onto frozen classifier heads, with zero training or calibration parameters.

## The Problem: Heterogeneity Collapse
When deployed in realistic serving environments, incoming queries arrive in heterogeneous batches (i.e., batches containing mixed tasks). To process the entire batch in a single forward pass, parameter-space merging techniques must average their dynamic routing coefficients over the batch dimension to construct a single set of merged weights. This batch-level averaging causes **heterogeneity collapse**: the merged model degrades to a static, sub-optimal uniform average of the experts, wiping out task-specific specialization and dropping accuracy down to near-uniform levels.

Prior state-of-the-art solutions like Micro-Batch Homogenization (MBH) attempt to mitigate this by wrapping the model in a stateful systems scheduling pipeline. MBH dynamically buffers, sorts, and partitions heterogeneous streams into homogeneous micro-batches. However, this introduces significant systems-level bloat:
1. **Serving Complexity:** Breaks the clean, stateless paradigm of deep learning inference.
2. **Queuing Latency:** Delays query streams in buffers, creating high tail latencies.
3. **Memory/Compute Overhead:** Grouping and partitioning algorithms consume significant auxiliary CPU and memory.

## The Proposed Solution: SABLE
Applying Occam's razor, the authors propose **SABLE (Sample-wise Activation Blending of Low-Rank Experts)**, a minimalist, network-level alternative that completely strips away stateful dynamic buffers, sorting steps, and scheduling wrappers. 

SABLE's core insight is that model ensembling does not have to occur in weight space. By leveraging the distributive property of matrix multiplication, SABLE shifts ensembling from parameter space to activation space. SABLE executes a single, shared forward pass through the pre-trained base model, alongside parallel, extremely lightweight low-rank ($r=8$) expert adapter passes. The intermediate activations are blended on-the-fly using sample-wise routing coefficients derived via non-parametric cosine similarity projections of features onto frozen task prototype centroids.

SABLE has several core features:
- **Perfect Heterogeneity Robustness:** Since ensembling occurs in activation space *after* projection, it is calculated per-sample, making SABLE natively immune to heterogeneity collapse (0.00% collapse).
- **System-Agnostic Simplicity:** Eliminates the need for stateful systems scheduling, allowing SABLE to be deployed in standard, stateless serving frameworks with zero systems-level modifications.
- **Top-$M$ Expert Pruning:** To maintain bounded serving latency when the expert pool scales, SABLE executes parallel adapters only for the top $M \ll K$ experts with the highest similarity coefficients, capping complexity at $O(M)$ instead of $O(K)$.
- **Dynamic Head Blending:** SABLE also applies Top-$M$ pruning to the final classification heads, making the entire network forward pass bounded and task-agnostic without requiring oracle task IDs.
- **Mid-Layer Routing (Late Adaptation):** SABLE's default configuration runs the first $L_{\text{route}}$ layers unadapted through the base network, extracting the penultimate features at $L_{\text{route}}$ to compute routing coefficients, and blending activations only across the late-stage layers. This resolves the *Representational Alignment Paradox* where early-layer features are semantically unaligned with final-layer classification heads.

## High-Dimensional Foundation Feature Validation (Section 4.4)
The authors recently updated SABLE with a rigorous physical experiment utilizing a pre-trained ImageNet **ResNet-18** as a frozen feature extractor. On top of the 512-dimensional extracted features, they build a 2-layer MLP adapter classification head ($\text{FC}_1 \in \mathbb{R}^{512 \times 128}$, $\text{FC}_2 \in \mathbb{R}^{128 \times 10}$) and train specialized experts on MNIST and FashionMNIST. They evaluate:
1. **SABLE Strict:** Both layers are ensembled using low-rank $r$ updates.
2. **Layer-Dependent Hybrid-Rank Protocol (SABLE Hybrid):** The massive hidden layer $\text{FC}_1$ is ensembled using low-rank $r$ adapters, while the final output projection layer $\text{FC}_2$ (which is parameter-negligible but representational-critical) is ensembled using full-precision (full-rank) updates.
3. **Three Centroid Types:**
   - *Support-16 Centroids:* Active activation averages over 16 support samples.
   - *Naive Zero-Data Centroids:* Row-wise average of expert parameters with zero support data.
   - *Refined Zero-Data Centroids (Ours):* L2-normalization applied to class weight vectors before averaging to prevent vector cancellation and preserve semantic orientation.

## Key Empirical Findings
- **Elimination of Collapse:** Under standard streams, SABLE exhibits 0.00% collapse. SABLE Late Adaptation achieves **68.10%** joint accuracy in the coordinate sandbox under both homogeneous and heterogeneous streams, outperforming the complex, stateful PFSR+MBH systems pipeline (**67.20%**).
- **Bypassing the Low-Rank Bottleneck:** Under SABLE Strict, constraining rank to $r=2$ degrades performance severely (57.20% with Support-16). Applying SABLE Hybrid surges joint accuracy at $r=2$ to **62.10%** (+4.90% absolute improvement over Strict), proving that keeping output projections full-rank completely bypasses the low-rank bottleneck.
- **The Low-Rank Regularization Paradox:** SABLE Hybrid at $r=2$ consistently and significantly outperforms its $r=4$ counterpart (e.g., 62.10% vs 58.90% with Support-16; 57.20% vs 55.60% with Refined Zero-Data). This non-monotonic trend occurs because constraining the hidden layer to $r=2$ acts as a powerful regularizer, pruning high-frequency representation noise, whereas $r=4$ introduces extra capacity that lets task-irrelevant features and cross-task adapter interference leak through.
- **Destructive Representational Interference of High-Capacity Experts:** Under domain-confounded blended streams (50-50 overlaid images), soft blending ($M=2$) outperforms hard routing ($M=1$) at extremely low ranks ($r=2$, SABLE Hybrid Soft achieves **26.00%** vs Hard **24.00%**), but this relationship reverses at higher ranks ($r=8$, Soft drops to **15.00%** while Hard is **17.00%**). At higher ranks, expressiveness reconstructs unregularized, highly specialized expert manifolds with near-perfect fidelity, causing incompatible task manifolds to collide and cancel each other out under soft blending. At $r=2$, the low-rank bottleneck acts as an aggressive low-pass filter, allowing constructive blending of smooth semantic coordinates.
- **Refined Zero-Data Centroids:** Consistently outperforms Naive Zero-Data Centroids by **+1.00% to +3.40%** absolute accuracy across all ranks, proving the mathematical validity of weight-space L2-normalization to prevent vector cancellation.
- **Wall-Clock Serving Latency:** On an NVIDIA A100 GPU, SABLE stateless single-pass execution achieves an average latency of **12.4 ms** and **412 MB** peak memory, representing a **6.8$\times$ latency reduction** and a **36.4% memory saving** over the stateful PFSR+MBH pipeline (84.6 ms, 648 MB).
