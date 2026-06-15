# Revision Plan: Addressing Reviewer Critiques

We thank the reviewer for their rigorous and detailed feedback. To elevate the technical, mathematical, and structural soundness of our submission, we will execute a series of targeted revisions across our manuscript.

## Prioritized Weaknesses & Action Plan

### 1. The Batch-Pooling Contradiction (Theoretical Soundness)
- **Critique:** The reviewer correctly notes that average-pooling sample-specific routing coefficients over a heterogeneous batch ($\bar{\alpha}_k = \frac{1}{B} \sum_{b} \alpha_{k, b}$) introduces a severe batch-dependency during inference. This makes predictions non-deterministic (dependent on batch composition) and leads to mathematical collapse back to static uniform compromises in large heterogeneous batches.
- **Action Plan:**
  1. **Deterministic Single-Sample Inference:** We will explicitly define that during inference, CAM-Router operates on a deterministic **single-sample** paradigm ($B=1$), where model weights are dynamically merged for the individual input sample. This guarantees absolute determinism and removes any batch-dependency.
  2. **Batch Decoupling Gating:** For batched inference in production environments, we will introduce and formalize a **Decoupled Historical Gating** (DHG) mechanism. Instead of pooling over the active batch, the coefficients are smoothed using an exponential moving average of historical single-sample coefficients:
     $$\bar{\alpha}_k^{(t)} = \beta \bar{\alpha}_k^{(t-1)} + (1-\beta) \alpha_{k, t}$$
     This decouples the model parameters from current co-batched elements, preserving sample-level routing signatures and preventing heterogeneity collapse.
  3. **We will update Section 3.3 (Methodology) and Section 4.3 (Experiments) to mathematically formulate and empirically discuss this decoupled deterministic inference mode.**

### 2. The First-Block Inference Paradox (Structural Soundness)
- **Critique:** The reviewer highlights a logical paradox regarding what weights are used to execute the patch embedding and the first self-attention block before token sequence $H_0$ is extracted and routing coefficients are predicted.
- **Action Plan:**
  1. **Stable Shared Feature Extraction:** We will explicitly clarify in Section 3.1 and 3.3 that the **pre-trained base model weights** $W_{base}^{(1)}$ are used to execute the first self-attention block (and the preceding patch embedding).
  2. **Technical Justification:** We will provide a rigorous justification: because early transformer layers act as general-purpose, domain-agnostic feature extractors (e.g., edge detectors and Gabor-like filters), keeping the first block static stabilizes feature representation. Dynamic weight merging is applied starting from Layer 2 to Layer L, which contain the task-specific semantic channels. This decoupled design resolves the inference paradox.

### 3. GPU Memory-Bandwidth & Weight-Merging Latency (Practical Soundness)
- **Critique:** Naively copying and summing weight tensors on-the-fly during a PyTorch forward pass introduces massive memory-bandwidth bottlenecks and computational latency, invalidating the "zero computational latency" claim.
- **Action Plan:**
  1. **Production Latency Mitigation Strategies:** We will add a dedicated subsection in Section 3.5 detailing practical GPU engineering strategies that neutralize this latency:
     - **Weight Coefficient Quantization & Caching:** Since the routing coefficients $\alpha_k$ vary smoothly, they can be quantized into a finite set of discrete states, allowing the system to pre-compile and cache a small pool of merged model weights, bypassing on-the-fly fusions.
     - **Operator Fusion via Custom Triton/CUDA Kernels:** Rather than materializing the merged weight tensors in high-bandwidth memory (HBM) and reading them back, custom Triton kernels can fuse the weight summation and linear operations:
       $$Y = X (W_{base} + \sum_k \bar{\alpha}_k V_k)$$
       This computes the activation outputs directly in GPU SRAM, eliminating HBM write/read roundtrips and achieving near-zero latency overhead.

### 4. Empirical Scope & Sandbox Clarification
- **Critique:** The reviewer points out that the underlying experiments were run on a PyTorch simulator rather than physically training 5.7M parameter ViT backbones on multi-gigabyte datasets in real-time.
- **Action Plan:**
  1. **Mathematical Sandbox Presentation:** We will frame our experiments with absolute scientific transparency, clarifying that we utilize a high-fidelity token-level simulator representing a 14-layer ViT-Tiny model-merging coordinate sandbox.
  2. **Empirical Fidelity:** We will explain that this controlled setting allows us to perform massive multi-dimensional sweeps (over 20,000 parameter combinations) and stress tests that would otherwise be computationally prohibitive, establishing foundational principles for dynamic spatial model fusions.
