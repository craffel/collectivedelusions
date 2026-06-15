# Peer Review of Conference Submission

## Summary of the Paper

This paper investigates test-time dynamic model merging (dynamic expert weight ensembling) at inference time and targets two critical operational failure modes:
1. **Calibration Data Scarcity ($N \le 32$):** Standard parametric linear routers overfit heavily on small calibration sets, leading to transductive collapse and high variance.
2. **Deployment Stream Batch Heterogeneity:** Mixed-task samples processed in the same execution batch smooth out routing logits across tasks, leading to "heterogeneity collapse" (degrading to static ensembling performance).

To address these vulnerabilities, the authors propose:
- **Confidence-Gated Hybrid Routing (CGHR):** A dual-pathway system that gates predictions on a sample-by-sample basis. It routes high-confidence samples via a trained parametric linear router (Pathway A) and falls back to a zero-shot, parameter-free subspace router (PFSR - Pathway B) when parametric confidence falls below a threshold $\gamma_{\text{conf}}$.
- **Micro-Batch Homogenization (MBH):** A dynamic batch-partitioning algorithm that groups incoming heterogeneous stream elements into task-homogeneous micro-batches on the fly, computes localized routing weights, merges expert parameters, and runs specialized model inference.

The authors also evaluate practical systems-level optimizations including **Fusion Weight Caching** (discretizing ensembling weights to cache fused adapters), **Homogeneity Bypass** (recovering baseline speed for single-sample/homogeneous batches), and **Warp Batch Padding** to prevent thread-warp divergence under highly skewed task streams. Evaluated in a synthetic 1-layer coordinate-isolated sandbox, the proposed CGHR and MBH methods consistently outperform standard parametric and static ensembling baselines.

---

## Strengths and Weaknesses

### Strengths

1. **Clear and Structured Formulations:** The paper is exceptionally well-written, mathematically rigorous, and highly structured. The descriptions of CGHR, PFSR, the three confidence metrics, and MBH are clear, precise, and easily reproducible.
2. **Highly Practical Systems Intuition:** The authors demonstrate excellent systems awareness by investigating and proposing concrete solutions to ensembling bottlenecks in the Appendices:
   - **Fusion Weight Caching:** Discretizing routing coefficients (step size $0.10$) achieves a **2.87$\times$** weight fusion speedup with a **98.2%** cache hit rate and absolutely zero accuracy loss.
   - **Homogeneity Bypass:** Bypassing MBH partitioning for single-sample ($B=1$) or homogeneous workloads completely recovers baseline execution speeds ($1.0\times$ overhead).
   - **Warp Batch Padding:** Proving that Warp Batch Padding under skewed streams improves effective GPU throughput by **1.63$\times$** while reducing grid latency by **38.8%** demonstrates excellent understanding of hardware-native parallel execution.
3. **Rigorous Empirical Sweeps:** The authors execute comprehensive parameters sweeps over 5 independent seeds for confidence thresholds (Fig 1), calibration sample complexities (Fig 2), and deployment batch sizes (Fig 3), validating their claims within the controlled sandbox.
4. **Refreshing Academic Honesty:** The authors are highly transparent and upfront about their synthetic sandbox assumptions and CPU-bound simulation artifacts, building high academic trust.

### Weaknesses

1. **Complete Lack of Real-World Model/Dataset Validation:** The paper's most critical weakness is that all quantitative experiments are conducted in a synthetic, 1-layer *Isolating Coordinate Sandbox* with MNIST/CIFAR-10 coordinate-isolated prototypes. There are **zero** evaluations on actual deep neural networks (e.g., pre-trained Transformers) or standard multi-task benchmarks (e.g., GLUE, DomainNet, Decathlon) with real fine-tuned LoRA adapters. Without showing how CGHR and MBH perform under highly overlapping, non-orthogonal representation spaces in real multi-layer networks, the practical deployability of these methods remains completely unproven.
2. **Privileged Coordinate-Slice Input in Sandbox:** In the synthetic sandbox, the parametric router (Pathway A) takes the global representation $z_b \in \mathbb{R}^{192}$ with high-dimensional noise, whereas the non-parametric PFSR (Pathway B) is provided with local, block-sliced representations $z_{k, b} \in \mathbb{R}^{48}$. This means PFSR is equipped with privileged architectural knowledge about coordinate task boundaries. If the parametric router also knew which coordinates to select, the routing problem would be trivial. This structural asymmetry creates an artificial gap, making standard parametric routers look worse under small $N$ than they would in an environment where both pathways share the same input space.
3. **High Latency Bottleneck of Raw MBH on GPUs:** Standard MBH partitions batches on host CPU and launches sequential forward passes for each micro-batch.
   - As modeled in Table 4, on GPU systems, standard MBH results in massive latency multipliers: **4.33$\times$** at $B=1$, **3.20$\times$** at $B=32$, and **1.86$\times$** at $B=256$.
   - In cloud-scale, high-throughput serving systems, a $2\times$ to $4\times$ latency penalty is highly impractical.
   - While the authors outline a custom **Triton Segmented-BGEMM kernel** (Appendix D.4) to solve this, this kernel is presented as a *qualitative proposal* and is NOT physically implemented or benchmarked in the paper. The latency measurements in Table 3 are from a CPU-bound simulator, and Table 4 is merely "simulated" GPU behavior. The lack of an actual parallel GPU implementation is a major systems-level gap.
4. **Combinatorial Memory Overhead of Caching:** The Fusion Weight Caching strategy scales combinatorially with the number of experts $K$ and discretization step size $h$. While caching is manageable for $K=4$, as the expert registry scales to standard settings ($K \ge 32$), storing all pre-fused combinations becomes completely intractable. The proposed LRU eviction policy is described qualitatively but not implemented or evaluated, leaving the scalability of weight caching unverified.

---

## Detailed Evaluations

### Soundness: Good
The mathematical formulations are sound, and the empirical results in the synthetic sandbox are thoroughly validated over multiple seeds. The UNC-PFSR Equivalence Theorem and class correlation derivations (Appendix A.1 & A.2) are mathematically elegant. However, the soundness of the empirical evaluation is limited by the artificial input asymmetry of the sandbox and the use of simulated/projected GPU latencies instead of actual hardware profiling on a parallel GPU serving engine.

### Presentation: Excellent
The paper is written with high rigor, clear logical flow, and professional styling. The figures are clean and directly support the claims. The Appendices are exceptionally detailed, addressing hyperparameters, algorithmic flows, and systems-level latency trade-offs with commendable thoroughness.

### Significance: Fair
The proposed hybrid routing (CGHR) and batch partitioning (MBH) architectures are highly intuitive and could guide the design of future multi-tenant serving engines. However, because the methods are validated *purely* in a 1-layer synthetic coordinate-isolated sandbox without any actual deep learning model or real-world dataset experiments, the immediate practical significance of this work is restricted. It operates as a speculative prototype rather than a deployment-ready framework.

### Originality: Good
The paper offers a well-thought-out ensembling of existing ensembling and ensembling-routing concepts (parametric routing, cosine similarities, and batch sorting/grouping). The combination of parametric and training-free pathways via confidence gating is elegant and highly logical. The systems-level Segmented-BGEMM Triton design is, however, highly derivative of advanced serving frameworks like S-LoRA and Punica.

---

## Overall Recommendation

**Rating: 3 (Weak Reject)**

### Justification of Rating
While the paper is technically elegant, exceptionally well-presented, and exhibits excellent systems awareness, the weaknesses currently outweigh the merits for a major machine learning conference. Validating a dynamic model-merging framework *purely* on a 1-layer synthetic coordinate sandbox without a single experiment on a real deep learning model (such as a pre-trained Transformer with LoRA adapters) or real-world multi-task datasets severely limits its credibility and immediate practical impact. The coordinate-isolation assumption in the sandbox represents an idealized, non-overlapping setup that does not translate directly to real-world representation spaces. Furthermore, the systems-level GPU latency solutions (Triton kernels) and memory-eviction policies (LRU caching) are entirely simulated or qualitative proposals rather than physical implementations.

To elevate this paper to an Accept, the authors must address these major gaps:
1. **Real-world Model Validation:** Implement and evaluate CGHR on a standard multi-task benchmark (e.g., ensembling 4 LoRA adapters on a pre-trained Transformer like BERT or LLaMA-3-8B on GLUE or DomainNet) to demonstrate that the confidence gating and "SVD subspace projection operators" successfully handle overlapping representation spaces.
2. **Actual GPU Profiling:** Provide physical latency, throughput, and GPU occupancy benchmarks on a real GPU accelerator (e.g., NVIDIA A100 or H100) using a real Triton Segmented-BGEMM implementation of MBH, rather than relying on CPU loops or simulated GPU models.
3. **Fair Sandbox Comparison:** Re-evaluate the sandbox experiment where the parametric router and PFSR share the exact same input space (e.g., both receive the global $D$-dimensional vector) to eliminate the coordinate-slice advantage of the non-parametric pathway.
