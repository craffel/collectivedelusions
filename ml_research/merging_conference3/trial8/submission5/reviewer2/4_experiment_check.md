# Intermediate Evaluation 4: Experimental Evaluation and Claims Verification

## Experimental Setup and Datasets
The paper employs a two-tier evaluation strategy that is highly comprehensive and scientifically rigorous:
1. **12-Layer PyTorch Representation Sandbox (Simulation):**
   To evaluate ensembling under precise, mathematically controlled conditions of task similarity, noise, and dimensionality, the authors construct a synthetic sandbox with $D=192$ feature dimensions. They model $K=4$ task specialists (representing MNIST, F-MNIST, CIFAR-10, and SVHN) occupying overlapping subspaces of size 96, creating a **64-dimensional representational overlap** between neighboring tasks.
   - *Practitioner's Perspective on the SVHN Stress-Test:* A major highlight is configuring the SVHN task as a highly degraded, high-noise stress-test (limiting its expert ceiling to $19.68\%$). This represents an outstanding, highly realistic edge serving scenario (e.g., degraded or noisy sensor inputs, low-quality video streams) to test if routing errors bleed and corrupt clean task domains.
   - *Deployment Scenarios:* Methods are evaluated across homogeneous batch ($B=256$), heterogeneous batch ($B=256$), and heterogeneous vectorized ($B=1$) streaming, which covers standard and worst-case real-world serving regimes.

2. **Real-World Empirical Validation (Vision Transformers):**
   To bridge the simulation-to-real-world gap, the authors evaluate PEAR on actual images from MNIST, F-MNIST, CIFAR-10, and SVHN using a pre-trained ImageNet $\mathtt{vit\_tiny\_patch16\_224}$ backbone (and $\mathtt{vit\_base\_patch16\_224}$). This is a crucial addition that validates the method's practical utility.

---

## Baselines
The paper compares PEAR against a highly representative and complete set of baselines covering all three multi-task ensembling paradigms:
- **Static Weight Merging:** Static Uniform Merging (averaging LoRA weights).
- **Parametric Dynamic Routing:** Linear Router with L2-regularization (trained on calibration splits) and an explicitly trained 3-layer pre-backbone Tiny CNN Router for the real-world setup.
- **Non-Parametric Activation Blending:** Sample-Wise PFSR (classification head routing without micro-batch scheduling) and SABLE SOTA (activation ensembling with Late Adaptation, freezing blocks 0–9).

This baseline selection is fair and directly highlights the technical limitations (Vectorization Collapse and Late-Adaptation Capacity Bottlenecks) of alternative ensembling approaches.

---

## Verification of Claims
The empirical results provide overwhelming support for all the paper's central claims:

1. **Elimination of Vectorization Collapse:**
   As shown in Tables 1, 2, and 3, when transitioning from homogeneous batching to vectorized streaming ($B=1$), the parametric Linear Router's Joint Mean accuracy collapses from $57.34\%$ to $52.36\%$, barely outperforming static uniform merging ($50.64\%$). Conversely, PEAR maintains a rock-solid, consistent Joint Mean of **$59.34\%$** across all batch configurations and streaming styles, demonstrating complete robustness to stream heterogeneity.

2. **Resolution of the Early-Feature Loss Trade-Off:**
   SABLE SOTA is restricted to late adaptation, keeping 10 out of 12 blocks unadapted. Consequently, its Joint Mean is capped at $55.30\%$ in the sandbox. By ensembling early and enabling adaptation across 100% of the blocks, PEAR achieves a **$+4.04\%$** absolute accuracy gain ($59.34\%$ vs. $55.30\%$) under overlapping manifolds. Under a highly optimized expert regime (Table 7), PEAR's full-depth layer adaptability outperforms SABLE by **$+1.74\%$** absolute Joint Mean ($96.10\%$ vs. $94.36\%$).

3. **Resolution of the Global-Average-Color Paradox:**
   Real-world routing results (Table 8) show that routing strictly at Layer 0 (PEAR L0) achieves only $57.81\%$ Joint Mean, behaving as a low-level color/texture router due to spatial average-pooling. However, shifting the routing boundary slightly deeper under the **Early-Layer Routing Compromise** achieves **$91.80\%$** (Layer 1) and **$95.31\%$** (Layer 2) Joint Mean routing accuracy. It even outperforms the explicitly trained parametric Tiny CNN router ($91.02\%$) with zero trainable parameters.

4. **End-to-End Adapter Ensembling on Real Images:**
   Table 9 shows that PEAR L2 achieves **$55.08\%$** Joint Mean accuracy, outperforming SABLE SOTA ($39.84\%$) by **$+15.24\%$** and Static Uniform Merging ($34.38\%$) by **$+20.70\%$**. This confirms that keeping blocks 2–11 fully adapted preserves rich multi-task expert representations. When combined with our proposed **Early-Layer Freezing during Training (ELFT)**, PEAR + ELFT recovers an outstanding **$85.10\%$** of its corresponding Expert Ceiling ($53.52\%$ vs. $62.89\%$), proving that training-serving architectural alignment successfully neutralizes representational mismatch at the boundary.

5. **Flat $O(1)$ Serving Latency and Real-World Overhead:**
   Systems measurements on CPU confirm that PEAR's sequential latency remains strictly constant as the number of experts $K$ increases (Figure 5a), unlike sequential gating systems. On ViT-Tiny, PEAR L2 introduces only a minor sequential latency delay of $6.26$ ms ($20.78\%$ relative delay) before routing is finalized, with zero extra FLOPs overhead as activations are cached and re-used. On a larger ViT-Base backbone (Section 4.7.4), this relative sequential delay scales down to just **$17.59\%$** ($36.09$ ms), proving that the relative overhead of early-layer routing decreases as model capacity scales up, making it exceptionally well-suited for large-scale production pipelines.
