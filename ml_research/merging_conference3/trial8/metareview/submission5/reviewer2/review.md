# Peer Review

## Strengths and Weaknesses

### Strengths
1. **High Practical Utility and Edge Feasibility:** The paper tackles a highly practical, systems-level problem in multi-task edge serving—orchestrating specialized, low-rank adapters sample-by-sample under memory and latency constraints. The proposed parameter-free, closed-form PEAR framework represents an incredibly elegant, deployment-ready solution with flat $O(1)$ sequential complexity.
2. **Double-Tier Empirical Validation:** The methodology is validated across both a controlled, high-fidelity 12-layer PyTorch representation sandbox (simulating overlapping task manifolds under realistic sharing conditions and SVHN noise stress-tests) and an end-to-end real-world visual pipeline using a pre-trained ImageNet $\mathtt{vit\_tiny\_patch16\_224}$ backbone on actual images.
3. **Surgical Resolution of the Color Routing Paradox:** The authors proactively identify that standard Layer 0 routing average-pools token projections, mathematically behaving as a low-level color router which fails on semantic datasets. The proposed "Early-Layer Routing Compromise" (routing at Layer 1 or 2) resolves this paradox, increasing real-world routing accuracy from $57.81\%$ to $95.31\%$.
4. **Training-Serving Alignment via ELFT:** Shifting the routing boundary deeper introduces a subtle training-serving representational discrepancy at the boundary. The authors propose and validate **Early-Layer Freezing during Training (ELFT)**, which freezes early blocks used for routing during fine-tuning. This successfully neutralizes representational mismatch, allowing the model to recover up to **$85.10\%$** of its corresponding expert ceiling.
5. **Outstanding Performance against Baselines:** Despite introducing zero trainable gating parameters, PEAR L2 outperforms explicitly trained linear gating networks and pre-backbone Tiny CNN routers on real images. It completely eliminates the Vectorization Collapse that plagues parametric routers under single-sample streaming ($B=1$), and outperforms SABLE SOTA by **$+15.24\%$** absolute end-to-end accuracy by unlocking full-depth layer adaptability.
6. **Detailed Systems-Level Profiling:** Latency measurements on CPU are provided for both ViT-Tiny and ViT-Base. The authors demonstrate that the relative sequential delay of early-layer routing scales down from $20.78\%$ (ViT-Tiny) to $17.59\%$ (ViT-Base), indicating excellent systems-level scalability.

### Weaknesses
1. **Lack of Physical Edge NPU Evaluation:** While the authors provide latency profiling on CPU and transparently analyze hardware scalability ceilings (e.g., thread concurrency and physical memory bandwidth ceilings of loading $K$ parallel adapters simultaneously), their evaluations are conducted on CPU. Profiling execution speeds and memory bandwidth footprints on physical edge NPUs (such as Apple Neural Engine, Google Edge TPU, or ARM Ethos) would strengthen their systems claims.
2. **Evaluation on Fine-Grained Overlap is Highly Speculative:** The authors note that under extreme intra-domain semantic overlap (e.g., distinguishing fine-grained visual subdomains), early-layer routing might suffer from representation bleed, and propose offline alignment strategies (CKA, Procrustes projection) as a solution. However, this is left as future work and lacks empirical verification in the current manuscript.
3. **Terminology of "Zero-Shot" Centroids:** The paper refers to PEAR's reference anchors as "Zero-Shot Patch Centroids." However, computing these centroids requires an offline calibration split ($B_{\text{cal}} = 64$) with explicit task class labels. While extremely low-shot, it is technically a weakly-supervised calibration phase rather than a purely "zero-shot" or unsupervised strategy.

---

## Soundness
**Rating: Excellent**

The submission is technically exceptionally sound. The mathematical formulations of PEAR—including Patch Embedding projection, token spatial pooling, Zero-Shot Patch Centroids, Unit-Norm Cosine similarity, and Intra-Task Dispersion Calibration (IDC)—are rigorous and complete. The authors are highly careful and honest about the limitations of their work, identifying the Global-Average-Color Routing Paradox on real images and proposing robust, mathematically aligned solutions (Early-Layer Routing Compromise and ELFT). 

The experimental design is highly rigorous, utilizing 5 independent random seeds with reported standard deviations across all sandbox evaluations. The SVHN task is intentionally configured as a highly degraded stress-test (19.68% expert ceiling) to evaluate robustness under corrupted data structures. The real-world ViT evaluations on actual images successfully bridge the simulation-to-real-world gap, and end-to-end adapter ensembling experiments thoroughly verify that PEAR's early-layer routing translates to correct multi-task classification.

---

## Presentation
**Rating: Excellent**

The submission is clearly written, highly structured, and the narrative flow is exceptionally easy to follow. The paper progresses logically from formalizing serving bottlenecks in existing ensembling literature to the PEAR methodology, synthetic simulation sweeps, real-world ViT evaluations, ablation studies, and systems latency profiling. 

All tables are complete, clearly captioned, and include explicit error bars. Figures 5a and 5b are of high quality and concisely convey the core systems-level advantages of PEAR (flat $O(1)$ latency scaling and robustness to stream heterogeneity). The authors properly position their work relative to static weight merging (TIES, DARE), parametric gating (Linear Router), and non-parametric activation blending (PFSR, SABLE), clearly detailing how PEAR resolves the early-layer restriction of SABLE and the Vectorization Collapse of parametric routers.

---

## Significance
**Rating: Excellent**

The paper addresses a highly significant and pressing practical bottleneck in multi-task edge serving. By proving that early-layer representations in Vision Transformers can serve as highly accurate, zero-cost non-parametric routers, PEAR completely resolves the Routing Paradox and bypasses SABLE's late-adaptation capacity bottleneck. This allows edge operators to deploy merged expert adapters with full-depth layer adaptability in a single parallel forward pass, maintaining flat $O(1)$ latency and zero dynamic state memory.

Given PEAR's simplicity, robustness to single-sample vectorized streams ($B=1$), and systems-aware fallbacks (Hard Edge Rejection with generalist heads), this framework is immediately deployable in real-world visual pipelines (e.g., smart cameras, robotics, edge servers). Deep learning systems researchers and practitioners are highly likely to build on PEAR's early-layer routing and training-serving alignment (ELFT) principles for production-scale deployments.

---

## Originality
**Rating: Excellent**

The paper demonstrates outstanding originality. Instead of proposing more complex parametric routing networks that require heavy training and overfit to calibration data, the authors rethink the architectural location of the routing boundary in Vision Transformers. Performing non-parametric routing inside Layer 0 or early layers represents a highly creative and original concept.

Additionally, the integration of Intra-Task Dispersion Calibration (IDC)—which standardizes similarity scales across asymmetric task manifolds—and Adaptive Task-Specific Thresholding—which scales OOD boundaries dynamically with task density—represents highly original, systems-aware algorithmic designs. The identification and resolution of the Global-Average-Color Routing Paradox through the Early-Layer Routing Compromise and ELFT further demonstrate a high degree of technical originality.

---

## Overall Recommendation
**Rating: 5: Accept**

**Justification:**
This is a technically solid, highly thorough, and exceptionally well-written paper that addresses a major practical bottleneck in parameter-efficient serving. Grounded in the minimalist philosophy of Occam's razor, the proposed PEAR framework achieves state-of-the-art ensembling performance and full-depth layer adaptability with zero trainable parameters and flat $O(1)$ sequential latency complexity. The paper's double-tier evaluation—combining a controlled synthetic sandbox with an end-to-end real-world Vision Transformer pipeline on actual images—rigorously validates its claims. Proposing the Early-Layer Routing Compromise and ELFT to resolve the Color Routing Paradox is a highly complete, systems-aligned engineering feat. 

While profiling on physical NPUs and empirical verification under extreme fine-grained overlap are left for future work, the current manuscript's strengths far outweigh its minor limitations. PEAR represents a highly significant contribution to the machine learning systems and PEFT serving communities, and is highly recommended for acceptance.
