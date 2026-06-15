# Peer Review of Conference Submission: PEAR (Patch-Embedding Activation Routing)

## Summary of the Paper
This paper introduces **PEAR (Patch-Embedding Activation Routing)**, a parameter-free, closed-form ensembling framework designed to dynamically blend specialized low-rank expert adapters (LoRAs) at runtime on resource-constrained devices. The paper identifies two key bottlenecks in current multi-task serving literature: (1) the *Routing Paradox / Early-Feature Loss Trade-Off* in non-parametric routers like SABLE, which perform routing in deep layers and leave early layers unadapted to avoid redundant backbone execution, and (2) *Vectorization Collapse* in parametric routers (e.g., linear gating layers) under vectorized streaming regimes (batch size $B=1$).

To address these limitations, PEAR proposes performing non-parametric routing inside the frozen Patch Embedding layer (Layer 0) of a Vision Transformer. It calculates cosine similarities of global-average-pooled Layer 0 features against offline-constructed task-specific Zero-Shot Patch Centroids (ZPC) over a small calibration set ($N_{cal}=64$). It then standardizes these similarities via Intra-Task Dispersion Calibration (IDC) to resolve asymmetric task densities, normalizes them via a temperature-scaled softmax to compute sample-specific routing weights, and dynamically blends active expert adapters across the remaining blocks in a single parallel forward pass.

While evaluating PEAR on real-world images from MNIST, Fashion-MNIST, CIFAR-10, and SVHN, the authors discover a **Global-Average-Color Routing Paradox** where Layer 0 routing fails on real-world images (57.81% accuracy) because spatial pooling over linear patch embeddings is equivalent to averaging pixel colors. To resolve this, they propose the *Early-Layer Routing Compromise*, shifting routing to Layer 1 or Layer 2. Combined with *Early-Layer Freezing during Training* (ELFT) to resolve boundary representational mismatch, they show that PEAR L2 recovers up to 85.10% of its Expert Ceiling and outperforms SABLE SOTA by 11.72%--15.24% on real-world images.

---

## Overall Recommendation
**Rating:** 2: Reject
**Justification:** While the paper is well-written and demonstrates a commendable effort to engage with practical systems and engineering challenges, it suffers from a fundamental logical contradiction (the Early-Layer Routing Trilemma), an extremely weak and statistically under-powered real-world evaluation, and unfair baseline comparisons.

---

## Detailed Evaluation

### 1. Soundness
**Rating:** Fair
**Justification:** The core mathematical framework is clear and logical. However, several critical technical flaws and unaddressed logical contradictions undermine the soundness of the methodology:

- **The Early-Layer Routing Trilemma:** The authors pitch PEAR as a framework that achieves dynamic ensembling "inside the Patch Embedding layer (Layer 0)... allowing expert adapters to be activated and blended across 100% of the network depth." However, on actual images, Layer 0 routing completely fails (57.81% accuracy) due to the "Global-Average-Color Routing Paradox." To fix this, the authors shift routing to Layer 1 or Layer 2 (the *Early-Layer Routing Compromise*). This introduces a fundamental trilemma:
  1. *Representational Mismatch:* If they run the early blocks using the unadapted base model (to extract features for routing) but keep early adapters active in the experts, they introduce a representational mismatch at the boundary block, which degrades performance.
  2. *Double-Pass Latency:* If they run the early blocks with the unadapted base model for routing and then re-run them with the adapted weights, they violate their flat single-pass $O(1)$ latency claim.
  3. *Early-Layer Freezing (ELFT):* If they freeze the early blocks during training and serving to align the paths, **they are no longer adapting 100% of the network depth**. They are doing the *exact same late adaptation* as SABLE SOTA, simply freezing 2 layers instead of 10.
  The authors present ELFT as a solution, but it is a conceptual concession: it proves that full-depth layer adaptability is practically impossible on real images without either doubling latency or suffering from representational mismatch.
- **Label Space Conflicts in OOD Fallback:** The "Hard Edge Rejection" fallback routes OOD queries to a "dedicated, task-agnostic, low-cost generalist classification head" trained directly on the unadapted base representations over the combined calibration splits ($K \times 64 = 256$ samples). In multi-task serving, separate tasks have completely disjoint label spaces (e.g., MNIST digits vs. CIFAR-10 objects). The authors fail to explain how a single generalist head can output coherent predictions across disjoint label spaces. Training a unified classifier on only 256 samples is heavily under-parameterized and will overfit aggressively. The authors provide zero accuracy results or evaluations for this head.
- **Asymmetric Task Densities and Adaptive OOD Rejection:** The authors claim that their adaptive OOD thresholding ($\gamma_{OOD, k} = \eta \cdot d_k$) resolves task density asymmetry. However, looking at Table 6, the adaptive threshold actually *increases* (degrades) the False Acceptance Rate (FAR) on both MNIST and SVHN compared to a global threshold of 0.15. Furthermore, the reported SVHN in-distribution accuracy improvement (13.60% vs. 10.00%) is evaluated on only 64 test samples. A difference of 3.6% represents exactly **2 extra samples** correct, which is statistically insignificant and does not justify the increased FAR.

### 2. Presentation
**Rating:** Good
**Justification:** The paper is highly polished, the math is clearly formulated, and the tables and figures are well-formatted. However, the presentation suffers from **conceptual inflation and over-marketing**:
- The authors repeatedly sell PEAR as a "strictly parameter-free routing framework designed for the frozen Patch Embedding layer (Layer 0)" that "adapts 100% of the network depth."
- This sales pitch is contradicted by their own empirical findings: Layer 0 routing fails on real images, and their best-performing setup is Layer 2 routing with ELFT (which freezes the first two blocks).
- The authors must tone down their claims in the abstract and introduction, and align their narrative with their actual, best-performing real-world pipeline.

### 3. Significance
**Rating:** Fair
**Justification:** Shifting the routing boundary slightly deeper to Layer 2 to preserve $\ge 83\%$ of the network's adaptability is a useful practical trick. However, because this setup relies on the exact same early-freezing paradigm as SABLE SOTA, the conceptual significance is incremental. 
Moreover, because the real-world evaluations are conducted on an extremely small scale (64 test samples per task) with highly unperformant experts (SVHN expert ceiling is 37.50%), the practical significance of the results is severely limited. Other researchers and practitioners cannot verify if these ensembling gains hold under realistic production conditions where experts are highly capable.

### 4. Originality
**Rating:** Fair
**Justification:** The individual components of the framework—such as cosine similarity, centroid matching, and freezing early layers—are heavily derived from prior work (SABLE, SPS-ZCA, etc.). 
While Intra-Task Dispersion Calibration (IDC) is a nice addition to balance task densities, the overall architecture is conceptually very close to SABLE when deployed on actual images (where early layers must be frozen via ELFT).

---

## Detailed Strengths and Weaknesses

### Strengths
1. **Engaging with Real-World Failure Modes:** The paper is highly commendable for proactively identifying and analyzing the "Global-Average-Color Routing Paradox" on actual visual manifolds, rather than hiding behind synthetic sandbox successes.
2. **Systems-Level Discussion:** The authors show a strong appreciation for physical hardware limitations, including detailed analyses of memory bandwidth, physical memory transfer serialization, and hardware concurrency ceilings on edge devices.
3. **Rigorous Sensitivity Analysis:** The ablation sweeps over the temperature parameter ($\tau$) and OOD thresholds ($\gamma_{OOD}$) provide useful engineering guidelines.

### Weaknesses
1. **Severe Statistical Under-Powering:** Evaluating the real-world classification pipeline on only **64 test images per dataset** (a total of 256 images across 4 tasks) is extremely weak. In public benchmark datasets like MNIST, CIFAR-10, and SVHN, test splits contain thousands of images. Restricting the evaluation to 64 samples makes the accuracy figures highly volatile and statistically unreliable.
2. **Strawman Parametric Baseline:** The "Tiny CNN Router" baseline is trained from scratch on only 256 total calibration samples. It is mathematically guaranteed to overfit aggressively. Comparing PEAR (which utilizes a pre-trained ImageNet ViT backbone) to this overfitted, small-scale CNN is an unfair comparison that does not prove PEAR's architectural superiority. A pre-trained MobileNet or a linear probe on the ViT features would be a much stronger and fairer baseline.
3. **Compromised Expert Specialists:** The "expert" adapters used in the real-world LoRA validation are trained on only 64 samples. Consequently, their classification ceilings are extremely low (e.g., SVHN expert ceiling is only 37.50%--39.06%). Evaluating an ensembling framework on "experts" that perform barely above random guessing limits the scientific value of the results, as a practical serving system is designed to orchestrate *highly performant* specialists.
4. **Lack of Actual Hardware Benchmarking:** Despite the extensive theoretical discussion regarding edge NPUs, LPDDR memory buses, and concurrency limits, all latency benchmarks are executed on a standard CPU. The paper lacks actual profiling (e.g., measuring physical throughput, memory transfer bottlenecks, or power draw) on standard edge hardware (like Jetson Nano or Raspberry Pi), leaving its systems-level claims unverified.

---

## Questions and Requests for the Authors
1. **Why is the real-world evaluation restricted to only 64 test samples per task?** Please provide results evaluated on the complete, standard test splits of MNIST, Fashion-MNIST, CIFAR-10, and SVHN.
2. **How does the "generalist classification head" resolve label space conflicts across disjoint tasks?** Please provide a detailed architectural description of this head, explain how its loss function is formulated across disjoint label spaces, and report its classification accuracy under OOD and in-distribution fallback scenarios.
3. **Why did you train a Tiny CNN from scratch as the parametric baseline?** A pre-trained CNN (e.g., ResNet-18) or a linear layer trained on the pre-trained ViT's Block 0/1 features would be much stronger. Please update your baseline evaluations to include these fairer comparison models.
4. **Please provide physical edge-hardware profiling.** Run your latency and memory benchmarks on an actual edge device (such as a Jetson Nano or Raspberry Pi) and report physical throughput (frames per second) and cache utilization as the number of experts $K$ scales, validating your LPDDR memory bus bottleneck discussion.
5. **Please tone down your marketing claims.** Given that Layer 0 routing fails on real images and you must freeze Blocks 0--1 via ELFT, please remove or modify claims of "routing inside the Patch Embedding layer (Layer 0)" and "adapting 100% of the network depth" from your abstract and introduction, as they are contradicted by your actual real-world pipeline.
