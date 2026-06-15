# Experimental Evaluation and Audit: SPS-ZCA

## Experimental Setup and Datasets
The experimental sandbox setup (Isolating Coordinate Sandbox, or ICS) models a Vision Transformer backbone ($L=14$ layer groups, $D=192$ dimensions) and $K=4$ task experts. The primary visual datasets used for evaluation are:
- **MNIST** (handwritten digits)
- **Fashion-MNIST** (clothing items)
- **CIFAR-10** (natural images)
- **SVHN** (street view house numbers)

For downstream text generalizability, the paper utilizes a pre-trained **GPT-2** model evaluated across three domains:
- **Legal**
- **Medical**
- **Code**

## Critical Evaluation of the Experimental Design

While the empirical evaluation is detailed, a rigorous analysis reveals several critical limitations and gaps in the experimental setup:

### 1. Toy and Simplistic Nature of Visual Datasets
The primary visual evaluation relies on highly simplistic, low-resolution benchmarks (MNIST, Fashion-MNIST, CIFAR-10, SVHN).
- These datasets represent **toy-like classification tasks** with highly distinct and easily separable geometric structures in early activation spaces.
- This extreme separation is reflected in the reported **Fisher Separability Criterion (FSC) of 47.50 at Layer 3**. Such a high FSC guarantees that a simple, training-free nearest-centroid router will achieve near-perfect routing accuracy.
- In realistic, complex, or fine-grained scenarios (e.g., ImageNet subclasses, CUB-200 birds, Stanford Cars), the semantic boundaries in the early representation space would be heavily overlapped and entangled. Under these realistic conditions, the FSC would collapse, leading to high routing error and on-the-fly "activation bleeding" that would degrade performance toward the static Uniform Merging baseline.
- Although the authors discuss this limitation conceptually under "Architectural Generalizability" and propose Hierarchical Centroid Clustering or Supervised Head Fine-Tuning, they do **not** provide any physical or simulated results on a fine-grained dataset to validate these mitigations.

### 2. Low Classification Baseline for the SVHN Expert
An examination of the classification accuracies reveals a major quality gap in the task-specific experts:
- In the simulated Table 1, the **SVHN Expert Ceiling is reported as only 31.20%**.
- In the physical PyTorch Table 4, the **SVHN Expert Ceiling is reported as only 29.78%**.
- For a standard 10-class classification task like SVHN, a well-tuned Vision Transformer model should easily achieve classification accuracies exceeding **90%**. An expert ceiling of under 30% indicates that the SVHN expert was either trained on an extremely truncated subset or suffered from severe optimization issues.
- Drawing strong conclusions about the robust recovery of the joint "Expert Ceiling" is mathematically and empirically weak when one of the primary task-specific components is performing barely better than random guess. It remains unproven whether the centroid alignment and activation blending framework can scale and preserve performance when all expert pathways are highly specialized and accurate.

### 3. The "Serving Gap" and Physical Framework Slowdowns at Scale
The physical execution profiling in Section 4.8.2 exposes a critical systems-ML bottleneck:
- Under large batch sizes ($B=256, G=4$), the proposed methods are actually **slower than the sequential MBH baseline** in physical PyTorch:
  - **MBH Baseline:** 303.33 ms
  - **SPS-FP (Ours):** 460.34 ms ($1.52\times$ slowdown)
  - **SPS-SG (Ours):** 346.67 ms ($1.14\times$ slowdown)
  - **SPS-VSG (Ours):** 341.20 ms ($1.12\times$ slowdown)
  - **SPS-Compiled (Ours):** 336.03 ms ($1.11\times$ slowdown)
- This physical slowdown is due to PyTorch's high framework overhead for dynamic boolean masking, tensor slicing, and scatter-gather indexing, which completely overrides the theoretical FLOP savings under sequential CPU execution.
- Although the authors project a **3.90$\times$ speedup** under an analytical cost model, this is an idealized, theoretical model that assumes native compiler-level fused loops.
- The actual physical wall-clock speedup is only achieved under small batch sizes ($B=16$), where sequential kernel dispatching delays dominate. For larger scales, the paper relies on a conceptual compiler-fused loop pseudocode (Appendix A) rather than an actual compiled binary. This creates a substantial gap between the paper's theoretical/analytical speedup claims and the physical reality in standard deep learning frameworks.

### 4. Small Calibration Size and Overfitting in OOD Rejection
The coordinate-space GMM for OOD rejection is fitted on only **64 calibration samples per task**.
- While the OOD threshold sensitivity sweep (Table 3) and the ROC curve (Figure 3) show strong performance (95.2% TPR at 4.3% FPR) over 1000 out-of-sample validation samples, fitting a Gaussian Mixture Model on such extreme data scarcity remains statistically risky.
- A small sample size is highly sensitive to outliers and covariate shifts. If the in-distribution tasks undergo even minor domain shifts, the routing coordinates will drift, potentially falling into the low-density GMM tails and causing an escalation in false OOD rejections (higher FPR).
- A more thorough evaluation would sweep the intensity of in-distribution noise and covariate shift to measure the stability of the coordinate GMM density boundaries under non-ideal, realistic serving conditions.
