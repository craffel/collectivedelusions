# Intermediate Evaluation 3: Soundness and Methodology

## Clarity of Description and Appropriateness of Methods
The paper's mathematical formulation of PEAR is clearly written and structured. The sequence of steps—Zero-Shot Patch Centroids (ZPC), Unit-Norm Cosine Projection, Intra-Task Dispersion Calibration (IDC), and Temperature-Scaled Softmax—is logical and easy to follow. 

However, a rigorous analysis of the methodology reveals several critical technical flaws, hidden assumptions, and logical contradictions that compromise the soundness of the framework.

## Potential Technical Flaws and Critical Observations

### 1. The Early-Layer Routing Trilemma
The paper pitches PEAR as a framework that achieves dynamic ensembling "inside the frozen Patch Embedding layer (Layer 0)... allowing expert adapters to be activated and blended across 100% of the network depth." 

However, on actual images, Layer 0 routing suffers from the **Global-Average-Color Routing Paradox** (57.81% accuracy) because spatial pooling over linear patch embeddings is mathematically equivalent to taking a projection of the global average color/brightness of the image.

To resolve this, the authors shift the routing boundary deeper to Layer 1 or Layer 2 (the *Early-Layer Routing Compromise*). This introduces a fundamental **trilemma** that the authors fail to resolve transparently:
- **Branch A: Representational Mismatch.** If they run Blocks 0--1 using the unadapted base model (to get activations for routing) but keep early adapters active in the expert LoRAs, they introduce a representational mismatch at Block 2, which degrades performance.
- **Branch B: Double-Pass Latency.** If they run Blocks 0--1 using the unadapted base model for routing, and then *re-run* Blocks 0--1 using the adapted weights of the selected expert, they violate their flat single-pass $O(1)$ latency claim.
- **Branch C: Freezing Early Layers (ELFT).** If they use Early-Layer Freezing during Training (ELFT), they freeze Blocks 0--1 during training and serving. This means **they are no longer adapting 100% of the network depth**. They are doing the *exact same late adaptation* as SABLE SOTA, simply freezing 2 layers instead of 10.

The authors cannot escape this trilemma. ELFT is presented as a solution, but it is a conceptual capitulation: it proves that full-depth layer adaptability (100% depth) is practically impossible on real images without either doubling latency or suffering from representational mismatch.

### 2. Generalist Classification Head as OOD Fallback
For OOD inputs, the paper proposes a "Hard Edge Rejection" fallback that routes queries through a "dedicated, task-agnostic, low-cost generalist classification head" trained directly on the unadapted base representations over the calibration split (K * 64 samples).
- **Label Space Conflict:** In a typical multi-task setting, separate tasks have completely disjoint and incompatible label spaces (e.g., MNIST digits 0--9 vs. CIFAR-10 classes). The paper does not explain how a single "generalist head" can make coherent predictions across disjoint label spaces. If the head must predict over the union of all classes, training a classifier on only 256 total samples (4 tasks * 64 samples) is heavily under-parameterized and will overfit aggressively.
- **Lack of Empirical Evidence:** The authors provide zero empirical validation or accuracy results for this generalist head under OOD scenarios.

### 3. Asymmetric Task Densities and Adaptive OOD Thresholding
In Section 4.5.2 (Table 6), the adaptive thresholding strategy ($\eta = 0.85$) is compared against global thresholds.
- **Degraded False Acceptance Rate (FAR):** The adaptive threshold yields an MNIST FAR of **5.47%** (worse than Global $\gamma_{OOD}=0.15$'s **4.68%**) and an SVHN FAR of **17.19%** (worse than Global $\gamma_{OOD}=0.15$'s **15.62%**).
- **In-Distribution Accuracy Noise:** The main advantage claimed for the adaptive threshold is that it preserves SVHN accuracy (13.60% vs. 10.00% under Global $\gamma_{OOD}=0.15$). However, given the test set size of 64 samples, 13.60% represents exactly **8.7 samples** correct, while 10.00% represents **6.4 samples** correct. A difference of **2.3 samples** is well within statistical noise and does not provide robust scientific evidence of the superiority of adaptive thresholding, especially when it actually worsens the False Acceptance Rate on both tasks.

### 4. Strawman Baseline for Parametric Gating
The Linear Router is trained on only 64 calibration samples per task. This extremely low-data split guarantees overfitting and poor generalization, which the authors use to claim "Vectorization Collapse." A fairer and stronger parametric baseline would perform a linear probe on the frozen ViT features, rather than training a router under highly compromised data-scarcity constraints.

## Reproducibility
The authors state that they evaluate their framework inside a "high-fidelity 12-layer synthetic representation sandbox in PyTorch" where "no actual image datasets, pixel-level inputs, or pre-trained Vision Transformers are executed" (except in the real-world validation section). 
While they declare this to preserve transparency, evaluating primarily on simulated, artificial 1D Gaussian vector representations makes the core empirical findings highly detached from real-world visual manifolds. The tiny scale of the real-world validation ($128$ total samples per task, 64 calibration / 64 test) further limits the reproducibility and statistical significance of the results.
