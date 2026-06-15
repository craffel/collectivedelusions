# 4. Experimental Evaluation Check

## Strengths of Experimental Design
The experimental design is highly structured, transparent, and exceptionally thorough across several dimensions:
1. **Strong Baseline Comparisons:** The authors compare FIOSR against five major baseline models, covering static merging, learned parametric models (Linear Router, L3-Softmax), SOTA quantum wave routing (QWS-Merge), and flat parameter-free projection (PFSR).
2. **Robustness Regimes:** Testing across varying stream batch sizes ($B=1$ to $512$) explicitly exposes the "Vectorization Collapse" of parametric models, providing strong support for the robustness of parameter-free routing with MBH.
3. **Statistical Significance:** Evaluating across 10 independent random seeds ensures that the reported improvements (e.g., the +8.56% gain over PFSR) are statistically significant and robust.
4. **Decisive End-to-End Physical Validation:** Section 4.8 introduces a physical validation setting using a pre-trained `ResNet-18` feature extractor and training specialized expert classifier heads on real image datasets (MNIST, FashionMNIST, and SVHN). This physical evaluation is a major strength that directly bridges the external validity gap, proving that FIOSR can be deployed on physical feature activations and handle dead or near-zero variance dimensions using scale-regularization.
5. **Real-World LoRA Simulation:** Section 4.7 evaluates FIOSR on highly anisotropic, correlated activation spaces from physical LoRA adapters, demonstrating a massive $+16.67\%$ routing accuracy improvement and recovering $98.30\%$ of the oracle performance.

## Crucial Experimental Weaknesses and Critiques

### Critique 1: Dataset Simplicity and Architecture Scaling (External Validity Gap)
While the addition of the physical ResNet-18 validation is a major strength, the datasets used (MNIST, FashionMNIST, and SVHN) are relatively simple and low-dimensional. 
- **Experimental Limitation:** The framework has not been evaluated on large-scale modern architectures (such as Vision Transformers like ViT-Huge, or Large Language Models like LLaMA-3-70B) or complex natural datasets (like ImageNet or GLUE). While the synthetic sandbox and ResNet-18 feature extraction isolate routing dynamics well, deep features of large-scale autoregressive transformers exhibit complex activation drift, token distribution shifts, and multi-layer attention dynamics that are not fully captured by these benchmarks.

### Critique 2: Heavy Computational and Systems Overhead of Micro-Batch Homogenization (MBH)
MBH dynamically partitions a heterogeneous batch of size $B$ into $G \le K$ homogeneous micro-batches based on dominant task coordinates. 
- **Systems Overhead Bottleneck:** While MBH successfully prevents heterogeneity collapse, executing up to $G$ separate sequential forward passes on $G$ dynamically merged models introduces significant latency and memory bandwidth overhead. If $G=K$ (all tasks active in a stream batch), model merging is computationally equivalent to running all individual expert models separately, completely negating the computational and parameter-efficiency benefits of test-time ensembling.
- **The Gating Compromise:** Although Top-$M$ Expert Gating with $M=1$ (hard Top-1 routing) restricts active forward passes to 1 and completely eliminates sequential MBH overhead, it collapses the model back to a hard task selector, losing the ensembling benefits for that sample. This represents a fundamental system-level trade-off (latency vs. accuracy) that is bypassed rather than fully resolved.

### Critique 3: Class Vocabulary Asymmetry and Fisher Storage Scaling
In Section 4.1, the authors define asymmetric task class sizes as $C_{\text{tasks}} = [10, 10, 10, 4]$ to validate CSC. 
- **Experimental Limitation:** While this successfully demonstrates a +1.15% gain from CSC, the vocabulary sizes (e.g., 4 or 10 classes) are microscopic compared to large language models (LLMs) with massive vocabularies ($C \approx 32\text{K}$ to $128\text{K}$) and high-dimensional representations ($d \approx 4096$). Storing $K \times C \times d$ Fisher coefficients in LLMs would introduce massive memory and storage overhead. Although the authors propose "Class-Grouped Pooling" and "Low-Rank FIM Factorization" as scaling strategies in Section 4.5, these strategies are merely discussed and not empirically validated, representing a significant scaling limitation.

## Experimental Rating: Excellent
With the addition of Section 4.7 (LoRA activation space validation) and Section 4.8 (end-to-end physical validation on a pre-trained ResNet-18), the authors have successfully resolved the major "synthetic sandbox over-reliance" critique. The experimental design is exceptionally strong, covering synthetic sandbox environments, realistic LoRA activation spaces, and end-to-end physical image classification.
