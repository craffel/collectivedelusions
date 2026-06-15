# 3. Soundness and Methodology

## Clarity and Completeness of Description
The paper's methodological description is exceptionally clear, rigorous, and complete. Every single equation, baseline, and optimization algorithm is laid out with precision:
* **Multi-Task Merging Formulation:** Explicitly defines the blending coefficient matrix $\Lambda \in [0, 1]^{K \times L}$ and how continuous unquantized layer weights $\theta^l_{\text{merged}}(\Lambda)$ are constructed via task vectors.
* **Post-Training Quantization (PTQ) Operators:** Formulates Uniform Asymmetric and Uniform Symmetric quantization mathematically. The authors provide crucial details on how scales ($s$) and zero-points ($z$) are dynamically recalculated at every forward pass, and clarify that PyTorch's autograd propagates gradients through active continuous scale ranges back to $\Lambda$ but is blind to zero-point rounding (asymmetric gradient flow).
* **Hardware Scenario Contextualization:** Contextualizes the target schemas with physical hardware architectures (e.g., Google Edge TPU requiring tensor-wise symmetric precision, Qualcomm Hexagon or Apple Neural Engine supporting asymmetric channel-wise precision).
* **Optimization Formulations:** Both first-order optimization (prediction entropy minimization with Elastic Spatial Regularization and Adam) and derivative-free optimization (1+1 Evolution Strategy with Rechenberg's 1/5th success rule) are mathematically detailed.
* **Hybrid Optimization Pipeline:** The proposed deployment-robust optimizer is fully formalized and presented as a high-quality pseudocode layout (Algorithm 1) in the appendix.

## Appropriateness of Methods
The evaluation design is highly appropriate and rigorous:
1. **Benchmark Choice:** Choosing a standardized `vit_tiny_patch16_224` backbone with MNIST, FashionMNIST, CIFAR-10, and SVHN heads represents a controlled, highly non-convex, and high-interference regime. Since unquantized Task Arithmetic baseline collapses from $>90\%$ individual performance to $35.12\%$ joint performance, this creates an extreme weight conflict. Stress-testing quantization-aware merging under such high conflict is an excellent way to expose the boundaries and limits of STE optimization.
2. **Control Baselines:** Including **Quantized AdaMerging** (unquantized FP16 search followed by post-hoc quantization) is a critical methodological baseline that isolates whether quantization-aware optimization is truly beneficial or if unquantized search is superior.
3. **Supervised Baseline:** Formulating a supervised calibration baseline (cross-entropy with true labels on the $N$-sample calibration stream) is an elegant control that successfully decouples data scarcity ($N$) from the structural failures of prediction entropy minimization.
4. **Empirical Extensions:** Incorporating a convolutional model (ResNet-18) and low-rank subspace projection (SVD rank-4) are highly appropriate to test architectural and parameter-space generalizability.

## Potential Technical Flaws and Transparent Limitations
The paper is technically flawless and handles potential limitations with high integrity and transparency:
* **SVD Subspace Proxy:** The authors acknowledge that their post-hoc SVD projection is a proxy for parameter-efficient fine-tuning (PEFT) and might not perfectly capture natively-trained LoRA. They transparently analyze that SVD-induced capacity degradation collapses absolute performance to $13.00\%$, cautioning the reader against interpreting the flat generalization gap as an active, robust alignment (the "Low-Capacity Generalization Illusion").
* **Scale-Up Bottlenecks:** The authors transparently explain the physical and computational bottlenecks that prevent empirical scale-up validation (requiring massive training resources to retrain matching expert checkpoints on multi-billion parameter backbones). They provide a robust scientific projection explaining why model scale is expected to expand, rather than contract, the Cross-Schema Generalization Gap due to the exponential growth of independent discrete rounding thresholds.

## Reproducibility
The reproducibility of the work is exceptionally high. The authors specify:
* Exact architectures (`vit_tiny_patch16_224`, `resnet18`).
* Exact number of parameters (5.7M for ViT-Tiny, 11.7M for ResNet-18) and layers ($L=14$ for ViT-Tiny).
* Specific datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).
* Optimization hyperparameters (Adam learning rate $10^{-2}$, moment coefficients, number of steps 100, initialization value $0.3$, search space size of 56 parameters).
* 1+1 ES parameters (initial step size $\sigma^{(0)} = 0.05$, generations 50, Rechenberg success multipliers).
* TV regularization scaling ($\alpha = 0.5$ default, and sweep values).
The comprehensive mathematical formulation of the quantization grids and algorithms ensures that an expert researcher could recreate this exact audit pipeline seamlessly.
