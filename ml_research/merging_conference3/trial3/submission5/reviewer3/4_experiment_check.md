# 4. Experimental Evaluation and Empirical Check

## Analysis of Experimental Setup and Baselines
While the paper presents multiple comparative baselines, the experimental setup has several severe limitations regarding scale, dataset choice, and practical relevance:

### 1. Toy-Scale and Outdated Benchmarks
The authors evaluate a Vision Transformer (ViT-Tiny, 5.7M parameters) on four classification benchmarks: **MNIST, FashionMNIST, CIFAR-10, and SVHN**.
* **Greyscale/Simple Datasets for ViT:** MNIST and FashionMNIST are greyscale $28 \times 28$ image datasets. Using a Vision Transformer pretrained on ImageNet ($224 \times 224$ RGB images) on these simple, low-resolution datasets is highly artificial and chemically mismatched. 
* **Lack of Real-World Edge Workloads:** Modern edge AI deployments typically handle natural images, speech/audio, or text processing using architectures like MobileNet, ResNet-50, or TinyLLMs/Llama-based models. Evaluating on MNIST and FashionMNIST does not reflect the complexity of actual edge workloads or the representational capacity of Vision Transformers.

### 2. Underfitted Experts due to Severely Limited Training Data
The individual experts are trained on only **512 samples per dataset** for 5 epochs.
* **Artificially Low Baselines:** Due to this extremely sparse training data, the baseline experts are highly underfitted. For example, the 8-bit individual experts achieve an average accuracy of only **79.05%** (MNIST: 83.55%, F-MNIST: 78.62%, CIFAR-10: 82.40%, SVHN: 71.63%). A standard ViT-Tiny fine-tuned on full datasets (e.g., CIFAR-10 or SVHN) should easily achieve over 90-95% accuracy. Evaluating model merging on weak, underfitted experts makes it unclear whether the proposed regularization benefits would hold for highly optimized, fully converged models where parameter distributions are much more structured.

### 3. Practical Value of On-Device TTA vs. Offline Optimization
A crucial baseline in the paper is **AdaMerging (Adam $\to$ 8-Bit/4-Bit)**, which represents full-precision, offline server-side optimization followed by post-hoc quantization.
* **Performance Gap:** In both 8-bit and 4-bit regimes, this offline baseline **strictly outperforms** the proposed direct on-device quantization-aware methods:
  - *8-Bit PTQ:* AdaMerging (Adam $\to$ 8-Bit) achieves **62.27%** vs. Q-PolyMerge (Adam) at **59.76%**.
  - *4-Bit PTQ:* AdaMerging (Adam $\to$ 4-Bit) achieves **50.20%** vs. Q-PolyMerge (Adam) at **48.87%**.
* **Conceptual Questioning of On-Device Adaptation:** The authors justify on-device adaptation by arguing that first-order gradient descent is "physically impossible" on low-power nodes due to SRAM limits (158.40 MB cached activations). While this memory constraint is real, **why is it necessary to adapt on-device at all if we can perform the adaptation offline on a server prior to deployment?** 
  If the practitioner has access to the unlabeled calibration stream (even if it is small, e.g., 16 images), they can run the optimization in full precision on a server/cloud instance, quantize the resulting merged model, and then flash the quantized model to the edge device. This offline pipeline avoids all SRAM and latency overheads on-device while delivering **superior accuracy (+1.33% to +2.51% absolute improvement)**. The paper's assumption that adaptation *must* occur on-device is highly forced and lacks a strong practical use case where cloud/edge-assisted offline adaptation is impossible.

### 4. Over-Regularization and Mixed Results
The continuous polynomial constraint ($d=2$) reduces the search space but can act as an over-regularizer:
* **8-Bit Regime:** In Table 1, under 8-bit Adam STE, Q-PolyMerge (59.76%) is outperformed by the unconstrained Q-Merge baseline (60.03%). Under 8-bit ES (zero-order), Q-PolyMerge (51.03%) is again outperformed by unconstrained Q-Merge (51.61%). This indicates that when the quantization noise is moderate, restricting the coefficients to a smooth polynomial actually *hurts* the model's capacity to find the optimal task blend.
* **4-Bit Zero-Order Regime:** In Table 3, under 4-bit ES (zero-order), the discontinuous **Block-wise Constant (ES) baseline slightly outperforms Q-PolyMerge (ES) by 0.28%** (43.33% vs 43.05%), indicating that the polynomial trajectory is not the most optimal parameter reduction strategy for derivative-free search.

## Statistical Rigor
The experiments are run on only **3 random seeds** (42, 100, 2026). Given the high standard deviations observed in the zero-order ES results (e.g., standard deviation of up to 11.75% for Q-PolyMerge ES on MNIST in Table 1, and 18.95% on CIFAR-10), 3 seeds are statistically insufficient to draw confident conclusions. The high variance indicates that the evolutionary search is highly unstable, and a larger sample of seeds (e.g., 5 or 10) is required to establish statistical significance.
