# Intermediate Evaluation 4: Experiment Check

## Critical Evaluation of the Experimental Setup

### 1. Extreme Scale Deficiency in Real-World Evaluation
The most egregious limitation of the experimental validation is the **extremely small dataset scale** used in Section 4.6.
- The authors evaluate their framework on only **128 real-world images per task** (64 for calibration and 64 for testing). 
- With only 64 test samples per dataset, the entire multi-task evaluation is conducted on a tiny pool of **256 images**.
- This makes the reported classification accuracies (Table 7) statistically highly volatile. A single correct or incorrect prediction changes the accuracy by **1.56%**.
- Drawing sweeping, definitive conclusions about "bridging the simulation-to-real-world gap" and "recovering 85.10% of the expert ceiling" from a test set of 64 images is statistically questionable. A standard, rigorous evaluation would test on the complete test splits of these public datasets (e.g., 10,000 images for MNIST/CIFAR-10 and 26,032 images for SVHN). The decision to restrict testing to 64 samples raises significant doubts about the generalizability of the findings and whether the authors are hiding performance decay over larger data distributions.

### 2. Strawman Baseline for Parametric Router
In the real-world routing comparison (Table 5), PEAR is compared against a **Tiny CNN Router**.
- This CNN router is a simple 3-layer model trained *from scratch* on only 256 total calibration samples. 
- Training a CNN from scratch on 256 samples is a recipe for extreme overfitting. 
- Comparing PEAR (which leverages a powerful, ImageNet-pretrained Vision Transformer backbone containing millions of parameters) to this tiny CNN trained from scratch on 256 samples is a textbook **strawman comparison**. 
- A rigorous evaluation would compare PEAR against a pre-trained feature extractor (such as ResNet-18 or MobileNet-V3) or a linear classifier/probe trained on the ViT backbone's early features. The fact that PEAR's pre-trained features outperform a tiny, overfitted CNN is trivial and does not justify PEAR's architectural superiority.

### 3. Compromised "Expert" Ceilings in Real-World LoRA Validation
Looking at Table 7 (End-to-end multi-task LoRA classification), the "Expert Ceiling" for several datasets is incredibly low:
- SVHN Expert Ceiling is only **39.06%** (Standard) and **37.50%** (ELFT).
- MNIST Expert Ceiling is only **71.88%** (Standard) and **57.81%** (ELFT).
- Joint Mean Expert Ceiling is only **66.80%** (Standard) and **62.89%** (ELFT).
In standard deep learning literature, basic classifiers easily achieve $>98\%$ on MNIST and $>90\%$ on SVHN. The extremely low "Expert Ceilings" in this paper occur because the authors fine-tune their expert adapters on only 64 samples for 15 epochs. 
Because the "experts" themselves are highly unperformant and barely better than random guessing on SVHN (37.50% vs. 10.00% random), the ensembling evaluation is conducted on highly compromised, weak models. This severely limits the practical relevance of the findings: an ensembling framework must prove it can combine *highly capable* specialists, not weak models that have barely learned the task.

### 4. Highly Artificial Synthetic Sandbox Design
In the synthetic sandbox experiments (Section 4.3), SVHN is configured with a "low expert classification ceiling of 19.68%." 
While the authors frame this as an intentional "stress-test" to evaluate robustness under degraded conditions, training a specialist expert that performs barely above random guessing is highly unrealistic. In practice, edge operators would never deploy or route to a corrupted, 19%-accurate specialist. This artificial setup seems designed to manipulate the evaluation to make PEAR's calibration and soft ensembling look more robust compared to hard routing.

### 5. Absence of True Edge Hardware Benchmarking
The paper repeatedly references "resource-constrained edge serving environments," "edge NPUs," "mobile devices," "narrow LPDDR memory bus widths," and "thread concurrency ceilings."
Despite this heavy systems-level marketing, **all latency measurements (Section 4.6.3) are performed on a CPU**. 
There is no hardware profiling, power measurements, or latency benchmarking on actual edge hardware (such as a Jetson Nano, Raspberry Pi, or mobile NPUs). CPU latency measurements do not reflect the specialized execution paths, thread scheduling, and hardware-cache constraints of edge NPUs, leaving the paper's systems-level claims completely unverified.
