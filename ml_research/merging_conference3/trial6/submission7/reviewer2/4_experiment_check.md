# Experimental Evaluation and Critical Audit

## 1. Experimental Setup and Baselines
The paper evaluates the proposed PFSR + MBH + UNC framework using:
- **A Synthetic Sandbox:** Designed as a diagnostic physical laboratory with $L=14$ layers, feature dimension $D=192$, and $K=4$ tasks (MNIST, FashionMNIST, CIFAR-10, and SVHN).
- **Computer Vision Domain:** Vision Transformers (ViT-Base, $D=768$) on 4 DomainNet domains (Quickdraw, Real, Sketch, Infograph).
- **Large-Scale NLP:** LLaMA-7B ($D=4,096$) on 4 NLP domains (Math/GSM8K, Coding/HumanEval, Translation/WMT, Instruction/Alpaca).

The baseline comparisons are highly comprehensive, including:
- **Static Merging Methods:** Uniform Merging, Task Arithmetic, and TIES-Merging.
- **Parametric Dynamic Routers:** Unregularized Linear Router, regularized Linear Router, QWS-Merge (SOTA wave-inspired router), and multiple layer-wise routing variations (L3-Linear, L3-Tanh, L3-Softmax).

This is an impressive range of baselines and task domains. However, a critical audit of the empirical methodology reveals deep, fundamental weaknesses.

## 2. Critical Evaluation & Methodological Gaps

### A. The "Simulation" Illusion (No Real-World End-to-End Evaluation)
The most severe, glaring weakness of the empirical validation is the **reliance on simulated representations instead of live model execution** for both the Vision Transformer (DomainNet) and LLaMA-7B benchmarks.
As explicitly disclosed in Sections 4.7 and 4.8:
- *DomainNet Benchmark:* "...these real-world benchmarks are evaluated using **simulated penultimate feature representation manifolds** modeled after actual ViT-Base domain feature distributions... rather than live fine-tuned Vision Transformer weights on full datasets during each simulation pass."
- *LLaMA-7B Benchmark:* "...these large-scale LLM evaluations are **simulated using representative feature embeddings and pre-calculated statistical expert ceilings** rather than running live 7-billion parameter active inference over raw text corpora..."

This is a massive empirical gap. The authors claim "Diverse Real-World Evaluations" and "validation on large-scale NLP experts (LLaMA-7B)," but they did not actually run these models or perform real weight-merging on live weights. Instead, they ran mathematical simulations of the penultimate representations. 
In a real deployment, representation manifolds are highly non-linear, dynamic, subject to noise, and prone to complex representation drift. Simulating feature spaces (presumably using simple parametric distributions) fails to capture the true complexity of live deep networks. Consequently, the claimed results on DomainNet (Table 5) and LLaMA-7B (Table 6) are speculative and lack scientific rigor, as they do not constitute genuine end-to-end empirical validation.

### B. The Tautological Nature of MBH Validation
The authors show that PFSR + MBH completely resolves heterogeneity collapse on mixed-task streams (Table 2 and Table 5). However, this result is **tautological and guaranteed by construction**.
By design, Micro-Batch Homogenization (MBH) partitions a heterogeneous batch into homogeneous micro-batches and dispatches them sequentially. Mathematically, this forces the input stream for each forward pass to be perfectly homogeneous. 
Therefore, the fact that MBH avoids heterogeneity collapse is not an algorithmic or representation-learning breakthrough; it is a direct consequence of simply running sequential, task-pure inference passes. The comparison against standard dynamic routers (which must process mixed-task batches in a single pass due to accelerator constraints) is unfair and trivial. It compares a single-pass model with a multi-pass, partitioned-inference system.

### C. Low Expert Ceiling on SVHN
In the synthetic sandbox (Table 1), the "Expert Ceiling" for the SVHN digit classification task is remarkably low at **31.20%**. 
SVHN digit classification is a standard benchmark where simple convolutional networks or lightweight classifiers easily achieve $>90\%$ accuracy. An expert ceiling of only $31.20\%$ suggests that either the specialized expert was severely under-trained, the base backbone architecture ($L=14, D=192$) was highly inappropriate, or the synthetic sandbox represents an excessively noisy, artificial setting. This raises serious doubts about the fidelity and representational validity of the "diagnostic physical laboratory" setup.

### D. Out-of-Distribution (OOD) Rejection Evaluation Limits
The authors propose a Gaussian Mixture Model (GMM) density estimator to perform robust OOD rejection (Table 9). While they report an impressive $95.20\%$ SVHN rejection rate on the synthetic sandbox, this GMM estimator is **not evaluated on the simulated DomainNet or LLaMA-7B benchmarks**. 
Fitting a multi-component GMM on low-dimensional coordinate spaces ($K=4$) might work well in a simplified synthetic sandbox, but its generalizability and scaling behavior in actual, complex high-dimensional feature spaces of ViTs and LLMs remains unproven.

## 3. Do the Results Support the Claims?
- **Claim 1: "Empirical Deconstruction of QWS-Merge"** -> **Yes.** The results clearly show that wave-inspired metaphors are unstable and suffer from overfitting, and that classical regularization on basic linear layers performs better.
- **Claim 2: "Analytical Proof of Layer-Averaging Collapse"** -> **Partially.** While the sandbox results show that a global single-layer router outperforms L3-Linear, the analytical proof is over-simplified, and the empirical results are on a highly simplified sandbox, leaving the claim unproven for complex, hierarchical real-world networks.
- **Claim 3: "Diverse Real-World Evaluations"** -> **No.** Since the DomainNet and LLaMA-7B benchmarks are simulated on synthetic feature distributions rather than executed live, they do not provide valid proof of real-world applicability or scalability.
- **Claim 4: "VRAM and Latency Viability via PEFT co-design"** -> **Partially.** The theoretical co-design is sound, and the SGMV PyTorch benchmark on an A100 GPU shows that parallel kernels can reduce dispatching latency. However, because SGMV requires highly custom CUDA pipelines, its general usability on standard serving frameworks is limited, and the sequential execution latency remains a non-trivial bottleneck on standard CPU/GPU nodes.
