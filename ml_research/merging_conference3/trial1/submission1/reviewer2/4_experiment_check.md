# 4. Experimental Rigor and Evaluation Check

## Critique of the Experimental Setup and Datasets

### 1. The "Toy Dataset" Constraint
The paper evaluates the proposed framework exclusively on a dual-task digit classification scenario: **MNIST** and **SVHN**:
- **MNIST** is an extremely simple, grayscale, $28 \times 28$ handwritten digits dataset. It has been virtually solved for over a decade, with simple linear classifiers or tiny convolutional networks easily achieving $>99\%$ accuracy.
- **SVHN** is a slightly more challenging street-view house numbers dataset ($32 \times 32$ RGB images), but remains a relatively low-complexity digit classification task.
- **Representative Gap:** In the model merging and multi-task learning literature, standard systems are evaluated on much more complex, high-dimensional benchmarks. For computer vision, this typically involves 8-task classification suites (including ImageNet, CIFAR-100, EuroSAT, RESISC45, etc.). For NLP, this involves multi-task benchmarks like GLUE or MMLU. Evaluating strictly on *two simple digit tasks* is highly non-representative. It raises serious concerns about whether the proposed method generalizes to more realistic multi-task scenarios containing complex, highly structured semantic representations.

### 2. Massive Model Overkill
The authors fine-tune and merge task vectors on a pre-trained **ViT-B-32** vision encoder (which contains approximately 86 million parameters and is pre-trained on massive datasets like ImageNet or CLIP):
- **Engineering Disconnect:** Deploying an 86M parameter transformer to solve a simple 10-class digit classification task (MNIST/SVHN) is massive engineering overkill. On resource-constrained edge devices where quantization is actually required, a practitioner would simply use a tiny MobileNet or ConvNet (e.g., $<1$M parameters) which would run orders of magnitude faster and fit comfortably in memory without requiring complex hybrid quantization schemes.
- **Artificial Ease:** Because the pre-trained ViT-B-32 representations are incredibly rich and general, fine-tuning them on MNIST and SVHN requires only microscopic weight updates. This explains why merging them is so clean, and why even direct INT8 quantization has zero loss ($94.93\%$). Evaluating on such an overpowered model on low-complexity tasks trivializes the merging and quantization difficulty, making the results look artificially polished.

## Evaluation of Baselines
The baselines used are limited:
- **Missing Merging Baselines:** The paper only uses standard Task Arithmetic (with uniform/optimized coefficients) as the merging foundation. It fails to compare QP-Merge against more advanced merging methods (like Ties-Merging, DARE, or SyMerge) *followed* by quantization. It is highly possible that advanced merging methods, which prune low-magnitude elements or resolve sign conflicts, naturally produce weight distributions that are much more resilient to quantization without requiring outlier decoupling.
- **The "SmoothQuant" Mockery:** The "SmoothQuant Baseline" is defined as optimizing diagonal scaling parameters on the unquantized merged model to minimize MSE, but *without* separating outliers or optimizing task weights. This is a custom, post-hoc optimization baseline, not the actual SmoothQuant algorithm (which applies activation scaling $D_l^{-1}$ to maintain mathematical equivalence). This makes it a weak "strawman" baseline rather than a true representative of state-of-the-art PTQ.

## Do the Results Actually Support the Claims?

### 1. Is Outlier-Residual Decoupling (ORD) Actually Needed?
The authors claim that ORD is a vital, core pillar of QP-Merge. However, their own ablation study (Table 3) and sensitivity sweep (Table 4) show that **ORD's empirical contribution is marginal and likely statistically insignificant**:
- In Table 3 (seed 2026), the "No ORD" ablation (which keeps all weights in homogeneous INT4 and just runs the QE-Calib optimization) achieves **$94.49\%$** average accuracy.
- The Full QP-Merge model (which includes ORD and routing outliers to a separate sparse FP16 path) achieves **$94.52\%$**.
- The delta is a microscopic **$0.03\%$** average accuracy.
- Since the standard deviation of QP-Merge across 3 seeds is $\pm 0.13\%$, a difference of $0.03\%$ is well within the noise level. This implies that ORD is empirically redundant, and that the performance recovery is almost $100\%$ driven by the QE-Calib scale optimization. The data does *not* support the claim that dense-sparse decoupling is necessary.

### 2. Is QP-Merge Actually "Hardware-Friendly" and Efficient?
The authors promote QP-Merge as a highly efficient, hardware-friendly solution. However, their physical GPU latency profiling reveals a major disconnect:
- The standard FP16 linear projection layer executes in **$10.48\ \mu$s**.
- The hybrid QP-Merge layer (dense INT4 + sparse FP16) executes in **$60.92\ \mu$s**.
- This represents an absolute **$5.8\times$ slowdown** compared to FP16 execution in PyTorch!
- The authors explain this as a high-level API and CUDA kernel launch latency overhead (~$50\ \mu$s) in PyTorch. While this is technically true, it means that for the actual model evaluated (ViT-B-32), QP-Merge is physically much slower than unquantized execution.
- Their claim of speedup relies entirely on an analytical scaling model of a LLaMA-7B size layer, assuming customized fused kernels on edge hardware. Therefore, the claim of "near-zero latency overhead" or "instantly deployable" is highly speculative and not supported by their actual empirical physical profiling on the evaluated model.
