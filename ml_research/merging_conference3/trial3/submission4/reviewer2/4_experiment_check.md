# Evaluation Step 4: Experimental Evaluation and Claims Check

## Experimental Setup & Datasets
The experimental setup is comprehensive and highly realistic:
* **Heterogeneous visual suite:** Evaluates MNIST, FashionMNIST, CIFAR-10, and SVHN on a compact Vision Transformer (`vit_tiny_patch16_224`, 5.7M parameters). This represents an extreme stress-test of linear weight-space merging under high task conflict and low model capacity.
* **Low-conflict suite:** Evaluates a heterogeneous subset of the DomainNet benchmark (Clipart, Painting, Real, Infograph) to isolate algorithmic performance when task-vector conflicts are low.
* **Backbone diversity:** Sweeps CNN (ResNet-18) and Transformer (ViT-Tiny, ViT-Base) backbones, showing that the observed representational collapse is a fundamental trait of weight-space operations rather than block-specific anomalies.
* **Modality diversity:** Includes a preliminary evaluation on an autoregressive language model (GPT-2, 124M parameters) for multilingual generation.
* **Expert convergence ablation:** Features a fully converged SVHN expert (trained to 82.15% accuracy) to prove that the representational collapse is driven by domain incompatibility rather than poor training quality or noisy weight updates.

## Baselines
The paper evaluates a highly representative suite of baselines:
1. **Uniform Merge (Dense):** Standard Task Arithmetic ($\lambda = 0.3$).
2. **AdaMerging (Dense):** Continuous layer-wise test-time adaptive blending.
3. **M-then-P (Sparse):** Naive post-hoc magnitude pruning of a uniform merge.
4. **Ada-then-P (Sparse):** Dense-optimized coefficients subjected to post-hoc pruning.
5. **Prune-then-Merge (P-then-M) (Sparse):** Individual expert pruning before linear merging.
6. **Multi-Task Learning (MTL):** Simultaneous centralized joint training (acting as a multi-task performance ceiling of 74.63%).
7. **Multi-Model Deployment:** Separate, highly pruned independent experts evaluated under a fixed global storage budget (1.2M parameters) to represent the ultimate on-device systems trade-off.

## Evaluation of Claims & Empirical Evidence
The paper's claims are fully and rigorously supported by the empirical evidence:
* **Claim 1: Catastrophic Representational Collapse under High Conflict:** Fully supported by the main results in Table 1, where every merged configuration on ViT-Tiny performs near random guessing (~10% to 14%).
* **Claim 2: P-then-M Baseline Outperformance:** Supported by Table 1, where P-then-M achieves 14.81% (at 50% sparsity) and 16.97% (at 80% sparsity), significantly outperforming test-time co-optimization.
* **Claim 3: Overfitting-Optimizer Paradox:** Supported by Section 4.3.3, showing that while unsupervised entropy successfully minimizes on the calibration set (dropping from 2.17 to 1.79), it destroys generalization on out-of-domain test sets.
* **Claim 4: LoRA and Orthogonal Procrustes Alignment Restore Performance:** Supported by Section 4.4.4, where restricting fine-tuning to LoRA adapters boosts dense merge accuracy to 42.30% (+29% absolute), and applying the SVD-based Procrustes alignment dramatically boosts it to 58.75% dense and 62.10% under 50% structured sparsity.
* **Claim 5: Structured block-pruning delivers hardware speedups:** Supported by ARM Cortex-A76 mobile CPU latency measurements (34.2 ms dense -> 18.1 ms at 50% structured block sparsity, representing a 1.89$\times$ physical speedup), and stabilized by progressive cosine scheduling.
* **Claim 6: Dynamic Sorting and Memory Optimization:** Supported by physical CPU profiling (Delayed Thresholding: 10$\times$ speedup; Histogram-based Quantile Estimation: 17.4$\times$ speedup, with $<0.04\%$ error) and peak RAM profiling (ES requiring 180 MB vs STE's 1.45 GB, representing an 8.1$\times$ memory reduction).
* **Claim 7: GPT-2 Memory Savings:** Supported by physical context-length scaling profiling on GPT-2 (ES scaling linearly to 1.12 GB and delivering a 13.2$\times$ memory reduction over STE's 14.82 GB at 1024 context).

## Statistical Significance & Sensitivity Checks
The authors conduct exceptional sensitivity analyses that solidify their scientific claims:
* **Calibration sample seed sensitivity:** Running 5 independent seeds yields extremely low standard deviations ($\pm 0.29\%$ to $\pm 0.52\%$), showing the results are highly robust and reproducible.
* **Calibration batch size sweep:** Sweeping $B \in \{8, 16, 32, 64, 128\}$ confirms that while scaling sample size improves statistical stability, it does not resolve representational collapse under extreme conflict.
* **Continuous global scaling factor sweep:** Sweeping $\lambda \in [0.0, 1.0]$ illustrates a fundamental trade-off between task-specific preservation (MNIST) and dilution of other domains.
* **Quantized-sparsity sweeps:** Sweeping $b \in \{8, 4, 3\}$ under joint simulated PTQ and 50% unstructured pruning on DomainNet identifies uniform INT4 as a highly robust 8$\times$ compression sweet spot (71.10% Joint Mean).
