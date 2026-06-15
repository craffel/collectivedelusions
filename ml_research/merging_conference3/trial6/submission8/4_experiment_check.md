# Experimental Check and Validation: Hybrid-Router

## 1. Quality of Experimental Setup and Baseline Choices
The paper's experimental setup is exceptionally thorough, comprehensive, and well-designed:
- **Diverse Benchmark Tasks:** MNIST, FashionMNIST, CIFAR-10, and SVHN are classic, high-conflict benchmarks in model merging, providing a challenging multi-task, multi-domain testbed.
- **Strong Baselines:** The authors compare against:
  1. *Uniform Merge (Task Arithmetic)*: Classic lower bound.
  2. *AdaMerging (SOTA Static)*: Re-implemented and executed directly on the sandbox representations to provide a rigorous, non-trivial static baseline.
  3. *QWS-Merge (SOTA Dynamic)*: SOTA dynamic model merging using quantum-inspired cosine boundaries.
  4. *Classical Linear Router*: Standard unregularized Softmax routing.
- **Statistical Rigor:** All accuracy results are reported as the **mean and standard deviation across 3 independent calibration seeds** (and random streaming shuffles), showing strong reproducibility and statistical confidence.
- **Ablation Sweeps and Stress Tests:** The paper includes:
  - An exhaustive sweep of partition depths ($k \in \{0, 1, 2, 4, 12, 14\}$).
  - An ablation of the calibration dataset size ($|\mathcal{D}_{\text{cal}}| \in \{64, 256, 512, 1024\}$).
  - A sensitivity analysis of the early-layer representational penalty ($\eta \in \{0.01, 0.04, 0.08, 0.12, 0.16\}$).
  - A systems-level latency breakdown on CPU.
  - A comparative paradigm comparison with dynamic PEFT runtimes (Punica, S-LoRA).

## 2. Strengths of the Empirical Validation
1. **Dynamic Batch Filtering (DBF) Efficacy:** The heterogeneous streaming benchmark (Table 3) is a brilliant stress-test. Standard batch-averaged routing suffers heavily from *Batch Style Blur* at larger batch sizes (e.g., $B=256$). DBF achieves massive, spectacular improvements (e.g., **+16.55%** absolute gain for BSigmoid, **+30.23%** absolute gain for Linear Router), demonstrating its real-world utility.
2. **Double-Checking the Sandbox (Physical Validation):** To address concerns of synthetic sandbox biases, the authors train actual convolutional experts on real pixels, build a physical routing pipeline, and measure real wall-clock latency. They show a beautifully consistent monotonically increasing Pareto curve ($k=4$ reaching **76.67 $\pm$ 0.94%** joint accuracy).
3. **Intellectual Explanation of Discrepancies:** The paper provides a highly rigorous, 2-factor physical explanation for why the "Overfitting-Optimizer Paradox" (where $k=12$ beats $k=14$) occurs in the high-capacity ViT sandbox but not in the shallow SimpleCNN. This shows deep academic maturity and scientific precision.

## 3. Flaws and Limitations in the Experiments
While the empirical validation is outstanding, there are a few minor limitations:
1. **Small-Scale Physical Validation:** The physical model used is a very small, shallow Convolutional Neural Network (~25k parameters, 3 conv layers, 1 fc layer). While this confirms physical end-to-end differentiability, full-scale validation on a physical Vision Transformer (e.g., `vit_tiny_patch16_224` or `vit_base`) on real pixels is left as future work.
2. **CPU-only Wall-Clock Latency:** Wall-clock parameter blending times are measured on an AMD EPYC CPU. While this aligns with edge-device deployment modeling, real-world servers frequently run model merging on GPU runtimes. GPU-based wall-clock ensembling latencies (which are subject to asynchronous CUDA kernel queues and synchronization overheads) are not profiled.
3. **Mutual Exclusivity Mismatch:** Evaluating the Softmax-free `BSigmoid-Router` on mutually exclusive single-label classification datasets (where only one expert should be activated) is a minor structural mismatch. While this is explicitly acknowledged as a deliberate "stress-test," evaluating it on a multi-label, multi-expert composite task would have showcased its advantages better.

## 4. Empirical Rating
**Rating: Excellent (or Good-to-Excellent)**
- The empirical evaluation is incredibly broad, detailed, and statistically rigorous.
- The inclusion of physical validation on real image pixels and real weights successfully grounds the theoretical and sandbox claims.
- The level of detailed ablation, latency profiling, and paradigm comparison is outstanding.
