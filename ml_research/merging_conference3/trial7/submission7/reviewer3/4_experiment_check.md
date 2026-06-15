# 4. Experiment Check

## Critical Evaluation of the Experimental Setup and Datasets

### 1. Synthetic vs. Real-World Datasets
- The primary evaluation is conducted on a **Hierarchical 14-Layer Sandbox** in PyTorch, which is a highly idealized, synthetic representation space. The tasks (MNIST, Fashion-MNIST, CIFAR-10, SVHN) are simulated as disjoint orthogonal coordinate blocks with added isotropic Gaussian noise.
- While Section 4.7 introduces a **Physical Vision Transformer (ViT-Tiny)** evaluation on real-world datasets and Section 4.5.2 conducts a **GPT-2 NLP** sequence routing benchmark, these real-world setups are evaluated under highly restricted, hyper-sparse conditions (16 samples per task).
- **The Core Flaw:** By training downstream expert adapters on only 16 samples per task, the resulting models have barely learned anything. For example, the Expert Ceiling (oracle) classification accuracy on MNIST is only **39.00%** (a task where a simple linear classifier easily gets >95%), CIFAR-10 is **29.00%**, and Fashion-MNIST is **20.00%**. Evaluating dynamic model merging on untrained, weak expert models makes the downstream classification results (where ELATI gets 21.50%) highly suspect and scientifically weak. It is questionable whether these findings generalize to fully fine-tuned expert models whose parameters have drifted significantly further, testing linear mode connectivity to its breaking point.

### 2. Symmetrical Benchmarking on CPU
- The physical wall-clock latency micro-benchmarks (Tables 2 and 3) are run on a **multi-core CPU**.
- While CPU benchmarking provides a clean, hardware-isolated environment to demonstrate mathematical FLOP reductions, CPU execution behaves completely differently from parallel accelerator hardware (GPUs).
- On GPUs, latency is heavily dominated by memory-bus bandwidth limits, register allocation, CUDA kernel launch overheads, and CUDA stream synchronization. Saving early-layer FLOPs on CPU may translate to negligible physical speedups on GPUs due to these hardware bottlenecks.
- The authors attempt to address this in Section 4.8 with a "Hardware-Level GPU Profiling Benchmark", but they transparently disclose that this is a **simulated and scaled benchmark** using a GPU memory-bus simulation, rather than physical execution on actual active GPU hardware.

## Appropriateness of Baselines
The baselines selected are comprehensive and highly rigorous:
- **Oracle (Expert Ceiling):** Represents the upper bound.
- **Static Merging Baselines (Uniform, DARE, TIES):** Represent the state-of-the-art in parameter averaging.
- **Parametric Routers (Linear Router Unregularized and L2-Regularized):** Evaluate supervised early routing.
- **Deep Routing (PFSR + MBH Penultimate):** Evaluates the state-of-the-art penultimate-layer routing.

This is a very strong baseline sweep. However, the evaluation is heavily biased by the low-data regime (64 samples). Linear Routers are shown to overfit under data scarcity, which favors ELATI's unsupervised centroid matching. Under a standard, full-data fine-tuning scenario where thousands of samples are available to train a parameterized linear router, the linear router would likely achieve near-perfect routing accuracy, narrowing or reversing ELATI's routing advantage.

## Do the Results Support the Claims?

### 1. "Saves Latency via Early-Layer Routing"
- **CPU Benchmarks:** Supported. End-to-end CPU execution decreases from 36.90 ms to 26.43 ms (1.40$\times$ speedup).
- **GPU Realities:** Weakly supported. Since the GPU profiling is simulated and the weight materialization overhead is catastrophic (112 seconds for LLaMA-7B, 2 seconds for 350M Medium), ELATI is **not** a low-latency serving framework when performing full-weight merging. It is only fast if it uses "low-rank PEFT arithmetic," which is just standard multi-tenant LoRA serving (like S-LoRA/Punica) and does not perform physical weight merging.

### 2. "Preserves Accuracy and Resolves Conflicts"
- **Sandbox:** Supported. ELATI achieves 56.89% joint accuracy compared to 48.27% for Uniform Merging.
- **Physical ViT-Tiny:** Weakly supported. While ELATI achieves 21.50% joint accuracy compared to 9.25% for Uniform Merging, the absolute accuracy of 21.50% is extremely poor (barely better than 10% random guessing). It does not prove that the merged model can perform complex multi-task inference robustly, only that it is slightly better than uniform averaging on practically untrained models.

### 3. "Robust to Domain Drift and Scarcity"
- **Supported.** The calibration size sweeps (Figure 7), out-of-distribution noise sweeps (Figure 6), and online adaptation drift tracking trajectories (Figure 9) are robustly modeled and provide convincing evidence of the data efficiency and robustness of unsupervised centroid projection.
