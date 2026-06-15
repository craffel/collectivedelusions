# Experimental Evaluation and Claims Check

This evaluation provides a critical analysis of the experimental setup, datasets, baselines, and empirical claims presented in the paper, evaluating whether the results truly support the paper's core assertions.

## Analysis of Experimental Setup and Datasets
The main evaluation in Section 4 is conducted inside a custom simulation environment: the **Isolating Coordinate Sandbox (ICS)**.
- **Model Backbone:** The model is a 12-layer Vision Transformer (ViT-Tiny) with 5.7M parameters.
- **Task Registry:** Consists of $K = 4$ downstream task experts targeting MNIST, Fashion-MNIST, CIFAR-10, and SVHN.
- **Ablation splits:** A very tiny 64-sample support split per task ($|D_{\text{cal}}| = 64$) is used for both routing (ZCA) and quantization calibration (QASC).

### Critical Weaknesses of the Setup:
1. **Toy-Scale Visual Domains:** The visual datasets used (MNIST, Fashion-MNIST, CIFAR-10, SVHN) are standard toy-scale benchmarks in machine learning. MNIST and Fashion-MNIST are grayscale $28 \times 28$ images, and CIFAR-10/SVHN are $32 \times 32$ images. Modern on-device foundation models serve much more complex, high-resolution visual tasks or high-dimensional language representations. While the authors attempt to scale their findings via LLM-scale simulations in Section 4.8, the primary classification results (Table 1) remain anchored in these simple toy-scale classification tasks.
2. **Synthetic Accuracy Generation:** The classification accuracies reported in Table 1 are simulated based on task representation projections under simulated quantization discretization noise, rather than physical forward passes of real datasets on actual neural network weights. While this is acceptable for a "hardware-calibrated analytical simulation study," it reduces the empirical weight of the accuracy claims, as the actual classifications are never executed end-to-end on real image tensors inside the main sweep.
3. **Extremely Deficient SVHN Baseline:** The unquantized "Expert Ceiling" for SVHN is exceptionally low at **31.20%**. In modern deep learning, even a lightweight Vision Transformer (ViT-Tiny) can easily achieve $>90\%$ in-distribution accuracy on SVHN. The extremely low 31.20% ceiling suggests that the SVHN adapter is heavily undertrained or completely capacity-bottlenecked. While the authors argue that this low performance serves as a "high-stress test case" to verify if quantization compounding collapses performance, it also highlights the lack of realism in the experimental baseline.

## Evaluation of Comparative Baselines
The comparative baselines are highly comprehensive and cover standard model-merging and routing strategies:
- **Parameter-space merging:** Uniform Merging (FP32), Quantized Uniform Merging (INT4).
- **Parametric routers:** Linear Router (Reg).
- **State-of-the-art edge pipelines:** PFSR + MBH SOTA.
- **Activation-space blending:** SPS-ZCA (FP32).
- **Post-training quantization:** Q-SPS (INT4, RTN / PTQ Baseline).

The comparative results clearly isolate the failure modes of other methods (such as the parameter-space collapse of quantized uniform merging and the linear scaling of MBH SOTA latency under mixed streams). However, because these baselines are also evaluated inside the synthetic coordinate sandbox, their performance is subject to the same simulation abstractions.

## Do the Results Support the Claims?
Within the boundaries of the simulated environment, the results logically support the claims:
- **Immunization against collapse:** The simulated accuracy of CG-Q-SPS is indeed identical to SPS-ZCA (recovering 99.5% of the ceiling), showing that activation-space blending protects against the parameter collapse of static merging.
- **Lossless conditional gating:** CG-Q-SPS achieves the exact same simulated accuracy as standard Q-SPS with QASC calibration (79.40% joint mean), supporting the claim that thresholding ($\theta = 0.01$) is mathematically lossless.
- **Pragmatic calibration:** QASC recovers the degradation of standard Round-to-Nearest (RTN) quantization (+0.96% absolute accuracy).

### The Physical Validation Contradiction:
However, **the core systems claim of a 3.97$\times$ physical speedup is NOT supported by the physical CPU micro-benchmarks in Section 4.9**:
- The authors show that when executing the low-rank projection on actual CPU hardware inside PyTorch (even with compilation), **uncompiled low-precision and compiled low-precision models are significantly slower than uncompiled FP32**:
  - Uncompiled single projection: FP32 executes in **0.0387 ms** vs. **0.1895 ms** for BF16 ($0.25\times$ slowdown).
  - Compiled single projection: FP32 executes in **0.0868 ms** vs. **0.2547 ms** for BF16 ($0.36\times$ slowdown).
- This means that in physical reality, the proposed low-precision PEFT execution actually leads to a **performance penalty** due to high-level framework dispatch overhead, casting stalls, and register unpacking delays.
- The projected 3.97$\times$ speedup is a **theoretical ideal** based on the assumption that custom fused C++ operators are compiled inside a runtime like ExecuTorch. Since the paper only presents a "systems compilation roadmap" rather than an actual physical compiled implementation, the physical latency advantages of CG-Q-SPS remain an unproven projection.

## Summary
The empirical evaluation is highly rigorous *as a simulation study*, and the authors are commendable for their intellectual honesty in including Section 4.9, which highlights the uncompiled framework slowdowns. However, the reliance on toy-scale datasets, synthetic accuracy projections, a deficient SVHN baseline, and the lack of physical, compiled on-device speedup validation limits the overall impact of the experimental findings.
