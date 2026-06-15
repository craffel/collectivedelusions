# Experimental Setup and Results Check: Scale-Aligned Quantized Activation Blending (SA-QAB)

## 1. Evaluation Setup and Datasets
The primary evaluation of SA-QAB is conducted in a custom-built, cycle-accurate simulation environment named the **Coordinate Sandbox**.
- The sandbox simulates a 14-layer sequential network with a feature dimension of $D=192$ and rank-8 LoRA adapters.
- It does **not** evaluate on raw pixels. Instead, it uses synthetic 192-dimensional vector coordinate profiles calibrated to simulate the noise and complexity profiles of four downstream tasks: MNIST ($\sigma=0.15$), Fashion-MNIST ($\sigma=0.25$), CIFAR-10 ($\sigma=0.50$), and SVHN ($\sigma=0.80$).
- To address this synthetic limitation, the authors include a brief "Real-World 4-Task Pixel Feasibility Study" evaluating a physical `vit_tiny_patch16_224` (ViT-Tiny) and a ResNet-18 backbone on real image pixels.

## 2. Baselines and Comparisons
The paper compares SA-QAB against an appropriate and comprehensive set of baselines:
- **Expert Ceiling (FP16):** The upper bound performance of unquantized task-specific experts.
- **Uniform Merging (FP16):** Simple parameter-space weight averaging.
- **Linear Router (Reg) (FP16):** A full-precision linear regression-based routing baseline.
- **PMQ (Static - 4bit):** Standard Post-Merge Quantization (weight averaging in FP16 followed by INT4 quantization).
- **Q-Merge (STE - 4bit):** A state-of-the-art parameter-space merging method that optimizes merging coefficients under quantization constraints using Straight-Through Estimators (STEs).
- **SPS-ZCA (FP16):** The unquantized activation-blending baseline on which SA-QAB is built.

The comparison is extensive, and the catastrophic collapse of PMQ and Q-Merge under non-linear activations (GELU) is clearly demonstrated, proving that parameter averaging fails under low-bit quantization.

## 3. Critiques and Discrepancies

### A. The Artificial Simplicity of Disjoint Coordinate Spaces
In the Coordinate Sandbox, the synthetic task representations are generated in disjoint (fully orthogonal) coordinate subspaces (e.g., Task 0 uses channels $[0:48]$, Task 1 uses $[48:96]$, and so on).
- This disjoint formulation makes the early-stage Q-ZCA routing task incredibly easy, as there is zero representational cross-talk between tasks in early layers.
- Real-world deep neural network representations do **not** partition themselves into neat, mutually orthogonal coordinate blocks. Features are highly correlated and overlapping.
- While the authors conduct a "Task Overlap Sweep" ($\Omega \in [0.00, 1.00]$), they show that as overlap increases, routing accuracy drops and the overall expert ceiling degrades. At complete overlap ($\Omega=1.00$), the joint accuracy drops significantly (from ~77% to ~61%). This indicates that the main results presented in Table 3 (which assume disjoint subspaces) are highly optimistic, and real-world performance under overlapping representations will be substantially lower.

### B. The Host CPU Framework Latency Gap (139.9% Overhead)
In Section 4.3, the authors discuss physical profiling latencies in PyTorch on the host CPU.
- While the emulated bare-metal CMSIS-NN latency overhead of SA-QAB over the Static model is a negligible **3.7%** (0.03 ms), the physical PyTorch CPU overhead of SA-QAB over Static is a massive **139.9%**!
- The authors attribute this to high-level Python framework overheads (dynamic dispatch and kernel launch).
- This exposes a major practical limitation: SA-QAB is **not** a plug-and-play solution for standard edge runtimes (like standard ONNX Runtime Mobile, PyTorch Mobile, or TensorFlow Lite). To achieve any of the claimed latency benefits, developers **must** write custom, bare-metal assembly-level mixed-precision SIMD kernels. In standard high-level frameworks, the overhead of dynamic routing and parallel mixed-precision GEMM branches completely dominates, making SA-QAB slower than even unquantized ensembling or static models.

### C. GMM Fallback Policy Complexity vs. Utility
The GMM fallback policy is evaluated over two choices: Standard Fallback (completely bypassing experts, setting $\alpha=0$) and Soft Fallback (falling back to uniform ensembling, $\alpha_k = 1/K$).
- The paper argues that Soft Fallback boosts overall classification accuracy.
- However, as shown in Table 3, the overall accuracy improvement is marginal:
  - No Rejection: **77.50%**
  - Optimal threshold ($\eta=-255.0$): **77.40%** (actually a 0.10% decrease!)
  - Threshold ($\eta=-245.0$): **78.00%** (a mere 0.50% improvement).
- This tiny 0.50% gain does not justify the immense software and hardware complexity of running a secondary GMM density estimator, setting thresholds, and branching execution, especially on memory-starved edge platforms.
