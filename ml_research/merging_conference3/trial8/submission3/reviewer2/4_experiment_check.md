# Experimental Evaluation and Results Scrutiny

## Critical Review of the Experimental Design and Benchmarks

### 1. High Reliance on Synthetic "Coordinate Sandbox"
The primary quantitative evaluation of the proposed framework (Table 2) is conducted within the **Coordinate Sandbox** rather than on real image pixels. 
- While the authors state that the sandbox noise profiles are calibrated to represent MNIST, Fashion-MNIST, CIFAR-10, and SVHN, the inputs are ultimately **synthetic 192-dimensional vector profiles** rather than actual images.
- In this synthetic environment, task representations are deliberately generated in **disjoint/orthogonal coordinate subspaces**. This design choice makes the dynamic routing and alignment task exceptionally easy. It is highly questionable whether the reported performance (+58.90% joint accuracy improvement over post-merge quantization) would transfer to real-world datasets where task-specific representations are heavily entangled and share the same feature spaces.

### 2. Omission of Real-Pixel Baseline Comparisons
In Section 4.3 (Expanded Real-World 4-Task Pixel Feasibility Study), the authors present a brief "feasibility study" on real-pixel image datasets using a pre-trained ViT-Tiny backbone and a ResNet-18 backbone. However:
- This feasibility study is presented entirely in text format and **completely lacks comparative baseline results** (e.g., there is no Table showing what PMQ, Uniform Merging, or Q-Merge achieve on the real pixel datasets).
- The authors only report that SA-QAB achieves "84.80%" joint accuracy on real pixels. Without showing the baseline performance under the exact same real-pixel setup, it is impossible to verify if static merging methods collapse as severely, or if SA-QAB's benefits are as pronounced, on real-world datasets.
- For the ResNet-18 evaluation, the authors only report the "routing accuracy" (87.00% average routing specificity), but **completely omit the final classification accuracy** of the quantized pipeline. This is a selective reporting issue that raises concerns about the actual end-to-end performance of ResNet-18.

### 3. Missing SOTA Model Merging Baselines
The authors compare SA-QAB against simple "Uniform Merging" (weight averaging) and "Q-Merge" (optimization under quantization constraints).
- They completely omit comparisons against widely established, state-of-the-art model merging baselines such as **TIES-Merging**, **DARES**, **ZipIt**, or **Task Arithmetic** (all of which are cited in Section 2). 
- While the authors claim weight-space merging collapses under non-linearities, they must empirically demonstrate this by benchmarking against these advanced methods rather than relying solely on simple uniform averaging.

### 4. Poor Performance on Noisier Datasets (SVHN)
In Table 2, under the synthetic sandbox, the performance of SA-QAB on the SVHN-calibrated profile is extremely poor:
- The Expert Ceiling (FP16) for SVHN is **65.60%**.
- SA-QAB (Ours) only achieves **39.20% joint accuracy** on SVHN.
- This represents a massive **26.40% absolute accuracy loss** compared to the unquantized ceiling, even after the 5-epoch Quantization-Aware Fine-Tuning (QAT).
- Under direct post-training quantization (PTQ, 50.00% joint accuracy), the performance on SVHN was likely near-random. This extreme performance drop on noisier profiles indicates that SA-QAB is highly sensitive to quantization noise and representation drift, severely limiting its utility on realistic, noisy edge datasets.

### 5. Massive Host CPU Latency Overhead
In Section 4.2, the authors disclose that on the host CPU in PyTorch, SA-QAB incurs a massive **139.9% latency overhead** compared to the Static 4-bit model (1.136 ms vs. 0.474 ms).
- While they claim that this overhead will disappear on bare-metal CMSIS-NN execution due to the elimination of Python's dynamic dispatch and kernel launch overheads, this claim is speculative since they did not perform physical on-board profiling.
- In many edge-deployment scenarios (e.g., Raspberry Pi, Jetson Nano, or standard mobile/edge platforms running PyTorch Mobile, ONNX Runtime, or TensorFlow Lite with Python bindings), the high-level framework overhead is a realistic constraint. In these environments, SA-QAB would be **more than 2x slower** than static post-merge quantization, completely undermining its efficiency claims.
