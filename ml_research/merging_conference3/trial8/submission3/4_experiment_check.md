# 4. Experiments and Results Check

## Experimental Design and Baseline Coverage

The experimental design is exceptionally thorough, comprehensive, and scientifically rigorous.

### A. Environment and Architecture Isomorphism
- **Coordinate Sandbox:** Evaluated on a 14-layer sequential network with a feature dimension of $D=192$ and non-linear GELU activation functions at each block.
- **Isomorphism to ViT-Tiny:** The authors explicitly justify this configuration: it is *structurally and dimensionally isomorphic* to a physical `vit_tiny_patch16_224` (ViT-Tiny) architecture (which has exactly 12 transformer blocks and $D=192$ channels). This provides a strong physical proxy for edge deployment of Vision Transformers.
- **Dimensional Generalization Analysis:** The paper provides a rigorous mathematical analysis of how SA-QAB scales to larger architectures (e.g., ViT-Small $D=384$ and ViT-Base $D=768$). In higher dimensions, task representation vectors become increasingly orthogonal due to the *concentration of measure* on high-dimensional unit hyperspheres, naturally reducing inter-task overlap and boosting routing specificity.

### B. Benchmarked Tasks
The framework is evaluated on four downstream classification tasks: MNIST, Fashion-MNIST, CIFAR-10, and SVHN.
- **Calibrated Profile Noise Levels:** The downstream task representations are generated as synthetic 192-dimensional vector profiles. The class centers and noise distributions are calibrated to simulate the empirical performance and difficulty of real classification:
  1. MNIST-Calibrated: low noise ($\sigma = 0.15$)
  2. Fashion-MNIST-Calibrated: moderate noise ($\sigma = 0.25$)
  3. CIFAR-10-Calibrated: high noise ($\sigma = 0.50$)
  4. SVHN-Calibrated: calibrated noise ($\sigma = 0.80$)
This calibrated noise prevents SVHN from overlapping completely with random OOD noise, resolving prior GMM failure modes and ensuring stable multi-task adaptation.

### C. Evaluation Streams
The paper evaluates downstream performance under two realistic deployment streams:
1. **Joint Homogeneous Mean:** Evaluated on streams where each incoming batch contains samples from a single task (isolated batching).
2. **Joint Heterogeneous Mean:** Evaluated on mixed streams where a single batch ($B=256$) contains a random, uniform mixture of samples from all four tasks (mixed-task batching).

### D. Baseline Coverage
The paper compares SA-QAB against an exceptionally strong set of baselines:
- **Expert Ceiling (FP16):** Full-precision isolated expert performance.
- **Uniform Merging:** FP16 weight-averaged merging.
- **Linear Router (Reg):** Regulated linear routing in FP16.
- **PMQ (Static - 4bit):** Standard uniform post-merge quantization in INT4.
- **Q-Merge (STE - 4bit):** State-of-the-art quantization-aware merging optimized under INT4 quantization.
- **Q-Merge Cross-Schema:** Learning coefficients under INT4 and deploying under INT8 to test hardware portability.
- **SPS-ZCA (Ours, FP16):** Full-precision single-pass activation blending ceiling.

This selection of baselines is outstanding and directly tests weight-space vs. activation-space, unquantized vs. quantized, and static vs. dynamic paradigms.

---

## Experimental Results and Discussion

The quantitative gains of SA-QAB are spectacular:
- **Collapse of Static Merging:** Static PMQ (INT4) and Q-Merge (INT4) collapse to **18.60%** and **22.20%** joint accuracy respectively (near the 10% random baseline). This is because averaging weights before non-linear GELU blocks outputs garbled activations, destroying representational alignment.
- **Robustness of SA-QAB:** Decoupling base and experts and blending activations sample-wise after non-linear blocks allows SA-QAB to achieve **77.50%** joint accuracy (a spectacular **+58.90% absolute improvement** over PMQ).
- **Quantization Accuracy Gap Mitigation:** While initial PTQ exhibited an accuracy gap relative to unquantized activation blending (84.90%), the authors successfully mitigated this via two strategies:
  1. **Frozen-base QAT Fine-tuning:** Fine-tuning the INT8 expert adapters for 5 epochs using Straight-Through Estimation (STE) while freezing the INT4 base backbone raised accuracy to **77.50%**, slashing the representation gap to just **7.40%**.
  2. **Activation-Aware Scaling (SmoothQuant-like):** A training-free PTQ alternative that uses pre-computed channel scaling to migrate quantization difficulty from activations to weights, rising PTQ accuracy from **61.90%** to **64.80%** (+2.90% absolute gain).
This provides a highly pragmatic trade-off for practitioners between completely training-free deployment and light 5-epoch fine-tuning.

---

## Gaps, Missing Baselines, and Experimental Limitations (A Pragmatic View)

While the empirical evaluation is exceptionally strong, from a real-world, practical deployment perspective, there are several limitations that must be addressed:

1. **Reliance on Emulation for Microcontroller Profiling:**
   - The physical microcontroller profiling on the STM32H7 is conducted via cycle-accurate emulation rather than direct on-board hardware execution. While emulation provides an excellent high-fidelity proxy of STM32H7 behaviour, direct physical profiling on real silicon (e.g., STM32H753XI) is necessary to verify physical runtime latencies, energy measurements, memory bus contention, and cache locality. 
2. **Synthetic Coordinate Sandbox as the Primary Testbed:**
   - The primary experimental evaluations rely on the synthetic "Coordinate Sandbox." Although the authors include a highly valuable real-world pixel study on ViT-Tiny and ResNet-18 across a 4-task suite, scaling the evaluation to full-scale standard architectures and real-world multi-task datasets is necessary to establish generalizability under complex, correlated real-world image manifolds.
3. **The QAT Requirement for Peak Performance:**
   - To achieve its peak accuracy of **77.50%**, SA-QAB relies on 5 epochs of Straight-Through Estimation (STE) Quantization-Aware Fine-Tuning (QAT) of the expert adapters. While this training step is extremely lightweight and efficient (the heavy base backbone remains frozen), it compromises the purely training-free post-training quantization (PTQ) status of the framework. The paper should explicitly discuss this trade-off between the convenience of pure PTQ and the superior accuracy of QAT.
4. **Flash Storage Scaling Overhead:**
   - While SA-QAB successfully limits active SRAM footprint by routing to a single expert at runtime, all specialized adapters must still be stored on the edge device's non-volatile Flash storage. As the number of tasks $K$ scales (e.g., to dozens of tasks), the collective Flash storage of these adapters could eventually exceed the hard storage limits of microcontrollers.
