# 4. Experimental Evaluation Check

## Evaluation of Experimental Setup & Datasets
The experimental validation is highly structured, multi-tiered, and very thorough:
1. **The Coordinate Sandbox:** The primary evaluation uses a 14-layer, 192-dimensional vector simulation suite calibrated to match the noise and complexity profiles of four standard image datasets (MNIST, Fashion-MNIST, CIFAR-10, SVHN).
   - *Skepticism:* A synthetic coordinate-space sandbox is fundamentally a toy setting that does not capture the raw spatial and representation complexities of real image pixels.
   - *Mitigation:* The authors do an outstanding job of justifying and mitigating this. They show that $D=192$ and a 12-block layout are dimensionally isomorphic to a physical `vit_tiny_patch16_224` (ViT-Tiny). They also conduct a systematic **task subspace overlap sweep** ($\Omega \in [0.0, 1.0]$) to stress-test feature overlap, which is a common real-world challenge.
2. **Real-Pixel Feasibility Study:** Crucially, the authors validate their findings on a real-world image manifold using a pre-trained physical ViT-Tiny model. This is an excellent addition that bridges the gap between the synthetic sandbox and practical applications.
3. **Convolutional Extension:** They also extend evaluation to a **ResNet-18** convolutional backbone, proving that the early-stage Q-ZCA routing generalizes across distinct network topologies (Vision Transformers vs. CNNs).

## Baselines
The paper includes a robust set of baselines:
- **Expert Ceiling (FP16)** and **SPS-ZCA (FP16)** represent the unquantized performance bounds.
- **Uniform Merging** and **PMQ (Static 4-bit)** represent standard static parameter-space merging.
- **Q-Merge (STE 4-bit)** represents state-of-the-art optimization-based weight-merging under quantization.
- **Linear Router (Reg)** represents a dynamic routing baseline.

The results clearly show that static parameter-space merging collapses catastrophically (under 23% accuracy) due to weight scale corruption and representation misalignment across non-linear GELU layers. In contrast, SA-QAB achieves **77.50%** accuracy (with QAT) or **70.10%** (with training-free SmoothQuant scaling), proving the necessity of activation-blending over weight-blending when dealing with non-linear networks.

## Systems & Microcontroller Profiling (Practitioner's Delight)
The hardware-focused evaluation is a major highlight:
- **Cycle-Accurate Emulation on STM32H753XI:** The authors report concrete physical metrics: Flash storage (KB), active SRAM footprint (KB), MAC operations, estimated latency (ms), and energy per inference (mJ). Showing that SA-QAB fits within the 1MB SRAM of the STM32H7 (requiring 360.8 KB) while FP16 Ensembling exceeds it (1224.8 KB) is a very strong and practical proof of utility.
- **Physical CPU PyTorch Profiling vs. Bare-Metal Emulation:** The analysis of the latency overhead (3.7% on bare-metal CMSIS-NN vs. 139.9% in PyTorch) is an outstanding systems insight. It correctly identifies that high-level Python framework overheads (dynamic dispatch, kernel launch latency) mask the true efficiency of $O(1)$ expert execution, which can only be fully realized on bare-metal hardware.

## Support for Central Claims
The results strongly support the central claims of the paper:
- **DHQ (INT4 base / INT8 adapter)** successfully reduces storage and SRAM overhead while maintaining representational safety.
- **Q-ZCA** achieves highly accurate routing (e.g., 90.00% joint routing accuracy on ViT-Tiny pixels).
- **QSR** and **QAT/Pre-scaling** effectively mitigate low-bit scale drift and quantization noise.
- **OOD GMM Gate** provides a robust fallback mechanism with high TPR and low FRR.
- SA-QAB successfully avoids the **batch-size heterogeneity collapse** that plagues static merging methods.
