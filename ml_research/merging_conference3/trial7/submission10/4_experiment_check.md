# Experimental Evaluation and Results Analysis

## 1. Scope of the Experimental Suite
The experimental evaluation in this paper is exceptionally thorough, rigorous, and diverse, spanning both simulated sandbox environments and extensive real-world physical hardware profiling:
* **The ICS Sandbox:** Simulates a $L=14$ layer, $D=192$ feature space with $K=4$ experts (MNIST, F-MNIST, CIFAR-10, SVHN). Primary sandbox results are reported in Table 1 (Homogeneous Performance Sweep) and Table 2 (Deployment Stream Audit).
* **Physical PyTorch ViT-Tiny Validation:** Evaluates a real pre-trained `vit_tiny_patch16_224` backbone from `timm` on real images. ZCA achieves exactly **100.0% routing accuracy** (zero task identification errors) on real image test sets. The merged model achieves 76.14% physical Joint Mean accuracy, recovering **100.0% of the physical Expert Ceiling** and outperforming static Uniform weight merging (37.33%) by a wide margin (Table 4). This completely bridges the simulation-to-reality gap.
* **Text Modality Generalizability (GPT-2):** Evaluates a three-expert text sequence classification and autoregressive generation task suite (Legal, Medical, Code domains) using pre-trained GPT-2 adapters. SPS-ZCA achieves 98.50% routing accuracy, 91.83% Joint Mean classification accuracy (matching the expert ceiling), near-perfect perplexity preservation (12.18 Joint Mean vs. 12.15 ceiling, compared to Uniform merging's catastrophic 84.50 perplexity), and outstanding text generation quality (ROUGE-L score of 92.40% vs. Uniform's 31.50%).

## 2. Baseline Comparisons and Fairness
The paper compares against five core baselines, covering both classical parametric routers and non-parametric state-of-the-art:
* **Expert Ceiling:** Establish the upper-bound of completely unmerged, task-isolated models.
* **Uniform Merging:** Static weight averaging.
* **Linear Router (Reg):** Parametric routing regularized with $L_2$ weight decay.
* **QWS-Merge SOTA:** Quantum Wavefunction Superposition Merging.
* **PFSR + MBH SOTA:** Prior non-parametric SOTA that splits batches into homogeneous micro-batches.
The baselines are evaluated under 100% fair and unpenalized conditions, making the comparison highly transparent and scientifically rigorous.

## 3. Comprehensive Ablations
The paper conducts a total of eleven distinct and highly rigorous ablation studies:
1. **Sensitivity to Batch Heterogeneity:** Sweeps $B \in \{16, \dots, 512\}$, demonstrating immunity to heterogeneity collapse (Figure 3a).
2. **Execution Cost and Throughput Scaling Audit:** Sweeps $B \in \{16, \dots, 512\}$, showing SPS-ZCA delivers over 1000 samples/sec at $B=256$ (compared to MBH's sub-270 samples/sec).
3. **Unit-Norm Calibration (UNC) under Scale Imbalances:** Demonstrates perfect Joint Mean accuracy recovery (from 79.22% back to 79.80%) under a $5\times$ scale imbalance.
4. **Intra-Task Dispersion Calibration (IDC):** Fully restores balanced routing (from 95.40% down to 47.00%) under highly asymmetric task manifold spreads.
5. **OOD Rejection Performance and Threshold Sensitivity:** Establishes a highly precise 95.2% TPR at a 4.3% FPR (Table 3), using out-of-sample disjoint calibration/validation partitions to ensure zero data leakage.
6. **Routing Temperature Sensitivity:** Measures soft-to-hard Softmax scaling, identifying $\tau=0.001$ as optimal (Figure 3b).
7. **Early-Layer LoRA Freezing Capacity Study:** Demonstrates that freezing Blocks 1--3 degrades joint accuracy by a functionally negligible **-0.02%**, validating the representational capacity of early-layer routing.
8. **Calibration Split Size Sweep ($|\mathcal{C}_k|$):** Sweeps $|\mathcal{C}_k| \in \{4, \dots, 128\}$, showing ZCA achieves perfect 100.0% routing accuracy with only 16 samples (Table 5).
9. **GMM Mixture Components Sweep ($M$):** Sweeps $M \in \{1, 2, 4\}$, confirming $M=2$ represents the optimal elbow-point and $M=4$ overfits the low-resource calibration split.
10. **High-Density Expert Scaling Sweeps up to $K=128$:** Sweeps expert registry sizes, demonstrating stable 100.0% routing accuracy up to $K=16$ and 96.80% at $K=64$ (Table 7).
11. **Proof-of-Concept Evaluation on CUB-200:** Evaluates fine-grained bird classification, demonstrating taxonomic super-class hierarchical grouping and low-resource Supervised Head Fine-Tuning (SHFT) restores routing accuracy from 74.20% to 98.40% (Table 6).

## 4. Empirical Weaknesses and Validation Gaps

### A. Lack of Complete Physical Baseline Evaluations
A major gap in the physical validation (Section 4.7) is the selective evaluation of baselines. While Table 4 compares the physical end-to-end classification accuracies of the "Expert Ceiling", "Uniform Weight Merging", and "SPS-ZCA (Ours)", it completely omits the other three key baselines:
* **Linear Router (Reg)**
* **QWS-Merge SOTA**
* **PFSR + MBH SOTA**
Although the authors argue that these baselines are fully evaluated in the simulated ICS sandbox, higher representational entanglement in the physical world could introduce routing jitter that affects these baselines differently. Omitting them from the physical validation weakens the empirical claim of outperforming them in practice.

### B. The Serving Gap and Hardware Dependency
The physical profiling on sequential edge CPUs reveals a significant "serving gap" for large batch sizes. Under $B=256, G=4$, uncompiled PyTorch SPS-SG suffers an 11% to 52% wall-clock latency slowdown compared to MBH. This is because PyTorch's dynamic indexing, boolean masking, and list slicing overheads completely override the theoretical memory bandwidth and FLOP savings.
Actual physical wall-clock speedup (1.17$\times$) is only achieved at low batch scales ($B=16$), where sequential scheduler delays and DRAM weight loading dominate.
For larger batch sizes, the speedup is highly dependent on custom compiler integrations (ONNX Runtime C++ CustomOp). Reframing the systems latency claims to make this clear is essential.
