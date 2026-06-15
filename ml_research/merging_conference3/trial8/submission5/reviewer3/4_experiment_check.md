# Critical Evaluation of Experimental Methodology: PEAR

## 1. Critique of the PyTorch "Representation Sandbox" Simulation
The primary empirical results of the paper (Tables 1, 2, and 3) are evaluated inside a synthetic 12-layer representation sandbox in PyTorch, rather than using standard real-world Vision Transformers and actual image datasets. 
While the authors are transparent about this design choice and justify it as a means to isolate representational overlap, a theory-minded reviewer must highlight several limitations of this setup:
* **Over-Simplification of Image Manifolds:** In the sandbox, queries are modeled as simple 1D Gaussian vectors centered around randomly generated class prototypes, and tasks are modeled as fixed orthogonal-like subspaces of size 96 with 64-dimensional overlaps. In reality, deep representation manifolds of actual visual datasets are highly non-linear, multi-modal, and entangled across dimensions. Success on a synthetic Gaussian mixture setup does not guarantee that the geometric assumptions (e.g., scale-invariant unit hyperspheres) hold on complex, real-world data manifolds.
* **Artificial SVHN "Stress-Test":** The SVHN task in the sandbox is configured with an expert classification ceiling of $19.68\%$ by setting its noise scale to $1.20$. In real-world machine learning, SVHN is a clean, structured dataset where standard CNNs or ViTs easily achieve $>95\%$ accuracy. Turning SVHN into an essentially random noise task ($19.68\%$ is barely above the $10\%$ random baseline for 10 classes) is a highly artificial configuration. While it evaluates the robustness of the Intra-Task Dispersion Calibration (IDC) in preventing noise bleed, this hand-crafted stress-test may artificially exaggerate PEAR's ensembling benefits compared to standard multi-task scenarios where all experts are highly performant.

---

## 2. Statistical Significance and Standard Deviation Overlap
The authors report classification accuracies as the mean $\pm$ standard deviation across 5 independent random seeds.
In Table 1 (Homogeneous Batch Deployment, $B=256$), the results under the overlapping subspace layout show:
* **SABLE SOTA:** $55.30 \pm 2.26\%$ (Interval: $53.04\%$ -- $57.56\%$)
* **PEAR (Ours):** $59.34 \pm 1.99\%$ (Interval: $57.35\%$ -- $61.33\%$)
The standard deviations of SABLE and PEAR overlap slightly. While PEAR demonstrates a higher mean accuracy ($+4.04\%$ absolute gain), the slight overlap in statistical intervals suggests that under certain representational seeds, the margin of improvement may be narrow, which weakens the claim of absolute dominance.

---

## 3. High Sensitivity to Hyperparameters and Overfitting Risk
The ablation and sensitivity analyses reveal that PEAR is highly sensitive to its hyperparameter selections:
* **Temperature Sensitivity (Table 5):** Under overlapping subspaces (Seed 10), sweeping the temperature $\tau$ from $0.10$ to a hard-routing regime ($0.0001$) collapses the Joint Mean accuracy from $59.00\%$ to $51.40\%$ (a loss of $-7.60\%$ absolute). Hard routing forces a single expert selection, meaning any routing error at Layer 0 scrambled representations across all 12 blocks.
* **OOD Threshold Sensitivity (Table 6):** Sweeping the global OOD threshold $\gamma_{\text{OOD}}$ from $0.05$ to $0.15$ collapses the Joint Mean accuracy from $56.90\%$ to $48.60\%$ (a loss of $-8.30\%$ absolute), due to over-rejection on noisier manifolds (like SVHN).
Given this extreme sensitivity, and because the non-parametric framework operates on a data-scarce calibration split ($B_{\text{cal}} = 64$ samples per task), selecting these parameters solely by maximizing calibration accuracy presents a severe risk of overfitting. If the calibration samples do not perfectly represent the test-time distribution, the selected $\tau$ and $\gamma_{\text{OOD}}$ can lead to catastrophic performance degradation.

---

## 4. The Representational Capacity Gap in End-to-End Adapter Serving
In Table 9, the authors report real-world end-to-end multi-task LoRA classification accuracies on actual images using a pre-trained ImageNet backbone:
* **Standard Setup:** PEAR (Ours) achieves $55.08\%$, while the Expert Ceiling is $66.80\%$.
* **ELFT Setup:** PEAR (Ours) + ELFT achieves $53.52\%$, while the Expert Ceiling is $62.89\%$.
While PEAR significantly outperforms SABLE SOTA ($39.84\%$) and Static Uniform Merging ($34.38\%$), there remains a substantial gap to the Expert Ceiling:
* **$-11.72\%$** absolute gap in the Standard Setup.
* **$-9.37\%$** absolute gap in the ELFT Setup.
This significant performance gap demonstrates that linear activation blending over multiple specialized LoRA experts still introduces **considerable representational distortion** that is not fully resolved by the proposed calibration heuristics. The blended activations do not perfectly preserve the expert representational structures, highlighting a fundamental limitation of non-parametric blending.

---

## 5. Idealized Systems Latency Model
The authors' claims of flat $O(1)$ sequential latency complexity (illustrated in Figure 1a) are theoretically idealized:
* **Memory Bandwidth Bottlenecks:** Executing $K$ parallel expert paths concurrently requires loading the parameters of all $K$ expert adapters into high-speed memory and executing them. While the sequential depth of the network remains constant, the computational (FLOPs) and memory bandwidth footprints scale as $O(K)$.
* **NPU Serialization:** On actual resource-constrained edge NPUs, smartphones, or microcontrollers with narrow memory buses, the physical memory transfer required to load $K$ adapters concurrently will exceed hardware cache capacities. This leads to physical serialization of memory access, which violates the ideal $O(1)$ flat latency model and results in linear latency scaling in practice. While the authors discuss this scaling ceiling and propose Hard Edge Rejection, the latency benchmarks shown in Figure 1a are evaluated on CPU, which does not fully capture the hardware-level bus serialization of highly parallel edge processors.
