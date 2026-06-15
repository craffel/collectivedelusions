# Intermediate Evaluation: 3_soundness_methodology.md

## Clarity of the Description
The methodology is described with exemplary clarity, precision, and rigor. The mathematical formulations are highly complete and structured logically. The paper provides complete formulations for:
- Symmetric uniform quantization and dynamic dynamic scaling factors for activations and weights.
- The sequentially decoupled optimization protocol of **Quantization-Aware Scale Calibration (QASC)**.
- **Zero-Shot Centroid Alignment (ZCA)** with **Intra-Task Dispersion Calibration (IDC)**.
- The **Coordinate GMM safety shield** for out-of-distribution (OOD) rejection, with explicit discussion of the diagonal covariance matrix regularizer.
- The **Conditional Gated Q-SPS (CG-Q-SPS)** execution masks and layer-wise blending step.
- System optimizations such as **Local Batch Re-Ordering**, **Temporal-Aware Routing Hysteresis** (EWMA coordinate filter), and **Adaptive Battery-Aware Gating**.
- Hardware-calibrated analytical cost modeling (DRAM transfer latency, execution latency with dynamic register unpacking and threading barriers, and active energy consumption models).

The appendix further enriches the description by providing full mathematical derivations for the QASC decoupling, L&ouml;wdin Symmetric Manifold De-Entangling (SMD) closed-form steps, and online EM updates for GMM adaptation.

---

## Appropriateness of Methods
Every methodological choice is grounded in physical deployment constraints, which is highly appropriate for a hardware-oriented edge-serving framework:
* **Symmetric vs. Asymmetric Quantization:** The authors choose symmetric uniform quantization and explicitly justify it over asymmetric quantization. While asymmetric quantization theoretically captures non-negative ranges better, its zero-point offset correction terms introduce substantial runtime ALU operations and memory load stalls on low-power CPUs. Symmetric quantization enables highly optimized, branchless, and cache-friendly fused C++ integer GEMM loops.
* **Sequentially Decoupled QASC:** Jointly searching for down-projection, up-projection, and activation scales is a computationally heavy $O(N^3)$ search. Decoupling this into sequential $O(N)$ searches is highly appropriate, collapsing calibration time to under 0.05 seconds per layer while yielding equal output reconstruction fidelity.
* **Coordinate-Space GMM:** Fitting the GMM on the low-dimensional coordinate space ($K=4$) instead of the raw early Layer 3 features ($D=192$) is a major statistical masterstroke. Early-stage high-dimensional representations carry high-frequency low-level texture noise that pollutes density estimation. Projecting features to ZCA-IDC coordinates first filters out this high-dimensional variance, enabling a lightweight diagonal GMM to achieve outstanding OOD detection.
* **Diagonal Covariance Regularizer:** Using diagonal covariance instead of full covariance restricts evaluation complexity to $O(K)$, ensuring fast, branchless execution on microcontrollers and preventing singular matrix inversion failures on compact calibration splits.

---

## Potential Technical Flaws and Intellectual Honesty
The paper is exceptionally robust and free of technical gaps. Rather than hiding limitations or presenting "just-in-case" idealized setups, the authors demonstrate an outstanding level of intellectual honesty and technical maturity:
1. **The Simulation-to-Hardware Gap:** The authors openly admit that physical speedups from low-bit quantization are not achieved out-of-the-box in uncompiled environments (such as eager-mode PyTorch or high-level `torch.compile` on CPU), which suffer from Python dispatch and dynamic memory allocation overheads. They transparently frame their evaluation as an *analytical simulation modeling compiled execution capabilities*, and provide a concrete systems compilation roadmap ( ONNX Runtime, ExecuTorch C++ CustomOps, Neon SIMD register-level pipelining) to bridge this gap.
2. **The Hysteresis-Latency-Cache (HLC) Pareto Frontier:** Under sequential $B=1$ serving, the authors sweep the EWMA coordinate smoothing coefficient $\gamma$. They openly show that while higher $\gamma$ values eliminate cache-thrashing routing flicker, they introduce a *temporal transition lag* when the stream switches to a new domain, causing transition-phase classification drops.
3. **The Early-Stage Representation Depth Trade-Off:** The authors acknowledge that routing task-agnostically at Layer 3 captures low-level features and may fail on visually entangled, fine-grained tasks. They propose a dynamic calibration-time block index selection to resolve this depth trade-off.
4. **Addressing Low Baseline Expert Ceilings:** The authors openly address the low SVHN expert ceiling (31.20%), explaining that the adapter capacity is deliberately restricted to low-rank ($r=8$) to serve as a high-stress test case for their ensembling mechanism under extreme capacity constraints.
5. **Ablation of Centroid Orthogonalization:** The authors explore Gram-Schmidt CCO and L&ouml;wdin SMD, but honestly report that explicit orthogonalization is mathematically redundant and even detrimental under noise due to "noise spillover" across joint projection spaces, establishing raw ZCA-IDC as the robust champion.

---

## Reproducibility
The work is highly reproducible. All mathematical derivations are explicitly written out in the appendix. The authors provide exact hyperparameters, hardware modeling parameters, network dimensions, dataset statistics, and calibration details in both the main text and Appendix B. The inclusion of empirical physical benchmarks and pre-trained ViT weight validations over real images (CIFAR-10) further guarantees that the claims are grounded in reproducible empirical data.
