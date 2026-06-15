# Peer Review

## Summary of the Paper
This paper addresses the challenge of serving concurrent Parameter-Efficient Fine-Tuning (PEFT) experts (specifically LoRA adapters) on resource-constrained edge CPUs and microcontrollers. Traditional parameter-space model merging suffers from "heterogeneity collapse" when evaluated on mixed-task input streams. Meanwhile, SOTA systems-level pipelines (such as PFSR + MBH) avoid interference by partitioning heterogeneous batches into task-homogeneous sub-batches, which requires sequential forward passes of the base model and scales latency linearly with the number of active tasks.

To resolve this latency-heterogeneity trade-off, this paper builds directly on the recently proposed **SPS-ZCA** (Single-Pass Activation-Space Dynamic Blending with Zero-Shot Centroid Alignment) framework. It introduces **Q-SPS** and **CG-Q-SPS** to optimize this paradigm for edge-device constraints:
1. **Low-Bitwidth Symmetric Quantization (Q-SPS):** Quantizing LoRA expert weights to 4-bit/8-bit symmetric integers and activations to 8-bit, executing low-rank additions in pure integer precision.
2. **Quantization-Aware Scale Calibration (QASC):** A training-free post-hoc calibration heuristic that sequentially decouples the MSE scale optimization of the down-projection and up-projection matrices.
3. **Conditional Expert Gating (CG-Q-SPS):** Skipping the execution of expert adapter pathways whose routing coefficients fall below a threshold ($\theta = 0.01$) under a low temperature setting ($\tau = 0.001$), reducing compute overheads.
4. **Intra-Task Dispersion Calibration (IDC):** Scaling routing cosine similarities by expected in-distribution averages to equalize coordinate scales.
5. **Coordinate GMM Safety Shield:** Fitting a diagonal GMM on the low-dimensional routing coordinates for out-of-distribution (OOD) task detection.
6. **Temporal-Aware Routing Hysteresis:** An EWMA coordinate-smoothing filter to stabilize cache residency and suppress routing flicker under sequential ($B=1$) streaming.

The authors evaluate these techniques primarily inside a custom, hardware-calibrated analytical simulation environment (the Isolating Coordinate Sandbox, or ICS) mapping a 12-layer Vision Transformer (ViT-Tiny) across four toy-scale domains (MNIST, Fashion-MNIST, CIFAR-10, SVHN). The paper reports simulated joint mean accuracy recovery (79.40% under 4-bit precision, recovering 99.5% of the unquantized float ceiling) and projects a 3.97$\times$ speedup and a 56.2% energy savings over SOTA sequential micro-batching. It also includes empirical single-layer reconstruction tests on pre-trained ViT weights and physical PyTorch CPU micro-benchmarks.

---

## Strengths and Weaknesses

### Strengths:
1. **Exceptional Intellectual Honesty:** The paper stands out for its transparency. The authors explicitly report physical CPU micro-benchmarks (Section 4.9) demonstrating that running low-precision PEFT inside eager PyTorch actually results in a **slowdown** (up to $4\times$) on physical hardware due to high-level framework dispatch and dynamic allocation overheads. They also candidly report that explicit centroid orthogonalization methods (Gram-Schmidt CCO and Löwdin SMD) are mathematically redundant and detrimental under noise due to representation coupling. This level of self-critical analysis is highly refreshing and scientifically valuable.
2. **Impressive Systems-ML Co-Design Depth:** The paper is written with outstanding technical rigor and shows a deep appreciation for physical edge-serving constraints. It explicitly analyzes and proposes practical solutions for register-unpacking penalties, log-sum-exp Softmax stabilization, cache line pollution under routing flicker, lock-free ring-buffer task queues, and heterogeneous Big.LITTLE CPU thread scheduling.
3. **Thorough Ablation Sweeps:** The evaluation includes exhaustive sweeps across critical operational parameters, including quantization bitwidths (FP32 to INT2), feature noise ($\sigma$), task coordinate entanglement ($\epsilon$), temporal coordinate smoothing scales ($\gamma$), and expert registry scale ($K$).
4. **Decoupled Calibration Efficiency:** Formulating QASC to optimize down-projection and up-projection scales sequentially is a highly practical choice that successfully reduces calibration complexity from $O(N^2)$ to $O(N)$ with no loss in representation fidelity.

### Weaknesses:
1. **Limited Conceptual Originality:** The paper lacks a major conceptual leap or a truly original, paradigm-shifting idea. The core paradigm—executing serving in a single forward pass by blending expert activations layer-by-layer on-the-fly and routing them in early layers via centroid alignment—was already fully established by the prior **SPS-ZCA** framework. The proposed additions are a collection of post-hoc engineering heuristics and standard statistical tools (symmetric uniform quantization, decoupled MSE-clipping bounds, threshold-based gating, expected scale division, diagonal GMMs, and first-order EWMA filters) stacked on top of the existing framework. While they are combined effectively, they represent incremental systems tuning rather than a fundamental change in how the community thinks about multi-task serving.
2. **Absence of a Physical Compiled Implementation:** The core performance claims of a 3.97$\times$ speedup and a 56.2% energy savings are generated entirely via analytical simulation models and algebraic hardware cost projections. However, the physical micro-benchmarks in Section 4.9 demonstrate that executing these methods in eager PyTorch actually degrades performance on real CPU. The paper would be significantly stronger if the authors had physically implemented their compiled fused custom C++ kernels in ONNX Runtime or ExecuTorch and measured the speedups on real physical edge CPUs (e.g., Raspberry Pi or Cortex-M microcontrollers), rather than presenting a "systems compilation roadmap" (Section 5.2). Without this physical compiled validation, the latency advantages of CG-Q-SPS remain entirely hypothetical.
3. **Reliance on Toy Vision Tasks and Sandboxed Data:** The main accuracy evaluations are confined to a synthetic sandbox operating on simple, toy-scale classification tasks (MNIST, Fashion-MNIST, CIFAR-10, SVHN). Grayscale $28 \times 28$ images and low-resolution RGB images do not reflect the complexity of modern on-device foundation models. To demonstrate true serving utility, the system should be evaluated on more challenging, real-world multi-task benchmarks—such as on-device multi-task language processing (e.g., LLaMA-3.2 fine-tuned on diverse instructions) or multi-task dense vision models.
4. **Extremely Low SVHN Expert Ceiling:** The unquantized SVHN Expert Ceiling of 31.20% is extremely low (barely above a random 10-class baseline of 10.0%) and practically unusable. Running coordinate evaluations and OOD GMM safety shields on a task that is barely trained reduces the realism of the experimental baseline.

---

## Soundness
**Rating: Good**

**Justification:**
The proposed mathematical formulations are correct, rigorous, and highly detailed. The authors are careful and honest about evaluating the limitations of their work, particularly regarding framework dispatch overheads and the redundancy of centroid orthogonalization. However, the rating is bounded at "Good" because the main classification and latency results are generated inside a custom coordinate simulation sandbox under idealized coordinate projections, rather than being validated on real, physical datasets propagating through an end-to-end compiled physical model.

---

## Presentation
**Rating: Excellent**

**Justification:**
The submission is exceptionally well-written, articulate, and mathematically rigorous. The progression from the introduction of edge-serving trade-offs to the detailed co-design of Q-SPS/CG-Q-SPS is highly logical and easy to follow. Figures and tables are polished, professional, and contain rich, detailed captions. Edge-cases and operating system constraints are analyzed with great clarity and professional maturity.

---

## Significance
**Rating: Fair**

**Justification:**
While the paper addresses an important problem (on-device multi-expert serving) and provides practical insights for systems-ML practitioners (such as the framework dispatch overheads of small CPU matrix operations and the redundancy of orthogonalization under noise), its overall significance is limited. Because the core latency and energy savings are entirely simulated, and the physical micro-benchmarks actually show performance slowdowns, the work does not yet offer a physically proven serving solution. Furthermore, the reliance on toy-scale vision datasets and sandboxed coordinates limits its immediate relevance to modern foundation model deployments.

---

## Originality
**Rating: Fair**

**Justification:**
The paper does not introduce a fundamentally new machine learning paradigm. The core concept of activation-space dynamic blending via early centroid routing is adopted directly from the prior SPS-ZCA framework. The technical additions are highly incremental, representing a collection of standard, well-known post-hoc engineering heuristics (symmetric uniform rounding, decoupled sequential MSE optimization, thresholding, diagonal GMMs, and first-order EWMA filters) stacked on top of an existing framework to optimize it for edge-device constraints.

---

## Overall Recommendation
**Rating: 3: Weak Reject**

**Justification:**
The paper is technically solid, exceptionally well-written, and demonstrates outstanding intellectual honesty. However, for a major machine learning conference, the merits are ultimately outweighed by the lack of conceptual novelty and the lack of physical compiled validation. 
- Conceptually, the work is highly incremental, as it builds directly on the established SPS-ZCA paradigm and relies on standard heuristics and classic statistical models. 
- Empirically, the core speedup (3.97$\times$) and energy savings (56.2%) are entirely simulated and projected. The physical micro-benchmarks actually show a performance slowdown on physical CPU due to high-level framework overheads. 
Without an actual physical compiled implementation (e.g., executing fused custom C++ kernels in ONNX Runtime or ExecuTorch on a physical edge CPU) and evaluations on more realistic, modern multi-task benchmarks (rather than toy vision tasks like MNIST and a heavily underperforming 31.20% SVHN baseline), accepting this work is premature. It is an excellent systems optimization paper that is currently at a conceptual/simulation phase rather than a physically proven deployment solution.

---

## Constructive Feedback and Questions for Authors

1. **Physical Compiled Feasibility:** Given that standard PyTorch (even compiled) exhibits a significant performance penalty under low-precision PEFT on CPU, why did you not implement and evaluate the fused C++ kernels mapped out in your "compilation roadmap"? A physical evaluation of CG-Q-SPS compiled inside ExecuTorch on a physical edge CPU (such as a Raspberry Pi) would dramatically elevate the paper's empirical soundness.
2. **SVHN Performance Ceiling:** Why is the SVHN expert ceiling extremely low at 31.20%? If the base backbone is pre-trained, fine-tuning even a lightweight $r=8$ LoRA adapter on SVHN should easily yield $>80\%$ accuracy. Does this low ceiling indicate a bug in the adapter fine-tuning or a fundamental representation bottleneck in the ViT-Tiny backbone? How do the ZCA-IDC coordinates and Coordinate GMM safety shield behave when the SVHN expert is highly optimized and operates at a realistic, high-accuracy ceiling ($>90\%$)?
3. **Covariate Shift and GMM Adaptation:** In physical on-device deployments, gradual domain drift or ambient environment changes can cause representation shift. While you mathematically discuss online EM updates for the GMM safety shield, did you evaluate this online adaptation empirically? How robust is the static GMM threshold $\eta$ under sudden, distinct environmental shifts (e.g., transitioning from indoor to outdoor lighting)?
4. **HLC Frontier Transition Lag:** Under sequential $B=1$ streaming, the EWMA filter stabilizes cache residency but introduces a transition lag of up to 12.67 steps at $\gamma=0.95$, leading to systematic accuracy drops during task switches. In highly dynamic environments where tasks switch frequently and unpredictably, how can this transition lag be mitigated without triggering cache-thrashing? Is there a dynamic adaptation strategy for the smoothing coefficient $\gamma$?
