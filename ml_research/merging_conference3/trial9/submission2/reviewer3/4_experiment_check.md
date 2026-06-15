# Evaluation Task 4: Experiment Check

## Experimental Setup and Datasets
The experimental evaluation is highly structured, thorough, and conducted over 10 independent random seeds:
1. **14-layer Analytical Coordinate Sandbox (ICS):** The primary testing environment, which simulates model execution in a 192-dimensional representation space. It models four downstream classification tasks: MNIST, Fashion-MNIST, CIFAR-10, and a sub-optimal, high-noise SVHN task (which acts as an adversarial stress-test representing night sensor streams with low standalone ceilings of $\approx 21.6\%$).
2. **MobileNetV3-Large & DomainNet Compiler Simulation:** A simulated compiler pilot modeled on the TVM runtime engine (v0.15) with static memory planning on a simulated ARM Cortex-M7 core (400 MHz). It uses four specialized rank-8 LoRA experts fine-tuned on four diverse domains of the DomainNet dataset (Real, Clipart, Painting, Sketch).
3. **Physical Board Profiling:** Physical bare-metal execution on an STM32H747I-DISCO dual-core microcontroller, with physical energy and latency measured using a high-resolution Joulescope JS110 power analyzer.

---

## Evaluation of Comparative Baselines
The paper compares RB-TopM against a highly comprehensive and solid set of baselines:
- **Expert Oracle:** The idealized performance upper bound.
- **Uniform Merging:** Standard parameter-space weight averaging.
- **TIES-Merging & DARE:** State-of-the-art static parameter-space merging.
- **SABLE SOTA & SPS-ZCA:** Un-gated dynamic activation blending.
- **Q-SPS:** Quantized activation-space blending with static gating.

**Fair Comparison Check:**
The authors are highly meticulous about ensuring a fair comparison. They recognize that different routing architectures optimize under different temperature scales ($\tau$). SPS-ZCA/Q-SPS are run in raw activation spaces and require a colder temperature ($\tau = 0.001$) to prevent a flat, near-uniform routing distribution, whereas SABLE/RB-TopM use Intra-Task Dispersion Calibration (IDC) and are optimized under a warmer temperature ($\tau = 0.05$). The authors evaluate each baseline under its literature-optimized temperature, ensuring absolute scientific integrity.

---

## Do the Results Support the Claims?
**Yes, the empirical results provide strong, convincing support for all primary claims:**
1. **Smooth, Monotonic Accuracy-Latency Frontier:** This is clearly supported by Table 2 and Figure 1, demonstrating that the system resource coefficient $C_{\text{budget}}$ successfully scales down ensembling capacity on-the-fly.
2. **Mitigation of "Activation Dilution":** The non-monotonic accuracy peak at $C_{\text{budget}} = 0.4$ ($75.85\%$ accuracy vs. $75.37\%$ at $C_{\text{budget}} = 1.0$) strongly validates the claim that dynamic pruning acts as an "activation regularizer". Zero-gating marginal, non-specialized pathways successfully blocks activation noise from bleeding into deeper layers.
3. **Substantial Latency and Resource Savings:** Supported by simulated TVM pilots ($76.2\%$ DRAM fetch saving and expert latency speedup) and physical STM32 profiling ($74.7\%$ physical latency speedup and $82.9\%$ energy saving on expert paths), proving that memory-bandwidth relief translates directly to hardware efficiency.
4. **OOD Safety Shield Rejection:** Supported by the GMM shield evaluation, which achieves a $38.04\%$ OOD rejection rate under high-noise queries while keeping the unseen test-set false-positive rate tightly bound to $5.26\%$ under the regularized calibration protocol.
5. **Linear Complexity Scaling:** Supported by the expert scaling sweep (Table 6, Appendix D), proving that early-stage similarity routing consumes microsecond-scale execution overhead ($45.85\ \mu\text{s}$ at $K=4$ and only $58.15\ \mu\text{s}$ at $K=24$ under HMD-GMM).
6. **Hierarchical HMD-GMM Scalability:** Supported by the scaling sweep, demonstrating that grouping tasks into macro-domains preserves high OOD rejection rates ($>93\%$) up to $K=24$ tasks, completely de-bottlenecking flat GMM coordinate overlap.

---

## Experimental Limitations and Critiques

### 1. Synthetic Nature of the Primary Sandbox (ICS)
- **Critique:** The primary quantitative results (Table 2, Sensitivity sweeps) are evaluated on the 14-layer Analytical Coordinate Sandbox (ICS). Although the authors mathematically justify the sandbox using intrinsic dimensionality theory (proving that real DNN layers naturally contract task representations to low-dimensional manifolds of $10 \le d \le 50$), a synthetic sandbox still assumes clean orthogonal subspaces and cannot capture the highly non-linear, twisted, and overlapping activation manifolds of real-world deep neural networks.
- **Mitigation:** The authors successfully mitigate this by providing the simulated MobileNetV3-Large TVM pilot on DomainNet (Table 3), demonstrating that the monotonic trade-off frontier and regularizing ensembling trends generalize perfectly to real visual activation manifolds.

### 2. Scale of Real-World Physical Validation
- **Critique:** The physical board profiling on the STM32 board and the simulated TVM pilot are conducted with a relatively small expert registry ($K=4$ tasks) on a lightweight backbone (MobileNetV3-Large). 
- **Mitigation:** The authors acknowledge this scale limitation. While they provide detailed analytical latency and DRAM bandwidth projections for multi-gigabyte models like LLaMA-3-8B (Appendix C, showing that RB-TopM prevents memory-bus choking by reducing FP16 adapter transfers by $78.5\%$, keeping queue occupancy below $21.8\%$), they admit that physical profiling of such large-scale LLMs is outside the scope of their current microcontroller-focused hardware setup. This is a very honest and acceptable limitation.
