# Intermediate Review Step 4: Experiment Check

## Evaluation of Experimental Setup
The authors evaluate PID-Merge across two distinct environments:
1. **Isolating Coordinate Sandbox (ICS):** A simulated environment designed to mimic multi-task query streams under high-frequency representation noise.
   - **Parameters:** $L = 14$ layers, hidden dimension $D = 192$, and $K = 4$ task experts. Anchored routing at $L_{\text{frozen}} = 3$.
   - **Workloads:** Evaluated under both Homogeneous (steady-state) and Heterogeneous (rapid step-to-step switches) streams, on both Orthogonal ($\rho=0, V=0$) and Overlapping ($\rho=0.5, V=12$) coordinate manifolds.
2. **Physical Validation on GPT-2 Small:** A physical hardware experiment running on an NVIDIA A100 GPU (80GB).
   - **Configuration:** GPT-2 Small (117M parameters, 12 layers) serving 3 task adapters (Sentiment Analysis, Text Summarization, and Machine Translation) fine-tuned on IMDB, SAMSum, and WMT16 respectively ($r=8, \alpha=16$).
   - **Workload:** $T = 100$ sequential independent queries under a rapidly switching heterogeneous stream.

## Baselines
The paper compares PID-Merge against a comprehensive and competitive set of baselines:
- **Expert Oracle:** The hypothetical upper bound representing 100% accurate routing.
- **Uniform Merging (Static):** A baseline applying static, uniform ensembling weights ($1/K$) across all layers.
- **SABLE (Stateless Raw):** A SOTA nearest-centroid dynamic stateless router.
- **ChemMerge (Kinetics ODE):** A SOTA stateful method modeling weights as continuous-time chemical kinetics.
- **Momentum-Merge (EMA):** A stateful method simplifying ChemMerge to open-loop Exponential Moving Average.
- **PAC-Kinetics:** A stateful method regularized via PAC-Bayesian complexity bounds.

## Analysis of Results and Claims Support
The empirical results strongly and rigorously support the paper's central claims:
1. **In Heterogeneous Streams (Overlapping Manifolds - Table 1):**
   - **PID-Merge (Calibrated)** achieves **94.82%** accuracy, which is almost identical to SABLE's stateless ceiling (**94.93%**).
   - In contrast, open-loop stateful methods collapse under rapid task transitions due to temporal inertial drag: **ChemMerge** drops to **88.42%** and **Momentum-Merge** drops to **86.17%**.
   - This proves that PID-Merge's closed-loop formulation and Derivative (D) acceleration successfully eliminate the tracking delay, yielding up to a **+6.40%** and **+8.65%** absolute accuracy improvement.
2. **In Physical GPT-2 Small Validation (Table 2):**
   - **PID-Merge (Calibrated)** achieves **88.64%** average accuracy (Oracle is $90.48\%$, SABLE is $89.14\%$).
   - It slashes depth-wise layer-to-layer jitter by **73.3%** (from $0.724$ down to $0.193$), demonstrating stable representation convergence under continuous physical noise.
   - **Serving Latency Overhead:** ChemMerge adds a massive **0.482 ms** of overhead per forward pass due to continuous ODE integrations. PID-Merge adds an imperceptible **0.012 ms** (a $40\times$ speedup), making it fully deployment-ready for latency-sensitive applications.
3. **Out-of-Sample Parameter Generalization (Table 4):**
   - The authors demonstrate that when PID-Merge is calibrated on a tiny 32-sample sequence, even under extreme task bias (80% Task 1 or 100% Task 1), the learned gains generalize perfectly on a balanced 200-sample test stream without any performance degradation ($92.17\%$ test accuracy). This is an outstanding result showing high parameter robustness and minimal calibration data requirements.

## Methodological Limitations and Gaps
Despite the strong results, we note several limitations in the experimental evaluation:
1. **Sandbox (ICS) Noise Propagation Constraint:**
   The authors honestly disclose a critical methodological limitation of the ICS sandbox: representation noise is injected *only* at the initial boundary layer, with zero subsequent propagation. Consequently, the stateless SABLE router transitions instantly and does not exhibit depth-wise oscillations in simulation. While the authors successfully address this gap through physical GPT-2 experiments (where physical layer-wise noise is present), the sandbox results should be interpreted with caution.
2. **Model Scale Limitation:**
   The physical experiments are validated on a relatively small **GPT-2 Small model (117M parameters, 12 layers)**. Modern production serving systems run multi-billion parameter models (e.g., LLaMA-3 8B with 32 layers or LLaMA-3 70B with 80 layers). While the authors theoretically analyze scaling dynamics to deeper architectures (Appendix C) and propose anti-windup clamping to handle deep networks, the absence of empirical validation on a multi-billion parameter model is a notable gap. Providing even a single-GPU benchmark of LLaMA-3 8B would dramatically strengthen the paper's systems-level claims.
3. **Lack of Hardware Throughput Measurements:**
   While the authors profile individual GPU forward-pass latency overhead (0.012 ms), they do not report overall end-to-end server throughput metrics (e.g., tokens per second, query throughput) under batched workloads in a production engine like S-LoRA. Since they heavily emphasize high-throughput systems integration and write a Triton kernel fusion design blueprint, empirical throughput evaluation of a batched server would be highly valuable for enterprise deployments.
