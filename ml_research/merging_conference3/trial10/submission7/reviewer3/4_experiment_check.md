# Paper Evaluation: 4_experiment_check.md

## 1. Experimental Setup and Benchmarks
The paper evaluates the proposed Tenant-Decoupled Stateful Routing (TDSR) framework inside a 14-layer simulation environment called the **Analytical Coordinate Sandbox (ICS)**.
- **Network Geometry:** Depth $L = 14$ layers, hidden dimension $D = 192$.
- **Routing Configuration:** Routing layer set at $l_{\text{route}} = 3$ (with layers 1-3 frozen).
- **Fleet and Pools:** $K=4$ specialized expert adapters corresponding to $K=4$ tasks; $M=4$ virtual slots to serve $4$ tenants.
- **Stream Workload:** $N_{\text{cal}} = 100$ calibration samples, $N_{\text{test}} = 400$ testing samples. An interleaved multi-tenant streaming workload is simulated by randomly selecting a tenant at each step to submit a query belonging to their specialized task.
- **Manifold Geometries:** Evaluated on two synthetic manifold structures: Orthogonal Manifolds ($overlap = 0$) and Overlapping Manifolds ($overlap = 12$).

### Critical Practitioner Critique of the Sandbox Setup:
While the ICS is highly precise, clean, and useful for isolating representation dynamics from generation noise, it is a **purely synthetic and stylized sandbox environment**. 
- **No Real-World Validation:** The evaluation contains **no real-world datasets** (e.g., GSM8K, MMLU, CIFAR, ImageNet) and **no real physically-trained models** (e.g., LLaMA, Mistral, ViT) running on physical GPU/TPU hardware.
- **Contrived Workload Structure:** In the main experiment, tenant IDs and task IDs are perfectly aligned ($y_t = u_t$). Although the authors discuss breaking this tenant-task conflation in Section 4.4, they do not present detailed tables for randomized, non-conflated user-to-task distributions. In real-world enterprise gateways, a single user session will submit a sequence of queries that span multiple tasks (e.g., translation, coding, text editing) dynamically, and multiple concurrent users will query the same task.
- **Conclusion on Benchmark Validity:** For a practitioner, this is the paper's largest weakness. While the mathematical proofs are solid, the lack of real-world PEFT ensembling experiments (e.g., on an LLM batching gateway running Punica or S-LoRA) limits the empirical proof of deployment readiness.

---

## 2. Compared Baselines
The paper compares TDSR against highly appropriate and strong control baselines:
1. **Static Uniform Merging:** Establishes the static, non-adaptive baseline.
2. **Stateless SABLE (Raw):** Represents the state-of-the-art stateless dynamic router, which is highly responsive but sensitive to sample-level feature noise.
3. **Global PAC-Kinetics:** Represents the state-of-the-art stateful routing baseline, which maintains a single global routing state and is subject to state contamination.
4. **Oracle Stateful Routing / Isolated Clean-Stream Baseline:** Represents the theoretical performance ceiling where tenant streams are entirely isolated and processed by independent stateful routers.

These baselines are excellent for isolating the impact of "state contamination" and showing how much of the performance ceiling TDSR is able to recover.

---

## 3. Empirical Support for Claims

### A. Resolution of the State Contamination Bottleneck
The quantitative results in Table 1 (Orthogonal Manifolds) and Table 2 (Overlapping Manifolds) strongly support the claim that TDSR resolves state contamination. 
- On Orthogonal Manifolds, standard Global PAC-Kinetics drops to **68.70%** classification accuracy, whereas TDSR Explicit Local achieves **70.60%** (outperforming the Global baseline by **+1.90%** absolute).
- On Overlapping Manifolds, Global PAC-Kinetics drops to **69.10%** accuracy, whereas TDSR Explicit Local achieves **70.85%** (outperforming the Global baseline by **+1.75%** absolute, within 0.50% of the isolated clean-stream ceiling of **71.35%**).

### B. Reduction of Routing Jitter (Stability)
The intra-session jitter analysis strongly supports the claim of dramatic routing stability improvements.
- TDSR Explicit Local slashes intra-session routing jitter to **0.232446** (Orthogonal) and **0.220341** (Overlapping) compared to stateless SABLE's high jitters of **0.552111** and **0.533918** respectively. This is a **2.4$\times$ relative stability improvement**, recovering near-Oracle levels of temporal smoothing.

### C. Concurrency and Latency Scalability
The concurrency scaling sweep in Table 3 supports the claim that TDSR scales gracefully with the number of concurrent tenants.
- As the number of concurrent tenants scales from $M=4$ to $M=256$, TDSR Local consistently outperforms Global PAC-Kinetics (by up to **+1.60%** at $M=256$).
- Latency remains exceptionally low: even at $M=256$ active tenants, the unoptimized PyTorch implementation adds only **67.39 microseconds** per forward pass. In a low-level C++ or CUDA compiled environment, this would run in less than 1.5 microseconds, verifying the lack of system-level bottlenecks.

### D. Memory Reclamation via Dual-Clock Decay
The Dual-Clock decay sweep in Table 4 validates the claim that a background physical timer can prune stale, inactive tenant slots to prevent memory leaks in production without degrading accuracy.
- Under a moderate timeout threshold of 100-200 steps, TDSR recovers to **74.90% - 75.00%** classification accuracy (within 0.20% of the infinite-retention ceiling of 75.10%), confirming that the self-cleaning mechanism is highly viable and safe for high-concurrency cloud servers.

### E. Statistical Rigour
All dynamic routing baselines and TDSR variants are evaluated across **5 independent random seeds**, with standard deviations reported. This confirms that the observed accuracy improvements and jitter reductions are statistically verified and robust against query stream noise.
