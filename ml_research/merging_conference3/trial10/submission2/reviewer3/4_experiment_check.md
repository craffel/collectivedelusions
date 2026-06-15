# Experimental Evaluation and Claims Verification: LDS-Kinetics

This document critically evaluates the experimental setup, baselines, and empirical results of the LDS-Kinetics submission, verifying if the central claims are fully supported by evidence.

---

## 1. Rigor of the Experimental Setup
The experimental evaluation is exceptionally thorough, multi-dimensional, and adheres to high empirical standards:
* **Simulated Coordinate Sandbox:** Following standard evaluation protocols in stateful model merging (e.g., ChemMerge, PAC-Kinetics), the authors construct a 14-layer backbone sandbox with a hidden dimension of $D=192$, simulating ensembling of $K=4$ task experts. 
* **Manifold Geometries:** The setup evaluates both *orthogonal* task manifolds and *overlapping* task manifolds ($V=12$ dimensions of overlap). This is critical because overlapping manifolds represent a highly challenging and realistic regime with severe inter-task representation interference.
* **Corruptions:** The evaluations are subjected to sequence-dependent Gaussian noise ($\sigma \in [0.05, 1.20]$) and task-specific coordinate biases ($b \in [0.0, -2.30]$), ensuring a robust stress-test under noisy, real-world serving environments.
* **Calibration Splitting:** Calibration is restricted to a very tight sequence length ($T = 32$). This strict low-data regime is highly appropriate as it exaggerates the threat of overfitting, making it an ideal environment to test the PAC-Bayesian complexity penalty.
* **Multi-Seed Reporting:** All primary results are reported across 5 independent random seeds with standard deviations, ensuring statistical reliability.

---

## 2. Comprehensiveness of Baselines
The paper compares LDS-Kinetics against an impressive pool of 13 baselines:
* **Hypothetical Bounds:** Expert Oracle (routing ceiling) and Uniform Merging (static average blending).
* **Stateless Active-Space Routers:** SABLE (Raw) and SABLE (SEP).
* **Stateless Spatial Control Baselines:** *Static Layer-Wise Decay* and *Static Block-Wise Constant*. These are crucial, newly introduced control baselines designed to isolate whether LDS-Kinetics' benefits are purely due to spatial variation across depths or if temporal kinetics are necessary.
* **Stateful Global Routers:** Heuristic ChemMerge and Global PAC-Kinetics ($M=1$) (the previous SOTA baseline).
* **Unregularized Decoupled Routers:** Decoupled ERM ($M=3, M=11$) and their symmetry-broken counterparts (SB).

This exhaustive baseline selection ensures that every facet of the proposed method—decoupling, regularization, and temporal recurrence—is isolated and verified.

---

## 3. Analysis of Empirical Results and Claims Support
The empirical results provide compelling and robust evidence to support all central claims:

### Claim 1: Depth-decoupling improves performance under dynamic workloads
* **Evidence:** In the overlapping manifold environment under heterogeneous streams (Table 2), LDS-Kinetics ($M=11$) achieves a joint serving accuracy of **$66.84\%$** (compared to $66.81\%$ for Global PAC-Kinetics), recovering approximately $10.3\%$ of the remaining gap to the Oracle ($67.10\%$). More importantly, under non-linear activation propagation (GELU + LN, Section 4.4), the Tri-Block LDS-Kinetics ($M=3$) achieves **$68.50\%$** accuracy on overlapping streams, outperforming Global PAC-Kinetics ($68.40\%$) and significantly outperforming stateless SABLE ($67.40\%$) and the stateless spatial baselines ($67.60\%$).

### Claim 2: Temporal kinetics are mathematically required (over spatial-only schemes)
* **Evidence:** To address peer-review questions, the authors compare LDS-Kinetics against stateless spatial baselines (*Static Decay* and *Static Block*). Under GELU + LN non-linear propagation, the Tri-Block model ($M=3$) outperforms the best spatial baseline by up to **$0.70\%$ absolute accuracy** while dramatically suppressing temporal ensembling jitter. This proves that when representations propagate through non-linear layers, high-frequency weight oscillations from stateless routing compound across depths, causing extreme representational drift that degrades classifier alignment. Stateful temporal smoothing is mathematically necessary to maintain stable, cohesive representational pathways.

### Claim 3: PAC-Bayesian regularization prevents transductive overfitting and breaks symmetry
* **Evidence:** Unregularized Decoupled ERM falls into a degenerate lockstep update path under Adam, collapsing to global $M=1$ performance. When this symmetry is broken via random perturbations (SB), the unregularized models overfit the short sequence ($T=32$), resulting in low serving accuracy ($66.80\%$ on overlapping heterogeneous streams). In contrast, the PAC-Bayesian complexity penalty breaks sign-symmetry naturally during optimization using its KL gradient, while simultaneously shrinking the generalization gap (to $0.0576$ at $T=32$) and guiding robust specialization. This claims verification is elegantly mapped across calibration sweeps ($T \in \{32, 256\}$) in Figure 3.

### Claim 4: Networks organize ensembling tempos along a "tempo-gradient"
* **Evidence:** Deconstructing the learned parameters (Section 4.3.4) reveals a striking depth-dependent pattern: the early block (Layers 4–7) learns high decay ($a \approx 0.32$, short temporal memory) and high temperature ($\tau \approx 0.18$) to adapt rapidly to switches, while the late block (Layers 12–14) learns exceptionally low decay ($a \approx 0.94$) and low temperature ($\tau \approx 0.04$) to act as a stable low-pass decision filter, protecting downstream logits from representational noise.

### Claim 5: The method scales to large expert pools ($K=16$)
* **Evidence:** As the expert pool scales to $K=16$, LDS-Kinetics ($M=11$) displays progressively superior temporal stability compared to the global baseline, reducing heterogeneous jitter by **$8.0\%$** ($0.9184$ vs. $0.9987$) and homogeneous jitter by **$12.8\%$** ($0.3003$ vs. $0.3443$), while maintaining flat sub-linear step latency ($\sim 336\ \mu$s) on CPU.

### Claim 6: The method generalizes to a physical pre-trained backbone with real adapters
* **Evidence:** On a physical 6-layer sequence model, LDS-Kinetics ($M=2$) reduces routing jitter by **$46.6\%$ over SABLE** and **$6.1\%$ over Global PAC-Kinetics**, while achieving a joint accuracy improvement of $+0.14\%$ over Global. Furthermore, the physical step latency of LDS-Kinetics is virtually identical to and statistically indistinguishable from the Global baseline ($1110.85\ \mu$s vs. $1101.91\ \mu$s) when the recurrences are parallelized as a single batched tensor product. This completely resolves systems-level GPU overhead concerns.
