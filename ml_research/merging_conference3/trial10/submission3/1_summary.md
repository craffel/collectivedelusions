# Paper Summary: Lotka-Volterra Competitive Serving (LVCS)

## 1. Core Objective
The paper introduces **Lotka-Volterra Competitive Serving (LVCS)**, a novel dynamic model ensembling paradigm that rejects traditional linear state-space formulations in favor of non-linear biological ecosystem modeling. Specifically, the paper targets the problem of dynamic mixture-of-experts (MoE) or parameter-efficient fine-tuning (PEFT, specifically LoRA) expert blending in sequential query serving. In this setting, a serving system must balance **responsiveness** (instantaneous adaptation at task boundaries) and **stability** (resilience to query-level activation noise).

## 2. Key Proposed Methods
To address this, the authors model the latent activation-space trajectories of task-specific experts as population densities of competing species governed by a discrete-time biological recurrence. The four main technical components are:
1.  **Discrete-Time Lotka-Volterra Ricker Recurrence:** Instead of linear decay-and-injection state updates, the paper proposes a layer-by-layer spatial recurrence modeled on the Ricker competition equation. This formulation uses exponential multiplicative factors, which mathematically guarantees that population densities (and hence ensembling weights) remain strictly positive, eliminating the need for mathematical clamping heuristics.
2.  **Diagonal Carrying Capacities:** Enforces a strict lower bound on diagonal self-limitation coefficients ($c_{kk} \ge 0.1$) to stabilize the population trajectories, avoiding unbounded growth and chaotic behaviors.
3.  **Adaptive Niche Plasticity (Disturbance-Gated Competition):** Measures local temporal stream homogeneity via the cosine similarity of resource coordinates between consecutive steps ($Sim_t$). When a sudden task transition occurs ($Sim_t \approx 0$), the off-diagonal inter-species competition coefficients ($c_{kj}$ for $k \neq j$) are scaled down (with a baseline floor $\delta = 0.1$). This lowers the "invasion barrier," allowing the colonizing (new) expert to establish itself rapidly without historical drag.
4.  **Systems-First Static Coordinate Approximation:** Rather than recalculating resource coordinates dynamically at every layer, the coordinates are extracted once at an early layer ($l_{\text{route}} = 3$) using Principal Component Analysis (PCA) projection matrices. This static approximation runs the spatial Ricker recurrence across subsequent layers with static inputs, reducing inference latency by over 51% compared to a fully dynamic model.

## 3. Evaluation Setup & Baselines
The methodology is evaluated in two distinct environments:
- **Coordinates Sandbox (CS):** A synthetic 14-layer representation simulation testbed with $K = 4$ task experts (MNIST, Fashion-MNIST, CIFAR-10, SVHN). Serving sequences are evaluated under Homogeneous (long task blocks of 250 queries) and Heterogeneous (random rapid transitions) streams, using both Orthogonal and Overlapping manifold structures.
- **Real-World BERT-Tiny GLUE Sequence Classification:** Fine-tunes LoRA adapters on SST-2, MRPC, and CoLA tasks, ensembling both activations and classification heads on a heterogeneous sequence of 1,200 total queries.

### Baselines Evaluated:
1.  **Oracle:** Theoretical upper bound.
2.  **Uniform Merging:** Static baseline with equal weights.
3.  **SABLE:** Stateless centroid router.
4.  **ChemMerge:** Continuous-time biochemical reaction stateful router (ODE-based).
5.  **Momentum-Merge:** Constant EMA stateful router.
6.  **PAC-Kinetics (Vanilla & Augmented):** SOTA linear stateful baseline.
7.  **Softmax (Static) & MLP (Static):** Expressive non-recurrent static classifiers.
8.  **GRU Router:** Unconstrained spatially recurrent baseline.

## 4. Key Quantitative Findings
- **Orthogonal Manifolds:** LVCS achieves competitive performance (e.g., 85.06% accuracy on heterogeneous streams), outperforming ChemMerge by +0.46% and PAC-Kinetics by +0.36%.
- **Overlapping Manifolds:** LVCS demonstrates significant performance gains. On homogeneous streams, LVCS (Dynamic) achieves 89.26% (outperforming PAC-Kinetics by +1.20%). On heterogeneous streams, LVCS (Static) achieves 90.06% (outperforming PAC-Kinetics by +1.34%).
- **Real-World Sequence Accuracy:** On the BERT-Tiny GLUE stream, LVCS (Static) achieves 61.25%, outperforming SABLE and PAC-Kinetics (60.25%), and MLP (Static) (61.00%), while remaining highly parameter-compact (24 params vs 115 for MLP and 404 for GRU).
- **Inference Latency & Scalability:** Vectorized PyTorch implementation of LVCS (Static) takes only 1.63 ms on CPU. Multi-batch scalability sweeps from $B=1$ to $1024$ show super-linear throughput scaling (up to 86,933 QPS on CPU) and a reduction in sequential recurrence overhead to ~20%.
