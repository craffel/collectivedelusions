# 1. Summary of the Paper

## Main Topic and Objective
The paper addresses the challenge of **test-time model ensembling across heterogeneous, non-stationary task streams**. In real-world model-serving deployment scenarios, incoming queries often arrive in continuous, sequential streams with high temporal coherence, rather than as independent and identically distributed (i.i.d.) batches. Fusing parameter-efficient fine-tuning (PEFT) adapters (e.g., LoRA) on-the-fly to construct specialized networks for each query is a highly practical and scalable way to serve multi-task applications. 

The paper's objective is to build a **stateful routing mechanism** that retains historical context to smooth out representation noise (ensuring stability) while remaining agile enough to pivot instantly under sudden task transitions (ensuring plasticity), without incurring unacceptable computational or latency overhead.

---

## Proposed Approach: Unitary Geodesic Routing (UGR)
To resolve the limitations of existing stateful routing methods—which perform updates in unconstrained flat Euclidean spaces and rely on post-hoc Softmax projection, introducing representational lag (hysteresis), scale mismatches, and high-frequency jitter—the authors propose **Unitary Geodesic Routing (UGR)**. UGR models the ensembling routing states directly on the curved $(K-1)$-dimensional unit sphere $\mathbb{S}^{K-1}$:

1. **Information-Geometric Born Mapping:** It leverages the square-root homeomorphism (Bhattacharyya/Hellinger mapping) $s_k = \sqrt{\alpha_k}$ to map coordinates from the positive orthant of the hypersphere $\mathbb{S}^{K-1}_+$ to the probability simplex $\Delta^{K-1}$ natively. This guarantees exact simplex constraints with zero geometric scale compression or activation distortion, bypasses Softmax projections, and mathematically corresponds to exact, closed-form Fisher-Rao geodesic flows.
2. **Closed-Form Geodesic Rotation (Spherical EMA):** It defines a Rodrigues-like geodesic rotation operator that interpolates along the shortest great-circle path of the hypersphere. This has a computational complexity of $\mathcal{O}(K)$ (where $K$ is the number of experts) and completely bypasses expensive matrix exponentials and virtual-time numerical ODE solvers.
3. **Torque-Driven Adaptive Agility:** It scales the rotational step size dynamically in proportion to the representational "torque" (the angular distance/mismatch $\phi$ between the current state and incoming activation signals). This automatically accelerates transitions during task switches and vanishes during stable streams to suppress jitter.
4. **Spatial-Temporal Geodesic Coupling:** It propagates the converged ensembling state smoothly across sequential query boundaries ($\mathbf{s}_t^{(L_{\text{frozen}})} = \mathbf{s}_{t-1}^{(L)}$), creating a continuous 2D spatial-temporal geodesic trajectory that utilizes mature deep semantic priors to stabilize early-layer routing.

---

## Key Findings and Empirical Evidence
The paper evaluates UGR across two highly rigorous, multi-seed benchmarks:
* **High-Fidelity Synthetic Coordinate Sandbox (ICS):**
  * Evaluated across 14 layers, 192 dimensions, and 10 independent random seeds.
  * **UGR** achieves a Joint Mean Classification Accuracy of **75.08%** (outperforming the SOTA continuous biochemical kinetics baseline ChemMerge Reset by **+5.43%** absolute and SABLE by **+0.34%**). This reaches **95.20%** of the theoretical expert ceiling (Oracle).
  * **Jitter Suppression:** UGR slashes layer-to-layer routing jitter ($L \ge 5$) to **19.51 $\times 10^{-4}$**, representing a **2.10x reduction** compared to ChemMerge Reset and **2.49x reduction** compared to ChemMerge Coupled.
  * **UGR (Hybrid Reset)** reduces the boundary transition shock (Jitter $L \ge 4$) by over **2.5x** to **425.55 $\times 10^{-4}$** and slashes intra-query routing jitter ($L \ge 5$) to a pristine **5.13 $\times 10^{-4}$**.
* **Real-World Multi-Task Text Classification Stream (20newsgroups):**
  * Evaluated on a continuous block-structured serving stream of 800 real text documents across 5 independent seeds.
  * **UGR** achieves a Joint Mean Accuracy of **92.25%** ($\pm$ 0.90%), outperforming Coupled Momentum-Merge by **+4.13%** absolute and Coupled ChemMerge by a massive **+21.60%** absolute margin.
  * **Jitter Suppression:** UGR slashes layer-to-layer routing jitter to **3.68 $\times 10^{-4}$** (a **1.63x reduction** over Coupled Momentum-Merge).
  * **UGR (Softmax-Free Target):** Using a fully Softmax-free target (ReLU + $L_1$-norm) achieves **87.40%** accuracy while reducing routing jitter to an exceptionally pristine **1.50 $\times 10^{-4}$** (a **4.0x reduction** over Coupled Momentum-Merge).
* **Wall-Clock Latency and Computational Overhead:**
  * Profiled under single-threaded sequential execution on an Intel Xeon Platinum CPU.
  * UGR adds less than **0.07 ms** of latency per query compared to the stateless SABLE baseline.
  * The fully Softmax-free variant, **UGR (Softmax-Free Target)**, slashes query latency to just **0.436 ms** and boosts system throughput to **2295.3 QPS**, completely bypassing the costly virtual-time numerical ODE solvers required by ChemMerge (0.460 ms / 2173.1 QPS).

---

## Explicitly Claimed Contributions and Verification
1. **Curved State-Space Formulation:** Modeling ensembling states on the non-Euclidean hypersphere mapping natively to the probability simplex via the information-geometric square-root homeomorphism. *Verified by exact, Softmax-free simplex constraint satisfaction and lack of scale-distortion.*
2. **Closed-Form Geodesic Updates:** Deriving a computationally efficient Rodrigues-like geodesic operator that performs spherical interpolation (Slerp) without numerical solvers or matrix exponentials. *Verified mathematically to be exactly norm-preserving, and empirically by low latency (0.487 ms/query).*
3. **Torque-Driven Agility:** A physics-inspired, self-regulating feedback loop where angular torque dynamically scales the routing inertia, resolving the stability-plasticity trade-off. *Verified by near-instantaneous transitions at task boundaries (Figures 1 and 3) and stable, lock-on trajectories during stationary periods.*
4. **Spatial-Temporal Coupling:** Initializing early-layer boundary conditions of the current query with the mature semantic beliefs from the final layer of the previous query. *Verified by the huge accuracy boost of memory-coupled stateful models (Table 3) and specific ablation analysis.*
5. **Controlled Text-Routing Validation:** Demonstrating UGR's outstanding ability to filter out document-level representation noise and lock onto the correct task expert. *Verified by the massive +21.60% accuracy improvement and 16.8x jitter reduction over ChemMerge on 20newsgroups.*
6. **Theoretical and Scalability Extensions:** Formulating differentiable training-time gradients, positive orthant persistence proofs, high-dimensional local top-$k$ sub-sphere routing, online centroid adaptation, and empirical validation of latent expert discovery. *Verified by successful PyTorch backpropagation, mathematical proofs, and centroid-recovery simulations (0.9965 cosine similarity).*
