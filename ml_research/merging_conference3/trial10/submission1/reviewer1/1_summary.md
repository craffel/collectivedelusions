# 1. Summary of the Paper

## Main Topic and Goal
The paper addresses the **accuracy-stability dilemma** encountered when deploying Mixture-of-Experts (MoE) models or dynamic adapter-merging systems on resource-constrained edge devices under sequential, heterogeneous query streams. Specifically, the paper targets the **routing jitter paradox**, where:
1. **Stateless routers** (e.g., SABLE, SPS-ZCA) suffer from violent, high-frequency spatial (layer-to-layer) oscillations of ensembling weights, which triggers representation drift and degrades serving quality.
2. **Stateful routers** (e.g., ChemMerge, PAC-Kinetics) introduce temporal filters (e.g., continuous-time biochemistry kinetics or differential moving averages) that smooth spatial jitter but introduce severe inertial lag (hysteresis) during rapid task switches, leading to downstream accuracy collapse.

The goal of the paper is to design a dynamic ensembling controller that achieves **globally smooth, jitter-free routing trajectories across network depth (spatial dimension) while maintaining zero temporal lag and zero serving hysteresis across samples (temporal dimension).**

---

## Technical Approach (QPathMerge)
The authors propose **Markovian Path-Integral Ensembling (QPathMerge)**, a training-free serving controller inspired by statistical physics and path-integral formulations. Key technical components include:

1. **Path-Integral Formulation:** 
   - Network depth is modeled as a discrete 1D lattice, and the routing trajectory is modeled as a discrete Euclidean path integral over the active network layers ($l \in \{L_{\text{start}}, \dots, L_{\text{end}}\}$).
   - The path probability is governed by a Boltzmann distribution $P(\mathbf{p}) = \frac{1}{\mathcal{Z}} \exp(-\mathcal{S}[\mathbf{p}] / \tau)$, where the Euclidean action $\mathcal{S}[\mathbf{p}]$ consists of:
     - **Matching potentials (potential energy):** Measuring semantic similarity between intermediate representations and pre-calibrated expert activation centroids.
     - **Transition barriers (kinetic energy):** Penalizing expert switches between adjacent layers ($\gamma \cdot \mathbb{I}[k_l \ne k_{l+1}]$).
   - This formulation maps the routing problem to a 1D chain-structured **Markov Random Field (MRF)**.

2. **Bidirectional Solver (Belief Propagation):**
   - The exact globally optimized marginal probabilities of expert selection are computed at each layer using the **Forward-Backward sum-product algorithm** (Belief Propagation) in $O(L K^2)$ time.
   - Symmetrical forward and backward message propagation cancels out single-directional representation drift (exponential sharpening), resulting in perfectly smooth trajectories.

3. **Recursive On-The-Fly QPathMerge (QPathMerge-Single):**
   - To avoid the dual-pass computational overhead of the exact algorithm, the authors propose a single-pass deployment candidate.
   - Forward messages $\alpha^{\text{fwd}}_l(k)$ are computed exactly using actual intermediate representations on-the-fly.
   - Backward messages $\beta^{\text{bwd}}_l(k)$ are recursively updated over a **Truncated Backward Horizon ($H \le 4$)** by speculatively assuming that future potentials match the current layer's potential.
   - The exponential convergence of this truncation is mathematically guaranteed via **Dobrushin's contraction theorem**, reducing the single-pass complexity to $O(L H K^2)$.
   - Relaxation variants, **Linear Extrapolation (LinearExtrap)** and **Rolling Average Extrapolation (RollingExtrap)**, are introduced to break power-iteration degeneracy under non-monotonic representation trends.

---

## Key Findings and Claims
1. **Resolution of the Accuracy-Stability Trade-off:** Under heterogeneous, rapid task-switching workloads, QPathMerge slashes spatial layer-wise routing jitter by over $3\times$ compared to SABLE and ChemMerge, while completely avoiding representation hysteresis (zero temporal lag) to achieve leading serving accuracy.
2. **Dobrushin Contraction Convergence:** A small, truncated backward horizon ($H = 4$) is sufficient to capture over 91% of the spatial smoothing benefits, validating that the speculative on-the-fly backward pass converges rapidly.
3. **Double-Edged Sword of Spatial Smoothness:** Spatial trajectory smoothing improves performance under noisy, homogeneous tasks but acts as a minor representation mismatch hazard under sharp, non-monotonic multi-task compositions across depth.
4. **Superiority of Linear Extrapolation:** On non-monotonic composite tasks, linear slope projection (\texttt{LinearExtrap}) successfully breaks power-iteration degeneracy to achieve leading serving accuracy ($99.67\%$), while historical rolling averaging (\texttt{RollingExtrap}) collapses to $91.42\%$.
5. **Physical Generalizability:** The framework generalizes to physical architectures (ResNet-18 evaluated on natural ImageNet-1K streams with 40 classes), demonstrating a $3.23\times$ to $5.83\times$ reduction in physical layer jitter while preserving top accuracy.
6. **Hardware Efficiency:** QPathMerge-Single adds negligible overhead (e.g., $1.35$ ms or $5.35\%$ end-to-end latency on ResNet-18) and actively prevents cache thrashing and DRAM weight-swapping energy overheads on physical NPUs.

---

## Claimed Contributions and Evidence
- **Contribution 1:** The first formulation of deep network ensembling as a discrete Euclidean path integral over depth. 
  - *Evidence:* Fully developed mathematical framework in Section 3 mapping network depth to a 1D chain MRF.
- **Contribution 2:** A linear-time, single-pass deployment variant (QPathMerge-Single) utilizing a Truncated Backward Horizon.
  - *Evidence:* Mathematical proof of convergence using Dobrushin's contraction theorem in Section 3.6, and empirical validation sweeping $H$ in Section 4.3 (Table 4).
- **Contribution 3:** Extensive evaluations in a high-fidelity 14-layer Coordinate Sandbox under three manifold configurations (Orthogonal, Overlapping, Composite).
  - *Evidence:* Quantitative results in Tables 1, 2, and 3 demonstrating leading accuracy and low jitter under rapid task switching.
- **Contribution 4:** Physical validation on a pre-trained ResNet-18 model using natural ImageNet-1K streams across 40 distinct classes.
  - *Evidence:* Evaluation on natural images with dynamic test-time augmentations (Table 5) showing significant jitter reduction and robust classification accuracy.
- **Contribution 5:** Self-contained production-grade PyTorch implementation and detailed hardware FLOPs/latency/energy analysis.
  - *Evidence:* Complete class implementation in Appendix A and detailed computational scalability profiling in Section 4.4 (Table 6).
