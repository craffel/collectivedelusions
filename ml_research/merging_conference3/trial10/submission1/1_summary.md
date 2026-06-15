# Paper Summary: Markovian Path-Integral Ensembling (QPathMerge)

## 1. Core Problem and Motivation
The paper addresses the **accuracy-stability dilemma** in dynamic Mixture-of-Experts (MoE) and parameter-merging systems when served under sequential, heterogeneous edge workloads.
- **Stateless routers** (e.g., SABLE, SPS-ZCA) suffer from **spatial (layer-to-layer) routing jitter**, where ensembling coefficients fluctuate violently across adjacent network layers. This spatial oscillation triggers representation drift and cascading representation collapse, degrading output quality.
- **Stateful routers** (e.g., ChemMerge, Momentum-Merge, PAC-Kinetics) use temporal low-pass filters (like ODEs or EMAs) to suppress spatial jitter, but introduce **temporal inertial lag (hysteresis)** during rapid task switches, leading to accuracy drops at task boundaries.

The core research question is: *Is it possible to achieve globally smooth, jitter-free routing trajectories across network depth while maintaining zero temporal lag and zero serving hysteresis across samples?*

---

## 2. Proposed Methodology (QPathMerge)
The authors propose **Markovian Path-Integral Ensembling (QPathMerge)**, a training-free serving controller inspired by discrete Euclidean path integrals in statistical physics.

### 2.1. Mathematical Formulation
- **Lattice Mapping**: The network's $L$ layers are represented as a discrete 1D lattice. A routing trajectory is a state path $\mathbf{p} = (k_{L_{\text{start}}}, \dots, k_{L_{\text{end}}})$.
- **Path Action**: The path cost is governed by an energy-minimization objective consisting of:
  - *Matching Potential (Potential Energy)*: Cosine similarity between representations and pre-calibrated expert centroids $\mu_k^{(l)}$.
  - *Transition Barrier (Kinetic Energy)*: A constant penalty $\gamma$ applied when changing experts between adjacent layers ($\mathbb{I}[k_l \ne k_{l+1}]$).
- **Sum-Product Solver**: Maps the path onto a chain-structured Markov Random Field (MRF). Exact marginal ensembling weights $\alpha_k^{(l)}$ are calculated using a scale-normalized Forward-Backward algorithm (Belief Propagation) in $O(L K^2)$ time per sample.
- **Symmetric Cancellation**: In the bidirectional formulation, when transition leakage $M \to 0$, the forward and backward passes exhibit exponential sharpening that perfectly cancels, yielding zero layer-wise trajectory jitter.

### 2.2. Single-Pass Variant (QPathMerge-Single)
To avoid the $2\times$ computational overhead of a trial pass, the authors propose **Recursive On-The-Fly QPathMerge**:
- Runs in **exactly one forward pass**.
- At layer $l$, computes the forward message $\alpha^{\text{fwd}}_l$ using the actual representation $h^{(l-1)}$.
- For the backward message, speculatively assumes that all future layer potentials are identical to the current layer's potential ($\psi_{l'} = \psi_l$).
- Solves a backward recurrence over a **Truncated Backward Horizon** of length $H \ll L$ (default $H = 4$), which cuts complexity to $O(L H K^2)$ linear scaling.
- To prevent "power iteration degeneracy" under the constant future assumption, they propose **Linear Extrapolation (LinearExtrap)** and **Rolling Extrapolations (RollingExtrap)** based on past layer potential trends.

---

## 3. Key Experimental Findings
The authors evaluate QPathMerge in a 14-layer Coordinate Sandbox and physically validate it on a ResNet-18 model using ImageNet-1K.

- **Sandbox Evaluation**:
  - *Orthogonal & Overlapping Manifolds*: QPathMerge-Single ($H=4$) slashes layer-wise jitter by over **$3\times$** compared to SABLE (slashing spatial jitter from 0.0105 to 0.0032 under orthogonal heterogeneous streams), while maintaining leading serving accuracy (97.47%) and zero serving hysteresis.
  - *Composite Manifolds (Task Shifts over Depth)*: Evaluates performance when the target expert shifts sharply at Layer 9. QPathMerge-LinearExtrap achieves leading accuracy (99.67%) and smooth layer transitions.
- **Physical Validation (ResNet-18 on ImageNet-1K)**:
  - Features 40 distinct classes across 4 tasks.
  - SABLE-Dynamic experiences high spatial jitter (0.252). QPathMerge slashes this jitter to 0.078 ($3.23\times$ smoother) while sustaining top classification accuracy (62.50%).
  - Bypasses the temporal hysteresis of ChemMerge under heterogeneous stream transitions.
- **Computational Overhead**: QPathMerge-Single ($H=4$) adds only **1.35 ms (5.35%)** of end-to-end latency overhead compared to SABLE-Dynamic on a standard CPU, maintaining strong practical feasibility.
