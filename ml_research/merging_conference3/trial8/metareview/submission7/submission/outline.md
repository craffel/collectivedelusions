# Outline: ChemMerge: Non-Equilibrium Chemical Reaction Kinetics for Dynamic Model Merging

## 1. Title, Authors, and Abstract
* **Title:** ChemMerge: Non-Equilibrium Chemical Reaction Kinetics for Dynamic Model Merging
* **Author Identity:** Choose a visionary-sounding persona, e.g., "Julian Vance" and "Aria Thorne" from the "Institute for Advanced Physical Computation, Zurich, Switzerland".
* **Abstract:**
  - Context of dynamic model merging under streaming, heterogeneous conditions.
  - The critical gap: stateless layer-wise routing leads to high-frequency coefficient jitter, representation saturation, and sharp switching spikes. MBH is stateful but has prohibitive $O(K)$ latency.
  - Our solution: ChemMerge, which treats representation flow as a multi-component chemical reactor.
  - Core mechanism: first-order non-equilibrium kinetics differential equation discretized via Euler steps across layers, maintaining continuous concentration vectors.
  - Key results: Joint mean of 78.11% (homogeneous) and 78.06% (heterogeneous), recovering 98.81% of Expert Ceiling, completely immune to Heterogeneity/Vectorization Collapse with $O(1)$ constant latency.

## 2. Introduction
* **Paradigm Shift Hook:** Challenge the traditional "decoupled layer" assumption of dynamic ensembling. Layers are sequential, so routing should be sequential, continuous, and possess "inertia."
* **Biological/Physical Inspiration:** Neural representations as reacting chemical species. Introduction of chemical kinetics to model smooth representation flow.
* **The ChemMerge Framework:**
  - Catalytic Zero-Shot Alignment (C-ZCA)
  - Non-Equilibrium Kinetic Routing (NEKR)
  - Catalytic Activation Blending (CAB)
* **Contributions Summary:** Highlighting the +6.18%-8.22% absolute gain, latency benefits, and physical elegance.

## 3. Related Work
* **Static Weight-Space Merging:** Task Arithmetic, TIES-Merging, DARE.
* **Dynamic Activation-Space Merging:** MoEs, QWS-Merge.
* **Test-Time Adaptation & Streaming Merging:** MBH (stateful scheduling), SABLE, SPS-ZCA (stateless early routing). Highlight limitations of both.

## 4. Methodology
* **Problem Setup & Architecture:** Vision Transformer, $L$ blocks, $K$ LoRA experts. Early boundary at Layer 3.
* **Catalytic Zero-Shot Alignment (C-ZCA):**
  - Unit-Norm and Dispersion calibration.
  - Computation of centroids $\mu_k^{(3)}$ and intra-task scales $s_k$.
* **Non-Equilibrium Kinetic Routing (NEKR):**
  - Temperature-scaled Arrhenius rate equation for forward reaction $k_{k,b}^{(l)} = A_0 \exp(u_{k,b}/\tau)$.
  - Non-equilibrium differential rate equation: $\frac{d C_k}{dt} = k_k(1-C_k) - k_{\text{decay}} C_k$.
  - Explicit Euler step discretization across layers: $C_k^{(l)} = C_k^{(l-1)} + \Delta t [k_k^{(l)}(1-C_k^{(l-1)}) - k_{\text{decay}}C_k^{(l-1)}]$.
  - Boundary condition: $C_k^{(3)} = 1/K$.
* **Catalytic Activation Blending (CAB):**
  - Normalization via Law of Mass Action: $\alpha_k^{(l)} = C_k^{(l)} / \sum_j C_j^{(l)}$.
  - Dynamic single-pass blending of hidden representations.

## 5. Experimental Evaluation
* **Experimental Setup:** Analytical Coordinate Sandbox (14 layers, 192 dimension, 4 tasks: MNIST, Fashion-MNIST, CIFAR-10, SVHN).
* **Main Results Sweep:** Detailed analysis of Table 1 (comparison with Expert Ceiling, Uniform, Linear, QWS-Merge, PFSR+MBH, SABLE, SPS-ZCA).
* **Robustness & Streaming Analysis:**
  - Complete immunity to Heterogeneity Collapse ($B=256$) and Vectorization Collapse ($B=1$).
  - Verification of constant $O(1)$ computational latency.
* **Ablations & Hyperparameter Analysis:**
  - Effect of virtual reaction step $\Delta t$, back-reaction/decay rate $k_{\text{decay}}$, and temperature $\tau$.
* **Qualitative Visualizations:**
  - Discussion of the generated plots (`results/fig1.png`, `results/batch_size_heterogeneity.png`, `results/layer_trajectory.png`). Specifically, show how layer-wise trajectories exhibit low-pass temporal smoothing.

## 6. Discussion and Future Directions
* **Visionary Outlook:** Physical and biological frameworks as the future of deep learning. Scaling up to large-scale language models (LLMs) and diffusion processes.

## 7. Conclusion
* Brief wrap-up of ChemMerge's success and potential.
