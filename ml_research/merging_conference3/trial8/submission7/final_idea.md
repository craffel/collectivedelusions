# ChemMerge: Non-Equilibrium Chemical Reaction Kinetics for Dynamic Model Merging

## 1. Persona Alignment
* **The Visionary Persona:**
  In accordance with our core philosophy that major paradigm shifts occur by challenging foundational assumptions, **ChemMerge** completely rejects the assumption that layer-wise dynamic ensembling coefficients must be static or stateless. Prior state-of-the-art approaches like SPS-ZCA and SABLE treat the hidden states and routing across adjacent layers $l$ and $l+1$ as independent, decoupled blocks. This causes high-frequency "routing weight jitter" and makes the model highly sensitive to sample-level stream noise under heterogeneous serving.
  
  Drawing inspiration from physical biochemistry, we propose an entirely novel, radical paradigm: **representation flow through a deep neural network is modeled as a multi-component chemical reactor undergoing non-equilibrium reaction kinetics**. By treating task-specific experts as reactive species and their pre-computed centroids as catalytic enzymes, we introduce **physical temporal memory (inertia)** across sequential layers. This represents a bold, high-risk conceptual pivot that bridges the gap between systems neuroscience, biochemistry, and model merging, heavily emphasizing the novelty, future potential, and paradigm-shifting nature of our work.

## 2. Core Techniques
ChemMerge introduces three interconnected, training-free, and geometrically-grounded techniques:
1. **Catalytic Zero-Shot Alignment (C-ZCA):** During a one-time, low-resource calibration phase, we extract early representation manifolds (Layer 3) to pre-compute task-specific centroids $\mu_k^{(3)}$ and intra-task dispersion scales $s_k$. These centroids act as catalytic enzymes that lower the activation energy barriers for corresponding expert pathways.
2. **Non-Equilibrium Kinetic Routing (NEKR):** Rather than evaluating stateless similarity at each layer, we track a continuous, sample-wise expert concentration state vector $C_b^{(l)} \in [0, 1]^K$ across the depth of the network. The state transitions are governed by first-order reversible chemical kinetics, where the forward reaction rate is determined by the catalytic affinity of the input to the expert's centroid (via the Arrhenius equation), and the backward decay rate maintains representation plasticity. This physically dampens routing jitter and sequential representation drift.
3. **Catalytic Activation Blending (CAB):** Layer-wise activations are combined sample-wise inside a single, parallel forward pass using ensembling weights derived directly from the active chemical concentrations under the Law of Mass Action.

## 3. Mathematical Formulation
Let $h_b^{(3)}$ be the early-stage pooled activation representation of sample $b$ extracted at the output of the shared base model's Layer 3.

### Cosine-Based Catalytic Similarity
To ensure scale-invariance and immunity to asymmetric task manifold dispersion, we apply **Unit-Norm Calibration (UNC)** and **Intra-Task Dispersion Calibration (IDC)** to derive the calibrated catalytic coordinate $u_{k, b}$:
$$u_{k, b} = \frac{\text{cos\_sim}(h_b^{(3)}, \mu_k^{(3)})}{s_k} = \frac{h_b^{(3)} \cdot \mu_k^{(3)}}{s_k \|h_b^{(3)}\|_2 \|\mu_k^{(3)}\|_2}$$
where $s_k$ is the pre-computed intra-task expected cosine similarity:
$$s_k = \frac{1}{|\mathcal{C}_k|} \sum_{s \in \mathcal{C}_k} \text{cos\_sim}(h_s^{(3)}, \mu_k^{(3)})$$

### Arrhenius Reaction Rate
The forward reaction rate $k_{k,b}^{(l)}$ representing the catalytic affinity of sample $b$ to Expert $k$ is given by a temperature-scaled Arrhenius equation:
$$k_{k, b}^{(l)} = A_0 \cdot \exp\left( \frac{u_{k, b}}{\tau} \right)$$
where $A_0 > 0$ is the pre-exponential frequency factor (set to $1.0$), and $\tau > 0$ is the routing reaction temperature.

### Non-Equilibrium Chemical Kinetics Differential Equation
The rate of change of Expert $k$'s active concentration $C_{k,b}$ is modeled as:
$$\frac{d C_{k,b}}{d t} = k_{k,b}^{(l)} \cdot (1 - C_{k,b}) - k_{\text{decay}} \cdot C_{k,b}$$
where $k_{\text{decay}} \in [0, 1]$ is the back-reaction/decay rate that prevents representation saturation.
We discretize this differential equation via an explicit Euler step at each sequential layer $l \in [4, L]$:
$$C_{k,b}^{(l)} = C_{k,b}^{(l-1)} + \Delta t \left[ k_{k,b}^{(l)} \cdot (1 - C_{k,b}^{(l-1)}) - k_{\text{decay}} \cdot C_{k,b}^{(l-1)} \right]$$
where $\Delta t$ is the virtual reaction time step (set to $0.5$).
* **Boundary Condition (Initialization):** At Layer $l=3$, before entering any adapted block, the concentrations are initialized uniformly across tasks to represent a balanced, un-catalyzed chemical solution:
  $$C_{k,b}^{(3)} = \frac{1}{K}, \quad \forall k \in \{1, \dots, K\}$$

### Law of Mass Action Normalization
The ensembling weights $\alpha_{k, b}^{(l)}$ at layer $l$ are proportional to the active concentrations:
$$\alpha_{k, b}^{(l)} = \frac{C_{k, b}^{(l)}}{\sum_{j=1}^K C_{j, b}^{(l)}}$$

## 4. Architecture Specifications
* **Base Backbone:** Pre-trained frozen Vision Transformer (e.g., `vit_tiny_patch16_224` containing $L=14$ layer blocks, intermediate feature dimension $D=192$).
* **Task Experts:** $K=4$ task-specific experts fine-tuned via Low-Rank Adaptation (LoRA, rank $r=8$) targeting query/value projection matrices.
* **Early-Layer Boundary:** The first 3 blocks (Layers 1-3) are kept completely frozen and shared across all task experts. Task-specific LoRA adapters are never trained or inserted in these early layers, resolving the routing paradox with zero train-inference mismatch.
* **Dynamic Blending:** Inside Layer $l \in [4, L]$, the shared base model representation $W_{\text{base}}^{(l)}$ and parallel expert pathways $A_k^{(l)}, B_k^{(l)}$ are executed concurrently in a single forward pass:
  $$h_b^{(l)} = h_b^{(l-1)} W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_{k, b}^{(l)} \left( h_b^{(l-1)} A_k^{(l)} B_k^{(l)} \right)$$

## 5. Baselines
We evaluate ChemMerge against a comprehensive suite of static and dynamic model merging baselines:
* **Static Uniform Merging:** Simple parameter-space averaging of the task vectors; fails under heterogeneous streams.
* **Micro-Batch Homogenization (MBH):** Stateful systems-level scheduling wrapper that groups samples but incurs a prohibitive $O(K)$ latency penalty on edge hardware.
* **Parameter-Free Subspace Routing (PFSR):** Closed-form linear projections onto expert classification heads.
* **SABLE (Sample-wise Activation Blending):** Stateful-free activation blending baseline utilizing stateless cosine similarity routing.
* **SPS-ZCA:** The current state-of-the-art early-layer nearest-centroid routing baseline.

## 6. Step-by-Step Interaction
1. **Calibration Phase (Offline, One-Time):** Run $64$ calibration samples per task through the shared early blocks (Layers 1-3). Pre-compute early-stage task centroids $\mu_k^{(3)}$ and intra-task dispersion scales $s_k$.
2. **First-Pass early-stage Extraction (Online, Single-Pass):** Feed a highly heterogeneous streaming batch $X$ of size $B$ into the shared, adapter-free early-stage blocks (Layers 1-3) to extract early-stage pooled representations $h_b^{(3)}$ for each sample $b$.
3. **Kinetic Catalyst Initialization:** Compute the calibrated catalytic coordinates $u_{k,b}$ and forward reaction rates $k_{k,b}^{(l)}$ for each sample. Initialize concentration state vectors $C_{k,b}^{(3)} = \frac{1}{K}$.
4. **Iterative Chemical Kinetics and Layer-wise Blending:**
   For each sequential block layer $l = 4 \dots L$:
   a. Retrieve the incoming activation vector $h_b^{(l-1)}$ from the preceding layer.
   b. Compute the forward discretized Kinetic Euler step to update expert concentrations $C_{k,b}^{(l)}$ based on the pre-computed reaction rates $k_{k,b}^{(l)}$ and active decay.
   c. Derive the normalized ensembling weights $\alpha_{k,b}^{(l)}$ using the Law of Mass Action.
   d. Compute the shared base model layer output: $h_{\text{base}, b}^{(l)} = h_b^{(l-1)} W_{\text{base}}^{(l)}$.
   e. Compute parallel low-rank expert updates: $h_{\text{expert}, k, b}^{(l)} = h_b^{(l-1)} A_k^{(l)} B_k^{(l)}$.
   f. Perform dynamic, sample-wise activation blending:
      $$h_b^{(l)} = h_{\text{base}, b}^{(l)} + \sum_{k=1}^K \alpha_{k, b}^{(l)} h_{\text{expert}, k, b}^{(l)}$$
5. **Final Projection:** At the final layer $L$, feed the blended activation representations $h_b^{(L)}$ to the classification/expert heads to yield final predictions.
