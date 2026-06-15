# Idea Proposal: GraviMerge (Orbital Gravitational Dynamics for Jitter-Free Dynamic Model Merging)

## 1. Persona Alignment
As **The Visionary** (curiosity-driven, seeking radical departures from conventional paradigms, willing to take high risks for immense conceptual rewards), we reject the safe, incremental approach of modeling representation flow through simple linear filters or static centroid distances. Instead, we propose an entirely novel, physics-inspired paradigm: **GraviMerge**. 

By drawing a profound analogy to celestial mechanics and general relativity, we treat the deep representation spaces of neural networks as a multi-body gravity field. We re-envision the intermediate activation state as a moving spacecraft (a coordinate probe) with physical inertia, velocity, and mass, and model the pre-trained task experts as massive celestial attractors (stars). This radical shift introduces **physical momentum** directly into deep representation trajectories. This provides a natural, physically-grounded mechanism to completely eliminate layer-to-layer ensembling jitter without requiring artificial low-pass filters or lossy optimization sweeps. It proves that deep learning trajectories can be understood and stabilized through the elegant, universal laws of physical dynamics.

---

## 2. Core Techniques
GraviMerge introduces four novel physical mechanisms into the forward pass of multi-expert model serving:

1. **Arrhenius Mass Activation (AMA):** Instead of static gravitational bodies, the "gravitational mass" $M_k$ of each expert centroid is determined dynamically at test-time. Using a temperature-scaled Gibbs/Arrhenius factor on early-layer zero-shot alignment, the expert most aligned with the sample is activated with a huge mass, establishing a strong gravitational pull, while unrelated experts remain low-mass dwarf stars.
2. **Multi-Body Gravitational Pull (MBGP):** At each adapted layer, the representation probe experiences a net attractive gravitational vector force, which is the vector sum of individual gravitational pulls exerted by each expert star according to Newton's universal law of gravitation.
3. **Inertial Trajectory Integration (ITI):** To resolve routing jitter, the representation probe tracks a continuous multi-dimensional *velocity vector* across sequential layers (acting as discrete time steps). By incorporating a drag/viscosity coefficient modeling the representation medium, ITI integrates velocity and acceleration to update the probe's position, ensuring smooth, differentiable, and physically stable trajectories.
4. **Gravitational Influence Blending (GIB):** The ensembling weight $\alpha_k^{(l)}$ for each adapter is computed as the relative magnitude of the gravitational force exerted by that specific expert on the probe. This ensures that the ensembling weights are a continuous, smooth function of the spacecraft's orbital path.

---

## 3. Mathematical Formulation

### 1. Dynamic Gravitational Mass (AMA)
For an input sample, the shared early layers (Layers 1--3) output a feature representation $h^{(3)}$. The dynamic gravitational mass $M_k$ of task expert $k \in \{1, \dots, K\}$ is set as:
$$M_k = \exp\left( \frac{\cos(h^{(3)}, \mu_k^{(3)})}{\tau} \right)$$
where $\mu_k^{(3)}$ is the pre-computed centroid for task $k$ at Layer 3, and $\tau > 0$ is the routing temperature parameter.

### 2. Angular Distance Metric
The distance $r_{k, l-1}$ between the probe's position $h^{(l-1)}$ at the entrance of layer $l$ and the expert attractor $k$ is formulated as the Euclidean distance on the unit hypersphere:
$$r_{k, l-1} = \sqrt{2 \left( 1 - \cos(h^{(l-1)}, \mu_k^{(3)}) \right)}$$

### 3. Softened Gravitational Force (Newtonian Gravity with Plummer Potential)
To avoid numerical singularity (division by zero) when a probe passes exactly through a task centroid, we use a softened gravitational force model. The force vector $\mathbf{F}_k^{(l)}$ pointing from the probe towards expert centroid $k$ is defined as:
$$\mathbf{F}_k^{(l)} = G \frac{M_k}{\left( r_{k, l-1}^2 + \epsilon^2 \right)^{3/2}} \hat{u}_k^{(l-1)}$$
where $G > 0$ is the gravitational constant, $\epsilon > 0$ is the Plummer softening parameter, and $\hat{u}_k^{(l-1)}$ is the attractive unit direction vector:
$$\hat{u}_k^{(l-1)} = \frac{\mu_k^{(3)} - h^{(l-1)}}{\|\mu_k^{(3)} - h^{(l-1)}\|_2}$$

### 4. Inertial Trajectory Integration (ITI)
The net force accelerates the probe. Under Newton's second law with unit probe mass ($m=1$), the acceleration is $\mathbf{a}^{(l)} = \sum_{k=1}^K \mathbf{F}_k^{(l)}$. We update the continuous velocity and coordinate position across layers $l \in [4, L]$ as:
$$\mathbf{v}^{(l)} = \gamma \mathbf{v}^{(l-1)} + \mathbf{a}^{(l)} \Delta t$$
$$\tilde{h}^{(l)} = h^{(l-1)} + \mathbf{v}^{(l)} \Delta t$$
$$h^{(l)} = \frac{\tilde{h}^{(l)}}{\|\tilde{h}^{(l)}\|_2}$$
where $\gamma \in [0, 1]$ is the friction/drag coefficient modeling the viscosity of the representation medium, and $\Delta t > 0$ is the virtual step size.

### 5. Gravitational Influence Blending (GIB)
The ensembling weight $\alpha_k^{(l)}$ is proportional to the relative magnitude of the gravitational pull exerted by expert $k$:
$$\alpha_k^{(l)} = \frac{\|\mathbf{F}_k^{(l)}\|_2}{\sum_{j=1}^K \|\mathbf{F}_j^{(l)}\|_2} = \frac{M_k / (r_{k, l-1}^2 + \epsilon^2)^{3/2}}{\sum_{j=1}^K M_j / (r_{j, l-1}^2 + \epsilon^2)^{3/2}}$$

---

## 4. Architecture Specifications
* **Backbone Network:** Sequential multi-layer network with $L = 14$ layers and intermediate feature dimension $D = 192$.
* **Expert Count:** $K = 4$ task-specific adapters (representing MNIST, Fashion-MNIST, CIFAR-10, SVHN).
* **Early Shared Layers (Layers 1--3):** Frozen baseline layers, serving as the task feature extractor, outputting $h^{(3)}$.
* **Adapted Serving Block (Layers 4--14):** GraviMerge routing updates and LoRA ensembling are executed at each of these layers.
* **Input Representation:** $h^{(0)} \in \mathbb{R}^{D}$.
* **Velocity Initialization:** At the boundary layer, the velocity vector is initialized to zero: $\mathbf{v}^{(3)} = \mathbf{0} \in \mathbb{R}^{D}$.
* **Hyperparameters:**
  - Gravitational constant $G = 1.0$
  - Medium drag/friction coefficient $\gamma = 0.5$ (prevents chaotic orbit escapes or runaway velocities)
  - Integration step size $\Delta t = 1.0$
  - Plummer softening factor $\epsilon = 0.1$
  - Reaction temperature $\tau = 0.05$

---

## 5. Baselines
To demonstrate the empirical superiority and physical stability of GraviMerge, we compare it against four major baselines:
1. **Uniform Merging:** Uses a static, equal blend of all experts ($\alpha_k^{(l)} = 1/K = 0.25$ for all $k, l$), representing the training-free lower bound.
2. **SABLE (Sample-wise Activation Blending):** A state-of-the-art stateless baseline that calculates ensembling weights via raw cosine similarities at each layer. It represents a highly accurate but high-jitter baseline.
3. **SPS-ZCA (Zero-Shot Centroid Alignment):** A stateless early-routing nearest-centroid baseline. It represents a simple but noisy baseline under out-of-distribution (OOD) stream noise.
4. **ChemMerge:** A state-of-the-art state-dependent baseline that models ensembling weights via first-order chemical concentration ODEs. It serves as our direct physical competitor.

---

## 6. Step-by-Step Interaction

1. **Step 1: Early Representation Extraction:** The input query sample flows through frozen Layers 1--3 of the model, yielding the early-stage representation $h^{(3)} \in \mathbb{R}^{192}$.
2. **Step 2: Gravitational Charge Allocation:** The cosine similarity between $h^{(3)}$ and each of the pre-computed catalytic centroids $\mu_k^{(3)}$ is calculated to determine the dynamic gravitational mass $M_k$ of each expert.
3. **Step 3: Initial Boundary Condition:** The probe's physical velocity vector is initialized to $\mathbf{v}^{(3)} = \mathbf{0} \in \mathbb{R}^{192}$.
4. **Step 4: Layer-by-Layer Gravitational Serving ($l = 4 \dots 14$):**
   - **Distance Computation:** Calculate the unit-sphere distance $r_{k, l-1}$ from the current probe position $h^{(l-1)}$ to each centroid $\mu_k^{(3)}$.
   - **Force Calculation:** Compute individual gravitational force vectors $\mathbf{F}_k^{(l)}$ and the relative ensembling weights $\alpha_k^{(l)}$.
   - **Expert Adapter Blending:** Form the blended model layer parameters using $\alpha_k^{(l)}$ to process the forward pass.
   - **Momentum Integration:** Sum forces to get net acceleration $\mathbf{a}^{(l)}$, update velocity $\mathbf{v}^{(l)}$ using drag $\gamma$, compute the next coordinate position $\tilde{h}^{(l)}$, and normalize to obtain $h^{(l)}$.
5. **Step 5: Final Serving Output:** The deep activation $h^{(14)}$ passes to the final classification layer to produce robust task predictions.
