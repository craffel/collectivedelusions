# Idea Proposal: QPathMerge (Quantum Path-Integral Ensembling)

## 1. Persona Alignment
This proposal strongly aligns with the traits and goals of **The Visionary** persona. Instead of making incremental, feedforward adjustments to existing routing models or applying heuristic smoothing (such as Exponential Moving Averages), we draw inspiration from quantum mechanics and statistical physics. We completely rethink the concept of stateful routing by modeling the layer-wise ensembling coefficients as a discrete Euclidean path integral over network depth. 

Rather than treating ensembling as a feedforward-only process, we view it as a global optimization problem across the space of all possible expert pathways. By applying quantum path integrals and solving them exactly via message-passing belief propagation, we propose a highly novel, elegant, and paradigm-shifting training-free ensembling architecture. It resolves the fundamental accuracy-stability trade-off by achieving optimal layer-to-layer smoothing (near-zero jitter) while remaining stateless across sequence samples (completely eliminating temporal lag and serving hysteresis).

## 2. Core Techniques
We introduce **QPathMerge** (Quantum Path-Integral Ensembling), a training-free and mathematically exact framework for layer-wise adapter ensembling:
1. **Network Depth as a Path Lattice:** We model the sequential layers of a deep neural network $l \in \{1, \dots, L\}$ as the discrete coordinate positions (imaginary time steps) of a 1D lattice.
2. **Euclidean Path Integral:** We define the routing trajectory as a state path $\mathbf{p} = (k_1, k_2, \dots, k_L)$ where $k_l \in \{1, \dots, K\}$ is the expert adapter chosen at layer $l$. Every path has an action $\mathcal{S}[\mathbf{p}]$ penalizing task mismatches (potential energy) and layer-to-layer expert transitions (kinetic energy).
3. **Exact Marginalization via Sum-Product Message Passing:** We map the path probability to a 1D chain-structured Markov Random Field (MRF). Rather than using continuous-time ODE integration (as in ChemMerge \cite{chemmerge_2026} and PAC-Kinetics \cite{pac_kinetics_2026}), we apply the **Forward-Backward Algorithm (Belief Propagation)** to calculate the exact, globally optimal marginal probability $\alpha_k^{(l)} = P(k_l = k)$ for each expert at each layer in $O(L K^2)$ time.
4. **Symmetric Layer-Space Smoothing:** By utilizing both forward messages (past routing choices) and backward messages (future routing choices), QPathMerge acts as a highly stable, symmetric low-pass filter across network depth. It eliminates high-frequency layer-wise routing oscillations without carrying any state across sequence samples, thus completely bypassing the "inertial lag" (hysteresis) that degrades performance under rapid task switches.

## 3. Mathematical Formulation
Let the sequence of layers be $l \in \{1, \dots, L\}$, and the set of expert adapters be $k \in \{1, \dots, K\}$. Let $h^{(l-1)} \in \mathbb{R}^D$ be the intermediate representation vector entering layer $l$, and let $\mu_k^{(l)}$ be the pre-trained anchor centroid of expert $k$ at layer $l$.

### 3.1. The Euclidean Action
A specific routing path is represented by $\mathbf{p} = (k_1, k_2, \dots, k_L)$. We define the Euclidean action $\mathcal{S}[\mathbf{p}]$ of this path as:
$$\mathcal{S}[\mathbf{p}] = \sum_{l=1}^L \mathcal{L}_{\text{match}}(h^{(l-1)}, k_l) + \sum_{l=1}^{L-1} \mathcal{L}_{\text{trans}}(k_l, k_{l+1})$$

where:
- **Matching Loss (Potential Energy):** Measures how poorly expert $k_l$ fits the representation at layer $l$. We define this as the negative log of the cosine similarity:
  $$\mathcal{L}_{\text{match}}(h^{(l-1)}, k_l) = -\log \left[ S(h^{(l-1)}, \mu_{k_l}^{(l)}) \right]$$
  where $S(A, B) = \max \left( \epsilon, \frac{\langle A, B \rangle}{\|A\|_2 \|B\|_2} \right)$ is the clamped cosine similarity, preventing numerical issues.
- **Transition Loss (Kinetic Energy):** Penalizes high-frequency layer-to-layer switching to suppress routing jitter:
  $$\mathcal{L}_{\text{trans}}(k_l, k_{l+1}) = \gamma \cdot \mathbb{I}[k_l \ne k_{l+1}]$$
  where $\gamma \ge 0$ is the transition barrier constant, and $\mathbb{I}$ is the indicator function.

### 3.2. Path Probability and Partition Function
Following the Feynman path integral formulation under a Wick rotation, the probability of the network following routing path $\mathbf{p}$ is governed by the Boltzmann distribution:
$$P(\mathbf{p}) = \frac{1}{\mathcal{Z}} \exp\left( -\frac{\mathcal{S}[\mathbf{p}]}{\tau} \right)$$
where $\tau > 0$ is the quantum temperature, and $\mathcal{Z}$ is the partition function summing over all $K^L$ possible pathways:
$$\mathcal{Z} = \sum_{\mathbf{p}} \exp\left( -\frac{\mathcal{S}[\mathbf{p}]}{\tau} \right)$$

### 3.3. Exact Marginals via Forward-Backward Message Passing
To compute the ensembling weights $\alpha_k^{(l)}$ at each layer $l$, we calculate the marginal probability $\alpha_k^{(l)} = P(k_l = k) = \sum_{\mathbf{p}: k_l = k} P(\mathbf{p})$. We define:
- **Node Potentials (State Likelihoods):** 
  $$\psi_l(k) = \exp\left( -\frac{\mathcal{L}_{\text{match}}(h^{(l-1)}, k)}{\tau} \right) = \left[ S(h^{(l-1)}, \mu_k^{(l)}) \right]^{1/\tau}$$
- **Edge Potentials (Transition Weights):**
  $$\phi_l(k, k') = \exp\left( -\frac{\mathcal{L}_{\text{trans}}(k, k')}{\tau} \right) = \exp\left( -\frac{\gamma}{\tau} \mathbb{I}[k \ne k'] \right)$$
  Letting $M = \exp(-\gamma/\tau) \in [0, 1]$ be the transition leakage parameter, we simplify the transition factors:
  $$\phi_l(k, k') = \begin{cases} 1 & \text{if } k = k' \\ M & \text{if } k \ne k' \end{cases}$$

We execute the sum-product message passing recursively:
1. **Forward Pass (Propagation of Past History):**
   - Initialize: $\alpha^{\text{fwd}}_1(k) = \psi_1(k)$
   - Recurrence: For $l = 2$ to $L$:
     $$\alpha^{\text{fwd}}_l(k) = \psi_l(k) \sum_{k'=1}^K \alpha^{\text{fwd}}_{l-1}(k') \phi_{l-1}(k', k)$$
   - Scale-Normalization: To prevent underflow, we normalize the forward messages at each layer:
     $$\tilde{\alpha}^{\text{fwd}}_l(k) = \frac{\alpha^{\text{fwd}}_l(k)}{\sum_{j=1}^K \alpha^{\text{fwd}}_l(j)}$$

2. **Backward Pass (Propagation of Future Commitment):**
   - Initialize: $\beta^{\text{bwd}}_L(k) = 1$
   - Recurrence: For $l = L-1$ down to 1:
     $$\beta^{\text{bwd}}_l(k) = \sum_{k'=1}^K \beta^{\text{bwd}}_{l+1}(k') \phi_l(k, k') \psi_{l+1}(k')$$
   - Scale-Normalization: We normalize the backward messages at each layer:
     $$\tilde{\beta}^{\text{bwd}}_l(k) = \frac{\beta^{\text{bwd}}_l(k)}{\sum_{j=1}^K \beta^{\text{bwd}}_l(j)}$$

3. **Marginal Probability Assembly:**
   At each layer $l$, the exact marginal probability (ensembling weight) is:
   $$\alpha_k^{(l)} = \frac{\tilde{\alpha}^{\text{fwd}}_l(k) \tilde{\beta}^{\text{bwd}}_l(k)}{\sum_{j=1}^K \tilde{\alpha}^{\text{fwd}}_l(j) \tilde{\beta}^{\text{bwd}}_l(j)}$$

## 4. Architecture Specifications
- **Input:** Globally pooled/normalized activation features $h^{(l-1)} \in \mathbb{R}^D$ at layer $l$, where $D=192$ and $L=14$ active layers.
- **Centroids:** Target task centroids $\mu_k^{(3)} \in \mathbb{R}^D$ extracted from Layer 3 (Global Early-Layer Anchoring Mode) to ensure stable representation anchors.
- **Hyperparameters:**
  - **Quantum Temperature $\tau > 0$**: Controls the sharpness of the probability distribution (default $\tau = 0.5$). Small $\tau \to 0$ leads to hard-routing selection; large $\tau \to \infty$ leads to uniform ensembling.
  - **Transition Leakage $M \in [0, 1]$**: Controls the coupling strength between adjacent layers. $M = 1.0$ completely disables layer smoothing (equivalent to independent layer-wise routing); $M \to 0$ enforces absolute layer-wise identity, forcing the router to choose a single expert across all layers. (Default $M = 0.05$ to $0.15$).
- **Output:** Exact ensembling weights $\alpha_k^{(l)} \in [0, 1]$ such that $\sum_{k=1}^K \alpha_k^{(l)} = 1.0$, which are used to merge the active LoRA adapters:
  $$W^{(l)}_{\text{merged}} = W^{(l)}_{\text{base}} + \sum_{k=1}^K \alpha_k^{(l)} \Delta W_k^{(l)}$$

## 5. Baselines
We will compare QPathMerge against the following key baselines:
1. **Uniform Merging:** The static average baseline where $\alpha_k^{(l)} = 1/K$ for all experts.
2. **SABLE (Stateless Cosine Router):** The nearest-centroid stateless routing baseline that suffers from high routing jitter.
3. **SPS-ZCA:** Stateless, dispersion-calibrated nearest-centroid routing.
4. **ChemMerge:** The stateful biochemical kinetics router solving continuous-time ODEs to achieve layer-wise smoothness at the cost of temporal serving lag.
5. **Momentum-Merge:** Constant Exponential Moving Average (EMA) layer smoothing across depth.
6. **PAC-Kinetics:** The state-of-the-art stateful ensembling method optimizing continuous chemical kinetics using a Catoni PAC-Bayesian bound under stationary mixing processes.

## 6. Step-by-Step Interaction
For each input sample $x_b$ in a sequence stream:
1. **Extract Representation Anchor:** Pass the input through early network layers up to Layer 3, and extract the activation representation $z_b = h_b^{(3)} \in \mathbb{R}^D$.
2. **Compute Node Potentials (Likelihoods):** For each active layer $l \in \{4, \dots, 14\}$ and each expert $k \in \{1, \dots, K\}$:
   - Compute the cosine similarity $S(z_b, \mu_k^{(3)})$.
   - Set the local node potential $\psi_l(k) = \left[ S(z_b, \mu_k^{(3)}) \right]^{1/\tau}$.
3. **Forward Message Passing:** Compute the scaled forward messages $\tilde{\alpha}^{\text{fwd}}_l(k)$ from layer $4$ to $14$.
4. **Backward Message Passing:** Compute the scaled backward messages $\tilde{\beta}^{\text{bwd}}_l(k)$ from layer $14$ down to $4$.
5. **Formulate Exact Marginals:** At each layer $l$, multiply the normalized forward and backward messages to find the final ensembling weights $\alpha_k^{(l)}$.
6. **Dynamic Adapter Activation Blending:** Blend the expert LoRA outputs using $\alpha_k^{(l)}$ to compute the layer-wise forward activations, and proceed to the next layer.
7. **Zero-Lag Handoff:** Since no state is carried over to the next sample, the system is immediately ready to process the next query in the stream with zero inertial lag.
