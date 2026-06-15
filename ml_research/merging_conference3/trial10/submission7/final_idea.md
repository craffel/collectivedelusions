# Tenant-Decoupled Stateful Routing (TDSR): Resolving State Contamination in Multi-Tenant serving of Dynamic Model Merging

## 1. Persona Alignment
*The Pragmatist* persona cares primarily about real-world usability, serving throughput, system-level robustness, and latency constraints under realistic deployment conditions. 
Our proposed **Tenant-Decoupled Stateful Routing (TDSR)**, also known as **Slot-Kinetics**, is directly inspired by these core traits:
- **Direct Real-World Impact:** In production environments (e.g., LLM gateways, multi-user edge nodes), sequential serve streams are rarely composed of a single, continuous user's queries. Instead, servers process *interleaved* streams from multiple independent users (tenants) submitting queries simultaneously. Existing stateful routers (such as ChemMerge or PAC-Kinetics) catastrophically fail under interleaved streams due to **state contamination (cross-talk)**—where the state from User A's task bleeds into User B's task, causing incorrect expert blending and severe accuracy drops. TDSR directly solves this major production blocker.
- **Ultra-Low Compute & Memory Footprint:** Maintaining a small pool of active routing state slots (e.g., $M = 4$ slots of dimension $K = 4$) requires storing only a microscopic $4 \times 4$ tensor. Comparing task coordinates and updating states involves basic vector operations that add sub-microsecond latency, preserving the high-throughput, resource-efficient serving profile of PEFT expert ensembling.
- **Robustness to Workload Volatility:** By isolating temporal smoothing within respective user sessions, TDSR eliminates the vulnerability of stateful merging to uncorrelated streams, ensuring robust, reliable, and smooth serving regardless of stream composition.

---

## 2. Core Techniques
- **Multi-Tenant State Decoupling (Slot-Kinetics):** Instead of a single, global routing concentration state vector, the router maintains a pool of $M$ active *state slots* $\mathcal{S} = \{\mathbf{s}_1, \dots, \mathbf{s}_M\}$ acting as independent virtual chemical reactors.
- **Explicit vs. Implicit (Tagless) Stream Partitioning:**
  - *Explicit Session Tagging:* Uses system-provided metadata (e.g., session ID, user token) to map queries directly to their corresponding state slots.
  - *Implicit Coordinate Clustering (Tagless):* If session tags are unavailable, the router dynamically infers the session context by computing the online cosine similarity between the current query coordinate vector $\mathbf{e}_t$ and the running slot centroids $\mathbf{c}_m$.
- **Stateful Recurrence with Passive Exponential Decay:** Updates *only* the active slot's concentration state using a stable first-order recurrence, while applying a passive exponential decay on all inactive slots to release session memory and prevent stale state retention.
- **Subspace Coordinate Projection & Gibbs Policy:** Projects unit-normalized early-layer activations onto offline PCA-constructed task-specific subspaces to extract task affinity coordinates, and maps states to ensembling weights via a multi-temperature Gibbs Softmax policy.

---

## 3. Mathematical Formulation
Let $\mathbf{e}_t = [e_{1, t}, \dots, e_{K, t}]^T \in [0, 1]^K$ be the PCA coordinate projection of the unit-normalized intermediate activation extracted at routing layer $l_{\text{route}} = 3$ at serving step $t$.
We maintain $M$ state slots $\mathbf{s}_m \in \mathbb{R}^K$ and $M$ running centroids $\mathbf{c}_m \in [0, 1]^K$ for $m \in \{1,\dots, M\}$.

### A. Tenant Assignment / Session Routing
At step $t$, the winning state slot $m^*_t$ is determined as follows:
1. **Explicit Tag Mode (Metadata Available):**
   Given a session tag $u_t \in \{1,\dots, M\}$ associated with the query:
   $$m^*_t = u_t$$
2. **Implicit Tagless Mode (Dynamic Inference):**
   If session metadata is unavailable, the query is routed to the slot whose running centroid has the highest cosine similarity to the query's coordinate vector $\mathbf{e}_t$:
   $$m^*_t = \arg\max_{m \in \{1,\dots, M\}} \text{Sim}(\mathbf{e}_t, \mathbf{c}_m)$$
   Where the similarity metric is:
   $$\text{Sim}(\mathbf{e}_t, \mathbf{c}_m) = \frac{\mathbf{e}_t^T \mathbf{c}_m}{\|\mathbf{e}_t\|_2 \|\mathbf{c}_m\|_2 + \epsilon}$$
   We then update the winning slot's running centroid online using a running exponential moving average:
   $$\mathbf{c}_{m^*_t, t} = \eta \mathbf{c}_{m^*_t, t-1} + (1-\eta) \mathbf{e}_t$$
   Where $\eta \in (0, 1)$ is the centroid inertia coefficient (we set $\eta = 0.90$ in practice).

### B. Stateful State Update with Passive Decay
Once the winning slot index $m^*_t$ is identified, the routing state tensor is updated:
- **Active Slot Update (First-Order Recurrence):**
  $$\mathbf{s}_{m^*_t, t} = \mathbf{A} \mathbf{s}_{m^*_t, t-1} + W \mathbf{e}_t$$
  Where $\mathbf{A} = \text{diag}(a_1, \dots, a_K)$ with $a_k = \sigma(u_k) \in (0, 1)$ are the learnable task-specific retention rates, and $W \in \mathbb{R}^{K \times K}$ is the learnable coordinate injection matrix.
- **Inactive Slots Update (Passive Exponential Decay):**
  $$\mathbf{s}_{m, t} = A_{\text{decay}} \mathbf{s}_{m, t-1} \quad \forall m \ne m^*_t$$
  Where $A_{\text{decay}} \in (0, 1)$ is a static passive decay rate (we set $A_{\text{decay}} = 0.95$ in practice).

### C. Gibbs Softmax Policy mapping
The active ensembling weights $\boldsymbol{\alpha}_t \in \Delta^{K-1}$ are computed using the winning slot's updated state $\mathbf{s}_{m^*_t, t}$ through the multi-temperature Gibbs Softmax policy:
$$\alpha_{k, t} = q_k(\mathbf{s}_{m^*_t, t}; \Theta) = \frac{\exp(s_{m^*_t, k, t}/\tau_k)}{\sum_{j=1}^K \exp(s_{m^*_t, j, t}/\tau_j)}$$
Where the task-specific temperatures $\tau_k = e^{w_k} + \tau_{\min}$ are learnable, and $\tau_{\min} = 0.01$ is a strict minimum temperature threshold.

---

## 4. Architecture Specifications
We evaluate TDSR inside the standard 14-layer high-fidelity Analytical Coordinate Sandbox (ICS):
- **Network Depth & Dimension:** $L = 14$ layers; hidden dimension $D = 192$.
- **Frozen early boundary:** Layers $1$ to $3$. Feature extraction and task coordinate projection occur at Layer 3:
  $$h_t^{(3)} = v'_k + \epsilon_t, \quad e_{k, t} = \|V_{k, d}^T \tilde{h}_t^{(3)}\|_2$$
- **State Pool Dimensions:** 
  - Stateful slots: $M$ slots of dimension $K$ (represented as a tensor of shape $M \times K$). In our simulation, we set $M = 4$ slots for $K = 4$ tasks.
  - centoids: $M$ running centroid vectors of shape $M \times K$.
- **Learnable Router Parameters $\Theta$:**
  - Log-retention rates: $\mathbf{u} \in \mathbb{R}^K$.
  - Coordinate injection matrix: $W \in \mathbb{R}^{K \times K}$.
  - Log-temperatures: $\mathbf{w} \in \mathbb{R}^K$.
  - Total parameter complexity: $2K + K^2 = 24$ parameters for $K=4$ tasks.
- **Dynamic Blending Layers:** For layers $l \in [4, 14]$, activations are blended sample-wise using the ensembling weights $\boldsymbol{\alpha}_t$ derived from the active slot:
  $$h_t^{(l)} = h_t^{(l-1)} + \sum_{k=1}^K \alpha_{k, t} \gamma_V (v'_k - h_t^{(l-1)})$$
  where $\gamma_V = 0.05$ is the expert scaling factor.
- **Unified Classification Head:** A negative squared Euclidean distance classifier at Layer 14:
  $$\text{logits}_{t, k} = - \| h_t^{(14)} - v'_k \|_2^2 + b_k$$
  where $b_k$ is the task bias.

---

## 5. Baselines
To demonstrate the empirical utility and robustness of TDSR under interleaved servings, we evaluate it against four foundational baselines:
1. **Stateless SABLE / Layer Centroids:** A stateless baseline that maps coordinates directly to ensembling weights ($\mathbf{s}_t = \mathbf{e}_t$). Highly responsive to changes, but suffers from high trajectory jitter under query-level noise.
2. **Global Stateful Kinetics (PAC-Kinetics / Global ChemMerge):** A standard stateful router that maintains a single, global routing state across all steps. This will serve to demonstrate the severity of **state contamination** under interleaved serving streams.
3. **Static Uniform Merging:** A baseline that blends all experts uniformly ($\alpha_k = 1/K$), serving as an unbiased control.
4. **Oracle Stateful Routing (Clean Stream Upper Bound):** A theoretical upper bound where queries from different tenants are manually separated into clean, non-interleaved sequential streams, and evaluated using separate stateful routers.

---

## 6. Step-by-Step Interaction
1. **Feature Extraction:** Input $x_t$ flows through frozen early layers 1 to 3. Extract the intermediate activation vector $h_t^{(3)} \in \mathbb{R}^{192}$.
2. **Coordinate Projection:** Unit-normalize the activation vector to $\tilde{h}_t^{(3)} = h_t^{(3)} / (\|h_t^{(3)}\|_2 + 10^{-6})$. Project onto the $K$ offline PCA projection matrices $V_{k, d}$ to obtain task coordinate vector $\mathbf{e}_t \in [0, 1]^K$.
3. **Session Identification & Slot Routing:**
   - *Explicit Mode:* Read query metadata $u_t \in \{1,\dots, M\}$, and set winning slot index $m^* = u_t$.
   - *Implicit Mode:* Compute the cosine similarity between $\mathbf{e}_t$ and each running slot centroid $\mathbf{c}_m$. Set winning slot index $m^* = \arg\max_m \text{Sim}(\mathbf{e}_t, \mathbf{c}_m)$. Update the winning centroid: $\mathbf{c}_{m^*} \leftarrow \eta \mathbf{c}_{m^*} + (1-\eta) \mathbf{e}_t$.
4. **Decoupled Stateful Kinetics Update:** Update the winning slot's concentration state:
   $$\mathbf{s}_{m^*, t} = \mathbf{A} \mathbf{s}_{m^*, t-1} + W \mathbf{e}_t$$
   And passively decay all other inactive slots:
   $$\mathbf{s}_{m, t} = A_{\text{decay}} \mathbf{s}_{m, t-1} \quad \forall m \ne m^*$$
5. **Gibbs Policy Mapping:** Pass the winning slot's updated state $\mathbf{s}_{m^*, t}$ through the multi-temperature Gibbs Softmax policy to obtain ensembling weights $\boldsymbol{\alpha}_t \in \Delta^{K-1}$.
6. **Parallel Expert Activation Blending:** Blend intermediate activations in all dynamic layers $l \in [4, 14]$ in a single forward pass using $\boldsymbol{\alpha}_t$.
7. **Unified Distance-Based Classification:** Evaluate the final blended representation $h_t^{(14)}$ against the distance-based classification head to produce classification logits.
