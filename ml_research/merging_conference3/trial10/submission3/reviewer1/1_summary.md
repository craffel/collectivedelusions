# 1. Summary of the Paper

## Main Topic and Objective
The submission addresses the problem of dynamic model ensembling when serving specialized Parameter-Efficient Fine-Tuning (PEFT) adapters (such as Low-Rank Adaptation / LoRA) under sequential streaming queries. The core challenge is managing the trade-off between **responsiveness** (the speed of transitioning to a new expert when task boundaries are crossed) and **stability** (resisting query-level activation noise and maintaining coherent representation trajectories across network layers and depth). 

The paper proposes **Lotka-Volterra Competitive Serving (LVCS)**, which rejects the linear state-space assumption of existing stateful routing frameworks (e.g., ChemMerge, Momentum-Merge, PAC-Kinetics). Instead, it models the depth-wise activation trajectories of specialized experts as the population densities of competing species in a localized biological ecosystem, governed by a discrete-time Lotka-Volterra Ricker competition model.

## Proposed Approach
The LVCS framework consists of several key architectural components:
1. **Resource Extraction via PCA Coordinate Projection:** At an early routing layer ($l_{\text{route}} = 3$), the intermediate activations are projected onto pre-computed task-specific PCA subspaces to obtain resource coordinates $R_{k, t} \in [0, 1]$. These coordinates represent "resource abundance" for each expert species.
2. **Expert Growth Dynamics:** The growth rate $r_{k, t}$ of each expert is a linear function of its local resource coordinate: $r_{k, t} = w_k^{\text{grow}} R_{k, t} + b_k^{\text{grow}}$, where $w_k^{\text{grow}} = \exp(s_k)$ and $b_k^{\text{grow}}$ are learned parameters.
3. **Discrete Ecological Recurrence across Depth:** The virtual population density $x_{k, t}^{(l)}$ of each expert evolves across network layers ($l \in \{l_{\text{route}}+1, \dots, L\}$) via the discrete-time Lotka-Volterra Ricker recurrence:
   $$x_{k, t}^{(l)} = x_{k, t}^{(l-1)} \exp\left( r_{k, t} - \sum_{j=1}^K c_{kj, t} x_{j, t}^{(l-1)} \right)$$
   This exponential formulation naturally guarantees population positivity (bypassing the need for ad-hoc clamping) and models non-linear competition.
4. **Parametric Constraints on Ecosystem Structure:** 
   - **Self-limitation (Carrying Capacity):** Diagonal elements of the competition matrix are lower-bounded to prevent boundless growth ($c_{kk} = \exp(u_k) + 0.1 \ge 0.1$).
   - **Niche Competition:** Off-diagonal elements are mapped to $[0, 1]$ via a sigmoid function ($c_{kj} = \sigma(v_{kj})$).
5. **Adaptive Niche Plasticity:** To eliminate representational lag during sudden task switches, the inter-species competition coefficients are dynamically scaled by the local temporal homogeneity of the query stream:
   $$c_{kj, t} = c_{kj} \cdot (Sim_t + (1 - Sim_t) \cdot \delta)$$
   where $Sim_t$ is the cosine similarity of coordinates between steps, and $\delta = 0.1$ is a baseline competition floor.
6. **Systems-First Static Coordinate Approximation:** Rather than re-projecting coordinates dynamically at every layer, LVCS (Static) extracts coordinate representations once at $l_{\text{route}}$ and holds them static to drive the spatial recurrence across depth.

## Key Findings and Evidence
- **Coordinates Sandbox Evaluation:** LVCS is evaluated on homogeneous and heterogeneous serving streams under both Orthogonal and Overlapping manifold structures. On Overlapping Manifolds (high representational interference), LVCS (Static) achieves **89.08%** accuracy on homogeneous streams and **90.06%** on heterogeneous streams, outperforming the state-of-the-art linear stateful baseline PAC-Kinetics by up to **+1.34%** absolute.
- **Real-World BERT-Tiny Generalization:** On a heterogeneous GLUE sequence classification stream using SST-2, MRPC, and CoLA tasks, LVCS (Static) achieves **61.25%** downstream sequence accuracy, outperforming SABLE and PAC-Kinetics (**60.25%**) and MLP (Static) (**61.00%**), while using $5\times$ to $16\times$ fewer parameters than MLP (Static) or GRU Router baselines.
- **Ablation of Dynamic vs. Static:** LVCS (Dynamic) (which re-projects coordinates at every layer) and LVCS (Static) achieve virtually identical accuracy (within 0.1%–0.2%), validating the systems-level static approximation.
- **Latency and Throughput Scalability:** Latency measurements show that LVCS (Static) runs in **1626.34 $\mu$s**, cutting the latency of the dynamic version by **over 51%**. CPU scalability benchmarks show that a vectorized implementation scales super-linearly and its recurrence overhead collapses from $51.88\%$ to $20.37\%$ as batch size increases to $1024$.

## Explicitly Claimed Contributions
1. **Biologically-grounded non-linear stateful router** bridging discrete population ecology and deep representation learning.
2. **Lotka-Volterra Ricker Recurrence** for layer-by-layer ensembling with guaranteed population positivity and self-regulation.
3. **Adaptive Niche Plasticity** to eliminate representational lag during task switches.
4. **Systems-First Static Coordinate Approximation** reducing serving latency by over 51% compared to a fully dynamic model.
5. **Unconstrained non-linear recurrent router baselines** (e.g., GRU Router) to deconstruct dynamic ensembling trade-offs.
6. **Exhaustive evaluation** across multiple seeds and manifolds, showing superior accuracy and parameter efficiency over existing baselines.
