# 1. Summary of the Paper

## Main Topic and Motivation
The paper addresses a critical challenge in Parameter-Efficient Fine-Tuning (PEFT) and Mixture of Experts (MoE) serving pipelines: dynamically ensembling multiple task-specific expert adapters (like LoRA modules) on-the-fly under a highly heterogeneous, sequential sample-by-sample serving stream where the task identities of incoming samples are unknown.

While stateless dynamic routing systems (such as SABLE) perform sample-wise routing independently at each layer, they suffer from high layer-to-layer routing weight oscillations (routing jitter) due to representational noise. This jitter causes cascading representational drift, degrading downstream performance. To resolve this, state-of-the-art stateful routing methods like ChemMerge model routing trajectories as continuous-time chemical reaction concentrations inside an ODE system with Arrhenius reaction kinetics, but this introduces extreme system-level complexity, virtual-time discretization limits, and multiple hard-to-interpret hyperparameters.

This paper approaches the problem through the lens of Occam's razor, questioning the necessity of these complex biochemical systems.

## Proposed Approach
The authors mathematically deconstruct ChemMerge and prove that under standard Euler discretization, its continuous rate equations are mathematically dual to a simple, standard constant Exponential Moving Average (EMA). 

They propose **Momentum-Merge**, a training-free stateful ensembling framework that stabilizes routing trajectories across depth using a constant, single-parameter EMA update:
$$\alpha_k^{(l)} = (1 - \beta) w_k^{(l)} + \beta \alpha_k^{(l-1)}$$
where:
* $w_k^{(l)}$ is the similarity-routing weight computed at layer $l$ using cosine similarity between intermediate activations $h^{(l-1)}$ and task centroids $\mu_k$.
* $\alpha_k^{(l)}$ are the final ensembling coefficients applied to blend expert activations.
* $\beta \in [0, 1]$ is a constant momentum parameter controlling the low-pass smoothing filter.

They also introduce an advanced variant that incorporates:
1. **Layer-wise Centroid Calibration:** Calibrating task centroids layer-by-layer across network depth to handle representational coordinate rotations.
2. **Raw Boundary Initialization:** Initializing the recurrence using the first adapted layer's similarity weight to eliminate transient jitter.

## Key Findings and Claims
1. **Mathematical Dualism:** Under uniform task activation energy and constant temperature, the biochemical kinetics and ODE systems of ChemMerge simplify exactly to a standard constant EMA (Theorem 3.1).
2. **The Accuracy-Stability Trade-off:** Evaluating ensembling methods inside the Analytical Coordinate Sandbox (ICS) reveals a trade-off. Stateless routing (SABLE + Layer Centroids) achieves the highest joint accuracy (77.24%) but suffers from high routing jitter (0.028517). Stateful temporal smoothing acts as a low-pass filter that trades a small fraction of accuracy for high trajectory stability.
3. **Empirical Performance:**
   * **Momentum-Merge (Base):** Achieves a Joint Mean Accuracy of **74.85%** and reduces layer-to-layer routing jitter to **0.012860** (a $5.7\times$ reduction over SABLE), outperforming the more complex optimal SOTA ChemMerge baseline in both accuracy (74.71%) and stability (0.015339) under perfectly synchronized random seeds.
   * **Momentum-Merge (Advanced):** Achieves **74.98%** joint accuracy and virtually eliminates routing weight oscillations, reducing jitter to an astonishing **0.000374** (a $195.7\times$ reduction over tuned SABLE and a $41.1\times$ reduction over tuned SOTA ChemMerge).
4. **Interpretability of Momentum:** Sweeping $\beta \in [0, 1]$ shows that $\beta$ acts as a clean physical controller, shifting ensembling from stateless routing ($\beta=0$) to static uniform merging ($\beta=1$), peaking at $\beta = 0.60$.

## Explicitly Claimed Contributions (with Evidence in Paper)
* **Mathematical Deconstruction of SOTA:** Proven via Theorem 3.1 (Biochemical Deconstruction) in Section 3.5.
* **Momentum-Merge Methodology:** Described in Section 3.4 and 3.5.
* **Empirical Validation and Trade-off Mapping:** Supported by detailed quantitative experiments in the coordinate-aligned Analytical Coordinate Sandbox (ICS) across 10 random seeds, documented in Section 4 and Table 1.
* **Pareto Analysis and Extensions:** Supported by parameter sweeps over $\beta$ (Section 4.5, Fig. 2), Softmax temperature $\tau$ (Appendix C), depth-wise scheduling (V-shaped Momentum in Appendix D), advanced minimalist ablations (Appendix E), task-asymmetric noise analysis (Appendix F), and task-pool scalability sweeps (Appendix G).
