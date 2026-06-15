# Intermediate Evaluation 1: Paper Summary

## Main Topic
The paper addresses the challenge of serving Parameter-Efficient Fine-Tuning (PEFT) expert adapters, such as Low-Rank Adaptation (LoRA) modules, on a heterogeneous, sequential serving stream where task labels are unavailable. 

In deep networks under sequential serving, stateless routers (e.g., SABLE) suffer from high-frequency layer-to-layer oscillations in ensembling weights, a phenomenon termed **routing jitter**. This jitter causes the blending of incompatible expert weights across successive layers, triggering cascading representational drift and degrading model accuracy. 

To solve this, state-of-the-art stateful routing (e.g., ChemMerge) models ensembling weights as chemical concentrations governed by biochemical kinetics, Arrhenius reaction rates, and continuous Ordinary Differential Equations (ODEs). The paper evaluates this complex physical metaphor through the lens of Occam's razor, proving its mathematical equivalence to a simple Exponential Moving Average (EMA). It proposes **Momentum-Merge**, a training-free, single-parameter stateful ensembling framework that stabilizes routing trajectories using constant EMA temporal smoothing across network depth.

## Proposed Approach
The authors propose **Momentum-Merge**, which strips away the continuous-time biochemical metaphor of ChemMerge. In Momentum-Merge:
1. **Unit-Norm Anchored Similarity Routing:** Cosine similarity is measured between early-layer task-specific centroids $\mu_k$ (pre-computed using 64 calibration samples per task) and intermediate hidden representations $h^{(l-1)}$ at each layer.
2. **Gated Softmax Routing:** Similarities are converted into raw routing weights $w_k^{(l)}$ via a gated Softmax with temperature $\tau > 0$.
3. **Momentum-Merge Dynamics:** Ensembling weights $\alpha_k^{(l)}$ are smoothed across network depth using a simple constant Exponential Moving Average:
   $$\alpha_k^{(l)} = (1 - \beta) w_k^{(l)} + \beta \alpha_k^{(l-1)}$$
   where $\beta \in [0, 1]$ is the constant momentum coefficient.
4. **Boundary Conditions:** The recurrence is initialized either uniformly ($\alpha_k = 1/K$) or via **Raw Boundary Initialization** ($\alpha_k^{(L_{\text{frozen}})} = w_k^{(L_{\text{frozen}}+1)}$) to eliminate transient startup jitter.
5. **Layer-wise Centroid Anchoring (Advanced Variant):** Anchors are pre-computed layer-by-layer ($\mu_k^{(l)}$) to account for representational rotation across depth.

## Key Findings
1. **Mathematical Equivalence:** The continuous-time ODE model of ChemMerge discretized via explicit Euler integration simplifies exactly to a constant-inertia EMA (Momentum-Merge) under uniform activation energy and constant temperature.
2. **Accuracy-Stability Trade-off:** There is a fundamental trade-off in dynamic model ensembling. Stateless routing with layer-wise calibration (SABLE + Layer Centroids) achieves the highest joint classification accuracy (77.24%) but suffers from high layer-to-layer ensembling oscillations (routing jitter). Adding stateful smoothing acts as a low-pass filter, trading a minor fraction of accuracy to stabilize routing trajectories.
3. **Empirical Superiority:** basic Momentum-Merge achieves a joint classification accuracy of 74.85% and reduces routing jitter to 0.012860 (a 5.7$\times$ reduction over tuned SABLE). Advanced Momentum-Merge achieves 74.98% joint accuracy and reduces routing jitter to 0.000374 (a 195.7$\times$ reduction over tuned SABLE and a 41.1$\times$ reduction over tuned ChemMerge), recovering 92.41% of the Oracle expert ceiling.
4. **Task-Asymmetric Noise Robustness:** Under task-asymmetric noise, while ChemMerge's dynamic reaction rates offer a minor accuracy buffer ($+0.15\%$ to $+0.30\%$ absolute) under extreme asymmetry, it incurs a massive increase in routing jitter (surging to 0.0260). Momentum-Merge Advanced maintains near-zero routing jitter (0.002955) and comparable accuracy with zero ODE solver overhead.
5. **Scalability:** Under $K=10$ tasks, the optimal momentum parameter shifts from $\beta = 0.60$ to $\beta = 0.80$ to suppress higher-dimensional noise.

## Explicitly Claimed Contributions (with Evidence)
1. **Mathematical Deconstruction:** A formal proof (Theorem 1) that the continuous-time chemical kinetics of ChemMerge are mathematically equivalent to a simple constant-inertia EMA under explicit Euler discretization.
2. **Minimalist Methodology:** Proposing Momentum-Merge, which replaces the biochemical ODE solver with a single constant EMA equation, requiring only one hyperparameter ($\beta$) and zero systems overhead.
3. **Empirical Validation:** Rigorous evaluation across 10 independent random seeds in the Analytical Coordinate Sandbox (ICS), demonstrating that Momentum-Merge matches or exceeds SOTA ChemMerge accuracy and routing stability under perfectly synchronized random seeds.
4. **Pareto and Sensitivity Analysis:** Comprehensive sweeps over the momentum parameter $\beta \in [0, 1]$, Softmax temperature $\tau$, calibration subset size $|\mathcal{C}_k|$, and depth-wise momentum scheduling (V-shaped Momentum), showing robust control over routing dynamics.
