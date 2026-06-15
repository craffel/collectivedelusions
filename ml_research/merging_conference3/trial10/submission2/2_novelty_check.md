# Novelty Check

## Originality and Novelty Analysis
The core conceptual contribution of this work—**decoupling stateful routing kinetics across network depth**—is highly novel and represents a significant advancement in the sub-field of test-time dynamic model merging. 

Existing stateful routers (such as ChemMerge, Momentum-Merge, and PAC-Kinetics) enforce **spatial homogeneity**: they calculate a single global routing coefficient vector $\alpha_t$ and apply it uniformly across all network depths. This overlooks the long-established deep learning principle that features at different layers evolve at different semantic scales and process representations at distinct temporal tempos. 

By treating network depth as an active variable in temporal-spatial ensembling, LDS-Kinetics introduces a theoretically grounded and empirically validated alternative. It enables different depths of the model to specialize their memory retention:
- **Early layers:** Adapt rapidly to input transitions (acting as high-pass spatial-representational alignment layers).
- **Deep layers:** Maintain stability to shield the classifier from noise (acting as high-inertia, low-pass decision-making filters).

## Positioning and Differentiation from Prior Literature
The paper is exceptionally well-positioned and thoroughly differentiates itself from related works across multiple areas:
1. **Static Model Merging:** Clarifies how dynamic test-time ensembling avoids representation collapse and inter-task interference common in static parameter averaging methods (Task Arithmetic, TIES-Merging, DARE, GitRe-Basin).
2. **Dynamic Subspace Routing:** Highlights how stateful kinetics addresses the "routing jitter paradox" observed in stateless activation-space projection methods (SABLE, PAC-ZCA).
3. **Stateful Kinetics Routing:** Directly builds upon and critiques recent stateful frameworks (ChemMerge, PAC-Kinetics). By demonstrating that global stateful methods suffer from a "stateful accuracy penalty" due to temporal lag at transition boundaries, this paper establishes why depth-decoupled kinetics is the logical next step.
4. **Layer-wise Depth Dynamics:** Connects the proposed approach to general neural network literature showing representation evolution along network depths.

## Novelty Validation and Overclaiming Check
- **No Overclaiming of Mathematical Core:** The authors are transparent that the continuous-time kinetics and state-retention scaling are derived from preceding works (such as PAC-Kinetics). The novelty lies in *decoupling* this kinetics formulation along network depths and mathematically extending the PAC-Bayesian bound to handle joint, multi-block high-dimensional regularization.
- **Symmetry-Breaking Insight:** The identification of the "Adam lockstep symmetry pathology" and how the PAC-Bayesian KL gradient naturally breaks standard optimization weight symmetry is an original, insightful contribution that bridges optimization and statistical generalization.
- **No Missing Citations Identified:** Major relevant baselines from recent venues (2024–2026) are correctly cited and integrated into the empirical evaluation (e.g., SABLE, PAC-ZCA, ChemMerge, PAC-Kinetics, Momentum-Merge).
