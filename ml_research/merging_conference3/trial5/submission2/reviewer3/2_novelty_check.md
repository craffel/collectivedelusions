# Evaluation Task 2: Novelty Check and Assessment of Delta

## delta from Prior Work

### 1. Delta from Static Weight-Space Heuristics (Task Arithmetic, TIES-Merging, DARE-Merging)
*   **Prior Work**: Methods like Task Arithmetic (Wortsman et al., 2022; Ilharco et al., 2022), TIES-Merging (Yadav et al., 2023), and DARE-Merging (Yu et al., 2023) rely on static, globally uniform merging coefficients. They treat coordinate-wise weight conflicts independently and apply coordinate pruning (by magnitude or randomly) before ensembling.
*   **RBPM Delta**: Instead of heuristic, coordinate-wise weight pruning (which can destroy specialized representation pathways, especially in attention maps), RBPM retains complete, dense backbone weights and instead constrains the optimization of continuous, layer-wise merging coefficients. Rather than a static flat scale, RBPM allows depth-dependent ensembling transitions to capture the varying representation roles of deep layers.

### 2. Delta from Adaptive Ensembling Paradigms (Online AdaMerging, Offline Few-Shot Tuning)
*   **Prior Work**: Online AdaMerging (Yang et al., 2023) optimizes layer-wise coefficients unconstrainedly on unlabeled test streams via prediction entropy minimization, which is highly vulnerable to transductive overfitting and degenerate class collapse on local stream noise. Offline Few-Shot Validation Tuning (OFS-Tune; Vance et al., 2025) optimizes independent parameters per layer ($K \times L$ parameters) on a tiny labeled calibration set ($M \approx 10$ samples), leading to severe overparameterization and high-frequency parameter oscillations.
*   **RBPM Delta**: RBPM provides the first formal, learning-theoretic solution to this overparameterization. It constrains the continuous coefficient space to follow a low-degree polynomial trajectory across network depth, which acts as an analytical low-pass filter to shatter local validation noise. It introduces a mathematically sound Consensus-Pulling Rademacher Penalty to bound capacity without causing parameter scale distortion.

### 3. Delta from Heuristic Trajectory Restrictions (PolyMerge)
*   **Prior Work**: PolyMerge (Croft & Vance, 2024) heuristically proposed restricting online coefficients to low-degree polynomial trajectories. However, PolyMerge operates entirely online without labeled supervision (remaining vulnerable to class collapse) and lacks a formal theoretical framework.
*   **RBPM Delta**: RBPM establishes the first formal statistical learning-theoretic foundation for trajectory-based merging. It derives empirical Rademacher complexity bounds for the trajectory space, provides spectrally-normalized margin bounds for the merged network, and applies local Rademacher complexity theory to prove fast generalization rates under few-shot calibration.

---

## Characterization of Novelty

The novelty of Rademacher-Bounded Polynomial Merging (RBPM) is **highly significant**. 

While the concept of using polynomial trajectories has been heuristically touched upon in online contexts, this paper is the first to ground parameter-space model ensembling in **Statistical Learning Theory**. The formal bridge established between the 1D trajectory constraint (Theorem 3.1) and the functional generalization bounds of the deep neural network classifier (Theorem 3.2 and the dimensional linearization bridge) represents a major conceptual breakthrough. 

The paper's novelty is characterized by several highly rigorous theoretical and methodological contributions:
1.  **Rigorous Rademacher Complexity Bounds**: The step-by-step proof of the Rademacher complexity of the polynomial trajectory space (including the shifted sigmoid Ledoux-Talagrand contraction and derivative smoothness guarantees via Markov's Theorem) is mathematically precise and original.
2.  **Local Rademacher Complexity Derivation**: The application of local Rademacher complexity and Bernstein class conditions to prove fast generalization rates ($\mathcal{O}(1/N_{\text{img}})$) under data scarcity brings statistical rigor to a field dominated by empirical heuristics.
3.  **Methodological Synthesis**: Combining polynomial trajectory projection, Consensus-Pulling regularizers, and multi-task gradient surgery (PCGrad) creates a complete, mathematically coherent, and highly effective framework that bridges learning theory with practical multi-task deep ensembling.
