# 2. Novelty Check

## Key Novel Aspects
The paper introduces two main procedural mechanics:
1. **Coordinate-wise routing based on standardized task vectors (Soft-EPA):** Routing individual parameter coordinates to a single "dominant" expert, while scaling the remaining experts' updates by a constant factor $\gamma$.
2. **Task-Level Coefficient Tuning (TLC-Tune):** Using a simple (1+1) Evolution Strategy to optimize $K$ global scaling factors based on a balanced minimax validation accuracy score.

## Delta from Prior Work

### 1. Delta from Task Arithmetic (TA)
- *Task Arithmetic* linearly adds task-specific updates (task vectors) to the base model. EPM differs by introducing coordinate-wise exclusivity: instead of linearly averaging updates across all experts at every coordinate, it routes each coordinate primarily to one expert (the one with the largest standardized magnitude) while heavily attenuating the others. 
- *Critical Assessment of Delta:* Soft-EPA is mathematically formulated as $(1 - \gamma) \cdot \lambda_{k^*(j)} \tau_{k^*(j), j} + \gamma \sum_{k=1}^K \lambda_k \tau_{k, j}$ (Equation 10). When $\gamma = 0.2$, EPM is a convex combination of pure coordinate-wise exclusivity ($\gamma = 0$) and standard Task Arithmetic ($\gamma = 1$). While the framing as a "routing protocol" is elaborate, the mathematical delta is a simple weighted sum where the maximum-magnitude update at each coordinate is weighted by 1.0 and all other updates are weighted by $\gamma = 0.2$.

### 2. Delta from TIES-Merging
- *TIES-Merging* resolves conflicts by trimming small-magnitude updates, electing a majority sign direction at each coordinate, and averaging only the updates that agree with the sign. 
- *EPM* does not resolve sign conflicts, nor does it perform sign-consistent averaging. Instead, EPM uses a soft coordinate-wise assignment where only the dominant task's magnitude is fully retained and non-dominant tasks are attenuated.
- *Critical Assessment of Delta:* While EPM avoids the sign-voting step of TIES-Merging, its coordinate routing is still based purely on magnitude. By selecting the maximum standardized magnitude, EPM assumes that the coordinate with the largest relative update is the only "expert" that should update that weight, which is an incremental heuristic shift from TIES's magnitude-based trimming.

### 3. Delta from DARE
- *DARE* utilizes random dropout to delete a fraction of task updates, rescales the remaining updates by $1/(1-p)$, and averages them.
- *EPM* does not use random dropout. It uses a deterministic, magnitude-based assignment at the coordinate level.
- *Critical Assessment of Delta:* EPM lacks DARE's expectation-value scale-preservation mechanism. Under high sparsity ($p=0.8$), DARE's scaling factor ($1/(1-p) = 5.0$) preserves activation scales, whereas EPM's lack of scaling causes severe activation magnitude decay, resulting in a huge performance gap (EPM achieves 26.41% vs. DARE's 40.90%). EPM's "Dynamic Coherence Scheduling" is an incremental hyperparameter heuristic to compensate for this capacity starvation, but it remains significantly inferior to DARE's mathematically grounded expectation scaling.

### 4. Delta from SOTA Optimization Baselines (AdaMerging and ZipMerge)
- *AdaMerging* and *ZipMerge* optimize layer-group-wise scaling coefficients (56 or 70 continuous parameters) using training-free or test-time adaptation split validation.
- *TLC-Tune* restricts the search space to only $K$ global scaling factors (4 parameters for 4 tasks) and uses a (1+1)-ES.
- *Critical Assessment of Delta:* TLC-Tune is conceptually identical to standard global hyperparameter search. Tuning global scaling factors ($\lambda_k$) is a standard practice in almost all model merging works (including Task Arithmetic and TIES-Merging). The only "novelty" is that TLC-Tune uses a (1+1)-ES to automate this global scale search based on a minimax objective, which is an incremental and straightforward application of a classic black-box optimizer to an extremely low-dimensional hyperparameter space.

## Characterization of Novelty
The overall novelty of this paper is **highly incremental**. 
- The core algorithm, **Soft-EPA**, is a minor heuristic modification of Task Arithmetic that applies a coordinate-wise soft-max-like mask (using hard argmax over scaled standard deviations) to prioritize the largest update. 
- **TLC-Tune** is a standard (1+1) Evolution Strategy applied to global task-vector scaling factors, which are already standard hyperparameters in prior literature.
- The paper heavily inflates the conceptual packaging of these minor modifications, using sophisticated terms like "coherence-preserved parameter routing," "topological glue," "scale override," "double-imbalance trap," and "Overfitting-Optimizer Paradox" to mask the simplicity of a coordinate-wise argmax and a 4-parameter hyperparameter search.
