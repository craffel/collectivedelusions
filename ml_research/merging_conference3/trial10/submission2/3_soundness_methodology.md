# Soundness and Methodology Review

## Technical Soundness and Equation Check
The methodology is mathematically elegant and technically sound. The formulas represent a correct and consistent extension of stateful kinetics to layer-decoupled blocks. 

Let's review the main mathematical components:
1. **Normalized Representation (Eq. 1):** Correctly projects the activations $z_t$ onto the unit sphere with stabilization parameter $\epsilon = 10^{-8}$ to prevent division by zero.
2. **Subspace Coordinate Signal (Eq. 2):** Correctly uses the Euclidean norm of the projected vector $e_{k, t} = \|P_k \tilde{z}_t\|_2$ as a scale-free coordinate affinity metric.
3. **Decoupled Recurrence (Eq. 3):** Formulates independent state space evolution for each block $m$, which is theoretically sound.
4. **Dynamic Similarity-based Retention (Eq. 4 & 5):** Scales the block-wise retention $a^{(m)}_k$ dynamically based on the rolling cosine similarity of incoming coordinate signals ($Sim_t$). When the sequential workload changes abruptly, $Sim_t \to 0$, flushing the state and eliminating inertial drag.
5. **Gibbs Softmax Policy & Temperature Parameterization (Eq. 6 & 7):** Standard and robust formulation. $\tau_{\min} = 0.01$ prevents underflow, and the exponential $e^{w^{(m)}_k}$ ensures temperatures are strictly positive.
6. **Activation Blending (Eq. 8):** Formulates the linear combination of frozen backbone weights and dynamic, weight-blended low-rank adapter outputs, matching the standard test-time ensembling paradigm.

## PAC-Bayesian Complexity Penalty and Assumptions
The learning-theoretic regularization is derived from Catoni’s PAC-Bayesian bounds for stationary, $\beta$-mixing stochastic processes.

- **Assumption of Stationarity:** The PAC-Bayesian bound assumes a stationary mixing process, which is technically violated during non-stationary online serving (where task transitions occur).
- **Separation of Roles Solution:** The authors present a highly robust defense of this mismatch:
  - The PAC-Bayesian complexity penalty acts as a robust **regularization prior** during the offline calibration phase (which uses a stationary or representative sequence).
  - The online similarity scaling $Sim_t$ physically manages the non-stationarity during serving, acting as a **dynamic flush mechanism** that resets the stateful memory during task switches.
  - This dual mechanism is theoretically elegant: it preserves the mathematical guarantees of the stationary bound during learning while using a physical state reset to handle non-stationarity during inference.

## Optimization and Weight Symmetry
One of the most impressive sections of the methodology is the deconstruction of the optimization behavior of unregularized *Decoupled ERM*. 
- Under identical SABLE-grounded initialization ($\Theta_0$), the gradients across blocks share the same sign direction.
- Under the Adam optimizer, the first parameter update is determined by the sign of the gradient, making the updates across all $M$ blocks identical.
- This forces a textbook **weight symmetry**, trapping the blocks in a permanent degenerate lockstep path where they behave identically to a global model ($M=1$).
- The authors show that the PAC-Bayesian KL gradient $\nabla_{\Theta} \text{KL}$ breaks this sign symmetry naturally, allowing block parameters to safely diverge to their optimal depth-dependent tempos while providing strong statistical regularization.

## Minor Methodological Limitations/Critiques
1. **Truncated Cross-Entropy Loss ($\mathcal{L}_{\max} = 5.0$):** Used in the empirical risk term of the PAC-Bayesian minimization objective. While the truncation is mathematically required to ensure bounded loss (a standard requirement for PAC bounds), the choice of $\mathcal{L}_{\max} = 5.0$ is an arbitrary hyperparameter. A discussion on how sensitive the optimization is to different truncation thresholds would be beneficial.
2. **Block Boundary Choice:** The partitioning of the 11 ensembling layers into $M=3$ blocks (L4-7, L8-11, L12-14) is static. While the authors explore alternative partitionings (such as "Early-Heavy"), there is no dynamic or automated method to learn block boundaries online. The authors provide a theoretical justification for why Gumbel-Softmax boundary optimization is non-convex and difficult, but this remains a practical limitation.
