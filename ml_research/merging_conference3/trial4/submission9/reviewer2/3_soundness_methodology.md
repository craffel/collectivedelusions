# 3. Soundness and Methodology Check

## 1. Major Technical and Theoretical Flaws

### A. The Scale Mismatch Dilemma (Standardized Decisions vs. Unstandardized Physical Integration)
Soft-EPA decouples the standardization scale between decision routing and physical parameter integration:
- Routing decisions are evaluated in a standardized space: $M_{k, j} = | \frac{\lambda_k \tau_{k, j}}{\sigma_k} |$ (Eq. 7).
- Physical parameter updates are integrated in the unstandardized weight space: $\tau^{\text{exclusive}}_{k, j} = \lambda_k \tau_{k, j}$ (if dominant) or $\gamma \lambda_k \tau_{k, j}$ (if non-dominant) (Eq. 9).

The authors frame this decoupling as a "deliberate and highly effective coordinate-wise regularizer" designed to avoid the "Rich Task" Dominance Trap. However, from a rigorous mathematical and optimization standpoint, this introduction of a scale mismatch is a fundamental flaw:
- If expert $A$ (e.g., MNIST) has a very small global standard deviation ($\sigma_A = 0.000653$) and expert $B$ (e.g., SVHN) has a large standard deviation ($\sigma_B = 0.000974$), then for any coordinate $j$ where their physical updates are equal ($|\tau_{A, j}| = |\tau_{B, j}|$), the standardized magnitude will heavily favor expert $A$ ($M_{A, j} > M_{B, j}$).
- Consequently, coordinate $j$ is routed to expert $A$. When physically integrated, the physical update applied to the model is the unstandardized, tiny update of expert $A$. Meanwhile, the physical update of expert $B$ is attenuated by $\gamma = 0.2$.
- This means that a large, physically significant update from a complex expert is heavily suppressed (multiplied by 0.2) based purely on a normalized score, while a tiny, physically insignificant update from a simple expert is preserved. This scale mismatch actively dilutes and damages the representations of more complex experts. 
- The empirical results strongly support this criticism: when EPM is tuned, the accuracy of CIFAR-10 collapses from **75.83%** (Task Arithmetic) to **36.98%** (TLC-Tune). This massive collapse is not a "regularization" effect; it is the direct consequence of this scale-mismatched routing destroying the large physical updates that are essential to sustain CIFAR-10's complex visual representations.

### B. Topological Scrambling of Neural Representations
Neural network parameters do not exist as independent, isolated scalars; they form cohesive structural layers (weight matrices) that define high-dimensional activation manifolds. 
- Soft-EPA performs routing independently at the individual coordinate level ($1 \times 1$ scalar level). By assigning each individual scalar parameter to a different expert, the spatial and topological structure of the weight tensors is completely fractured.
- The authors' claim that a small background blend ($\gamma = 0.2$) serves as a "topological glue" or "coherence retention factor" that aligns the activation manifolds is a purely heuristic, post-hoc explanation. 
- In reality, scrambling 13.8% of the model's 5.52 million parameters (as cited in the scale override analysis) via coordinate-wise argmax destroys the joint covariance structure of the weight matrices, resulting in massive representation drift in deeper layers. The coherence retention factor $\gamma = 0.2$ is merely a heuristic band-aid that slightly cushions this severe topological damage, rather than preserving true representation coherence in a mathematically rigorous manner.

### C. Capacity Starvation and Lack of Activation Scale Preservation under Sparsity
Under high target sparsity (e.g., $p = 0.8$), EPM pruned weights are simply zeroed out.
- Unlike DARE, which rescales the remaining updates by $1/(1-p) = 5.0$ to maintain the expectation value of the activation scales, EPM does not perform any scale adjustment.
- This results in severe activation magnitude decay under high pruning, as 80% of the parameter updates are deleted without compensation.
- The authors' "Dynamic Coherence Scheduling (DCS)" attempts to mitigate this capacity starvation by increasing $\gamma$ to $0.71$. However, this is a clear admission of failure: as $\gamma$ increases to $0.71$, EPM ceases to be "exclusive" and becomes extremely close to standard Task Arithmetic ($\gamma = 1.0$), which lacks exclusivity entirely. 
- Even with DCS, EPM collapses to **26.41%** joint mean at $p=0.8$, whereas DARE (which preserves scales mathematically) maintains a highly robust **40.90%**. This proves that EPM's approach to sparse model merging is technically flawed because it lacks a proper scale-preservation framework.

### D. Methodological Strawman in Baseline Optimization
The authors construct a significant methodological strawman to support their claims regarding the "Overfitting-Optimizer Paradox" and "Optimization Failure":
- AdaMerging and ZipMerge were designed to optimize 56 and 70 continuous parameters, respectively, using **first-order gradient descent** on differentiable validation cross-entropy losses.
- The authors restricted the validation metric to a non-differentiable accuracy minimax score and forced all baselines to be optimized via a zero-order (1+1) Evolution Strategy (ES).
- A 56- or 70-dimensional non-convex continuous search space is mathematically expected to fail under a greedy single-point random mutation search like (1+1)-ES. 
- Labeling this failure as "absolute optimization failure (under-convergence)" of SOTA methods is a severe misrepresentation. The failure is entirely an artifact of the authors' crippled evaluation protocol, which forced these baselines to use an inappropriate optimizer. If AdaMerging and ZipMerge had been optimized using their native first-order gradient descent, they would have converged efficiently.

## 2. Reproducibility and Clarity of Description
- **Clarity:** The mathematical formulation of Soft-EPA, Task Vector Standardization, and TLC-Tune is written with high clarity. Equations 1 through 13 are explicitly stated and easy to follow.
- **Reproducibility:** While the equations are clear, reproducing the exact results requires the specific validation splits and the random seed configurations. The authors state that they run across 5 random seeds, which is good, but the code for the routing operator and TLC-Tune must be made publicly available to ensure true reproducibility. Crucially, the "scale override" statistics (761,836 parameters) are highly specific to their exact runs, and a minor difference in fine-tuning runs or validation splits would yield different numbers.
