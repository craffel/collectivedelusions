# Soundness and Methodology Evaluation

## Clarity of Description
The mathematical steps of the algorithm are described clearly. The notation is mostly standard, and the 4-line vectorized PyTorch code confirms that the implementation is straightforward. However, the conceptual and theoretical foundations are severely lacking in rigor.

## Critical Theoretical and Methodological Flaws

A theory-minded analysis of the proposed methodology reveals several critical flaws, arbitrary assumptions, and lack of mathematical rigor.

### 1. Scale Sensitivity and Lack of Task-Vector Normalization
The core assumption of WTA-Sign is that magnitude is a direct proxy for task confidence:
$$k^*(j) = \arg\max_{k} |T_{k,j}|$$
This assumption is theoretically unsound because it ignores the scale of individual task vectors, which depends heavily on the optimization hyperparameters of each expert:
- Let $\Delta w_k$ be the task vector of expert $k$. If expert $A$ was trained with a larger learning rate $\eta_A$, or for more epochs $E_A$, or with less weight decay, its parameter updates will naturally have larger absolute magnitudes than those of expert $B$, which might have been trained with a smaller learning rate $\eta_B$ or stronger regularization.
- In this scenario, we will have $|T_{A,j}| > |T_{B,j}|$ at almost all coordinates $j$. Under WTA-Sign, expert $A$ will "win" the sign election across the entire network, completely overriding expert $B$'s updates.
- The method does not perform any normalization or scale-standardization of the task vectors (e.g., standardizing by task-wise variance or normalizing to unit $L_2$ norm) prior to the comparison. This scale sensitivity makes the sign election arbitrary and highly dependent on training-time hyperparameter choices rather than any intrinsic task "confidence".

### 2. Arbitrary "Gradient-Space Justification"
The authors present a "Gradient-Space Justification for Magnitude-as-Confidence" in Section 3.1:
$$\Delta w \propto \sum_t \eta_t \nabla_{w} \mathcal{L}_t$$
This is a hand-waving physical analogy rather than a rigorous proof.
- In modern deep learning, optimizers like Adam modify the gradients using first and second moments:
  $$\Delta w \approx -\eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
  Because of the denominator term (second moment), the update magnitude in Adam is bounded and rescaled, heavily decoupling the update magnitude from the raw cumulative gradient pressure.
- For overparameterized models, a large parameter update does not necessarily indicate "task confidence." It can instead indicate that the pre-trained parameter was in a region of high local curvature (unstable loss landscape) or that the optimization path was noisy. WTA-Sign actively prioritizes these unstable, noisy updates over stable, well-behaved ones, which is theoretically counterproductive.

### 3. Masking-Induced Functional Degradation and Lack of Error Bounds
In Step 3, if an expert's update sign opposes the winning sign, it is set to zero by the mask $M_{k,j} = 0$:
$$M_{k, j} = \mathbb{I}(\text{sign}(T_{k, j}) == s_j)$$
- For any expert $k$ whose critical task-specific updates are masked out, there is no theoretical analysis or mathematical bound on the resulting function approximation error. 
- Discarding updates that disagree with the winner means that for any parameter coordinate where experts disagree, the non-winning experts will have their updates completely zeroed out. This can lead to a severe functional degradation of those experts. The paper offers no theoretical guarantees or analysis of the error introduced by this aggressive masking.

### 4. Overclaiming "Closed-Form" Elegance
The authors label WTA-Sign as a "mathematically closed-form conflict resolution method." While the algorithm can be written in a closed-form formula, a formula for a heuristic is not a mathematical proof of optimality. The paper lacks any convergence proofs, bounded error guarantees, or optimization-theoretic formulations that would justify why selecting the coordinate-wise maximum absolute change is the correct mathematical way to merge neural representations.
