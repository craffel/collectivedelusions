# 3. Soundness and Methodology Check

## Soundness of Technical Claims
The technical claims of the paper are mathematically and empirically sound, well-supported, and rigorously developed. The authors clearly formulate the optimization objectives, initialization strategies, and evaluation parameters. The core hypothesis—that classical routing collapse in the sandbox is an artifact of poor baseline setup (specifically, lack of proper initialization and regularization)—is evaluated systematically under controlled conditions.

Furthermore, the final version of the manuscript has successfully addressed and resolved several key critical nuances and potential areas of critique:

### 1. The "ICS Sandbox" as a Closed, Linear System
The 14-layer Analytical Coordinate Sandbox (ICS) is an elegant, coordinate-based simulation where activations are iteratively updated via:
$$h_b^{(l)} = h_b^{(l-1)} + \sum_{k=1}^K \alpha_{k, b} \gamma_V (v'_k - h_b^{(l-1)})$$
While this linear coordinate-attraction dynamical system allows for precise tracking of semantic trajectory, it lacks the complex, highly non-linear representation mappings characteristic of real-world deep neural networks (e.g., non-linear activations like GeLU/ReLU, layer norms, and multi-head attention projections). The authors honestly discuss this in Section 5.3 as a generalizability limitation. By modeling representation propagation as a simple attraction vector pulling toward task prototypes, the sandbox oversimplifies the geometric trajectories of real foundation models, making linear routing boundaries artificially clean.

### 2. Hand-Crafted Unbalanced Noise Structure
The sandbox is parameterized with task-specific noise standard deviations $\sigma = [0.05, 0.15, 0.40, 1.20]$. 
- This means Task 0 (MNIST) has virtually no noise ($\sigma = 0.05$), while Task 3 (SVHN) features extreme noise ($\sigma = 1.20$).
- This highly unbalanced noise structure is a key reason why ChemMerge's stateful low-pass filter dynamics are highly effective. When noise is extremely high and asymmetric, a temporal low-pass filter prevents ensembling weights from fluctuating wildly.
- The authors have added a detailed discussion in Section 5.2 analyzing how a realistic, balanced noise profile (where all experts achieve $>80\%$ accuracy) would narrow the ensembling stabilization premium, making the simpler stateless classical router the globally optimal, latency-efficient choice. This demonstrates superb scientific honesty.

### 3. Discretization Instability of the ChemMerge ODE Solver
The continuous-time kinetics in ChemMerge are formulated as an ODE. In practice, this is solved via a discretized Euler step:
$$C_{next} = C + \Delta t \left( R(1 - C) - K_{\text{decay}} C \right)$$
Under the sandbox configuration, the authors use a step size of $\Delta t = 1.5$ and decay rate $K_{\text{decay}} = 0.3$. 
- Mathematically, in standard numerical analysis, an Euler step size of $\Delta t = 1.5$ is extremely large and highly prone to overshoot and numerical instability. For instance, if $C = 0$ and $R = 1$, then $C_{next} = 1.5 > 1.0$.
- To prevent this instability, the code applies a hard clamp: `C = torch.clamp(C_next, 0.0, 1.0)`.
- The paper now explicitly acknowledges this large Euler step size as a numerical hazard prone to overshoot, which necessitates an ad-hoc hard-clamping "numerical hack" in practice. This further demystifies the continuous kinetics metaphor, exposing it as a hand-crafted discrete heuristic rather than a pure physical simulation.

### 4. Direct Classifier Logit Blending Incompatibility
In the BERT-Tiny implementation, the joint serving model evaluates joint predictions by taking a weighted sum of the task-specific classifiers' logit outputs:
$$\text{logits} = \alpha_0 \cdot \text{classifier}_0(\text{pooled}) + \alpha_1 \cdot \text{classifier}_1(\text{pooled})$$
- This design assumes that both classifiers output logits of the exact same dimensionality (2 output classes). 
- If Task 0 was a 2-class classification (SST-2) and Task 1 was a 3-class classification or 10-class classification, this blending equation would crash immediately with a shape mismatch error.
- The authors have added a dedicated paragraph "Architectural Limitation of Direct Logit Blending" in Section 4.9, outlining this exact shape mismatch constraint and discussing how realistic multi-task serving would require routing inputs to their respective heads without blending or implementing dynamic label adapters.

### 5. Resolution of the $N_{\text{cal}} = 32$ Low-Data Discrepancy
In previous drafts, there was an empirical contradiction where the unregularized router outperformed all other methods under $N_{\text{cal}} = 32$ on BERT-Tiny, which seemed to contradict the "overfitting bottleneck" claim.
- The authors have resolved this beautifully in Section 4.9. They explicitly discuss the $N_{\text{cal}} = 32$ results and explain them through **task geometry and representation separability**.
- Because SST-2 and QQP are semantically completely disjoint (movie reviews vs duplicate questions), their pre-trained token representations map to highly separated and non-overlapping subspaces within BERT-Tiny. Consequently, a simple linear parametric router with only $2 \times 128 = 256$ parameters can easily learn a clean separating boundary on 32 samples without overfitting.
- This represents a highly valuable, nuanced scientific explanation: the synthetic "overfitting bottleneck" is highly dependent on task selection/alignment and does not generalize to disjoint multi-task regimes. It demonstrates that proper regularization or training-free inductive priors are primarily mandatory when task boundaries are densely entangled or representations exhibit massive geometric overlap.

## Mathematical Precision and Notation
The math is exceptionally precise, clean, and perfectly aligned with the codebase:
- ChemMerge's continuous ODE kinetics, discretization, and normalization are mathematically formalized.
- Standard zero-initialization and L2 weight decay are explicitly defined early in the methodology.
- Trajectory Jitter is formally defined mathematically as the mean L2-norm of adjacent-layer blending weight differences.
- Gating functions (competitive Softmax vs. cooperative Sigmoid) are cleanly detailed.
