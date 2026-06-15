# 3. Soundness and Methodology

## Clarity of the Description
The mathematical and algorithmic description of **ThermoMerge** is exceptionally clear, detailed, and structured:
* The paper provides an explicit, step-by-step algorithm (Algorithm 1) detailing the entire test-time adaptation loop.
* The preconditioned SGLD (Adam-SGLD) update equations (Equations 17, 18, 19) are mathematically rigorous and appropriately scaled according to the Fluctuation-Dissipation theorem.
* Complex implementation details—such as freezing normalization layers, pre-allocating noise buffers to prevent PyTorch memory fragmentation, and performing layer-wise functional grouping of weights and biases—are described with outstanding transparency.

## Appropriateness of Methods
* **SGLD for Non-Convex Landscapes**: Modeling the test-time model merging loss surface as non-convex and using Stochastic Gradient Langevin Dynamics (SGLD) as an exploratory global optimization mechanism is highly appropriate. Isotropic noise injection is a well-established method to escape local minima.
* **Dimensionality-Scaled Langevin Noise (DSLN)**: The proposed DSLN is an elegant and necessary solution. High-dimensional parameters (like classification heads) would experience a catastrophic thermal "boiling" effect under standard coordinate-wise SGLD because the aggregate noise norm scales with the dimension ($d$). Scaling the coordinate-wise standard deviation by $1/\sqrt{d_j}$ maintains a dimension-invariant expected total kinetic energy ($2\eta T_t$), which stabilizes joint optimization.
* **Layer-wise Functional Parameter-Group Scaling**: Grouping weights and biases of the same functional layer to avoid a "thermodynamic imbalance" (where biases would be perturbed orders of magnitude more heavily than weights due to their low dimensionality) is a theoretically sound and highly practical stabilization technique.

## Potential Technical Flaws and Critical Concerns
1. **The Gap Between Synthetic Simulation and Deep Learning**:
   * The synthetic 1D loss landscape (Equation 22) is highly engineered to have a sharp trap at $\Lambda=0.2$ and a wide flat minimum at $\Lambda=0.6$, decorated with high-frequency sinusoidal ripples. While this serves as a clear proof of concept for ThermoMerge's global search capability, it represents an artificial and simplified landscape.
   * The "Specific Heat Capacity Peak" at $T_c \approx 0.02$, while conceptually beautiful, is only numerically computable on this 1D toy simulation. As the authors admit in Section 4.5, computing the partition function $Z(T)$ and thermodynamic quantities is strictly intractable in deep neural networks with thousands or millions of dimensions. Thus, the "thermodynamic phase transition" is a toy-scale mathematical demonstration and does not strictly translate to real deep networks, where the temperature decay is simply a hyperparameter schedule.
2. **Vulnerability to Teacher-Bias & Confirmation Bias**:
   * The self-labeling proxy loss $\mathcal{L}_{TT}$ relies on fixed, unmerged expert predictions. If these experts make systematic errors, are highly uncertain, or are biased on difficult or out-of-distribution test samples, SGLD will reinforce these incorrect predictions (confirmation bias).
   * While the authors propose three mitigation strategies (Confidence-Based Filtering, Entropy-Based Weighting, and Predictive Agreement Monitoring) in Section 3.2, they are mostly presented as speculative conceptual recommendations. The paper lacks systematic empirical tables in Section 4 evaluating these strategies on the deep learning benchmarks, making it unclear how effective they are in practice under varying levels of teacher noise.
3. **Hyperparameter Calibration Complexity**:
   * The initial temperature calibration heuristic (Equation 12) depends on estimating the expected gradient norms during the first few adaptation steps. In a streaming test-time environment, if the first few batches are highly non-representative or contain outliers, this calibration could be unstable, potentially leading to an excessively high or low $T_0$.

## Reproducibility
The overall reproducibility is rated as **high**:
* Every formula, hyperparameter value, and cooling schedule rate is transparently documented.
* The paper includes concrete PyTorch code snippets for efficient noise buffer pre-allocation.
* The experimental setups for both the 1D simulation and the deep learning (MLP & LoRA) tasks are clearly explained, down to the network layers, task splitting, and test-time streaming batch protocols.
