# Evaluation of Soundness and Methodology

## Clarity of Description and Reproducibility
* **Clarity:** The paper is exceptionally well-written, with highly structured sections, clear mathematical formulations, and a complete code snippet for the `PolyMergeGenerator` in the Appendix. The narrative is easy to follow, and the terminology is precise.
* **Reproducibility:** The pseudocode, mathematical formulations of the emulator, and hyperparameter listings provide a good foundation for reproducibility. However, because the primary results (Table 1) are generated inside a custom-calibrated synthetic simulator, exact replication depends on the specific, unreleased codebase of this simulator.

## Major Technical Flaws and Methodological Weaknesses

### 1. Unrealistic Additive Noise Model in the Mathematical Proof
The mathematical proof for Proposition 3.1 (Appendix B.1) assumes that the transductive noise $\boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ is added **directly** to the coefficient vector:
$$\boldsymbol{\lambda}_k = \boldsymbol{\lambda}^*_k + \boldsymbol{\eta}_k$$
This is a highly flawed and unrealistic assumption:
* In test-time adaptation, noise does not enter as a random, zero-mean white noise vector added directly to the parameters. 
* Instead, noise arises from the **stochasticity and distribution shift of the unlabeled target data stream** (e.g., small batch size, label imbalance). It enters the optimization trajectory through the **gradients of the non-linear, unsupervised loss function** (predicted Shannon entropy).
* Because the relationship between the data stream, the model's logits, the entropy loss, and the merging coefficients is highly non-linear, the linear projection argument ($\mathbf{P}\boldsymbol{\eta}$) does not hold in a physical setting. 
* Representing transductive noise as a simple additive spatial vector allows for an elegant trace-reduction proof, but this proof lacks physical meaning and scientific validity for actual deep neural networks.

### 2. Ridiculously Small Sample Sizes in Physical Validations
While the paper includes physical validations (Sections 4.6 and 4.7) to address the limitations of the synthetic simulator, the scale of these physical experiments is so small that it borders on trivial:
* **DeepResMLP Validation (Section 4.6):** The TTA stream consists of exactly **24 unlabeled samples** (12 per task). Drawing conclusions about the generalization of test-time adaptation from a dataset of 24 samples is statistically highly suspect and prone to high-variance artifacts.
* **CLIP Foundation Model Validation (Section 4.7):** The test stream consists of only **50 images** from CIFAR-10 and **50 images** from GTSRB, optimized for only 15 steps. On a stream of 50 images, a change in a single sample represents a massive 2.0% shift in accuracy.
* Evaluating an 86-million-parameter Vision Transformer on 50 images does not represent a realistic test-time adaptation deployment, which typically runs continuously over thousands of streaming samples. These "toy" physical setups are insufficient to validate the claims of a robust, general-purpose regularizer.

### 3. Severe Underfitting of the Primary "PolyMerge" Method
In the physical CLIP validation (Table 4), the primary proposed method, **PolyMerge ($d=2$, Quadratic)**, suffers from a catastrophic **12% accuracy drop** on CIFAR-10 compared to the static Task Arithmetic baseline (dropping from $92.00\%$ to $80.00\%$).
* This is a massive performance regression. The authors attempt to soften this finding by calling it an "underfitting bottleneck of global polynomials" and recommending SplineMerge instead.
* However, if the global polynomial parameterization (which is the main title and core contribution of the paper, "PolyMerge") actively degrades performance on a simple task like CIFAR-10, then the core thesis of the paper is heavily compromised.
* SplineMerge recovers the performance ($92.00\%$), but as discussed in the novelty check, SplineMerge (Piecewise Constant) is conceptually equivalent to simple block-wise merging—a baseline that has been widely used in model-merging literature.

### 4. Over-Engineered and Non-Convex "Stress-Test" Simulator
In Section 3.5.2, the authors introduce "Model II," which incorporates a "Non-Convex Rastrigin Loss Landscape" and "Multi-Scale Overfitting Noise." 
* This appears to be an extremely over-engineered synthetic testbed. Designing a custom loss function with sinusoidal ripples (Rastrigin function) and Brownian walk noise to "emulate" deep learning loss landscapes is highly artificial.
* In physical neural networks, the non-convexities are shaped by the architecture (skip connections, attention maps) and the data distribution. Adding arbitrary mathematical functions (like cosines) to a distance loss does not make the simulation "physically grounded."
* It creates a highly artificial environment specifically tailored to show that unconstrained gradient descent fails (as Rastrigin has thousands of local minima) and that smoothing (PolyMerge) succeeds by bypassing high-frequency ripples. This undermines the scientific credibility of the "Finding 4" claims.

### 5. Missing Baselines in Physical Validation
In both physical validation tables (Table 3 and Table 4), the paper fails to compare against standard state-of-the-art model merging baselines, such as **TIES-Merging** or **RegMean**. 
* They only compare against Task Arithmetic, unconstrained AdaMerging, and TV-regularized Adam.
* To prove that PolyMerge/SplineMerge is a viable framework, the authors must compare against these established methods on the physical datasets, rather than limiting the comparison to simple Task Arithmetic.
