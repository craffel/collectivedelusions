# Intermediate Evaluation 3: Soundness and Methodology

## 1. Clarity of Description
The methodology is presented with a high degree of clarity, mathematical precision, and logical structure:
- **Problem Formulation (Section 3.1)**: Clearly defines the base model, task-specific experts, and task vectors.
- **Input Phase State Extraction (Section 3.2)**: Formulates how the input representations are projected onto a low-dimensional phase-space using a frozen random projection matrix $P$ and normalized to a unit sphere.
- **Quantum Phase-Coherent Overlap (Section 3.3)**: Fully details the layer-wise cosine projection mechanism, including learned scaling amplitudes $R_k^{(l)}$, phase offsets $\phi_k^{(l)}$, and the fixed wave frequency scaling factor $\omega = \pi$.
- **Wavefunction Collapse and Weight Measurement (Section 3.4)**: Explicitly defines how sample-level routing coefficients are averaged across the batch dimension to form a single batch-level classical weight matrix, resolving a critical accelerator throughput bottleneck.
- **Linear Router Baseline (Section 3.5)**: Formally defines the classical soft-routing baseline, explaining its unconstrained parameters and why it differs from QWS-Merge.

All symbols are well-defined, and the dimensions of all vectors and matrices are mathematically consistent.

## 2. Appropriateness of Methods
The choice of methods is highly appropriate for the problem at hand:
- **Low-dimensional space ($d=4$)**: Keeping the phase-space dimension low and matching the number of tasks restricts the projection to a highly constrained manifold, preventing overfitting on small calibration sets.
- **Frozen Random Projection**: Projecting the high-dimensional feature representations ($D=192$) to a lower dimension ($d=4$) using a frozen random projection matrix is a parameter-free, lightweight way to reduce dimensionality without introducing extra learnable parameters.
- **Cosine Wave Projection**: The use of a non-monotonic cosine function provides a bounded, regularized subspace. Unlike a classical softmax or linear layer which can grow unbounded or shift aggressively with tiny parameter changes, the cosine projection restricts coefficient values to $[-R, R]$, acting as a strong, physically-grounded regularizer.
- **Tiny Validation Set (64 samples)**: Standard optimization in model merging often suffers from the Overfitting-Optimizer Paradox. Restricting the parameter space to just $336$ parameters allows optimization on 64 samples to succeed without transductive overfitting, which is highly appropriate for low-resource deployment.

## 3. Potential Technical Flaws, Limitations & Discussion
While the method is mathematically sound, there are several key technical considerations and limitations:

### A. The "Quantum" Analogy vs. Classical Mechanics
The paper relies heavily on quantum mechanics terminology (Hilbert space, eigenstates, wavefunction superposition, collapse, measurement). Mathematically, however, the formulation is a classical input-dependent routing network that uses a specific cosine projection and batch-averaging mechanism.
- *Discussion*: While the quantum analogy is an inspiring, highly creative, and successful metaphor that guides the design, the authors should be careful to make it explicit that no actual quantum mechanics or complex-valued numbers are used in the implementation (e.g., probability amplitudes here can be negative, whereas in quantum mechanics, the probability is the square of the absolute value of a complex-valued amplitude, ensuring it is always non-negative).

### B. Batch Dependency and I.I.D. Violation
The "wavefunction collapse" averages coefficients across the batch (Equation 8).
- *Discussion*: This is a practical compromise to avoid reconstructing different weight matrices for each individual sample in a batch (which would require batch size $B$ forward passes through separate weights, destroying GPU parallelism). However, this introduces **batch dependency**, violating the independent-and-identically-distributed (I.I.D.) assumption of standard machine learning. A sample's prediction depends on which other samples are present in its batch, leading to prediction inconsistency. The authors honestly and transparently address this in Section 4.5, which is commendable, but it remains a notable limitation for single-sample stream deployment ($B=1$).

### C. Extreme Conflict vs. Low-Conflict Trade-off
Under low-conflict datasets (MNIST, FashionMNIST), the classical Linear Router achieves higher performance than QWS-Merge ($91.20\%$ vs $77.60\%$ on MNIST).
- *Discussion*: The heavy wave regularization of QWS-Merge bounds its capacity, resulting in a slight performance penalty on simpler tasks compared to an unconstrained linear router. This Capacity-Regularization trade-off is mathematically expected but should be recognized as a limitation when maximum peak performance on simple tasks is desired.

## 4. Reproducibility
The reproducibility of the paper is **excellent**:
- The paper details the exact parameter count (336 parameters) and their initialization (amplitudes initialized to 0.3, phase offsets to 0.0).
- The training routine is straightforward: AdamW on the calibration set for 100 steps.
- The hyperparameters (such as $\omega = \pi$) are explicitly specified.
- The model backbone ($\mathtt{vit\_tiny\_patch16\_224}$) and dataset configurations are standard, making it easy for an expert to replicate the results.
