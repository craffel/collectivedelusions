# 5. Impact and Presentation Quality

### Strengths
1. **Mathematical Rigor**: The paper is exceptionally well-written from a mathematical perspective. The derivations are elegant, and the formalization of SVD-induced gauge issues (Kernel and Spectrum Distortion Theorems) is highly precise.
2. **Clear Writing and Visualization**: The paper is well-structured and easy to follow. Figure 1 provides a helpful flowchart of the pipeline, and the tables are comprehensive.
3. **Honesty on Failures**: The authors are commendable for their transparency. They explicitly show and analyze the catastrophic collapse of their initial idea (RIMO spectral balancing, $13.66\%$) and discuss the persistent performance gap with Task Arithmetic.

### Weaknesses (The Minimalist Lens)
1. **High Complexity, No Practical Payoff**: The paper is a textbook case of over-engineering. It introduces multiple Lie algebra operations, Cayley transforms, SVDs, Schur decompositions, complex Hermitian solvers, and training-time orthogonal constraints, yet fails to outperform a simple Euclidean linear average (Task Arithmetic).
2. **Self-Created Problems**: The core theoretical focus of the paper—the "spectral balancing pitfall"—is a mathematical problem that the authors created by attempting to perform an unnecessary spectral modification in a curved tangent space. The proposed mitigations (Schur, complex solvers, rank pruning) are layers of complexity added to fix this self-induced issue.
3. **Toy Scale**: The experiments are limited to Split-MNIST and Split-CIFAR-10 with tiny MLPs and a micro-ViT ($d=32$). There is no empirical validation on modern foundation models where model merging is a vital, practical technique.

### Potential Impact
The potential impact on the machine learning community is low. Practitioners seeking to merge models are highly unlikely to adopt a method that:
- Requires non-standard training-time orthogonal constraints.
- Introduces multiple heavy matrix decompositions (SVD/Schur/Complex Eigen-decomposition).
- Yields inferior performance compared to simple Euclidean averaging.
