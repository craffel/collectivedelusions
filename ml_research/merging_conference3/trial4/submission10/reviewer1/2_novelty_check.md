# Intermediate Evaluation 2: Novelty Check

## 1. Key Novel Aspects & Conceptual Leaps
The primary novelty of this paper lies in its **bold conceptual leap**—shifting the fundamental assumption of parameter-space model merging from **static, input-independent averaging** to a **dynamic, quantum-inspired parameter wave superposition**.

Specifically, the paper introduces several highly original mechanisms:
1. **Model Weights as Task Eigenstates**: Modeling specialized network checkpoints as eigenstates $|\psi_k\rangle$ in a parameter Hilbert space.
2. **Coherent Parameter Wave Superposition**: Instead of simple linear weight blending, the model leverages wave-like phase-interference. Merging coefficients are generated as dynamic probability amplitudes determined by the alignment (cosine inner product) between an input phase state and a set of learned, layer-wise, task-specific phase-basis vectors.
3. **Physical-Subspace Regularization**: The non-monotonic nature of the cosine wave function provides a highly bounded, robust projection manifold. This represents a distinct departure from unconstrained linear or softmax routing mechanisms.
4. **Wavefunction Collapse as Batch Measurement**: Resolving sample-wise accelerator bottlenecks by averaging sample-level amplitudes across the batch, interpreting it as a wavefunction collapse to a batch-level classical weight state.

## 2. Delta from Prior Work
The proposed method stands apart from two primary lines of existing literature:

### A. Static Parameter Merging (e.g., Task Arithmetic, TIES, DARE, OFS-Tune)
- **Prior Work**: These methods find a single, fixed set of weights to represent the merged model. Even supervised static calibration methods like OFS-Tune only search for static coefficients.
- **Delta**: QWS-Merge is **sample-conditioned and dynamic**. It evaluates each input stream on-the-fly and assembles different classical parameter configurations on a batch-level or sample-level basis. This allows a single compact backbone to support highly conflicting tasks simultaneously without representational cancellation.

### B. Classical Dynamic Routing & Test-Time Adaptation (e.g., AdaMerging, Linear Routers, MoE)
- **Prior Work**: Methods like AdaMerging use unsupervised online entropy minimization to find coefficients, which is prone to drift. Mixture-of-Experts (MoE) dynamically routes input tokens but requires scaling the physical size of the model. Classical routing (e.g., our Linear Router baseline) uses unconstrained linear projections mapping representations to routing weights.
- **Delta**: QWS-Merge uses a highly regularized, low-dimensional (d = 4) phase state space. Its routing parameters (336 in total) are bounded on a unit sphere and projected via a non-monotonic cosine function. This is mathematically and conceptually distinct from unconstrained linear routing. It provides localized, layer-wise specialization while remaining robust to overfitting in data-scarce (few-shot) regimes.

## 3. Characterization of Novelty
The novelty of this work is **significant, creative, and highly original**. It is not a marginal or incremental enhancement (e.g., slightly modifying scaling coefficients or adding heuristic alignment losses). Instead, it introduces an entirely new physical paradigm to parameter space:

- **Conceptual Originality**: The application of quantum-like phase coherence to resolve weight conflicts in compact neural networks is highly creative and intellectually ambitious. It forces the community to think about parameter merging as a wave-interference and state-collapse problem rather than a standard static optimization compromise.
- **Methodological Innovation**: Designing layer-wise, low-dimensional phase-basis projections that act as a regularized coordinate system is an elegant solution to the Overfitting-Optimizer Paradox that plagues test-time adaptation.

This level of conceptual ambition represents a substantial advancement that has the potential to inspire a new class of "physics-inspired" or "wave-inspired" dynamic merging techniques.
