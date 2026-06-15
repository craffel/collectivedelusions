# Intermediate Evaluation 2: Novelty Check

## Assessment of Key Novel Aspects
The paper's core novelty is conceptual and analytical rather than algorithmic. It introduces **parsimony** and **Occam's razor** to stateful model merging.
1. **Simplification of Stateful Routing:** Prior SOTA (ChemMerge) argued that modeling ensembling weights as concentrations in a chemical reactor with Arrhenius rates and ODE solvers was necessary for stateful routing. The paper's most novel aspect is the mathematical deconstruction showing that this complex physical metaphor is equivalent to a classical Exponential Moving Average (EMA).
2. **Raw Boundary Initialization:** The proposed initialization of the stateful recurrence at the first adapted layer's similarity weight ($w_k^{(L_{\text{frozen}}+1)}$) instead of a uniform $1/K$ prior is a small but highly effective novel addition. This is shown to eliminate transient startup jitter and reduce routing jitter by 70.1$\times$.
3. **V-shaped Momentum Scheduling & Adaptive Specificity:** The exploration of a depth-wise momentum schedule (applying higher inertia at the boundaries and higher plasticity at the middle semantic bottleneck) and its dynamic, variance-based estimation is a highly insightful extension of stateful smoothing.

## The 'Delta' from Prior Work
- **From ChemMerge (SOTA stateful):** The delta is the complete removal of the biochemical ODE system, Arrhenius rate equations, virtual time steps, and numerical solvers. Momentum-Merge replaces this entire pipeline with a single line of standard code (the EMA equation) while achieving comparable or superior accuracy and significantly lower routing jitter.
- **From SABLE (SOTA stateless):** The delta is the addition of temporal statefulness via depth-wise momentum smoothing, which suppresses high-frequency routing jitter (reducing it by up to 195.7$\times$), and the introduction of layer-wise centroid anchoring to account for representational rotation across depth.

## Characterization of Novelty
The novelty of this paper is **significant but conceptually reductive**. 
Rather than introducing a highly complex new method, the paper makes a profound contribution by **deconstructing and simplifying** an existing complex method, proving that a classical, well-understood baseline (EMA) actually outperforms the convoluted state-of-the-art. 

From a theory-minded perspective, this type of novelty is highly valuable. It exposes a "pseudo-physical" metaphor in deep learning as redundant complexity and establishes a clean, mathematically grounded baseline. The addition of Layer-wise Centroid Anchoring and Raw Boundary Initialization further refines the theoretical understanding of representational flow and recurrence initialization in deep networks.
