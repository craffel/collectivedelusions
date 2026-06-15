# 5. Impact and Presentation

## Major Strengths
* **Highly Pragmatic Motivation:** Bypassing the memory-bandwidth bottleneck of executing multiple low-rank experts on resource-constrained edge devices is a highly valuable contribution to TinyML.
* **The "Activation Dilution" Insight:** The discovery and mathematical formulation of the non-monotonic accuracy-latency trade-off (where pruning marginal experts regularizes deep representations and improves accuracy) is a highly interesting and counter-intuitive finding.
* **Outstanding Structural Clarity:** The paper is exceptionally well-structured, featuring an instructive data flow diagram, clear tables with standard deviations across 10 random seeds, and a complete notational glossary.
* **Exhaustive Appendix and Sensitivity Studies:** The appendix provides deep-dive analyses, including hardware Roofline models, closed-loop kernel driver mappings, scaling studies up to $K=24$ tasks, and hyperparameter sweeps, ensuring a complete and transparent engineering blueprint.

## Areas for Improvement
* **Enhancing Mathematical Rigor:** The mathematical appendix should address the simplifying independence assumptions in the covariance derivations and provide formal bounds on GMM support leakage and EM convergence stability.
* **Transitioning from Simulation to Physical Evaluation:** The main text should prioritize actual bare-metal physical hardware measurements (such as those from the STM32 and Joulescope profiling) over synthetic sandbox and compiler-level simulations.
* **Validating on Complex Manifolds:** Evaluating the routing and OOD mechanisms on non-linear, non-orthogonal deep representations of real-world large models (e.g., Vision Transformers or LLMs) would strengthen the paper's generalizability claims.

## Overall Presentation Quality
The presentation is **excellent**. The writing style is professional, concise, and direct. The arguments are well-developed, and the tables and figures are clean, highly informative, and academically transparent.

## Potential Impact and Significance
This work has **high potential impact** for edge computing and mobile deep learning. The training-free, plug-and-play nature of RB-TopM makes it immediately useful for practitioners deploying multi-task PEFT adapters in volatile edge environments. However, its academic impact in core machine learning theory might be limited by its highly heuristic design and lack of formal mathematical guarantees or convergence proofs.
