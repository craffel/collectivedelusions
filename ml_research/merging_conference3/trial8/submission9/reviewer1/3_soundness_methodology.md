# Evaluation Phase 3: Soundness and Methodology

## 1. Clarity of Description
The methodology is written clearly and uses precise algebraic notation to outline the proposed algorithms. The step-by-step description of Zero-Shot Expert Entropy Routing (EER), Entropy-Pseudo-Labeled Online Centroid Adaptation (EPL-OCA), Centroid-Gated Entropy Routing (CG-EER), and the complexity and systems-level profiling under post-activation divergence is easy to follow.

## 2. Theoretical Soundness and Gaps
While the methodology is clearly described, it suffers from significant theoretical gaps and a lack of rigorous mathematical analysis:

- **Complete Lack of Formal Mathematical Guarantees:** 
  The paper contains **no theorems, lemmas, propositions, or formal mathematical proofs**. Despite the authors' use of terms like "mathematically and empirically motivating" and "formally demonstrate," the mathematical content is restricted to algebraic restatements of their algorithms and simple arithmetic calculations for computational complexity. 

- **No Theoretical Justification for prediction entropy as a task routing metric:**
  The central premise of EER—that minimum prediction entropy corresponds to the correct task expert—is a heuristic assumption. There is no formal theoretical framework establishing that the correct expert's prediction entropy is bounded below the entropy of incorrect experts under any reasonable assumptions (such as well-separated task distributions, bounded classification error, or Lipschitz continuity of representation manifolds). 
  Indeed, the paper's own empirical results on real-world ResNet-18 embeddings *disprove* this heuristic assumption! The authors identify the **Entropy Calibration Discrepancy**, where simpler experts (like MNIST) exhibit severe OOD overconfidence, predicting on complex domains (like SVHN or CIFAR-10) with lower entropy than on their own in-distribution data. Because the paper lacks a theoretical model of this discrepancy, it patches the issue empirically using a semi-supervised gating mechanism (CG-EER) that requires pre-computed offline centroids, undermining the goal of complete calibration-freedom.

- **No Convergence or Stability Analysis of the Self-Referential Pseudo-Label Loop:**
  In EPL-OCA, the running centroids are updated on-the-fly using EER pseudo-labels, which are in turn used for routing. This creates a self-referential feedback loop. 
  On real features, this loop catastrophically collapses (as demonstrated by EPL-OCA Hard collapsing to 27.45%, EPL-OCA Soft collapsing to 31.52%, and UCG-EER collapsing to 28.45%). The authors empirically describe this failure mode but do not provide any analytical or mathematical model of this instability. A theoretically rigorous treatment would formulate this online update as a stochastic approximation or dynamical system and analyze the stability conditions or convergence bounds under representation shift and calibration discrepancy.

- **Under-Formalized "Representational Sparsity Paradox":**
  The authors coin this phrase to explain why centroid-based ensembling (EPL-OCA) performs poorly in the synthetic sandbox. They state that class orthogonality within task subspaces introduces high spatial sparseness, causing running centroids to jitter. However, this is an intuitive, qualitative description. A rigorous theoretical reviewer would expect a formal proof or bound showing how the variance of the running centroid is bounded as a function of the dimensionality and class orthogonality, and how this variance translates to a bound on the cosine routing error.

- **Systems-Level Simplifications:**
  The computational complexity analysis ($0.25 + 0.75K$ passes) is neat and the amortization formula ($1.0 + \frac{0.25 + 0.75K}{N_{\text{amortize}}}$) makes sense, but it is a deterministic calculation that assumes perfect synchronization and no memory bottlenecking in parallel hardware. Although the energy analysis is interesting, it is purely theoretical and relies on arbitrary hardware assumptions ($0.5\text{W}$ at $1.5\text{GHz}$, $50\text{pJ}$ for DRAM, $1\text{pJ}$ for SRAM).

## 3. Reproducibility
The mathematical description of the algorithms is complete enough to allow reproduction of the code and setup. The parameters ($\beta$, $\tau = 0.001$ for hard routing, $\tau = 0.5$ for soft blending, gating threshold $\delta \ge 0.7$) are explicitly provided.
