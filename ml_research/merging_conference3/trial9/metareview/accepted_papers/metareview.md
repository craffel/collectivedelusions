# Meta-Review and Acceptance Report

**Date:** June 15, 2026  
**Conference:** International Conference on Machine Learning (ICML) 2026 - Track on Model Merging, Ensembling, and Dynamic Serving  

---

## 1. Meta-Review Process Overview

The goal of this meta-review is to evaluate 10 paper submissions focused on the active and high-impact area of dynamic, test-time model ensembling and activation-space expert serving (specifically, utilizing parameter-efficient experts like LoRA). 

Each submission has been evaluated by up to three independent peer reviewers. In this meta-review process, we conducted a systematic, dual-level analysis:
1. **Quantitative Evaluation:** Aggregating and analyzing the numeric ratings and score profiles of each submission to establish a primary ranking.
2. **Qualitative Content Audit:** Critically reading the individual reviews to evaluate the scientific substance, mathematical soundness, experimental rigor, and practical significance of each paper. 

Based on the instructions, we have selected exactly **three submissions** to accept. When resolving ties or borderline cases, we prioritized the theoretical integrity of the proofs, the realism and ecological validity of the experimental benchmarks, and the presence of high-signal contributions that apply Occam's razor to demystify complex or "metaphorical" architectures.

---

## 2. Comprehensive Cohort Summary and Rankings

Below is the aggregated summary of all 10 submissions, ordered by their average peer-review score.

| Rank | Submission ID | Paper Title | Active Reviewers | Reviewer Scores | Average Score | Meta-Decision |
| :---: | :---: | :--- | :---: | :--- | :---: | :---: |
| **1** | **4** | **Momentum-Merge** | Reviewer 1, 2, 3 | R1: `5 (Accept)`, R2: `6 (Strong Accept)`, R3: `5 (Accept)` | **5.33** | **ACCEPT** |
| **2** | **9** | **PAC-Kinetics** | Reviewer 1, 2, 3 | R1: `6 (Strong Accept)`, R2: `3 (Weak Reject)`, R3: `5 (Accept)` | **4.67** | **ACCEPT** |
| **3** | **5** | **Methodological Audit & Deconstruction of Dynamic Activation-Space Model Merging** | Reviewer 1, 2, 3 | R1: `5 (Accept)`, R2: `5 (Accept)`, R3: `3 (Weak Reject)` | **4.33** | **ACCEPT** |
| **4** | **2** | **Resource-Budgeted Top-$M$ Expert Serving: Dynamic Activation-Space Gating for Low-Power Edge Model Merging** | Reviewer 1, 2, 3 | R1: `5 (Accept)`, R2: `3 (Weak Reject)`, R3: `5 (Accept)` | **4.33** | REJECT |
| **5** | **1** | **PAC-Bayesian Smooth Trajectory Merging (PAC-STM)** | Reviewer 2, 3 | R2: `5 (Accept)`, R3: `3 (Weak Reject)` | **4.00** | REJECT |
| **6** | **3** | **Contraction-Regularized Router (CR-Router) for Fixed-Point Convergence** | Reviewer 1, 2, 3 | R1: `2 (Reject)`, R2: `5 (Accept)`, R3: `5 (Accept)` | **4.00** | REJECT (Math Error) |
| **7** | **10** | **Dirichlet-PAC: Simplex-Constrained PAC-Bayesian Complexity Control** | Reviewer 1, 2, 3 | R1: `5 (Accept)`, R2: `2 (Reject)`, R3: `5 (Accept)` | **4.00** | REJECT (Empirical Failure) |
| **8** | **6** | **Continuous Riemannian-Geometric Homotopical Model Merging via Grassmannian Geodesic Blending (C-Lie-MM)** | Reviewer 1, 2, 3 | R1: `3 (Weak Reject)`, R2: `4 (Weak Accept)`, R3: `3 (Weak Reject)` | **3.33** | REJECT |
| **9** | **7** | **Lyapunov-Stable Active Representation Coupling for Dynamic Model Serving (L-ARC)** | Reviewer 1, 2, 3 | R1: `3 (Weak Reject)`, R2: `2 (Reject)`, R3: `5 (Accept)` | **3.33** | REJECT |
| **10** | **8** | **GraviMerge** | Reviewer 1, 2, 3 | R1: `2 (Reject)`, R2: `3 (Weak Reject)`, R3: `5 (Accept)` | **3.33** | REJECT (Evaluation Gap) |

---

## 3. Individual Submission Analysis

### Submission 1: PAC-Bayesian Smooth Trajectory Merging (PAC-STM)
* **Active Reviewers:** Reviewer 2, Reviewer 3
* **Ratings:** R2: `5: Accept` | R3: `3: Weak Reject` | **Average:** `4.00 / 6`
* **Strengths:** High mathematical beauty; connects continuous depth-wise Gaussian random walk priors analytically to a first-order finite-difference smoothness regularizer (Theorem 3.1). Introduces local uncentered Kernel PCA projections to preserve task centroid identity.
* **Weaknesses:** Highly vulnerable to severe citation and scholarly attribution gaps (does not cite core evaluated baselines like SABLE and PAC-ZCA, nor does it cite contemporary PAC-Bayesian model-merging works like Kim et al., UAI 2026). More critically, its empirical results are limited to image classification on a ViT-B/16 backbone where it achieves the *exact same* classification accuracy (86.25%) as unregularized baselines, showing zero practical performance advantage on real-world activations.
* **Rigor:** Highly solid theoretical exposition, but very weak real-world validation and poor literature contextualization.

### Submission 2: Resource-Budgeted Top-$M$ Expert Serving
* **Active Reviewers:** Reviewer 1, Reviewer 2, Reviewer 3
* **Ratings:** R1: `5: Accept` | R2: `3: Weak Reject` | R3: `5: Accept` | **Average:** `4.33 / 6`
* **Strengths:** Outstanding Systems-ML co-design. Uses a detailed Roofline Model to demonstrate that expert serving is strictly memory-bandwidth bound on edge devices and models DRAM fetch reductions. Validated on a compiled TVM pilot using MobileNetV3-Large on DomainNet across 10 random seeds, achieving a 17.5% real system latency reduction.
* **Weaknesses:** The theoretical proofs in the appendix rely on highly unrealistic shortcuts, such as modeling complex, non-linear activation features as isotropic white Gaussian noise and assuming mutual independence where strong noise coupling exists. Additionally, fitting a standard GMM over a strictly bounded domain $[-1, 1]^K$ is mathematically inconsistent.
* **Rigor:** Exceptional systems engineering and physical edge profiling, but compromised by shortcuts in its mathematical/statistical derivations.

### Submission 3: Contraction-Regularized Router (CR-Router) for Fixed-Point Convergence
* **Active Reviewers:** Reviewer 1, Reviewer 2, Reviewer 3
* **Ratings:** R1: `2: Reject` | R2: `5: Accept` | R3: `5: Accept` | **Average:** `4.00 / 6`
* **Strengths:** Grounded in Banach's Fixed-Point Theorem. Proposes Update-Space Quasi-Contractions and Test-Time Temperature Annealing to enforce stability in recurrent ensembling routing.
* **Weaknesses:** **Fatal Mathematical Error.** Reviewer 1's detailed proof check exposes that under the paper's own sandbox hyperparameters, the derived global contraction condition simplifies to requiring a spectral norm less than a negative number ($\|W_{\text{route}}^{(l)}\|_2 < -19.5 \tau_l$), which is mathematically impossible. Furthermore, at test time, temperature annealing ($\tau_l \to 0$) completely discards the regularized stability guarantees, and the router suffers up to a 17% absolute accuracy drop compared to simple non-parametric baselines.
* **Rigor:** Though highly mathematical in tone, it is severely flawed due to fundamental theoretical contradictions and severe empirical underperformance.

### Submission 4: Momentum-Merge
* **Active Reviewers:** Reviewer 1, Reviewer 2, Reviewer 3
* **Ratings:** R1: `5: Accept` | R2: `6: Strong Accept` | R3: `5: Accept` | **Average:** `5.33 / 6`
* **Strengths:** A powerful lesson in mathematical parsimony (Occam's razor). Deconstructs the highly complex, ODE-based *ChemMerge* SOTA framework and proves mathematically that its convoluted kinetics are equivalent to a simple, lightweight Exponential Moving Average (EMA). Reduces routing jitter by up to $195.7\times$ over SABLE and $41.1\times$ over ChemMerge with zero systems or solver overhead. Tested extensively across 10 random seeds with paired t-tests.
* **Weaknesses:** Evaluation is restricted to the synthetic Analytical Coordinate Sandbox (ICS). Dynamic scheduling of momentum in Appendix D re-introduces some system complexity.
* **Rigor:** Outstanding. It is a highly rigorous, conceptually elegant, and practically important paper that replaces unnecessary complexity with standard, efficient mathematical primitives.

### Submission 5: Methodological Audit & Deconstruction of Dynamic Activation-Space Model Merging
* **Active Reviewers:** Reviewer 1, Reviewer 2, Reviewer 3
* **Ratings:** R1: `5: Accept` | R2: `5: Accept` | R3: `3: Weak Reject` | **Average:** `4.33 / 6`
* **Strengths:** Performs an invaluable service to the community by exposing that contemporary dynamic model merging baselines are compared against under-regularized, uncalibrated "straw-man" classical baselines. Shows that a simple, properly regularized linear gating head with Zero-Initialized Softmax and L2 regularization is highly competitive and often superior. Outstandingly clear, deconstructing ChemMerge's ODE kinetics as a closed-loop low-pass filter and exposing a hidden concentration-clamping hack in its solver.
* **Weaknesses:** Lacks formal mathematical proofs (generalization or stability bounds). Real-world validation is restricted to BERT-Tiny (4 layers, hidden size 128), which does not fully capture modern large-scale LLM hardware constraints.
* **Rigor:** Superb empirical deconstruction, diagnostic clarity, and control-theoretic analysis, offering exceptional conceptual value.

### Submission 6: Continuous Riemannian-Geometric Homotopical Model Merging via Grassmannian Geodesic Blending (C-Lie-MM)
* **Active Reviewers:** Reviewer 1, Reviewer 2, Reviewer 3
* **Ratings:** R1: `3: Weak Reject` | R2: `4: Weak Accept` | R3: `3: Weak Reject` | **Average:** `3.33 / 6`
* **Strengths:** Outstanding differential geometry; models representation spaces on Grassmannian manifolds to avoid coordinate collapse. Uses an offline-online split and Chebyshev polynomial approximations to translate expensive SVD operations into hardware-accelerated GEMMs.
* **Weaknesses:** Severe case of over-engineering. Solves a "coordinate collapse" problem that is already naturally mitigated by standard residual identity connections and LayerNorm. Under realistic overlapping task conditions, the highly complex framework gains a marginal $+0.30\%$ over simple flat gating ($70.30\%$ vs $70.00\%$), which is within standard deviation and does not justify the massive GPU kernel overhead.
* **Rigor:** Technically correct and elegant differential geometry, but practically irrelevant due to extreme complexity.

### Submission 7: Lyapunov-Stable Active Representation Coupling for Dynamic Model Serving (L-ARC)
* **Active Reviewers:** Reviewer 1, Reviewer 2, Reviewer 3
* **Ratings:** R1: `3: Weak Reject` | R2: `2: Reject` | R3: `5: Accept` | **Average:** `3.33 / 6`
* **Strengths:** Integrates classical control theory (Lyapunov stability) to stabilize representation propagation. Commendable scientific honesty in reporting negative results (where active warping is statistically redundant). Introduces ECG-Reset and RASC, which effectively mitigate sensor dropouts and persistent router bias.
* **Weaknesses:** Derivations heavily rely on the "Layer-Identity Approximation" ($S(h^{(l-2)}, \mu) \approx S(h^{(l-1)}, \mu)$), which is severely violated by non-linear transformer layers (like SwiGLU MLPs). When residual updates are scaled to realistic levels ($\gamma \ge 0.5$), L-ARC's performance benefits completely collapse and become statistically insignificant ($p > 0.10$). Very weak real-world evaluation (pilot of only 100 queries).
* **Rigor:** Demonstrates solid control-theoretic design but suffers from brittle assumptions that collapse under realistic transformer scales.

### Submission 8: GraviMerge
* **Active Reviewers:** Reviewer 1, Reviewer 2, Reviewer 3
* **Ratings:** R1: `2: Reject` | R2: `3: Weak Reject` | R3: `5: Accept` | **Average:** `3.33 / 6`
* **Strengths:** Elegant differential geometry applying continuous, second-order Newtonian mechanics (mass-spring-damper physics with geodesic exponential maps and parallel transport) to parameter blending. Outperforms baselines in the synthetic RDS coordinate simulation.
* **Weaknesses:** Massive evaluation gap. Designed for multi-task edge serving of deep foundation models, but evaluated *strictly* on a toy projection of scikit-learn's $8\times8$ handwritten digits. Contains absolutely zero downstream validation on actual transformer models or standard language tasks.
* **Rigor:** Elegant mathematical pipeline, but severely undermined by a highly artificial toy digit evaluation.

### Submission 9: PAC-Kinetics
* **Active Reviewers:** Reviewer 1, Reviewer 2, Reviewer 3
* **Ratings:** R1: `6: Strong Accept` | R2: `3: Weak Reject` | R3: `5: Accept` | **Average:** `4.67 / 6`
* **Strengths:** Landmark multidisciplinary paper combining Lyapunov stability, biochemical kinetics, and Catoni-type PAC-Bayesian bounds under non-i.i.d. $\beta$-mixing data streams (solved via Even/Odd Block Splitting). Substantiated by a physical PyTorch neural network validation on real MNIST/Fashion-MNIST datasets, showing a $+21.50\%$ absolute accuracy improvement over Uniform Merging and a $2.59\times$ routing jitter reduction.
* **Weaknesses:** The mixing coefficient is practically unverifiable online. Physical evaluation, while better than pure simulators, is still restricted to a shallow, 3-layer MLP.
* **Rigor:** Exceptionally strong. It balances outstanding mathematical rigor with robust, non-i.i.d. learning-theoretic guarantees and physical deep network validation.

### Submission 10: Dirichlet-PAC: Simplex-Constrained PAC-Bayesian Complexity Control
* **Active Reviewers:** Reviewer 1, Reviewer 2, Reviewer 3
* **Ratings:** R1: `5: Accept` | R2: `2: Reject` | R3: `Good` (plotted as `5: Accept` by R1/R3) | **Average:** `4.00 / 6`
* **Strengths:** Models ensembling weights directly on the probability simplex $\Delta^{K-1}$ using a Dirichlet posterior and derives the exact analytical Dirichlet KL divergence. Proposes PEM-Div for unsupervised serving. Evaluates on real-world BERT backbones.
* **Weaknesses:** Reviewer 2 points out a major theory-practice mismatch: the PAC-Bayes bounds assume Stochastic Expert Routing (queries routed to a single expert), but the actual experiments evaluate continuous activation-space blending. Under real BERT-Medium experiments, the proposed method consistently underperforms simple baselines, suffering a severe **$-6.67\%$** absolute accuracy degradation compared to SABLE Norm and Uniform Merging.
* **Rigor:** Highly ambitious formulation, but severely penalized due to negative empirical results and a mismatch between the core routing theory and experimental execution.

---

## 4. Final Meta-Review Decisions and Justifications

### 1. **Submission 4: Momentum-Merge** (Recommendation: ACCEPT)
* **Justification:** This is the highest-rated paper in the cohort (Average: 5.33/6, with ratings of 5, 6, 5). It stands out as an exemplary application of scientific parsimony (Occam's razor). By rigorously deconstructing SOTA continuous-time kinetics frameworks (ChemMerge) and demonstrating that their ODE solvers are mathematically equivalent to a simple Constant Exponential Moving Average (EMA), the paper clears away unnecessary complexity. It delivers a $195.7\times$ reduction in routing weight jitter with zero computational overhead. The evaluation is highly robust (10 random seeds, paired t-tests). Acceptance is a clear and unanimous decision.

### 2. **Submission 9: PAC-Kinetics** (Recommendation: ACCEPT)
* **Justification:** This is the second highest-rated submission (Average: 4.67/6, with ratings of 6, 3, 5). It represents a landmark multidisciplinary work that bridges biochemical kinetics, control theory (Lyapunov stability), and statistical learning theory. Critically, it addresses the realistic non-i.i.d. nature of test-time serving streams by deriving a Catoni-type PAC-Bayesian bound under stationary $\beta$-mixing processes, resolving the exploding TV penalty through Even/Odd Block Splitting. Unlike other purely synthetic papers, it validates its claims on physical PyTorch neural networks on real datasets, showing an impressive $+21.50\%$ accuracy gain. It easily warrants acceptance.

### 3. **Submission 5: Methodological Audit & Deconstruction of Dynamic Activation-Space Model Merging** (Recommendation: ACCEPT)
* **Justification:** Tied for third place in average score (4.33/6), we recommend accepting Submission 5 over Submission 2. 
  
  **Comparative Analysis:** 
  While both submissions have a score profile of `{5, 5, 3}`, **Submission 5** performs a highly valuable and necessary corrective service to the machine learning community. It exposes a widespread methodological flaw in contemporary model-merging literature—specifically, comparing new routing methods against poorly initialized, unregularized "straw-man" classical baselines. It demonstrates that a simple, properly regularized linear gating head with Zero-Initialized Softmax and L2 regularization is competitive and often superior to complex SOTA ODE architectures.
  
  In contrast, while **Submission 2** boasts a very strong systems-ML co-design and mobile-hardware evaluation, its theoretical appendix relies on severe mathematical shortcuts (modeling complex deep activations as isotropic white Gaussian noise, assuming mutual independence where strong noise coupling exists, and fitting GMMs over strictly bounded domains). 
  
  Therefore, we accept **Submission 5** for its outstanding conceptual contribution, diagnostic clarity, control-theoretic deconstructions, and its high-signal role in raising evaluation standards in the dynamic model-merging field.

---

## 5. Summary of Rejection Rationale for Key Borderline/Low-Score Submissions

* **Submission 3 (CR-Router) - REJECT due to Fatal Theoretical Flaw:** Despite starting with strong scores, the paper contains a fatal mathematical error. Under its sandbox hyperparameters, the derived global contraction condition simplifies to requiring a spectral norm less than a negative number ($\|W_{\text{route}}^{(l)}\|_2 < -19.5 \tau_l$), which is mathematically impossible. This compromises its entire theoretical foundation.
* **Submission 10 (Dirichlet-PAC) - REJECT due to Empirical Failure:** Although conceptually ambitious in modeling simplex constraints, its core theoretical bounds assume stochastic expert routing, which is mismatched with its continuous activation-space blending experiments. Consequently, under real-world BERT-Medium experiments, it suffers a catastrophic **$-6.67\%$** absolute accuracy degradation compared to simple uniform merging.
* **Submission 6 (C-Lie-MM) - REJECT due to Disproportionate Complexity:** This represents extreme over-engineering. It builds a dense differential-geometric framework (Grassmannian manifolds, tangent mappings, Rauch comparison theorems, Chebyshev approximations) to solve a coordinate collapse problem already mitigated by LayerNorm and residual connections. The resulting performance gain over flat ensembling is a statistically insignificant $+0.30\%$ accuracy, which does not justify its massive Triton GPU kernel overhead.
* **Submission 8 (GraviMerge) - REJECT due to Catastrophic Evaluation Gap:** Despite elegant mathematics applying second-order Newtonian physics, the framework—designed for edge serving of deep foundation models—is evaluated *strictly* on a projection of scikit-learn's $8\times8$ toy handwritten digits. It features zero evaluation on real transformer models or NLP datasets, rendering its claims unverified in any realistic deep learning context.
