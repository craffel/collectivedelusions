# Conference Meta-Review Process and Decisions Report

## 1. Introduction and Process Overview

This document presents the official meta-review process and final acceptance decisions for 10 paper submissions evaluated for the Merging Conference. The core objective of this meta-review is to identify and accept exactly 3 outstanding submissions from the pool of 10 candidates.

The selection process was conducted under a rigorous, multi-stage assessment protocol:
1. **Individual Review Compilation**: All 3 independent peer reviews for each of the 10 submissions were systematically compiled and parsed to extract overall scores, recommendations, and detailed qualitative assessments.
2. **Analysis of Strengths and Weaknesses**: For each paper, we evaluated both the numerical scores and the depth, validity, and constructiveness of the reviewers' critiques. This step was crucial to distinguish between high scores on minor contributions and lower scores that might be easily addressed or represent pioneering research.
3. **Consensus and Divergence Auditing**: Submissions with highly divergent reviews (e.g., strong accept mixed with reject) were closely analyzed to understand the core point of tension and determine if the concerns were fatal or addressable.
4. **Comparative Ranking**: All submissions were compared side-by-side using a multi-dimensional criteria framework (soundness, presentation, empirical validation, conceptual novelty, and practical utility) to finalize the top 3 papers for acceptance.

---

## 2. Comprehensive Submission Registry

The table below lists all 10 evaluated submissions, their paper titles, the individual ratings from Reviewers 1, 2, and 3, their mean scores, and a brief summary of the reviews.

| Submission # | Paper Title | Reviewer Scores | Mean Score | Review Summary & Key Feedback |
| :---: | :--- | :--- | :---: | :--- |
| **1** | Riemannian Curvature-Regularized Test-Time Model Merging | R1: 3 (Weak Reject)<br>R2: 2 (Reject)<br>R3: 6 (Strong Accept) | 3.67 | **Key Strengths**: Pioneering problem definition of "Overfitting-Optimizer Paradox" and elegant differential geometry/spectral graph theory framing. <br>**Key Weaknesses**: Severe reliance on custom 1D synthetic emulators with a lack of standard real-world benchmarks (e.g., ImageNet-C, GLUE) and questionable toy-scale BERT/ViT pilot studies. |
| **2** | Rademacher-Bounded Polynomial Merging: Provable Generalization Bounds for Adaptive Model Merging | R1: 6 (Strong Accept)<br>R2: 3 (Weak Reject)<br>R3: 6 (Strong Accept) | **5.00** | **Key Strengths**: Pioneering and technically flawless paper that establishes the first rigorous learning-theoretic foundation for adaptive model merging with elegant Rademacher complexity bounds. <br>**Key Weaknesses**: Unexplained omission of CUB-200 benchmark, MNIST task dominance, and idealized linearization assumptions. |
| **3** | Robust Linear Routing: Deconstructing Complex Dynamic Model Merging via Occam's Razor | R1: 5 (Accept)<br>R2: 4 (Weak Accept)<br>R3: 4 (Weak Accept) | 4.33 | **Key Strengths**: Convincing, highly transparent "sanity check" deconstructing a complex quantum baseline (QWS-Merge), proving classical linear routing beats quantum wavefunction collapse. <br>**Key Weaknesses**: Low conceptual/algorithmic novelty (standard L2 and softmax temperature scaling) and severe citation omissions in the bibliography. |
| **4** | Demystifying Dynamic Model Merging via Bounded Classical Routing | R1: 6 (Strong Accept)<br>R2: 5 (Accept)<br>R3: 5 (Accept) | **5.33** | **Key Strengths**: Exceptionally strong, conceptually bold deconstruction showing simple classical routing with L2 regularization or independent Sigmoids (BSigmoid-Router) outperforms complex quantum/SOTA methods with zero test-time latency and ultra-compact parameter footprint. <br>**Key Weaknesses**: Restrictive empirical scope (compact ViT backbone and four vision tasks). |
| **5** | Demystifying Quantum-Inspired Model Merging: Layer-Wise Low-Dimensional Classical Routing Beats "Quantum" Wavefunction Collapse | R1: 6 (Strong Accept)<br>R2: 5 (Accept)<br>R3: 5 (Accept) | **5.33** | **Key Strengths**: Paradigm-shifting conceptual breakthroughs (proving layer-averaging collapse and exposing "robustness-accuracy" illusions) that challenge community assumptions. Rigorous evaluation on real-world CLIP parameters and a Triton compilation roadmap. <br>**Key Weaknesses**: Minor theoretical overclaims regarding universality and slight ambiguity in baselines. |
| **6** | Sparse Low-Rank Dynamic Merging: Enabling Batch-Independent and Parameter-Efficient Multi-Task Inference | R1: 4 (Weak Accept)<br>R2: 5 (Accept)<br>R3: 3 (Weak Reject) | 4.00 | **Key Strengths**: Resolves batch-dependency and soft-interference bottlenecks of dynamic model merging, enabling RAM-efficient inference. <br>**Key Weaknesses**: Conceptually misaligned (more like multi-LoRA/MoE than weight merging), O(K) computational bottleneck, and toy-scale evaluation. |
| **7** | Pruned Gradient Merging (PG-Merge): Deconstructing Complexity in Test-Time Model Fusion | R1: 3 (Weak Reject)<br>R2: 4 (Weak Accept)<br>R3: N/A | 3.00 | **Key Strengths**: Conceptual elegance in applying Top-k gradient selection to test-time fusion. <br>**Key Weaknesses**: Severe literature positioning errors (missing EATA/MECTA), toy-scale evaluation, and marginal improvement over zero-compute static baselines. |
| **8** | EpiMerge: Epigenetic Weight Masking for True Sample-Wise Dynamic Model Merging | R1: 4 (Weak Accept)<br>R2: 3 (Weak Reject)<br>R3: 2 (Reject) | 3.00 | **Key Strengths**: Engaging and highly original biological metaphor linking epigenetics to weight-space merging. <br>**Key Weaknesses**: Major empirical discrepancies in baseline tables, routing gates are virtually flat (acting statically), and consistently outperformed by a simpler static baseline (OFS-Tune) while adding 2-3x latency and parameter overhead. |
| **9** | Grassmannian Subspace Consensus Merging: A Spectral Filter for Multi-Task Parameter Alignment | R1: 3 (Weak Reject)<br>R2: 5 (Accept)<br>R3: 3 (Weak Reject) | 3.67 | **Key Strengths**: Beautiful and rigorous mathematical grounding in Grassmannian projection and spectral filtering. <br>**Key Weaknesses**: Catastrophic performance gap under task-conditional settings compared to expert ceilings, task-agnostic collapse, and toy-scale evaluations. |
| **10** | Chaos-Theoretic Attractor Merging (ChaosMerge): Dynamic Model Fusion via Coupled Map Lattices | R1: 2 (Reject)<br>R2: 3 (Weak Reject)<br>R3: 4 (Weak Accept) | 3.00 | **Key Strengths**: Highly creative connection between chaos theory and parameter-space model merging. <br>**Key Weaknesses**: Underperforms simple static and linear routing baselines by large margins, fails under task-agnostic mixed-batch settings, and has broken/lazy citations. |

---

## 3. Final Selection and Comparative Justification

Exactly three submissions have been selected for acceptance at the Merging Conference:
1. **Submission 4** (Demystifying Dynamic Model Merging via Bounded Classical Routing)
2. **Submission 5** (Demystifying Quantum-Inspired Model Merging: Layer-Wise Low-Dimensional Classical Routing Beats "Quantum" Wavefunction Collapse)
3. **Submission 2** (Rademacher-Bounded Polynomial Merging: Provable Generalization Bounds for Adaptive Model Merging)

### Comparative Analysis & Rationale:

- **Why Submissions 4 & 5?**
  Submissions 4 and 5 achieved the highest average reviewer score (**5.33/6.00**), with identical profiles of one `Strong Accept` and two `Accept` ratings. Both papers represent exceptional scientific contributions that apply Occam's razor to deconstruct exotic mathematical metaphors (such as quantum wavefronts and quantum wavefunction collapse) that have dominated SOTA model merging. They prove both theoretically and empirically that simple, properly regularized classical routing heads (like BSigmoid-Router or L3-Router) achieve superior performance and efficiency without complex, over-engineered mechanics. They are flawless in writing, highly rigorous in empirical evaluation, and carry major practical utility for edge deployments. Their selection was unambiguous and highly supported by all reviewers.

- **Why Submission 2 over Submission 3?**
  Submission 2 (mean score: **5.00**, ratings: **6, 3, 6**) was selected over Submission 3 (mean score: **4.33**, ratings: **5, 4, 4**). 
  - Although Submission 2 has a `Weak Reject` from Reviewer 2, its other two reviews are outstanding `Strong Accepts` (6/6), highlighting it as a pioneering, mathematically flawless paper that establishes the first rigorous statistical learning-theoretic foundation for adaptive model merging. The theoretical contributions (Rademacher complexity trajectory bounds, local Rademacher fast rates, margin bounds) are extremely robust and paradigm-shifting. The critiques raised by Reviewer 2 (such as the omission of CUB-200 and discussion of task dominance) are easily addressable revisions that do not undermine the paper's immense theoretical and empirical value.
  - In contrast, while Submission 3 is a solid deconstruction of QWS-Merge, all reviewers noted that its algorithmic novelty is highly incremental (applying standard L2 weight decay and softmax temperature scaling to a linear gating layer) and lacks conceptual ambition. Furthermore, it suffered from severe bibliography omissions (broken citations in references.bib). Thus, the pioneering theoretical and empirical depth of Submission 2 far outweighs the incremental nature of Submission 3, making Submission 2 the clear choice for the final acceptance spot.

- **Why other submissions were excluded?**
  - **Submission 1 (RCR-Merge)** (Mean: 3.67): Despite a creative geometric framing and a Strong Accept, two reviewers gave it Rejects. The paper suffers from severe empirical shortcuts, relying almost entirely on a handcrafted 1D synthetic emulator and toy-scale BERT/ViT pilot studies that lacked statistical significance.
  - **Submission 6 (SLD-Merge)** (Mean: 4.00): Suffers from a major conceptual misalignment, acting as a multi-LoRA/MoE architecture that keeps separate adapter pathways in memory rather than performing true parameter-space merging.
  - **Submissions 7, 8, 9, 10**: All received mean ratings of 3.67 or below. They exhibit severe empirical gaps, technical flaws, or performance collapses under task-agnostic settings, and fail to outperform simple static baselines while introducing significant computation and memory overheads.

---

## 4. Individual Meta-Review Summaries for Accepted Papers

### Submission 2: Rademacher-Bounded Polynomial Merging: Provable Generalization Bounds for Adaptive Model Merging
- **Overall Recommendation**: **Accept (Pioneering Theoretical Contribution)**
- **Meta-Review**: 
  This paper is an exceptionally strong, pioneering contribution that establishes the first rigorous statistical learning-theoretic foundation for adaptive model merging. The authors model the merging coefficients as a continuous global trajectory across network depth and derive elegant Rademacher complexity bounds and margin-based generalization rates. The empirical validation is exhaustive, supporting the core claims across both convolutional and Vision Transformer backbones with zero inference overhead. While one reviewer raised valid concerns regarding the omission of the CUB-200 benchmark, task dominance effects, and linearization assumptions, these represent constructive areas for minor revision rather than fatal flaws. The sheer mathematical depth, conceptual ambition, and outstanding clarity of this paper make it a stellar addition to the conference program.

### Submission 4: Demystifying Dynamic Model Merging via Bounded Classical Routing
- **Overall Recommendation**: **Strong Accept (High-Impact Deconstruction & Algorithm)**
- **Meta-Review**: 
  This submission is an outstanding, conceptually bold, and scientifically rigorous work that applies Occam's razor to demystify exotic "quantum" metaphors in dynamic model-merging protocols. The authors prove that a simple, properly regularized classical routing head (BSigmoid-Router) completely resolves reported representation collapses (outperforming the SOTA wave formulation by $+12.00\%$) with zero test-time latency and an ultra-compact parameter footprint (772 parameters). The reviewers praise the paper's flawless writing, exceptional latency profiling, and rare scientific honesty regarding generalist-specialist tradeoffs. The restricted empirical scope on a compact ViT backbone is well-justified for isolating parameter-routing dynamics. This represents a flawless, high-impact contribution that is accepted with the highest priority.

### Submission 5: Demystifying Quantum-Inspired Model Merging: Layer-Wise Low-Dimensional Classical Routing Beats "Quantum" Wavefunction Collapse
- **Overall Recommendation**: **Strong Accept (Paradigm-Shifting Critique & Framework)**
- **Meta-Review**: 
  This paper represents an outstanding, methodologically rigorous, and highly complete work that makes a significant deflationary contribution to the weight-space model merging literature. By deconstructing "quantum-inspired" metaphors, exposing critical baseline omissions, and proving layer-averaging collapse, the paper provides deep scientific clarity and challenges fundamental assumptions in the community. The authors introduce the highly effective L3-Router framework, a mixed-task stream audit, and a hardware-grounded Triton compilation roadmap, validating their findings on real-world CLIP parameters. The reviewers commend the paper's exemplary scientific hygiene and thorough audits. This work will serve as an essential cautionary tale and a guiding framework for simpler, more robust model merging designs.
