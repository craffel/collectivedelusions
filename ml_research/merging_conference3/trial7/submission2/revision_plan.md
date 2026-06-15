# Revision Plan: Information-Geometric Subspace Routing (FIOSR) - Iterative Refinement Round 10

Based on the highly constructive feedback from the Mock Reviewer in Round 9 (which gave us a stellar Accept with Score 5/6), we have formulated a plan to further perfect the manuscript by addressing the final minor suggestions and strengthening the academic presentation of our systems-level and empirical analyses.

## Prioritized Weaknesses & Action Plan

### 1. Refine the MBH System-Level Trade-off Explanation (Weakness 1)
- **Critique:** MBH requires executing up to $G \le K$ sequential forward passes under heterogeneous streams. Setting $M=1$ avoids this but loses true weight-space ensembling (collapsing to hard routing selection). The reviewer asks to make this trade-off even more explicit.
- **Action Plan:** We will add a dedicated, prominent paragraph in Section 5 (Conclusion \& Discussion) of `05_conclusion.tex` explicitly highlighting this system-level trade-off, clarifying how hard Top-1 routing serves as a computational safeguard at the expense of parameter ensembling, and suggesting future directions in concurrent multi-adapter kernel execution.

### 2. Concrete Roadmap for End-to-End Physical Evaluation (Weakness 2)
- **Critique:** Although simulated LoRA validation is realistic, the experiments are conducted inside simulated coordinate spaces. Physical validation on actual backbones is a critical next step.
- **Action Plan:** We will expand Section 5 ("Conclusion \& Discussion") of `05_conclusion.tex` to lay out a detailed, actionable, 4-step roadmap for end-to-end evaluation on physical models (such as LLaMA-3-8B and ViT-Base). We will specify the libraries (e.g., PEFT, Hugging Face), exact dataset suites (e.g., GLUE for LLMs, ImageNet sub-tasks for ViTs), activation extraction layers (e.g., MLP outputs), and how on-the-fly dFIM estimation will be conducted.

### 3. Fisher Vector Compression for Massive LLM Vocabularies (Weakness 3)
- **Critique:** Scaling the framework to LLMs with large vocabularies ($C \approx 32\text{K}$) and high dimensions ($d \approx 4096$) can introduce storage costs for routing coefficients.
- **Action Plan:** We will add a dedicated paragraph in Section 4.5 ("Ablation Studies and Analysis" of `04_experiments.tex`) proposing and discussing specific compression strategies: (1) **Class-Grouped Pooling** (grouping the $C$ vocabulary tokens into $G_v \ll C$ semantic clusters using hidden-state embeddings, reducing storage to $K \times G_v \times d$); and (2) **Low-Rank Factorization of Fisher Scales** (representing FIM scales as a product of low-rank matrices).

### 4. Direct Highlight of the Calibration Size ($N_c$) Phase Transition in Main Text (Weakness 4)
- **Critique:** The calibration size is fixed at $N_c=16$ in the main text. While Appendix B.2 contains a detailed sensitivity sweep (Table 6), the main text does not sufficiently showcase these insights.
- **Action Plan:** We will add a brief sentence in Section 4.5 ("Sensitivity to Calibration Size" of `04_experiments.tex`) explicitly outlining the quantitative results from the Appendix (i.e., showing that accuracy leaps from 65.16% at $N_c=4$ to 74.34% at $N_c=8$, and saturates at 75.61% at $N_c=16$) to ensure that the reader immediately grasps the statistical phase transition without needing to jump to the appendix.

---
This systematic update will resolve all remaining feedback with absolute rigor and maximum scholarly depth.
