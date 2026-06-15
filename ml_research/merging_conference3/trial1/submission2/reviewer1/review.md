# Peer Review

**Paper Title:** Deconstructing Sharpness-Aware Isotropic Merging: A Methodological Analysis of Component Contribution and Optimization Flatness

---

## 1. Strengths and Weaknesses

### Strengths

* **Exemplary Commitment to Elegant Simplicity:** The paper represents an outstanding methodological audit that de-bloats an overly complex, over-engineered deep learning pipeline. By systematically deconstructing Sharpness-Aware Isotropic Merging (SAIM), the authors show that a much simpler, more elegant, and unconstrained approach—standard globally perturbed Sharpness-Aware Minimization (SAM) paired with naive Task Arithmetic—frequently outperforms or matches a highly convoluted, coordinate-restricted training scheme and expensive post-processing SVD steps. This clear proof that simplicity is superior is a massive service to the community.
* **Outstanding Scientific Rigor and Transparency:** The authors prevent selective reporting by executing a complete, symmetric $5 \times 3$ grid crossing 5 optimizers and 3 merging strategies under two distinct mixing parameter regimes ($\lambda=0$ and $\lambda=0.2$). The inclusion of standard deviations across multiple random seeds provides excellent statistical transparency.
* **Rigorous and Elegant Theoretical Grounding:** Rather than relying solely on empirical sweeps, the paper provides a sound and elegant mathematical proof (Proposition 3.1) using second-order Taylor expansion and the Rayleigh-Ritz theorem. This proof mathematically binds the loss increase from post-hoc weight consolidation (e.g., pruning) to the spectral norm of the Hessian, beautifully connecting optimizer-driven flatness to post-hoc structural robustness.
* **Exposing and Correcting Literature Discrepancies:** The paper identifies and mathematically diagnoses a fatal algebraic typo in SAIM's published SA-BCD optimizer formula, which causes immediate model divergence. Implementing and evaluating two corrected variants is highly constructive and ensures absolute scientific reproducibility.
* **Practical GPU-Aware Computational Insights:** The paper highlights that coordinate-restricted optimizers like SA-BCD are actually 18.5% *slower* in wall-clock training time compared to standard global SAM, despite updating only 30% of parameters. Exposing that sparse indexing/masking breaks GPU parallelization and thread-coalescing is an invaluable, practical finding that warns researchers against a common over-engineering trap.
* **Highly Scalable PEFT Integration (LoRA-SAM):** Introducing LoRA-SAM as a lightweight, elegant solution ($<2.5\%$ wall-clock and $<1.5\%$ VRAM overhead) and proving that SVD post-hoc merging is completely redundant on flat, low-rank manifolds offers a highly practical, zero-overhead, and SVD-free consolidation pathway for large foundation models.
* **Thorough Capacity Scaling Validation:** Validating their deconstruction findings on an 86M parameter ViT-Base backbone successfully demonstrates that the synergistic benefits of pre-merging flatness remain highly robust as model capacity scales by over $17\times$.

### Weaknesses

* **Single-Seed Scale Validation on ViT-Base:** While the absolute performance gains are highly pronounced and far exceed the tight standard deviations observed in the ViT-Tiny sweeps, the scale validation results on the 86M parameter ViT-Base backbone (Table 3) are based on a single seed due to the extreme computational cost of full-parameter sequential fine-tuning.
* **Exclusively Task-Incremental Setting:** The paper’s empirical evaluation is restricted to the Task-Incremental continual learning setting, where an oracle task ID is provided during evaluation to swap in the task-specific classification head. Discussing or evaluating on the more challenging Class-Incremental setting would expand the scope and generalizability of the findings.
* **Proposed NLP Experiments Are Unexecuted:** The authors outline an excellent and feasible experimental design for NLP practitioners (BERT-Base on GLUE tasks) to test cross-domain generalizability. However, actually executing a subset of these experiments would have significantly strengthened the paper’s multi-modal, cross-domain conclusions.

---

## 2. Soundness

**Rating: Excellent**

**Justification:**
The submission is methodologically and technically flawless. The authors decouple the multi-component SAIM pipeline into modular, independent elements and cross-evaluate them on a symmetric $5 \times 3$ grid. They introduce several highly rigorous, custom-designed baselines (Scalar Update Decay, Norm-Matching, Scale-Calibrated, TIES, and DARE). 

The mathematical derivations are exceptionally clean and correct. The proof of Proposition 3.1 is elegant and correct. Furthermore, their mathematical analysis of the compounding scale shrinkage in their Norm-Matching baseline (Appendix C) is a brilliant, high-signal finding: they prove that due to the near-orthogonality of high-dimensional sequential updates, adding updates increases the combined norm ($\approx 1.442 N$), but averaging their norms yields a smaller target norm ($N$). This forces a scaling factor of $s_t \approx 0.693$ at each step, compounding exponentially to under $24\%$ of the proper update magnitude by step 5. This deep geometrical insight demonstrates outstanding scientific and technical soundness.

---

## 3. Presentation

**Rating: Excellent**

**Justification:**
The presentation quality is outstanding. The paper is exceptionally well-structured, clearly written, and the overall narrative is extremely easy to follow. The transition from problem formulation to component-level deconstruction is seamless and logical. 

The tables are beautifully formatted, complete, and report standard deviations for all primary sweeps. The figures (sensitivity sweeps and comparison bar plots) are highly clear, high-signal, and easy to interpret. The appendix provides outstanding scholarly depth, including CPU and GPU SVD execution time benchmarks across diverse model dimensions (up to LLaMA-7B) and a comprehensive hyperparameter sensitivity analysis of the perturbation radius $\rho_{\text{LoRA}}$ for LoRA-SAM.

---

## 4. Significance

**Rating: Excellent**

**Justification:**
The significance of this work is exceptionally high. In deep learning, there is a constant tendency to build increasingly complex, over-engineered pipelines and post-processing steps. This paper serves as a critical, highly valuable counterweight, proving that a simpler, more elegant, and unconstrained method (standard global SAM + naive Task Arithmetic) is highly competitive or superior, and that post-hoc SVD transformations are boundary-condition sensitive and completely secondary to training-stage flatness.

Exposing the 18.5% training-time bottleneck of coordinate-restricted optimizers on modern GPUs will save researchers and practitioners from wasting significant compute on redundant training schemes. Additionally, the LoRA-SAM paradigm offers a highly practical, zero-overhead, and SVD-free merging path for large language models and other foundation architectures. This paper will undoubtedly raise the bar for empirical rigor and establish standard global SAM as a mandatory pre-merging baseline for future weight consolidation algorithms.

---

## 5. Originality

**Rating: Excellent**

**Justification:**
While the paper is primarily analytical and methodological rather than algorithmic, its originality is significant and highly valuable. Rather than proposing an overly engineered new merging pipeline, the authors take a highly refreshing, meta-scientific approach: they deconstruct an existing complex framework to isolate the true causal drivers of performance. 

The formalization of the flatness-pruning synergy (Proposition 3.1), the mathematical deconstruction of the Norm-Matching baseline's compounding shrinkage, and the introduction of the SVD-free LoRA-SAM paradigm on flat low-rank manifolds represent substantial, highly original conceptual contributions to the field of deep model merging.

---

## 6. Overall Recommendation

**Rating: 5: Accept**

**Justification:**
This is an outstanding, technically solid, and methodologically beautiful paper that makes a high-impact contribution to the area of deep model merging and continual learning. It serves as a model for how methodological audits should be conducted in deep learning, demonstrating that simplicity, clarity, and unconstrained optimization-stage flatness are the true causal drivers of merging performance. 

The paper identifies and corrects fatal mathematical and code-level errors in the existing literature, provides a rigorous and elegant theoretical proof connecting Hessian curvature to post-hoc weight consolidation, and introduces a highly practical and computationally lightweight PEFT merging paradigm (LoRA-SAM). The minor weaknesses (single-seed validation on ViT-Base, lack of Class-Incremental CL or NLP experiments) are minor in comparison to the immense scientific depth, rigor, and clarity of the deconstruction. I strongly recommend accepting this paper for publication.
