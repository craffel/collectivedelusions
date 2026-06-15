# 5. Impact and Presentation Evaluation

This section assesses the paper's overall presentation quality, identifies major strengths and concrete areas for improvement, and evaluates its potential significance and impact on the research community.

## 1. Major Strengths
1. **Conceptual Simplicity & Philosophical Alignment:** The paper is strongly motivated by Occam's razor, offering a refreshing critique of the trend toward highly convoluted and hyperparameter-heavy regularization schemes (like RegCalMerge's Class-Capacity Normalization and Scale-Normalized Entropy Weighting). PG-Merge's simple gradient masking is highly intuitive.
2. **Clear Mathematical Exposition:** The formulation in Section 3.4 is clear, precise, and easy to follow. The definition of the dynamic sparse gradient mask and the post-update parameter projection (Equations 8-13) are mathematically sound.
3. **Thorough Ablation and Trajectory Analyses:** The ablation study over the sparsity ratio $p$ (Table 2) is very systematic, and the trajectory plots (Figure 3) provide excellent empirical evidence showing how unconstrained AdaMerging overfits while PG-Merge achieves stable, controlled adaptation.
4. **Insightful Optimizer Analysis (Appendix A):** The theoretical discussion in Appendix A identifying the internal state mismatch and momentum decay of adaptive optimizers like Adam when paired with sparse masking is highly insightful and of high academic quality.

## 2. Major Areas for Improvement
1. **Scholarly Rigor & Citation Practices (Critical):**
   - **Fix the MECTA error:** Correctly characterize MECTA as a test-time adaptation method rather than standard supervised fine-tuning. Include a formal citation tag and add its full bibtex entry to `references.bib`.
   - **Cite EATA:** Discuss and cite EATA (ICML 2022) as a key precursor that pioneered updating a sparse subset of parameters to stabilize the test-time adaptation loop.
   - **Clean up the bibliography:** Remove the dozens of uncited "ghost" references in `references.bib`. Ensure every paper mentioned in the text (such as QWS-Merge) is properly cited and has an entry in the bibliography.
2. **Resolve the Adam vs. SGD Contradiction:** Since Appendix A strongly advocates for pairing PG-Merge with standard SGD without momentum to avoid momentum decay and state mismatch, the authors should evaluate PG-Merge under SGD and compare it directly against the Adam-based implementation. It is highly inconsistent to theoretically advocate for SGD while empirically evaluating the entire paper on Adam.
3. **Scale Up the Experimental Evaluation:** To establish generalizability, the method must be evaluated on standard, large-scale model-merging benchmarks:
   - Use standard backbone architectures like **CLIP-ViT-B/32** or **CLIP-ViT-L/14**.
   - Evaluate on full, standard datasets (like the 8 tasks from the Task Arithmetic paper) rather than 1,024-image subsets.
   - Test on other modalities, such as Large Language Models (LLMs) like LLaMA-7B or Mistral-7B.
4. **Acknowledge and Discuss the Computational Trade-offs:** The authors must moderate their claims of PG-Merge being "training-free" and having "zero computational overhead." True training-free model merging has zero runtime overhead. PG-Merge requires 100 steps of backpropagation at test time, which introduces massive computational latency for a very modest ($0.54\%$) accuracy improvement over the static baseline. This trade-off must be discussed honestly.

## 3. Overall Presentation Quality: Good
The paper is exceptionally well-written, structured, and easy to read. The flow from the introduction of the Overfitting-Optimizer Paradox to the methodology and experiments is seamless. However, the poor scholarly rigor in the Related Work and references section (missing citations, mischaracterizations, ghost references) significantly compromises its academic quality and presentation rating.

## 4. Potential Impact & Significance: Fair to Poor
In its current state, the potential impact of the paper is quite limited:
- **Conceptual Overlap:** The core concept of using coordinate/parameter sparsity to stabilize TTA is already highly popular in the literature (e.g., EATA, MECTA), meaning the novelty is somewhat incremental.
- **Marginal Practical Utility:** A $0.54\%$ improvement over a zero-compute static uniform merging baseline, achieved at the cost of 100 steps of full backpropagation on a test-time calibration set, represents very low practical utility.
- **Toy Scale:** Evaluating only on a toy `vit_tiny` model on tiny subsets makes it difficult for researchers to trust that these results would hold for the large-scale foundation models where model merging is actually applied in practice.

If the authors can scale up their evaluation to CLIP-ViT or LLMs, show substantial gains over static baselines with a much smaller step budget (e.g., 5-10 adaptation steps), and resolve the scholarly and optimizer contradictions, the paper would have a much higher chance of making a significant impact.
