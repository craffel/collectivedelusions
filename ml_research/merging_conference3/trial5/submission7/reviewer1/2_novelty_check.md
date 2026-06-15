# 2. Novelty and Literature Positioning Check

An essential requirement of scientific publication is that a submission must accurately position itself within the broader context of prior and concurrent literature, credit foundational ideas properly, and clearly articulate its true "delta" or novelty. Evaluating the paper from this perspective reveals significant concerns in literature positioning, citation completeness, and claims of novelty.

## 1. Characterization of Novelty (The "Delta")
The paper claims that PG-Merge is a pioneering, minimalist approach that resolves test-time model fusion overfitting by applying a dynamic sparse gradient mask. While applying this specific gradient-sorting mask to *merging coefficients* in the model-fusion domain is technically a new application, the underlying concept—restricting updates to a sparse subset of critical parameters or layers during test-time adaptation (TTA) to prevent overfitting and transductive collapse—is already highly established in the broader TTA literature.

Specifically:
- **EATA (Efficient Test-time Adaptation, ICML 2022):** This seminal work proposed selecting and updating only a sparse subset of parameters (using Fisher Information or high gradient magnitudes) during test-time adaptation to prevent representation decay, collapse, and catastrophic forgetting. EATA is a direct conceptual precursor to the philosophy of PG-Merge, yet it is completely omitted from the bibliography and discussion.
- **MECTA (Memory-Economic Continual Test-Time Model Adaptation, ICLR 2023):** This work proposed dynamically selecting and updating a subset of "drift-sensitive" layers while keeping others frozen during test-time adaptation. 

The core conceptual "delta" of the paper—that parameter/coordinate sparsity acts as a stabilizer for the test-time adaptation loop—is therefore an incremental application of established TTA principles (EATA, MECTA) to the specific problem of merging coefficients, rather than a fundamentally new theoretical or conceptual discovery.

## 2. Incorrect Citation & Mischaracterization of MECTA
The paper mentions MECTA in Section 2.4 (Related Work, "Gradient Pruning and Sparse Optimization"):
> *"In the context of optimization, gradient pruning and coordinate selection (e.g., MECTA) have been used to select only high-magnitude gradient components to update during fine-tuning. However, prior gradient selection methods have primarily focused on reducing backpropagation memory or speeding up training in standard supervised regimes."*

This is a major scholarly error:
1. **Mischaracterization:** MECTA stands for **Memory-Economic Continual Test-Time Model Adaptation**. It is explicitly a **test-time adaptation (TTA)** framework, not a "supervised fine-tuning" method or "standard supervised regime."
2. **Missing Citation:** While the text mentions "MECTA", the authors have omitted a formal citation tag (e.g., `\cite{...}`) in the text, and there is **no corresponding entry for MECTA in `references.bib`**. 

Ignoring the test-time nature of MECTA allows the authors to falsely claim that they are the first to "repurpose" gradient selection and sparse masking as a test-time regularizer. Correctly characterizing MECTA reveals that using layer/coordinate selection as a test-time regularizer is exactly what MECTA did, significantly diminishing the paper's claims of conceptual pioneering.

## 3. Disconnected Bibliography ("Ghost" Citations)
The bibliography (`references.bib`) contains a massive number of entries that are never cited in the body of the paper. Out of 50+ bibliography entries, only 13 are actually cited. Uncited entries include key papers in parameter-efficient fine-tuning (e.g., `gu2024advancing`, `zaken2022bitfit`, `he2021towards`) and dataset references. Conversely, some methods mentioned in the text (such as MECTA and QWS-Merge) do not have corresponding citations or bibliography entries. This reflects a lack of scholarly rigor and suggesting that the bibliography was compiled carelessly (perhaps copied from a generic template) rather than being carefully integrated with the text.

## 4. Inaccurate Description of Concurrent Work
In the Related Work, the authors mention:
> *"Other concurrent works like QWS-Merge introduce quantum wave analogies using frozen random projections, normalized phase states, and interference equations..."*

However, there is no citation for `QWS-Merge` anywhere in the paper, and `references.bib` instead contains an entry for `qmerge` (Q-Merge: Quantization-Aware Model Merging via Straight-Through Estimators, ECCV 2024), which is a quantization-aware model merging paper and has nothing to do with "quantum wave analogies." This further demonstrates a careless handling of literature.

## Summary of Novelty and Positioning Rating: Poor
While the empirical application of coordinate-sparsity to merging coefficients is interesting, the paper fails to acknowledge that its core stabilizing philosophy is heavily borrowed from established TTA literature (like EATA). Furthermore, the paper suffers from serious scholarly errors, including the mischaracterization and non-citation of MECTA, the mention of uncited concurrent works, and a sloppy, disconnected bibliography.
