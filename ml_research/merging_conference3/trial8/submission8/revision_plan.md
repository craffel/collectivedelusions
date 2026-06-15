# Revision Plan: Addressing Peer Critique (Reviewer 2) - Round 5 (Session 29)

We present a systematic, prioritized plan and successful execution log addressing the final remaining presentation suggestions identified by our mock reviewer in the fifth round of review. In alignment with **The Methodologist** persona, we have resolved all minor points with complete mathematical rigor and dynamic cross-referencing.

---

## 1. Suggestion 1: Reference Labeled Equations in Section 3
*   **Critique:** Equation 1 (cosine similarity projection, `eq:cos_sim`) and Equation 2 (GMM variance estimation, `eq:gmm_variance`) are labeled but never explicitly referenced or cited in the text of Section 3.
*   **Action Taken:**
    *   Surgically edited `submission/sections/03_method.tex` to add explicit textual citations and dynamic references (`Equation~\ref{eq:cos_sim}` and `Equation~\ref{eq:gmm_variance}`) directly into the respective paragraphs.
    *   Integrated these references seamlessly into the narrative flow of the methodology section.

---

## 2. Suggestion 2: Dynamic Downstream Accuracy Referencing & Labeling
*   **Critique:** Section 4.4 references a hard-coded "Equation 4" on line 217 to refer to the downstream system-level classification accuracy equation ($\mathcal{A}_{\text{sys}}$). Additionally, the main accuracy equation and baseline evaluations are formatted as equation blocks but lack LaTeX `\label` commands.
*   **Action Taken:**
    *   Added LaTeX `\label` commands to the main downstream classification accuracy equation (`submission:eq:sys_accuracy`) and the corresponding baseline evaluations for unregularized (`submission:eq:sys_accuracy_unreg`), Ridge (`submission:eq:sys_accuracy_ridge`), and SRC-DE (`submission:eq:sys_accuracy_srcde`) variants in `submission/sections/04_experiments.tex`.
    *   Replaced the hard-coded string "Equation 4" on line 217 with the dynamic LaTeX reference `Equation~\ref{submission:eq:sys_accuracy}`.

---

## 3. Suggestion 3: Point Main Text to Dynamic Noise Estimator
*   **Critique:** While the main text references noise adaptation assuming oracle representation noise variance $\sigma^2$, Appendix A.3 presents a highly practical running noise estimator ($\hat{\sigma}^2_{\text{runtime}}$) using an EMA over closest-centroid distances. Pointing the reader to Appendix A.3 raises the practical systems relevance.
*   **Action Taken:**
    *   Added a dedicated sentence in Section 4.14 of `submission/sections/04_experiments.tex`: *"Alternatively, a practitioner can deploy the dynamic online noise estimator ($\hat{\sigma}^2_{\text{runtime}}$) presented in Appendix~\ref{sec:dynamic_noise_estimator} to estimate and adapt to serving noise on-the-fly."*
    *   This provides a direct bridge between theoretical oracle upper bounds and practical runtime deployment.

---

## 4. Suggestion 4: Broaden Backbone Generalization Scope in Main Conclusion
*   **Critique:** Pointing to Appendix A.5's discussion of backbone generalization in the main conclusion or introduction would make the paper's broad applicability even more prominent to readers from other sub-fields.
*   **Action Taken:**
    *   Updated Section 5.1 ("Generalization to Other Modalities and Scale") of `submission/sections/05_conclusion.tex` to append a direct citation to the appendix: *"A comprehensive discussion on generalization to convolutional backbones (e.g., ConvNeXt) and heavier vision transformers (e.g., ViT-Large) is provided in Appendix~\ref{sec:backbone_generalization}."*
    *   This ensures readers immediately appreciate the broad applicability of the coordinate-space density audit framework.

---

## 5. Execution & Verification Steps
1.  **Modified LaTeX Files:** Surgically updated `submission/example_paper.tex`, `submission/sections/03_method.tex`, `submission/sections/04_experiments.tex`, and `submission/sections/05_conclusion.tex`.
2.  **Clean Compilation:** Verified that the entire document compiles flawlessly using `tectonic` in `submission/` with zero layout issues, underfull/overfull boxes, or unresolved reference warnings.
3.  **Mock Review Validation:** Re-ran `./run_mock_review.sh` to confirm that all criticisms were successfully addressed. The paper remains in its pristine, unanimous **Strong Accept (Score: 6)** state with high praise across all criteria.
