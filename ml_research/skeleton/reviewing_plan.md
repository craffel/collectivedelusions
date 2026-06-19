# Reviewing Agent Operating Plan

Complete a full peer review of the conference submission. The compiled `submission.pdf` and the entire LaTeX source are provided in the `submission/` subdirectory. Since you have access to the LaTeX source, converting the PDF to plain text should be unnecessary.
You have been assigned a specific reviewer persona, described in `persona.md`. You **must** adopt this persona completely and evaluate the paper from the perspective and priorities of this persona.
**IMPORTANT: Your assigned persona is a secret internal motivation. Do NOT mention your persona (e.g., "The Critic", "The Empiricist") anywhere in your written review.**
Closely follow the reviewing criteria in `reviewing_criteria.md`, but filter them through the lens of your assigned persona.

**Page Limit Enforcement:** Before starting your detailed review, you MUST check the page limit of the submission by running `python ../check_page_length.py submission/example_paper.tex` (or whatever the main tex file is named). If the script outputs FAILED (i.e., the main text exceeds 8 pages), you must assign the lowest possible score for the submission and note this violation as the primary reason for rejection in your review.

To ensure a thorough and systematic review, you MUST break down your evaluation into component tasks before writing the final review.
Please generate the following intermediate files in your working directory based on the sections in the paper:
1. `1_summary.md`: A comprehensive summary of the paper's main topic, approach, key findings, and explicitly claimed contributions (with evidence).
2. `2_novelty_check.md`: An assessment of the key novel aspects, the 'delta' from prior work, and the characterization of novelty (e.g., incremental, significant).
3. `3_soundness_methodology.md`: An evaluation of the clarity of the description, appropriateness of methods, potential technical flaws, and reproducibility.
4. `4_experiment_check.md`: A critical evaluation of the experimental setup, datasets, baselines, and whether the results actually support the claims.
5. `5_impact_presentation.md`: A list of major strengths, areas for improvement, overall presentation quality, and potential impact/significance.

As part of the reviewing process, you can search for prior work covering related themes and methods using Google Search or the Semantic Scholar API as described in `semantic_scholar.md`.
Use the API key stored in the `SEMANTIC_SCHOLAR_API_KEY` environment variable.

After you have completed the intermediate evaluations (`1_summary.md` through `5_impact_presentation.md`), synthesize your findings into a final, cohesive review.
Your final review *must* include an overall recommendation.
Save the text of your final review as `review.md`.
