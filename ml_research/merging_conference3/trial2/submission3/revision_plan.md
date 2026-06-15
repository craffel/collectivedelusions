# Revision Plan - Addressing Minor Polish Suggestions

In response to the Mock Reviewer's minor suggestions for further polish (which rated our paper **5: Accept**), we have executed a final set of revisions to make the manuscript flawless and conference-ready.

## Prioritized Weaknesses & Action Items

### 1. Discussion on Deeper Architectures in Methodology
*   **Critique:** Move a brief sentence or pointer to B-Splines from the Appendix to the end of Section 3.2 (or 3.3) to show readers immediately how the framework scales to deeper models (32- or 80-layer networks).
*   **Action Item:** We have added a dedicated sentence at the end of Section 3.2 pointing out that for deep 32-layer LLaMA-7B or 80-layer LLaMA-3 networks, our continuous polynomial parameterization scales gracefully by employing B-splines and piecewise continuous local trajectories, referring directly to the formal scaling analysis in Appendix B.

### 2. Dynamic Programming Boundary Selection
*   **Critique:** In Section 4.4's Discussion on Spline Block Selection, elaborate slightly on the automated dynamic programming formulation to partition deep models by minimizing test-time prediction entropy.
*   **Action Item:** We have expanded the "Spline Block Selection and Automated Partitioning" paragraph in Section 4.4. We formulated the boundary search formally as a dynamic programming problem, wrote out the exact DP recurrence relation (Equation 11), and showed that it can be solved in $O(B L^2)$ time. This mathematically demonstrates that automated optimal block partitioning is computationally trivial (requiring only a few thousand operations) even for 32-layer or 80-layer models.

### 3. Epsilon Definition in Methodology
*   **Critique:** Meticulously verify that $\epsilon = 10^{-8}$ is explicitly defined around Equation 4 (Shannon entropy loss) in Section 3.2.
*   **Action Item:** We verified that $\epsilon = 10^{-8}$ is indeed explicitly defined and detailed in Section 3.2 as a microscopic numerical stability parameter to prevent taking the logarithm of zero during entropy backpropagation.

---

## Rebuttal and Progress Log Update
All three polish suggestions have been successfully implemented and compiled. Tectonic compiled the revised manuscript with zero errors. The final peer review score remains at a celebrated **5: Accept**.
