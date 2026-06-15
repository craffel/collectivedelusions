# Presentation and Impact: CR-PolySACM

We evaluate the quality of the presentation, the clarity of the figures/tables, the expected impact of the work, and the viability of the proposed future research directions.

---

## 1. Quality of Writing, Structure, and Narrative
The paper is exceptionally well-written, clear, structured, and logical:
- **Narrative Flow:** The narrative progresses seamlessly from identifying the practical challenge of post-training quantization in test-time adaptive merging, to a rigorous theoretical analysis of the quantization-induced loss gap, to diagnosing the task-vector norm scale pathology, and finally to presenting the CR-PolySACM framework and its empirical validation.
- **Terminology & Definitions:** Terms such as "Quantization-Operator Overfitting", "task-vector norm scale pathology", "CR-SACM", and "PolySACM" are clearly defined and consistently used throughout the text.
- **Professionalism:** The tone is objective, scholarly, and direct, avoiding ungrounded hype while highlighting major empirical breakthroughs.

---

## 2. Quality and Clarity of Figures and Tables
The visual aids and tabular representations are of publication-grade quality:
- **Figure 1 (Teaser Plot):** Successfully captures the core motivation, illustrating the dramatic performance differences between different merging paradigms across various precisions. It clearly shows the collapse of unregularized AdaMerging under low-precision, and highlights CR-PolySACM's robust performance.
- **Figure 2 (Sensitivity Plot):** Highly effective sensitivity analysis of the different model merging methods across the six quantization schemas. It clearly shows that CR-PolySACM achieves the absolute best balance between continuous performance and post-training quantization stability.
- **Table 1 (Merging Results):** Extremely clean, detailed, and informative. Bold numbers highlight the top performances in each column, and the columns represent a natural progression of quantization formats.
- **Table 2, 3, 4, 5 (Ablation Tables):** Clear, precise, and highly informative. They provide raw numerical data that directly validates the core theoretical mechanisms (the norm scale pathology, calibration size $N$, computational overhead, and the clipping-regularization threshold $\beta$).

---

## 3. Expected Impact on the Machine Learning Community
This work is highly significant and likely to influence future research and practice in several key areas:
- **Model Merging:** By establishing the connection between test-time adaptation, structural subspace constraints, and sharpness-aware optimization under quantization, the paper provides a powerful new toolkit for designing robust post-hoc composition methods.
- **Test-Time Adaptation (TTA):** The paper provides a deep warning about the fragility of unconstrained test-time adaptation under downstream quantization, encouraging future TTA research to incorporate flatness and subspace regularizations.
- **Edge Deployment & Quantization:** As edge deployment of large, merged multi-task models becomes increasingly common, CR-PolySACM provides a practical and highly efficient framework to compile these models to low-precision formats (like INT8 or INT4) without losing representational quality.

---

## 4. Practicality of Future Directions (LLMs/VLMs and Percentile-based Blueprint)
The author outlines highly promising and concrete future directions:
- **Scaling to LLMs/VLMs:** Large Language Models and Vision-Language Models are primary candidates for weight merging (e.g., merging specialized task experts). Extending CR-PolySACM to these models is a logical and high-impact next step.
- **Percentile-Based Automated Scale-Balancing:** In larger, deeper models with thousands of layers, manual tuning of the clipping threshold $\beta$ is fragile and impractical. The author proposes a highly elegant **percentile-based blueprint**: by dynamically setting $\beta$ as a lower percentile (e.g., the 10th percentile) of the empirical task-vector norm distribution across all layers, the scale-balancing mechanism automatically isolates and clips only the highly sensitive, near-zero tail layers.
- **Empirical Validation of the Percentile Blueprint:** The author added a preliminary empirical proof-of-concept of this blueprint on the Vision Transformer backbone in Appendix A.1. Setting $\beta$ to the 10th percentile automatically yielded $\beta \approx 0.098$ and achieved a Joint Mean Accuracy of **19.05%** under INT4, extremely close to the optimal hand-tuned setting of $\beta=0.10$ ($19.07\%$). This successfully demonstrates that a percentile-based dynamic threshold successfully and automatically isolates and scale-balances the sensitive, near-zero tail of the task-vector norm distribution across different model scales without requiring manual parameter sweeps.
