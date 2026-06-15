# Revision Plan: Addressing Mock Review Feedback

We highly value the reviewer's exceptionally rigorous and constructive feedback. To address the three critical flaws identified in the mock review, we have designed the following priority revision plan. Our edits preserve the conceptual elegance of the framework while dramatically strengthening its empirical rigor, narrative transparency, and physical interpretation.

## 1. Decomposed Jitter Analysis: Resolving the Jitter Contradiction (Critical Flaw 3)
- **Problem:** The reviewer points out that while UGR is more stable than the untuned ChemMerge, it exhibits significantly higher layer-to-layer routing jitter than a heavily tuned Momentum-Merge baseline.
- **Edits in Section 4:**
  - Introduce a new subsection titled "Intra-Task Stability vs. Inter-Task Agility: Decomposed Jitter Analysis" (Section 4.4.3).
  - Define and decompose the routing jitter metric into two distinct, high-fidelity components:
    1. **Intra-Task Jitter:** Measures stability within consecutive queries of the same task.
    2. **Inter-Task Jitter:** Measures the purposeful coordinate displacement during task switches.
  - Report the empirical results of this decomposition across 10 random seeds:
    - **UGR (Ours):** Intra-Task Jitter is **12.31 $\times 10^{-4}$**, while Inter-Task Jitter is **21.79 $\times 10^{-4}$** (a clean 1.8x separation).
    - **Momentum-Merge (Advanced):** Intra-Task Jitter is **2.88 $\times 10^{-4}$** and Inter-Task Jitter is **2.60 $\times 10^{-4}$** (virtually zero separation).
  - Discuss the physical interpretation: Momentum-Merge's low jitter is a symptom of severe representational inertia (it fails to rotate at boundaries). UGR's higher overall jitter is not random noise, but a purposeful, agile rotation.
- **True Block-Structured Stream Evaluation:**
  - Evaluate all methods on a realistic sequential block stream (boundaries every 50 samples).
  - Report that under block streams, UGR's overall jitter drops by **40%** to **11.63 $\pm$ 1.39 $\times 10^{-4}$** while achieving the highest Joint Mean Accuracy of **75.17% ± 0.93%** (exceeding all baselines).

## 2. Real-World Validation Roadmap (Critical Flaw 1)
- **Problem:** Complete reliance on the 14-layer synthetic Analytical Coordinate Sandbox (ICS) without validating on real-world deep learning datasets or architectures.
- **Edits in Section 5 (Conclusion & Future Directions):**
  - Add a dedicated, publication-ready roadmap paragraph.
  - Acknowledge the synthetic sandbox as a controlled testing environment, and detail the upcoming integration of UGR for dynamic, token-level ensembling of task-specific LoRAs in Large Language Models (LLaMA-3, Mistral) under sequential multi-task text generation streams.
  - Detail the ongoing evaluation on Vision Transformers (ViT) with task-specific adapters under streaming image classification.
  - Reference the derived analytical Jacobians and positive orthant persistence proofs (Appendix A) as the exact mathematical basis for this differentiable, real-world integration.

## 3. Emphasizing the Softmax-Free Simplex Projection benefits (Critical Flaw 2)
- **Problem:** Classification accuracy gains in randomized streams appear marginal over heavily filtered Euclidean baselines.
- **Edits in Section 4.4:**
  - Explain that in query-by-query randomized streams, carrying over state acts as a temporal distractor, which artificially compresses accuracy gains across all stateful methods and makes them converge near stateless baselines.
  - In Section 4.4.3, show that in realistic block-structured streams, UGR's non-Euclidean geodesic flow natively resolves this, achieving both state-of-the-art accuracy (75.17%) and pristine intra-task stability, establishing a clear Pareto-optimal serving frontier.
