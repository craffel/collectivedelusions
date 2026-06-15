# 5. Impact and Presentation Evaluation

## Major Strengths
1. **Outstanding Empirical Rigor:** Sweeping across **30 independent random seeds** in simulation and **5 random seeds** for physical convolutional networks sets a very high standard for empirical correctness. The paper is exceptionally thorough, analyzing standard streams, adversarial stream shifts, validation selection bias/domain shifts, task scalability, and physical weight-space optimization dynamics.
2. **Exposing Crucial Research Blindspots:** The paper identifies and exposes the "no-data" strawman and the extreme fragility of unsupervised online test-time adaptation (TTA) under realistic shifts. This is a very valuable and timely service to the machine learning community.
3. **The "Overfitting-Optimizer Paradox" Conceptualization:** The paper's systematic analysis of how optimization capacity and sample size interact (demonstrating that unconstrained high-dimensional spaces overfit catastrophically, while Nelder-Mead's apparent resistance is actually a failure to optimize) is highly insightful and technically sound.
4. **Physical CNN Grounding and Entropy Landscape Sweep:** Supplementing the continuous simulation with a physical CNN experiment and visualizing the actual, non-convex 2D prediction entropy landscape (Figure 5) provides excellent empirical grounding, proving that the simulated surrogates are realistic.
5. **Polished and Self-Critical Presentation:** The manuscript is exceptionally well-written, clear, and includes an honest, transparent Limitations section that discusses toy-scale limits and simulation abstractions.

---

## Areas for Improvement
1. **Address the Scale Gap of Physical Evaluation:** The physical validation is performed on a toy-scale CNN (~100k parameters) with MNIST and FashionMNIST. To fully prove that these findings translate to modern overparameterized models, a physical experiment on a pre-trained Vision Transformer (ViT-B/32, 86M parameters) or a small LLM would have been highly valuable.
2. **Elucidate the Dramatic Physical AdaMerging Collapse:** In Table 5, Online AdaMerging collapses to 42.94% on a clean physical stream, which is far below Uniform TA (55.27%). This is a much more severe drop than what was simulated (Table 1), indicating that unsupervised entropy minimization is highly unstable on actual physical CNN weights. The authors should analyze and explain the root cause of this massive discrepancy.
3. **Highlight Inductive Bias over Sample Regularization:** The results in Table 4 show that even with abundant validation data ($M=50$), Poly-Val outperforms unconstrained layer-wise search. This suggests that low-dimensional parameterizations like polynomials represent a beneficial structural inductive bias for weight-space model merging, rather than just acting as a shield against small-sample noise. This distinction should be highlighted and discussed in more detail.
4. **Physically Scale Task Cardinality:** The physical CNN experiments are restricted to a simple $K=2$ task setup. Given that the paper focuses on the scalability of Nelder-Mead and PyTorch Adam across $K \in [4, 64]$ tasks, evaluating a $K=4$ physical CNN setup (e.g., adding CIFAR-10 and SVHN) would have bridged the simulation and physical setups more cohesively.

---

## Overall Presentation Quality
The presentation quality is **Excellent**:
- The logical flow of the paper is seamless: it transitions from related work to problem formulation, search space definitions, optimization design, simulated evaluations, scalability sweeps, advanced ablations, physical validations, and finally a highly transparent discussion.
- The tables (Tables 1, 2, 3, 4, 5, 6) are incredibly dense, well-organized, and informative.
- The figure captions are detailed, explaining the setups and highlighting the key takeaways.

---

## Potential Impact and Significance
The potential impact of this paper is **High**:
- **Baseline Correction:** Weight-space model merging has become a key technique for LLMs, vision-language models, and diffusion models. Online TTA methods have recently gained immense attention. This paper acts as a vital methodological course correction, showing that a simple, zero-test-overhead baseline (OFS-Tune) can completely dominate active, complex methods.
- **Methodological Standard:** Proposing that researchers must evaluate TTA methods under adversarial stream shifts and compare them against few-shot validation tuning (whenever few-shot data is available) will likely improve the rigour and standards of the model-merging literature.
- **Zero-Overhead Deployment:** For machine learning practitioners, OFS-Tune offers a highly practical, simple, and deterministic alternative to active test-time optimization, which has massive engineering appeal due to its zero test-time compute.
