# Peer Review Report: Impact and Presentation Evaluation

## 1. Major Strengths
1. **Critical and Refreshing Methodological Perspective:** The paper takes a step back and challenges the foundational premise of an entire subfield (online test-time adaptation for model merging). Exposing the "no-data" strawman and the transductive fragility of TTA methods is a vital contribution that will prevent researchers from over-complicating deployment pipelines.
2. **The Overfitting-Optimizer Paradox:** The conceptualization and empirical proof of this paradox is a highly original and significant contribution. It beautifully explains why high-capacity optimization models fail in few-shot regimes, and why low-dimensional constraints (like polynomial profiles) are mathematically necessary to filter out validation noise.
3. **Rigorous and Extensive Evaluation:** Evaluating the simulation across **30 independent random seeds** and stress-testing under three distinct adversarial shifts (extreme label shift, bursty task streams, and small batch sizes) provides exceptional statistical strength.
4. **Physical Neural Network Validation:** The inclusion of a physical 5-layer Convolutional Neural Network experiment on real images (MNIST/FMNIST) completely bridges the gap between simulated abstraction and physical neural weights.
5. **Outstanding and Deep Sensitivity Analyses:** The appendices contain an exceptional amount of high-signal analyses:
   - *Task Scalability Sweep:* Analyzing scaling up to $K=64$ tasks, proving the catastrophic dimensionality collapse of Nelder-Mead and demonstrating that PyTorch Adam scales smoothly.
   - *Validation Selection Bias Sweep:* Analyzing isotropic and structured late-layer semantic shift up to $30\%$ validation bias.
   - *Domain Diversity & Landscape Roughness Sweeps:* Validating the simulation's task interference and ruggedness modeling.
   - *Optimization Budget tracking:* Showing that OFS-Tune GT-Merge converges in as few as 2 Adam steps.
6. **Intellectual Honesty and Scientific Transparency:** The paper is incredibly transparent about its boundaries, discussing the toy-scale of its physical network and the abstract nature of its simulation landscape in depth.

---

## 2. Areas for Improvement
1. **Scale of Physical Validation:** While the 5-layer DeepCNN on MNIST/FMNIST is a highly effective proof-of-concept, it is still a toy-scale experiment. Evaluating OFS-Tune on a pre-trained Vision Transformer (ViT-B/32) or a lightweight Large Language Model (e.g., LLaMA-1B or LLaMA-3-8B task experts) on more complex datasets (like ImageNet-1k, GLUE, or GSM8k) would provide absolute, industry-scale confirmation.
2. **Integration with Other Merging Frameworks:** The paper evaluates OFS-Tune on standard linear weight-space merging (Task Arithmetic). It would be highly valuable to discuss how OFS-Tune integrates with or extends to other popular structural merging methods like **TIES-Merging** (Yadav et al., 2023) or **DARE** (Yu et al., 2023). For example, can we optimize polynomial coefficient profiles on top of TIES-Merging's sign consensus?
3. **Analysis of More Complex Search Space Parameterizations:** The paper proposes GT-Merge ($d=0$) and Poly-Val-Merge ($d \in [1, 3]$). In Section 6.4 (Future Work), the authors suggest Block-wise Constancy, Piece-wise Splines, and Low-Rank matrices. Providing even a simple simulation or preliminary experiment for one of these advanced parameterizations (like block-wise stage-grouping) would further enrich the technical depth.

---

## 3. Overall Presentation Quality
The presentation of this paper is **excellent** and easily meets the bar for top-tier machine learning conferences (such as ICML or NeurIPS).
- **Structure and Flow:** The narrative flows beautifully from the introduction of model merging, through the critique of online TTA, to the formulation of OFS-Tune, the rigorous simulation results, and the physical validation.
- **Visual Clarity:** The figures and plots (such as Figure 3 showing the physical non-convex prediction entropy contour) are of extremely high quality, very readable, and directly support the text.
- **Academic Writing Style:** The tone is professional, scholarly, and scientifically rigorous. There is a clear alignment with existing literature, proper attribution of prior ideas, and deep, methodologically skeptical argumentation.

---

## 4. Potential Impact and Significance
This paper is highly significant and has the potential for **broad impact** in the machine learning community:
1. **Course-Correction for Model Merging:** It will act as a major methodological course correction, urging researchers to move away from overly complex, compute-heavy, and fragile online adaptation schemes, and instead focus on robust, simple offline baselines.
2. **Establishing New Evaluation Standards:** It establishes few-shot validation tuning as a mandatory baseline that future online TTA model-merging papers must evaluate against to justify their complexity and compute.
3. **Practical Utility for Practitioners:** For machine learning practitioners in industry, OFS-Tune provides a simple, zero-overhead, and reliable tool to instantly find optimal static merging coefficients without the fear of representational collapse or test-time backpropagation cost.
4. **Inspiration for Regularized Weight-Space Optimization:** The "Overfitting-Optimizer Paradox" and low-dimensional search trajectories will likely spark a new line of research into regularized optimization manifolds in deep weight spaces.
