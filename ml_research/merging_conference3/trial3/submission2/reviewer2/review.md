# Peer Review: Offline Few-Shot Validation Tuning (OFS-Tune) for Model Merging

## 1. Summary of the Paper
This paper presents a rigorous methodological critique of the recently popularized paradigm of online Test-Time Adaptation (TTA) for weight-space model merging. Online TTA methods dynamically adjust merging coefficients at test-time on an unlabeled target-task stream by minimizing unsupervised objectives (prediction entropy). The authors argue that this paradigm relies on a "no-data" strawman (comparing active, backpropagation-dependent adaptation solely against a naive, unoptimized uniform baseline) and collapses catastrophically under realistic, safety-critical stream shifts (such as class imbalance, temporal task clustering, and ultra-small batch sizes).

As a simple and robust alternative, the authors propose **Offline Few-Shot Validation Tuning (OFS-Tune)**. OFS-Tune leverages a tiny labeled validation set (as few as 5 to 10 samples per task) to find static, optimal merging coefficients offline. Under standard target streams, OFS-Tune consistently outperforms or matches SOTA online TTA methods with zero test-time compute. Crucially, under adversarial stream shifts, OFS-Tune remains perfectly robust, providing a stable, zero-overhead baseline. 

The paper also conceptualizes the **"Overfitting-Optimizer Paradox"**, demonstrating that unconstrained high-dimensional search spaces overfit catastrophically under scarce validation samples, whereas low-dimensional parameterizations (e.g., modeling coefficients as continuous low-degree polynomials across network layers) act as powerful analytical noise filters. The findings are backed up by a continuous weight-merging simulation calibrated on Vision Transformer (ViT-B/32) statistics (evaluated across 30 random seeds) and a physical proof-of-concept experiment on real Convolutional Neural Networks trained on MNIST and FashionMNIST (evaluated across 5 random seeds), including physical visualization of the non-convex prediction entropy landscape.

---

## 2. Key Strengths
- **Exemplary Empirical Rigor:** Evaluating the continuous simulation across **30 independent random seeds** and the physical CNN experiments across **5 random seeds** is an outstanding standard of empirical verification. Reporting mean and standard deviation for all key tables guarantees statistical significance and reproducibility.
- **Realistic Stress-Testing:** Rather than evaluating solely on clean, stable i.i.d. streams, the paper subjects online TTA methods to three highly realistic adversarial stream shifts (extreme label shift, bursty task streams, small batch sizes) and sweeps systematic validation selection bias up to 30%. This reveals critical vulnerabilities in the current TTA paradigm that are typically obscured in standard setups.
- **Deconstruction of "SOTA" Claims:** The paper successfully demystifies online TTA's reported gains. By running "sterile" noiseless simulations, the authors show they can replicate prior literature's claims, but prove that these gains vanish and collapse when realistic transductive noise and landscape non-convexity are introduced.
- **The Overfitting-Optimizer Paradox Insight:** The systematic comparison between Random Search, Nelder-Mead, and PyTorch Adam across different parameter dimensionalities is a high-signal conceptual contribution. It clearly exposes how local search algorithms like Nelder-Mead appear resistant to high-dimensional overfitting only because they fail to optimize (dimension stalling), whereas capable optimizers like Adam overfit catastrophically unless restricted to low-dimensional structures.
- **Physical Neural Network Validation:** Grounding the simulated findings in actual CNN weights on physical datasets (MNIST and FashionMNIST) and visualizing the actual 2D prediction entropy landscape (Figure 5) provides ironclad empirical validation for the paper's core claims.
- **Excellent Transparency:** The paper includes a very honest and detailed Limitations section that addresses the toy-scale nature of the physical validation and the abstract limits of the continuous simulation.

---

## 3. Weaknesses and Areas for Improvement
- **Scale Gap in Physical Validation:** While the physical validation is highly valuable, it is performed on a toy-scale CNN (~100k parameters) trained on MNIST and FashionMNIST. Modern model-merging is primarily deployed on massive foundation models (e.g., Vision Transformers with 86M+ parameters or Large Language Models). Although the authors include a transparent discussion of this limitation, a physical experiment on a standard pre-trained ViT-B/32 would have conclusively proven that these findings scale to overparameterized transformer hierarchies.
- **Discrepancy in Physical AdaMerging Performance:** In Table 5 (physical CNN experiments), Online AdaMerging collapses catastrophically to **42.94%** under a *clean, standard* test stream, which is significantly below the Uniform TA baseline of **55.27%**. In contrast, in the simulated Standard Stream (Table 1), Online AdaMerging achieves **79.72%** (closer to Uniform's 84.44%). The root cause of this massive physical collapse is not fully analyzed. Is prediction entropy minimization fundamentally unstable on simple convolutional weights, or was the learning rate ($lr=10^{-3}$) unstable? Elaborating on this discrepancy would strengthen the empirical analysis.
- **Inductive Bias vs. Sample Regularization:** Table 4 shows that even when validation data is abundant ($M=50$ samples per task), PyTorch Adam on Poly-Val ($d=2$) achieves **87.69%** simulated accuracy, outperforming the unconstrained 48-D Layer-wise search which gets **87.22%**. At $M=50$ samples, validation noise should be highly minimized. This indicates that low-dimensional trajectories like polynomials represent a highly beneficial **structural inductive bias** for model merging, rather than just acting as a regularizer against validation sample size noise. The paper should highlight and discuss this distinction in more detail.
- **Toy-Scale Task Cardinality in Physical Validation:** The physical CNN experiment is restricted to merging only 2 experts ($K=2$). Given that the simulation highlights catastrophic dimensionality collapse for $K \ge 16$ tasks under Nelder-Mead, evaluating a $K=4$ physical CNN setup (e.g., adding CIFAR-10 and SVHN) would have bridged the simulation and physical setups more cohesively.

---

## 4. Questions and Suggestions for the Authors
1. **Physical ViT-B/32 Feasibility:** Have the authors attempted to run physical weight-merging and few-shot validation tuning on a physical pre-trained ViT-B/32? If so, did the physical experiments replicate the same Overfitting-Optimizer Paradox observed in the simulation and CNN setups?
2. **Learning Rate Sensitivity in Physical AdaMerging:** What was the performance of physical Online AdaMerging under smaller learning rates (e.g., $lr = 10^{-4}$)? Could the catastrophic collapse in Table 5 be partially mitigated by slower, more conservative adaptation, or is prediction entropy minimization fundamentally broken on this DeepCNN architecture?
3. **Validation Set Selection Variance:** In the physical CNN experiments, how were the 10 few-shot validation samples selected? Were they randomly sampled and varied across the 5 independent random seeds, or were they fixed? If they were randomized, how much did the performance of OFS-Tune fluctuate between seeds, and how can practitioners guarantee stable validation splits in practice?

---

## 5. Formal Ratings

### Soundness: Excellent
The paper is technically flawless and shows an exemplary commitment to statistical validation. The multi-seed sweeps (30 seeds in simulation, 5 seeds in physical CNN), rigorous baseline tuning, and extensive sweeps of validation bias, task cardinality, loss-landscape roughness, and domain diversity provide ironclad support for all core claims.

### Presentation: Excellent
The manuscript is exceptionally well-written, clear, and logical. The tables are dense, informative, and beautifully structured. The figures are detailed and highly communicative, especially the physical 2D prediction entropy landscape visualization.

### Significance: Excellent
This work serves as a vital and timely methodological course correction for the model-merging community. By establishing a simple, robust, and zero-overhead baseline (OFS-Tune), it exposes the fragile assumptions of complex online TTA methods and will likely influence future research standards.

### Originality: Excellent
While the individual components are standard, the critical deconstruction of online TTA in model merging, the conceptualization of the "Overfitting-Optimizer Paradox," and the unique adaptation of low-dimensional trajectories for offline few-shot regularization are highly novel and original.

---

## 6. Overall Recommendation
**5: Accept**  
This is a technically solid, highly rigorous, and exceptionally well-evaluated paper that exposes major blindspots in the model-merging literature. The authors' deconstruction of "SOTA" online TTA claims under realistic stream shifts, combined with the conceptualization of the Overfitting-Optimizer Paradox, makes this paper a valuable contribution. Although there is a scale gap in physical validation (toy CNN vs. Transformers), the sheer volume and statistical rigor of the empirical evaluations make this an outstanding submission. I highly recommend acceptance.
