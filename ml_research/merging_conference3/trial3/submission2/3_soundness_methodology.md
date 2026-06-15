# Soundness and Methodology Check

## 1. Evaluation of Soundness
The submission is mathematically sound, technically rigorous, and intellectually honest. The authors combine mathematical formulations, a highly calibrated continuous simulation, and physical convolutional weight-space experiments, creating a very strong, cohesive, and credible methodological pipeline.

The soundness of each core component is evaluated below:

### A. Mathematical Formulation
- **Weight-Space Model Merging:** The layer-wise weight merging formulation is standard and correct:
  $$W_{merged}^{(l)}(\theta) = W_{base}^{(l)} + \sum_{k=1}^K \alpha_k(l; \theta) V_k^{(l)}$$
  The representation of layer-wise coefficient parameterization via low-degree polynomials of normalized depth ($l/L$) is modeled correctly and builds elegantly on PolyMerge, but here it is uniquely repurposed offline as an analytical noise filter.
- **Validation Selection Bias & Domain Shift:** The authors formalize target mismatch by introducing a systematic validation bias vector $v_{bias} \sim \mathcal{N}(0, \sigma^2_{bias} I)$ during offline optimization. Evaluating both isotropic Gaussian shift and structured late-layer semantic shift (concentrated solely in the high-level representation/classification layers) represents a highly rigorous, formal treatment of validation domain shifts.
- **Domain Diversity & Task Interference:** The formulation of simulated accuracy $\text{Acc}_k(\theta; D)$ under domain diversity index $D \ge 0$ (Equation 10) mathematically formalizes representational conflict:
  $$\text{Acc}_k(\theta; D) = B_k + \Delta_k (1.0 - R_k(\theta)) - D \cdot R_k(\theta)$$
  This is physically sound: combining highly diverse or conflicting experts causes severe representational interference (maximizing the interference penalty $-D$ under naive Uniform merging), which optimized coefficients successfully bypass.

### B. Controlled Simulation Landscape Calibration
The simulation setup is highly detailed and justified. The authors explicitly defend the scientific validity and necessity of this simulated approach:
- **Scientific Necessity:** It abstracts away hardware-level and framework-level conflated variables, making exhaustive sweeps (30 seeds, multiple noise levels, and dimensions) computationally tractable and reproducible.
- **Non-Convex prediction entropy Surrogate:** The online TTA optimization objective is modeled using a coupled quadratic surface combined with a high-frequency cosine wave penalty:
  $$\mathcal{L}_{TTA}(\theta) = \sum_{k=1}^K \left[ \frac{1}{2} + \frac{3}{2} e_k(\theta)^T \Sigma_{inv} e_k(\theta) + 0.03 \sum_{l=1}^L \left(1.0 - \cos(10 \pi e_{k, l}(\theta))\right) \right]$$
  This cosine penalty models the sharp local minima and localized non-smoothness characteristic of physical entropy landscapes under distribution shift, preventing the simulation from being unrealistically benign.

### C. Physical Convolutional Neural Network Validation
The physical validation bridges the simulation-empirical gap with complete rigor:
- **Differentiable Coefficient Optimization:** The use of PyTorch's functional call API (`torch.func.functional_call`) to perform exact, differentiable backpropagation through physical merging coefficients is technically elegant and flawless.
- **Demanding Stress Tests:** Training MNIST/FashionMNIST CNN experts across 5 independent seeds and evaluating OFS-Tune under $30\%$ random validation label noise provides a highly realistic, challenging validation environment.
- **Appropriate Baselines:** Comparing weight-space coefficient tuning directly against high-capacity supervised baselines (Few-Shot Joint FT and Few-Shot Head Tuning) directly validates the Overfitting-Optimizer Paradox on physical weight spaces.

---

## 2. Identified Methodological Weaknesses / Criticisms
Despite the high level of rigor, a few minor methodological limitations should be noted:

1. **Toy-Scale Nature of Physical Validation:** The physical CNN evaluated contains only 5 convolutional layers and ~100,000 weights on relatively simple datasets (MNIST and FashionMNIST). In practice, model merging is primarily applied to large-scale Vision Transformers (ViT) or LLMs (e.g., LLaMA, Mistral). While the authors acknowledge this limitation in Appendix E.1 and argue that the Overfitting-Optimizer Paradox is mathematically expected to be even more critical in overparameterized models, evaluating OFS-Tune on a pre-trained ViT-B/32 or similar model on standard visual tasks (such as CIFAR-100 or ImageNet-subsets) would make the empirical claims completely unassailable.
2. **Surrogate Landscape Abstraction:** Although the Coupled Model II continuous landscape is calibrated on empirical ViT-B/32 statistics, it remains a mathematical abstraction. Physical deep learning weight spaces exhibit discontinuous, high-dimensional topological boundaries and complex activation-level co-adaptations that cannot be fully modeled in a closed-form quadratic simulator.
3. **Static vs. Dynamic Streams:** OFS-Tune produces a static merged model, which is completely robust to stream shifts but cannot adapt to highly non-stationary continuous streams where target distributions drift dynamically over time. Unsupervised online TTA possesses a theoretical capability to adapt on-the-fly, which static coefficients lack.

---

## 3. Summary of Soundness Rating
**Rating: Excellent (or Good-to-Excellent)**
The submission is exceptionally sound. The mathematical formulations are precise, the simulation calibration is well-reasoned and defended, and the physical CNN validation provides an ironclad proof-of-concept that successfully replicates the Overfitting-Optimizer Paradox and non-convex prediction entropy landscapes on actual deep neural weights.
