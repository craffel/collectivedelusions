# 4. Experimental Evaluation and Claims Check

This evaluation is conducted from the perspective of an **Empiricist** who prioritizes rigorous experimental design, strong and regularized baselines, thorough ablations, statistical significance, and whether empirical results truly back up the paper's central claims.

---

## 1. Experimental Setup and Dataset Evaluation
- **Task Selection & Model Capacity**: The choice of a highly compact 5.7M-parameter Vision Transformer ($\mathtt{vit\_tiny\_patch16\_224}$) is excellent for simulating parameter-space conflict. In compact networks, weight-sharing is highly constrained, making them ideal testbeds for representational collapse.
- **Calibration Set Size**: The calibration set size (16 samples per task, 64 total) is extremely small. 
  - *Empiricist Critique*: While this shows parameter-efficiency, the choice of these 64 samples introduces a massive source of variance. The paper contains **no statistical error bars, confidence intervals, or standard deviations**. It does not specify if the experiments were run over multiple random seeds or if a single, cherry-picked calibration split was used. For an empiricist, evaluating on a single run with a 64-sample calibration set is statistically unsound.

---

## 2. Baseline Strength and Fairness
- **Absence of Competitive Static Baselines**: While Uniform Merging and AdaMerging are included, the paper does not compare against state-of-the-art static model merging techniques like **TIES-Merging** or **DARE**, which are specifically designed to resolve parameter sign and gradient conflicts. Comparing QWS-Merge only to Uniform Merging makes the "representational collapse" baseline artificially weak.
- **The "Strawman" Linear Router**:
  - The Linear Router is an unregularized dynamic baseline with 772 parameters optimized on 64 samples. It is highly prone to overfitting.
  - Furthermore, the Linear Router is restricted to **global routing** (the same coefficients for all 14 layers), whereas QWS-Merge utilizes **layer-wise routing** (allowing different coefficients per layer group).
  - *Empiricist Critique*: This is an unfair comparison. QWS-Merge's superior performance on SVHN could simply be a result of its layer-wise flexibility rather than its "wave-like cosine projections." To make this comparison fair, the authors must include a **Layer-wise Linear Router** with equivalent regularization (such as weight decay or dropout) to control for the degrees of freedom and the layer-wise routing capacity.

---

## 3. Critical Analysis: Do the Results Support the Claims?

### Claim 1: "QWS-Merge completely resolves representational collapse"
- *Empiricist Critique*: **No, the results do not support this claim.** 
  - In Table 1, the specialized expert ceiling is $70.52\%$, while QWS-Merge achieves $59.32\%$ joint mean accuracy—a significant $11.20\%$ gap.
  - On the simple MNIST task, the expert ceiling is $92.50\%$. The Linear Router gets $91.20\%$ (almost preserving the ceiling), while QWS-Merge drops to **$77.60\%$** (a massive $14.90\%$ performance drop on a basic task).
  - On FashionMNIST, the Linear Router gets $67.00\%$, while QWS-Merge gets $63.50\%$.
  - On CIFAR-10, the Linear Router gets $71.40\%$, while QWS-Merge gets $64.60\%$.
  - **QWS-Merge is actually significantly worse than the classical Linear Router baseline on 3 out of the 4 tasks (MNIST, FashionMNIST, and CIFAR-10)**. It only outperforms the Linear Router on SVHN ($31.60\%$ vs $15.30\%$). 
  - This suggests that QWS-Merge does not "resolve representational collapse" across the board. Instead, it acts as an extremely heavy regularizer (due to the constrained cosine space) that prevents overfitting on the hard task (SVHN) at the cost of **severely degrading performance** on the easier, high-performing tasks.

### Claim 2: "Wave-Like Subspace Regularization prevents parameter-space collapse"
- *Empiricist Critique*: **This claim is partially supported, but highly conflated.** 
  - While QWS-Merge avoids the SVHN collapse ($31.60\%$) seen in the Linear Router ($15.30\%$), the paper does not isolate whether this is due to the *wave-like cosine projection* or simply because QWS-Merge has a much smaller, highly constrained parameter footprint ($336$ parameters vs $772$ parameters). 
  - To support this claim, the authors must show that a Linear Router regularized with standard L2 regularization (weight decay) or constrained to a comparable parameter footprint still collapses on SVHN. Without this ablation, the "wave-like" aspect of the regularization remains an unproven hypothesis.

### Claim 3: "Transparent Heterogeneity Benchmark"
- *Empiricist Critique*: **This claim is fully supported, and it actually exposes the fatal flaw of the method.**
  - Table 2 shows that under mixed-task heterogeneous streams at batch sizes of 16 and 256, QWS-Merge collapses to $48.80\%$ and $48.70\%$, respectively.
  - This is **worse than static AdaMerging ($57.20\%$) and OFS-Tune ($55.60\%$)**, and is virtually identical to uniform merging ($49.20\%$).
  - This means that in any realistic deployment scenario where tasks are mixed in input streams, **the dynamic routing mechanism collapses completely**, and the model is less effective than standard static merging methods.

---

## 4. Omitted Experiments and Ablations
An empiricist review requires several critical experiments that are currently entirely missing from the paper:
1. **Regularization Ablation**: A comparison against a regularized Linear Router (using weight decay, dropout, or spectral normalization).
2. **Layer-wise vs. Global Routing Ablation**: A comparison against a Layer-wise Linear Router to isolate the benefit of layer-wise specialization.
3. **Hyperparameter Sensitivity**:
   - Sensitivity of QWS-Merge to the fixed frequency $\omega$ (currently fixed at $\pi$).
   - Sensitivity to the initialization of the amplitude scaling factors $R_k^{(l)}$ (currently initialized to $0.3$).
   - Sensitivity to the projection dimension $d$.
4. **Statistical Robustness**: Evaluation over at least 5 random seeds with different calibration sample splits, reporting mean and standard deviation for all results.
5. **Scale Ablation**: Evaluation on a larger model (e.g., ViT-Base or ResNet-50) to see if "representational collapse" and the need for QWS-Merge's regularization persist as model capacity increases.
