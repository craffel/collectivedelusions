# 4_experiment_check.md: Critical Evaluation of Experiments

## Experimental Setup and Statistical Rigor
The paper's experimental protocol is **exemplary in its statistical rigor**. While standard machine learning papers often report results averaged over 3 or 5 random seeds, this audit evaluates all methods across **30 independent random seeds** ($42 \le \text{seed} \le 71$) to capture statistical confidence. Standard deviations are explicitly reported for all results, enabling a rigorous evaluation of method stability. 

One minor simulator artifact is noted and honest-to-goodness explained by the authors: the standard deviation of the Uniform baseline in the simulation study is exactly $0.00\%$. This is because the normalized ratio ($\mathcal{R}_k = 1.0$ in Equation 4) becomes invariant to task seed when the merging coefficients are static and uniform.

---

## Evaluation of Datasets and Task Suites
The authors use a pool of four classic image classification datasets: MNIST, FashionMNIST, CIFAR-10, and SVHN. While these are relatively standard, small-scale datasets, the core novelty lies in **how they are systematically partitioned**. By dividing them into five suites representing different axes of domain distance and representational conflict, the authors construct a powerful multi-axial diagnostic benchmark that successfully exposes "Task Suite Bias":
* **Suite A (Homogeneous):** Low domain distance and low representational conflict.
* **Suite B (Heterogeneous):** High conflict (CIFAR-SVHN), exposing where unconstrained online methods overfit to noise.
* **Suite C (Cross-Domain Digits):** Massive grayscale-to-RGB domain shift.
* **Suite D (Cross-Domain Objects):** Grayscale-to-RGB natural object conflict.
* **Suite E (Control):** The monolithic standard benchmark.

Evaluating on these five configurations provides absolute empirical confirmation that single-suite benchmarks lead to false generalizability claims.

---

## Comprehensiveness of Baselines
The evaluation features an exceptionally comprehensive set of baselines, including:
1. **Static Uniform Baseline:** Represents traditional hand-tuned/uniform merging (Task Arithmetic).
2. **Online AdaMerging:** Represents state-of-the-art unconstrained layer-wise online Test-Time Adaptation.
3. **Online PolyMerge:** Represents polynomial-constrained online Test-Time Adaptation.
4. **Offline OFS-Unconstrained (Ablation Baseline):** Isolates the role of the trajectory constraint by performing unconstrained optimization offline using the same validation data.
5. **OFS-Tune (Linear, $d=1$):** Our proposed stable, continuous offline trajectory tuning.
6. **OFS-Tune (Quadratic, $d=2$):** Our proposed higher-capacity continuous offline trajectory tuning.

Additionally, in Appendix C and Section 4.6, the authors evaluate two alternative localized trajectory baselines (**Piecewise Linear Splines** and **Block-wise Parameter Sharing**) and compare them directly. In Section 4.4, they construct symmetrical optimization budget baselines (**OFS-Tune via Adam** and **AdaMerging via L-BFGS-B**), representing an outstanding effort to isolate confounding variables.

---

## Do the Results Support the Claims?
Yes, the empirical results provide robust, multi-seed support for all core claims made in the paper:

### Claim 1: "Task Suite Bias" masks limitations.
* *Evidence:* Table 2 shows that on Suite A, Uniform merging is highly competitive (89.50%) and TTA is stable. However, on Suite B, Uniform collapses to 51.50%. Online AdaMerging's relative ranking drops on Suite B compared to PolyMerge and OFS-Tune ($62.58\%$ vs. $68.51\%$ and $68.62\%$), showing that performance is highly sensitive to representational conflicts.

### Claim 2: Unconstrained online TTA suffers from transductive overfitting to stream noise.
* *Evidence:* In the high-conflict Suite B, unconstrained online AdaMerging lags behind the polynomial-constrained PolyMerge by $5.93\%$ and exhibits more than double the standard deviation ($\pm 5.71\%$ vs. $\pm 2.52\%$). This is visually supported by Figure 3, where AdaMerging's coefficient trajectory oscillates wildly across layers compared to the ground-truth optimal curve.

### Claim 3: OFS-Tune acts as a robust analytical noise filter.
* *Evidence:* OFS-Tune ($d=2$) matches or outperforms the online PolyMerge across all five suites (achieving **$68.62\%$** in Suite B and **$84.85\%$** in Suite E). Crucially, the **OFS-Unconstrained** baseline lags behind OFS-Tune ($d=2$) by **$8.20\%$** in Suite B and exhibits much higher variance ($\pm 4.98\%$ vs. $\pm 2.45\%$), proving that few-shot validation data alone is highly vulnerable to noise, and the trajectory constraint is a vital and necessary regularizer.

### Claim 4: Physical weight-space deep networks reveal catastrophic online collapse and OFS-Tune superiority.
* *Evidence:* In Section 4.5, physical validation on actual neural network weights demonstrates that:
  - In *scratch-trained independent basins*, simple merging collapses to random guessing ($12.20\%$), and online TTA collapses completely to $15.50\%$ (Unsupervised) and $15.10\%$ (Privileged). OFS-Tune acts as a "safe-by-default" algorithm, identifying incompatibility and preserving a single expert's accuracy offline ($51.70\%$).
  - In a *pre-trained connected basin*, online AdaMerging and PolyMerge actively degrade the robust Uniform weights (dropping accuracy by up to $3.40\%$) due to local stream noise overfitting. OFS-Tune successfully avoids stream noise, outperforming AdaMerging by **$4.20\%$** and PolyMerge by **$3.70\%$**, while matching or exceeding the Uniform baseline ($83.00\%$ vs. $82.20\%$) with zero test-time compute.
  - Standard temporal parameter smoothing (EMA) helps online methods slightly in the connected basin but fails to rescue them in independent basins, and still incurs test-time compute.
  - Symmetrical budget analysis (Section 4.4) proves that online TTA under second-order L-BFGS-B optimization actually *degrades* performance by overfitting to prediction entropy noise, while OFS-Tune converges perfectly using restricted first-order Adam.
