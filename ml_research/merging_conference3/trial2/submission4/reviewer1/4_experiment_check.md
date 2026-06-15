# Experimental Evaluation and Claims Verification

This document provides a critical evaluation of the experimental setup, datasets, baselines, and whether the empirical results actually support the core claims of the submission.

---

## 1. Experimental Setup and Datasets
The empirical evaluation is designed on a robust and standard machine learning benchmark:
- **Diverse Multi-Task Benchmark:** The evaluation spans **8 diverse classification tasks** (SUN397, Stanford Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, DTD), representing a wide range of domains (scenes, fine-grained objects, satellite imagery, digits, traffic signs, textures). This is a comprehensive and appropriate testing ground for multi-task model merging.
- **Backbone Model:** The choice of the modular, pre-trained Vision-Language CLIP ViT-B/32 architecture is standard and appropriate.
- **Fast Evaluation Setting:** Evaluating accuracy on a representative subset of $N=1024$ images per task is a practical choice that accelerates hyperparameter sweeps. The authors' rigorous statistical standard error analysis ($SE_{avg} \approx 0.51\%$) mathematically guarantees that this subset size provides stable, low-variance accuracy rankings, ensuring that the results are highly representative of the full validation sets.

---

## 2. Baselines and Comparisons
The paper is exceptionally commendable for comparing EdgeMerge against a wide array of strong, honest, and diverse baselines, avoiding the common pitfall of comparing against weak baselines:
- **Task Arithmetic (TA):** Instead of using an unoptimized baseline, the authors thoroughly optimized Task Arithmetic by performing a grid search over $\lambda \in [0.10, 0.80]$, establishing a strong empirical peak of **68.74%** at $\lambda = 0.20$.
- **Advanced Static Alignment Baselines:** They compared against advanced weight-space alignment methods: Git Re-Basin (**41.50%**), ZipIt! (**49.30%**), and TIES-Merging (**61.20%**).
- **Server-Grade Adaptive Baselines:** They compared against the state-of-the-art gradient-based test-time adaptation framework, SyMerge (**89.74%**).

This breadth of comparison provides a highly clear and honest picture of where EdgeMerge sits on the performance-resource Pareto frontier (as illustrated in Figure 1).

---

## 3. Do the Results Support the Claims?

### Claims Supported by Evidence:
1. **Extreme Latency and Memory Efficiency:** The resource profile (Table 7) strongly supports the claim of extreme efficiency. Bypassing backpropagation allows EdgeMerge to compute coefficients in **11.95 seconds** on a single forward pass, providing a massive $50\times$ speedup over SyMerge (10 minutes) while keeping GPU memory overhead at a negligible $\sim$100 MB (forward-only).
2. **Transparent Performance-Accuracy Trade-off:** The authors are highly transparent and intellectually honest about the massive accuracy gap compared to server-grade, backpropagation-based methods like SyMerge (68.69% vs. 89.74%). This frames EdgeMerge appropriately as an extreme-efficiency exploration for on-device/edge settings rather than a raw accuracy competitor under unconstrained server environments.
3. **Representational Invariance of CLIP Latent Spaces:** The quantitative results in Table 6 show that running calibration using base features ($X_k^{base}$) vs. expert features ($X_k^{expert}$) yields virtually identical results. This empirically validates the forward shortcut, proving that independent expert visual encoders remain highly aligned.
4. **Decoupled Scale Routing (DSR) Efficacy:** The results in Section 4.3.3 support the claim that decoupling the projection scale ($\lambda_{proj}$) from the static scale ($\lambda_{static}$) resolves the softmax scale discrepancy. DSR increases average accuracy to **68.82%** in the Scale-Aligned regime, and to a peak of **69.58%** in the Bottleneck Regularization regime, representing a statistically significant +0.84% improvement over standard Task Arithmetic.
5. **Hyperparameter Plateau Preservation:** The grid sweeps (Table 5 and Figure 3) support the claim that EdgeMerge provides a wider, more stable performance plateau across scaling factors $\lambda$ than standard Task Arithmetic, which has a fragile and risky peak at $\lambda=0.20$.

### Claims NOT Supported by Evidence:
1. **Efficacy of Activation-Guided Channel-Wise Routing:** The core conceptual claim of the paper is that Scale-Normalized Delta Activation Salience (SNDAS) and Channel-Wise Softmax Gating (CWSG) resolve inter-task weight conflicts by dynamically routing channels to the experts that find them most salient. **The experimental results do not support this claim.**
   - The ablation studies (Section 4.3.4) show that replacing this elaborate channel routing with a completely flat, static distribution ($\alpha_k = 1/K$, uniform gating) yields **exactly the same accuracy (69.58%)**.
   - If the activation statistics were doing meaningful work in routing individual channels to appropriate experts, EdgeMerge (CWSG) should outperform uniform blending, which simply averages all experts uniformly and does not filter out conflicting channel updates. Because they perform identically, the results demonstrate that the proposed channel-routing machinery has **zero functional utility**. 
   - Under standard coupled scaling (coupled EdgeMerge), the best accuracy is **68.69%**, which is actually *worse* than standard Task Arithmetic (**68.74%**). This further shows that the activation-guided routing does not provide any performance benefits over simple weight averaging unless saved by the decoupled scale tuning (DSR).
2. **Robustness to Gating Temperature:** The coupled EdgeMerge sweep shows a severe, localized non-monotonic temperature sensitivity at $\tau = 1.00$, where average accuracy collapses to **51.49%** (requiring a suboptimal global scaling of $\lambda=0.50$). This extreme sensitivity in a key hyperparameter undermines the broader claim of "hyperparameter robustness" under coupled scaling, although the authors correctly demonstrate that DSR resolves this instability.

---

## 4. Summary of Experimental Check
The experiments are conducted on an excellent, diverse benchmark and evaluated with exceptional statistical and baseline comparison standards. 

However, a critical look at the results reveals that the main empirical successes—the performance boost to 69.58% and the stabilization of the hyperparameter plateau—are driven **entirely** by Decoupled Scale Routing (DSR). The elaborate activation-guided channel gating machinery, which is the paper's primary conceptual novelty, is shown to be functionally inert and does not outperform a simple, static uniform scaling baseline.
