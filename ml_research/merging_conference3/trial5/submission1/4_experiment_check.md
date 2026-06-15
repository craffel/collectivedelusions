# 4. Experiment Check

## Quality and Rigor of Experiments
The empirical evaluation is highly rigorous, utilizing several admirable design choices:
- **30-Seed Statistical Validation:** Executing all simulation runs across 30 independent random seeds ensures that the reported improvements (mean and standard deviation) are statistically significant, which is rare and highly commendable in deep learning papers.
- **Unbiased Decoupled Evaluation:** The inclusion of the **Decoupled Isotropic Euclidean Metric** ($\boldsymbol{\Sigma} = \mathbf{I}$) is an outstanding and highly rigorous detail. Because the simulator's standard covariance inverse ($\boldsymbol{\Sigma}^{-1}$) behaves exactly as a graph Laplacian, it inherently favors any spatial smoothing method (including RCR-Merge), creating potential circularity. Evaluating all baselines on the uncoupled metric completely breaks this circularity, proving that RCR-Merge's gains are driven by physical representation protection rather than a self-serving evaluation metric.
- **Stage-wise Modular Transition Landscape:** Evaluating PolyMerge on a modular landscape with discrete boundaries is an excellent way to expose the scientific limitations of rigid global polynomial constraints (curve deformation / Runge's phenomenon).
- **Strong Baselines:** The baselines are highly representative, and unconstrained AdaMerging is evaluated under multiple learning rates ($lr=0.01$ and $lr=0.05$) to ensure a fair comparison.
- **Thorough Ablation and Sensitivity Studies:** Tables 2, 3, 4, and 5 cover hyperparameter sensitivity, regularizer ablations (separating TV and absolute anchoring), drift robustness, and long-term adaptation.
- **Real-World Multimodal Pilots:** Running functional autograd pilot studies on full-scale architectures (`bert-base-uncased` and `vit-base-patch16-224`) demonstrates actual representation collapse and successful stabilization on real physical parameters.

## Empirical Gaps and Flaws

Despite the excellent quality, several empirical limitations must be pointed out:

1. **Scale and Modality of Real-World Evaluation (Gap 1):**
   - Although the real-world pilots on BERT-Base (NLP) and ViT-B/16 (CV) are highly successful and show functional autograd working on 100M-scale models, they remain **small-scale proof-of-concept pilot studies** (evaluated on small, simulated streams of Task 1 inputs).
   - They lack evaluation on standard, high-dimensional out-of-distribution streaming benchmarks, such as ImageNet-C, ImageNet-R, or GLUE streaming domain shifts.
   - For a top-tier machine learning conference (such as ICML, NeurIPS, or ICLR), a complete evaluation on these standardized benchmarks is expected to fully confirm real-world utility.

2. **Lack of Real-world Baselines in Pilots (Gap 2):**
   - In the BERT-Base and ViT-B/16 pilot studies, the authors only compare **Unconstrained AdaMerging** against **RCR-Merge**.
   - They do not implement or evaluate other baselines (like PolyMerge or flat TV-regularized AdaMerging) on the actual BERT or ViT models.
   - It is highly important to verify whether PolyMerge actually suffers from the predicted curve deformation (Runge's phenomenon) on real networks, or whether flat TV-regularization is statistically inferior on real weights. This remains unverified on physical models.

3. **Presentation Flaw: Missing Figure Code and Broken References:**
   - In `04_experiments.tex` (L101 and L104), the text explicitly references `Figure~\ref{fig:visualizations}` (left and right) for qualitative visualizations of coefficient trajectories across depth and sensitivity sweeps over $\beta$.
   - However, **there is no `\begin{figure}` block with label `fig:visualizations` in the entire LaTeX source code!**
   - The image files `rcr_beta_sensitivity.png` and `rcr_merge_trajectory.png` exist in the directory but are completely omitted from the compiled document, resulting in undefined reference warnings and a major presentation gap for readers.

4. **K-FAC and Triggered Pilots are Simulated-Only:**
   - The K-FAC extension and the Threshold-Triggered curvature re-estimation are only tested on the synthetic simulator, leaving their real-world scalability and computational feasibility unverified on actual models.

## Verdict on Experiments
The experiments are **good to excellent** in terms of statistical rigor and scientific design (especially the circularity-breaking decoupled metric). However, they are held back by the lack of full-scale standard benchmark evaluations, missing baselines on real models, and a critical presentation error (the missing figure block).
