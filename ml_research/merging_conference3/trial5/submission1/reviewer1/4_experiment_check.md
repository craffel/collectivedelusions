# 4. Experimental Setup and Evaluation Check

## Evaluation of the Experimental Setup and Datasets

As an empirical reviewer, the experimental evaluation is the most critical and highly scrutinized aspect of this paper. While the authors have constructed a detailed and mathematically rich framework, the empirical execution exhibits several significant limitations that prevent a full endorsement of the paper's claims in their current state.

1. **Predominant Reliance on a Handcrafted Synthetic Simulator**:
   The primary quantitative results (Table 1, Table 2, Table 4, Figure 2) are generated on a synthetic **Coupled Model II Landscape** emulator. 
   While the authors defend this in Section 4.4, arguing that controlled simulations enable high-throughput statistical validation (30 seeds) and causal variables isolation, the fact remains that a synthetic emulator is an idealized, low-dimensional mathematical toy model. A 12-layer, 4-task emulator does not capture the complex, highly non-convex, and multi-dimensional loss landscapes of real deep networks. The transferability of these simulated results to actual deep learning systems remains highly questionable.

2. **Toy Scale and Lack of Rigor in Real-World Pilot Studies**:
   To address the simulation-only constraint, the authors describe two real-world pilot studies in Sections 4.5 and 4.6 (BERT-Base and ViT-B/16). While these are welcome additions, their execution is extremely toy-like:
   - **BERT-Base Pilot Study**: The authors fine-tune sentiment classification and topic classification experts, but evaluate on a tiny simulated stream with simulated expert perturbations. They report a "perfect 100.00% accuracy across both tasks" under RCR-Merge. In real-world NLP, achieving a *perfect 100% accuracy* is virtually impossible and highly suspicious. This suggests that the evaluation was performed on a trivial, small, or heavily sanitized subset of data, which lacks empirical credibility.
   - **ViT-B/16 Pilot Study**: This experiment is executed in "less than 15 seconds on a standard CPU," indicating that it was likely run on only a single batch of size 16 or a tiny handful of samples. This is far from a rigorous empirical validation.
   - **Complete Absence of Statistical Significance in Real-World Settings**: While the authors run 30 seeds on the synthetic simulator, they report zero statistical significance, zero standard deviations, and zero random seeds for the BERT and ViT pilot studies. This makes it impossible to know if the observed stabilization is statistically sound on real models or just a lucky run.

3. **Absence of Standard Out-of-Distribution (OOD) Benchmarks**:
   Standard test-time adaptation and adaptive model-merging papers (e.g., AdaMerging, PolyMerge, Tent) evaluate their methods on widely accepted, large-scale OOD benchmarks. For vision, this includes **ImageNet-C**, **ImageNet-R**, **ImageNet-A**, or **DomainNet** streams. For language, this includes **GLUE**, **MMLU**, or **GSM8K** streams under temporal drift.
   The complete lack of evaluations on these standard benchmarks is a major empirical gap. Without running on actual standard benchmark datasets, we cannot verify if RCR-Merge actually delivers robust performance under real distribution shifts.

## Evaluation of the Baselines

The choice and tuning of baselines are solid in the synthetic simulator but severely lacking in the real-world pilot studies:

- **In the Synthetic Simulator**: The authors compare against a static Uniform Baseline, unconstrained AdaMerging (under both standard $lr=0.01$ and elevated $lr=0.05$ to ensure a fair comparison), PolyMerge ($d=2$), and TV-Regularized AdaMerging. This is a strong and well-tuned set of baselines. The analysis of PolyMerge's failure on the Stage-wise Modular Transition Landscape (Table 2) is a valuable contribution, showing the limitations of rigid polynomial constraints relative to RCR-Merge's local soft barriers.
- **In the Real-World Pilot Studies**: On BERT-Base and ViT-B/16, the authors *only* compare RCR-Merge against unconstrained AdaMerging. They completely omit PolyMerge and TV-Regularized AdaMerging. This is a critical omission. Without comparing RCR-Merge against flat spatial Total Variation on real architectures, we cannot verify if the proposed *curvature-weighting* ($\sqrt{c_l c_{l-1}}$) actually provides any real-world benefit over isotropic smoothing, or if the stabilization is driven purely by flat Total Variation.

## Do the Results Support the Claims?

The empirical results on the synthetic simulator strongly support the core claims of the paper:
- **Catastrophic Overfitting (Table 1)**: Unconstrained AdaMerging drops from 87.45% (Uniform Baseline) down to 77.14% (on coupled) and 84.82% (on decoupled), validating the formulated Overfitting-Optimizer Paradox.
- **RCR-Merge Superiority (Table 1 & 2)**: RCR-Merge successfully resolves this collapse, achieving 90.51% (coupled) and 90.50% (decoupled) on the standard simulator, and 93.53% (coupled) and 93.85% (decoupled) on the modular landscape.
- **Ablation of Regularizers (Table 2 & Table 4)**: The authors' ablation study successfully disentangles spatial TV from absolute anchoring, proving that both are necessary to prevent joint-drift and spatial oscillations.
- **GNB Effectiveness (Table 2)**: Fully automated GNB anchor scaling achieves identical performance (93.85%) to manually tuned hyperparameters.

**However, the real-world claims are NOT sufficiently supported by the data.** The small, toy-scale, single-batch CPU pilot studies on BERT and ViT with simulated stream inputs and "perfect 100% accuracy" are insufficient to prove that RCR-Merge is physically viable and scalable to actual deep vision and language models. To fully support these high-dimensional deployment claims, the authors must conduct a rigorous, multi-seed evaluation on standard real-world datasets and benchmarks, comparing against all baselines with proper statistical significance.
