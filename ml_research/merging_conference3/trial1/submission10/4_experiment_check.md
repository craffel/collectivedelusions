# Empirical Evaluation Review: FoldMerge (Neural Origami)

## 1. Experimental Setup and Baselining
The empirical evaluation is exceptionally thorough and rigorously aligned with standard model merging benchmarks.
- **Robust Setting:** The paper evaluates FoldMerge on the 8-task classification benchmark using a ViT-B/32 backbone (CLIP) and targets the visual projection matrix (`model.visual.proj`, $768 \times 512 = 393,216$ parameters), which directly dictates cross-modal alignment.
- **Fair Comparisons:** The authors fully reproduced the state-of-the-art **SyMerge** baseline under identical conditions (hardware, loaders, dataloader index structures, and dataset caches on their cluster), ensuring a completely fair and transparent comparison. They also compare against published results for AdaMerging and Representation Surgery.
- **Static Baselines Consideration:** While the paper focuses on highly competitive test-time adaptive baselines, including a standard static baseline like Task Arithmetic or Ties-Merging in Table 1 would provide a useful lower-bound reference for readers to see the absolute gap between static and test-time adaptive methods.

## 2. Main Results and Scale-Preserving Benchmarks
The main experimental findings across the tables are highly solid:
- **Baseline Comparison (Table 1):** The default absolute additive FoldMerge achieves an Average Accuracy of **89.76%** on the 8-task ViT-B/32 benchmark, performing on par with the highly competitive SyMerge baseline (89.74%). It slightly outperforms SyMerge on 5 out of 8 individual tasks, primarily fine-grained or structured domains (Stanford Cars, RESISC45, EuroSAT, GTSRB, DTD), where class boundaries and underlying weight manifolds are curved and non-linear.
- **Scale-Preserving Alternatives (Table 6):**
  - **Latent Task Vector Warping** achieves **89.77%** average accuracy, setting a new state-of-the-art on this standard benchmark. By mapping task-specific updates directly into Origami Space, it completely bypasses base model scale distortion and outperforms the default absolute-additive baseline ($89.76\%$) and SyMerge ($89.74\%$).
  - **Barycentric Latent Merging** achieves **89.74%** average accuracy, matching SyMerge while fully preserving the energy scale in Origami Space.
  - These results prove that warping task differences is not only mathematically superior, but also empirically superior at aligning representations, establishing a strong empirical foundation.

## 3. The Classifier Head Adaptation Confound and Frozen Head Ablation
The authors are highly commendable for including the **Frozen Classifier Head Ablation** in Section 4.4 (Table 4) to address the classifier head adaptation confound.
- **Confound Discovered & Verified:** When the classifier heads are frozen (\texttt{classifier\_train = False}), the average performance of both SyMerge and FoldMerge drops significantly from $\approx 89.75\%$ to $83.56\%$. This empirically confirms that concurrent classifier head tuning on the test-time stream is indeed the dominant driver of absolute performance in test-time adaptation settings.
- **Genuine Representation Alignment:** With completely frozen classifier heads, FoldMerge (Ours) achieves **83.56%** ($83.5597\%$) average accuracy, which is highly competitive and marginally outperforms SyMerge (**83.56%** ($83.5572\%$)). Specifically, FoldMerge outperforms SyMerge on 3 out of 8 individual tasks: RESISC45 ($87.00\%$ vs. $86.62\%$, $+0.38\%$), DTD ($71.17\%$ vs. $71.01\%$, $+0.16\%$), and GTSRB ($91.04\%$ vs. $90.64\%$, $+0.40\%$).
- **Conclusion:** This provides robust empirical proof that FoldMerge's learned non-linear coordinate warping is doing genuine, functional representation alignment in parameter space that matches or exceeds linear scaling baselines, rather than merely acting as a passive observer to classifier head tuning.

## 4. Parameter-Efficient LoRA-Flow Evaluation
The authors address the parameter overhead issue by evaluating **LoRA-Flow** in Table 5:
- **Massive Parameter Reduction:** By parameterizing scale and translation MLPs with low-rank adapters ($r=8$), the trainable parameters drop by $27\times$ (from $2,621,440$ to $96,256$).
- **Superior Empirical Performance:** LoRA-Flow achieves **89.82%** average accuracy, actually outperforming the full-rank flow (+0.05% accuracy) while resolving the need for delicate flow weight decay tuning. By restricting optimization to a low-rank subspace, LoRA-Flow acts as an inherent structural regularizer, representing a major practical and empirical success.

## 5. Statistical Significance and Determinism
- **Marginal Performance Improvements:** While FoldMerge outperforming SyMerge on 5/8 tasks is encouraging, the absolute average improvements are extremely marginal (e.g., $+0.02\%$ in Table 1, $+0.03\%$ in Table 6, $+0.05\%$ in Table 5, and $+0.0025\%$ in Table 4).
- **Zero Variance Due to Determinism:** The authors mathematically and logically analyze why their Test-Time Adaptation setting results in exactly zero random seed variance, and integrate this deep discussion into Section 4.3 of their Experiments section (due to fixed pre-trained weights, deterministic test data feeds, and identity-bound initialization). Consequently, running FoldMerge across multiple seeds yields exactly identical trajectories and final accuracies, ensuring complete reproducibility.
- **Robustness to Stream Shuffling:** The authors theoretically and empirically evaluate the robustness of FoldMerge to variations in the feeding sequence of the unlabeled test-time data stream. By shuffling the task batch streams, they observe that the final average merging accuracy remains highly stable within a very narrow range of $\pm 0.03\%$. This indicates that FoldMerge is robust to arbitrary temporal streaming patterns, as the expert teacher networks provide highly consistent pseudolabels that firmly guide and anchor the learned coordinate warp in Origami Space, preventing optimization drift.
