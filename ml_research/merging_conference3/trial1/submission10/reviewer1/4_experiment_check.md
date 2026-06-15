# 4. Experimental Check

This section critically evaluates the experimental setup, datasets, baselines, and whether the results presented in the paper actually support the core claims.

---

## 1. Experimental Setup and Baselines
- **Datasets & Backbone:** The authors use a standard 8-task classification benchmark on a ViT-B/32 backbone. This is a solid, standard setup in the model merging and test-time adaptation literature.
- **Baselines:** The paper compares FoldMerge against several competitive baselines, including Task Arithmetic, Ties-Merging, AdaMerging, Representation Surgery, and SyMerge (the current state-of-the-art). This list of baselines is appropriate and representative of the state-of-the-art.

---

## 2. Microscopic Empirical Delta over Linear Scaling
The main quantitative results in Table 1 show:
- SyMerge (SOTA linear baseline): **89.74%** Average Accuracy.
- FoldMerge (Ours - default unnormalized): **89.76%** Average Accuracy.
- **The Empirical Delta:** The absolute difference between the highly complex, non-linear FoldMerge and the simple linear SyMerge is a microscopic **+0.02%** in average accuracy.
- **Statistical Significance:** In deep learning, a $0.02\%$ difference is completely within standard statistical noise. Although the authors claim that the test-time adaptation process is deterministic and produces zero run-to-run variance, this determinism is simply a consequence of fixing seeds and sequential dataloading on a single cluster run. It does not imply that $+0.02\%$ is a generalizable, statistically significant improvement on unseen data.
- **Performance Trade-offs:** Out of the 8 individual tasks, FoldMerge actually *underperforms* SyMerge on 3 tasks (SUN397 by $-0.45\%$, SVHN by $-0.17\%$, and MNIST by $-0.05\%$). This shows that the non-linear "Origami" folding is not a robust panacea and can degrade performance even compared to standard linear scaling.

---

## 3. The Classifier Head Adaptation Confound
The paper is exceptionally honest in identifying a major experimental confound in Section 4.4:
- During test-time adaptation, classifier head training is enabled (`args.classifier_train = True`), meaning that the **$388\text{K}$ parameters of the classification heads** are optimized directly on downstream test data using expert teacher predictions.
- **The Frozen Classifier Ablation (Table 5):** When the classifier heads are held completely frozen (`classifier_train = False`), the average accuracy of both methods drops from $\approx 89.75\%$ to **83.56%**.
- **Comparing the Isolated Core Merging:**
  - SyMerge (Frozen): **83.56%** ($83.5572\%$)
  - FoldMerge (Frozen): **83.56%** ($83.5597\%$)
- **Crucial Theoretical Insight:** This ablation is devastating for the empirical claims of FoldMerge. When we isolate the true representation alignment capability of weight-space merging (by freezing the classifier head), the highly complex, non-linear coordinate-warping framework performs **identically** to simple, linear scaling. 
- This empirically proves that the entire non-linear diffeomorphism framework provides **no measurable functional advantage** over standard linear scaling. The vast majority of the test-time adaptation gains are driven by basic, linear gradient optimization of the classifier heads, not by "Neural Origami" warping.

---

## 4. Parameter and Computational Overhead
- **Parameter Footprint:** FoldMerge introduces a 4-layer normalizing flow network of **$\approx 2.6\text{M}$ parameters** ($3.0\%$ of the entire ViT-B/32 backbone). SyMerge, in contrast, optimizes extremely lightweight linear scale factors.
- **Computational Cost:** FoldMerge requires **$1.28$ seconds per optimization step** on high-end NVIDIA H100 GPUs, which translates to **10.6 minutes** of joint optimization for just a single projection layer. 
- **Efficiency and Scalability:** In a real-world test-time adaptation scenario, dedicating 10 minutes of H100 compute to adapt a single layer of a model is highly impractical, especially when simple linear methods (which take seconds or milliseconds) achieve identical accuracy.
- **LoRA-Flow Compression:** While the proposed LoRA-Flow (rank $r=8$) successfully compresses the trainable parameter footprint to $96\text{K}$ and achieves **89.82%** average accuracy ($+0.08\%$ over SyMerge), the optimization step time remains high at **$1.26$ seconds per step** (due to the forward/backward passes through the 4-layer flow network and the analytical inverse). Thus, the computational bottleneck remains largely unresolved.

---

## 5. Summary of Experimental Check
1. The quantitative performance of FoldMerge is virtually identical to the state-of-the-art linear baseline SyMerge ($89.76\%$ vs. $89.74\%$).
2. The frozen classifier ablation reveals that the non-linear coordinate warping provides zero functional improvement over linear scaling when isolated.
3. Almost all test-time adaptation gains are driven by classifier-head optimization, which is a major confound.
4. FoldMerge introduces massive parameter ($\approx 2.6\text{M}$ parameters) and computational (10.6 minutes on H100) overhead without any corresponding performance payoff.

**Experimental Rating:** **Fair** (The experiments are well-designed and the paper is highly transparent, but the results ultimately fail to demonstrate the practical utility or empirical superiority of the proposed method).
