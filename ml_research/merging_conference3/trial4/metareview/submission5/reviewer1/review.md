# Peer Review

## 1. Summary of the Paper
This paper presents **Sparsity-Guided Task Arithmetic (SG-TA)**, a training-free post-hoc weight-space consolidation framework designed to merge multiple task-specific neural network experts into a single, unified multi-task model. The central thesis is that naive Task Arithmetic causes catastrophic representational collision because fine-tuned models adapt their parameters in orthogonal directions, resulting in destructive interference. SG-TA decouples weight-space sparsification by applying absolute magnitude-based binary masks to task vectors prior to scaling and addition, surgically removing low-magnitude updates which act as optimization noise. 

The authors formulate and evaluate:
1. **Global Quantile (GQ) Masking:** A single global magnitude threshold is computed across the entire model, allowing different layers to retain varying densities of parameters.
2. **Layer-wise Quantile (LQ) Masking:** Independent layer-specific thresholds are computed, enforcing a homogeneous parameter budget across layers.
3. **Task Vector Magnitude Normalization (TV-Norm):** Pre-scales task vectors by the inverse of their mean absolute magnitude to address task dominance in joint weight-space.
4. **Sigmoid-Gated Soft Masking (SG-TA-Soft):** Applies continuous sigmoid gates to smooth weight-space boundaries and stabilize hyperparameter optimization.
5. **Non-Uniform Coordinate Search (CS):** A highly scalable, linear-time $\mathcal{O}(T)$ coordinate descent algorithm to optimize task-specific parameters ($k_i, \alpha_i$) in high-dimensional spaces.

Evaluated on a 4-dataset visual classification benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN) using a compact Vision Transformer (`vit_tiny_patch16_224`, 5.7M parameters) backbone, SG-TA GQ achieves a Joint Mean Accuracy of **61.40% $\pm$ 1.39%**, which represents a $+15.08\%$ absolute improvement over Naive Uniform TA ($46.32\%$) and $+2.17\%$ over Optimized TA ($59.23\%$).

---

## 2. Key Strengths (Rigorous Empirical Design)
The paper is exceptionally strong from an experimental and empirical validation perspective:

* **Rigorous and Fair Baseline Optimization:** 
  The authors did not just tune their own method; they fully and fairly optimized all baselines (Naive TA, Optimized TA, TIES-Merging, DARE-Merging, P-then-M, L-Scale, and Fisher-Weighted Averaging) using the exact same Offline Few-Shot Validation Tuning (OFS-Tune) grid across 5 random calibration seeds. This represents an exceptionally high standard of scientific fairness.
* **Inclusion of Joint MTL and Expert Ceilings:** 
  To provide absolute bounds, the paper includes the Dense Experts Ceiling ($95.91\%$) and trains a physical Joint Multi-Task Learning (MTL) baseline ($95.55\%$). This establishes a clear, rigorous target for zero-shot consolidation and defines the true multitask upper bound.
* **Deep Diagnostic and Ablation Studies:** 
  The paper goes beyond reporting simple joint accuracy averages to dissect the mechanics of weight sparsification. These deep dives include:
  * **Keep-ratio sensitivity curves** that contrast GQ and LQ scopes across $k \in [0.1, 1.0]$.
  * **Validation pool size sweeps** ($N_{\text{val}} \in [10, 20, 50, 100]$) to analyze calibration sensitivity.
  * **Direct empirical verification of the Orthogonal Noise Hypothesis** by measuring task vector pairwise cosine similarity (finding high orthogonality, similarities of 0.015 - 0.033) and the even lower similarity of pruned low-magnitude updates (0.0099 - 0.0169).
  * **Landscape stabilization sweeps** under continuous Sigmoid-Gated Soft Masking.
  * **A pilot simulation study** modeling transformer layer specialization in NLP/LLMs to show how GQ dynamically allocates budget compared to LQ.
* **Exceptional Scientific Honesty:** 
  The authors are exceptionally transparent about the limitations of their work. They openly admit that:
  1. The improvement over TIES-Merging is not statistically significant due to overlapping standard deviations ($61.40\% \pm 1.39\%$ vs. $60.64\% \pm 1.30\%$).
  2. A massive absolute performance gap ($34.51\%$) remains between the merged model ($61.40\%$) and the expert ceiling ($95.91\%$), rendering the merged model practically undeployable for high-stakes applications.
* **Practical Calibration Solutions:** 
  The proposed **TV-Norm** successfully balances domain representation (boosting MNIST from $36.74\%$ to $53.70\%$), and **Coordinate Search (CS)** successfully scales high-dimensional task-specific parameter search to linear time $\mathcal{O}(T)$ while rebalancing task performance.

---

## 3. Major Weaknesses and Constructive Critiques
While the empirical rigor is outstanding, several weaknesses limit the paper's overall impact and generalizability:

* **Severe Scale and Generalizability Limitations:**
  The evaluation is restricted to a compact Vision Transformer backbone (`vit_tiny`, 5.7M parameters) on low-resolution toy datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). Modern model merging is primarily utilized for Large Language Models (LLMs) with billions of parameters or larger Vision-Language models like CLIP-ViT-B/16 (86M to 300M+ parameters) on high-dimensional text and image domains. In large-scale models, parameter redundancy is much higher, which can fundamentally alter weight-space collision dynamics. The pilot simulation study with 72 simulated NLP tensors is a valuable step, but it is not a substitute for a physical evaluation on a real language model (e.g., Llama-3, Gemma-2, or Mistral-7B) on standard downstream tasks.
* **Overstated Claims of Superiority:**
  In several prominent locations (Abstract, Introduction, Conclusion), the authors assert that SG-TA "outperforms" state-of-the-art baselines like TIES-Merging and DARE-Merging. However, as Table 1 shows, the Joint Mean Accuracy of SG-TA GQ ($61.40\% \pm 1.39\%$) and TIES-Merging ($60.64\% \pm 1.30\%$) differ by only **$0.76\%$**. Since their standard deviations overlap significantly, this improvement is **not statistically significant**. The authors must soften these superiority claims and instead frame SG-TA as *achieving comparable performance to TIES-Merging while being conceptually and computationally much simpler* (no sign election, no sign-compatibility checks). Simplicity is a major selling point in its own right.
* **Unproven "Layer-Starvation" Hypothesis:**
  In Section 4.3, the authors discuss a crossover phenomenon where LQ masking outperforms GQ at larger keep-ratios ($k \ge 0.7$). They hypothesize that because GQ is unconstrained, at large $k$ it may allow certain layers to become completely dense (retaining 100% of updates) while other layers are excessively pruned (starved), disrupting global representation flow. While highly plausible, **the authors provide no empirical data to back this up**. A plot or table showing the actual distribution of layer budgets under GQ at $k=0.7$ is required to empirically validate this hypothesis.
* **Inadequacy of the Default Few-Shot Validation Size ($N_{\text{val}}=10$):**
  The default OFS-Tune calibration uses $N_{\text{val}}=10$ samples per task. The authors' sweeps show that this extremely small size introduces substantial calibration noise, resulting in a high standard deviation for SG-TA (GQ-Norm) of **$\pm$ 4.56%**. When they double the pool size to a highly manageable $N_{\text{val}}=20$, the variance is slashed by more than 4x to $\pm 1.10\%$, and Joint Mean Accuracy increases to $63.73\%$. Given how cheap it is to collect and evaluate 20 samples per task (still highly "few-shot"), using 10 samples as the default baseline seems unnecessarily restrictive and volatile. The authors should establish $N_{\text{val}}=20$ as the default calibration standard.
* **Oracle Routing Simplification:**
  The setup assumes a "test-time task routing oracle" that routes samples to their corresponding task-specific heads. While common in the literature, this is a major structural simplification that inflates apparent multi-task capabilities. In real-world deployment, the task label is unknown. Reporting joint accuracy under a unified shared head (or a compact routing model) would make the empirical results far more convincing and realistic.

---

## 4. Evaluation Ratings

* **Soundness: Excellent**
  The experimental design is exceptionally thorough, incorporating 5 random seeds, mean and standard deviations, fully and fairly optimized baselines, diagonal Fisher curvature weighting, and a physical Joint MTL baseline. The diagnostic and ablation sweeps (validation pool sweeps, cosine similarity analysis, soft-gating sweeps) represent a textbook example of empirical rigor.
* **Presentation: Excellent**
  The paper is beautifully written, mathematically precise, logically organized, and highly transparent about its performance limits and statistical insignificance. 
* **Significance: Good**
  The systematic insights (global budget flexibility, soft-gating landscape stabilization, TV-Norm, and Coordinate Search) are highly valuable and likely to influence future merging research. However, because the physical experiments are restricted to a tiny model on toy datasets, the practical utility of these findings for LLM practitioners is currently restricted until physically verified at scale.
* **Originality: Good**
  While magnitude-based binary masking and task arithmetic are mature techniques, the specific combination and the systematic dissection of GQ vs LQ, continuous soft gating, and linear-time Coordinate Search represent a highly original, well-reasoned, and valuable set of contributions.

---

## 5. Overall Recommendation and Justification
**Overall Recommendation: 4: Weak Accept**

### Justification:
The paper is a technically solid, exceptionally well-written, and scientifically honest contribution that significantly advances our understanding of spatial regularization in weight-space model merging. Its empirical rigor—demonstrated by fully optimized baselines, 5 random seeds, and extensive ablations—far exceeds the typical standard. 

However, its impact is currently limited by the scale of its evaluation (a 5.7M parameter ViT on low-resolution toy datasets). Because modern model merging is primarily applied to LLMs and large foundation models, it remains an open question whether these exact trends and layer budget allocations generalize to large-scale settings. Therefore, a "Weak Accept" is highly appropriate: it is a solid paper whose contributions are valuable for others to build on, but whose generalizability and immediate impact are restricted by its experimental scale. If the authors can physically verify their method on a real language model or larger foundation model, this work would easily deserve a strong "Accept".

---

## 6. Questions and Actionable Feedback for the Authors

1. **Tone Down Superiority Claims:** Please soften assertions of "outperforming" TIES-Merging in the Abstract, Introduction, and Conclusion. Frame the method instead as *achieving comparable performance to TIES-Merging while being conceptually and computationally much simpler* due to the complete removal of sign election and compatibility protocols.
2. **Empirically Validate GQ Layer Budgets:** To support your "layer-starvation" hypothesis at larger keep-ratios ($k \ge 0.7$), please add a plot or table displaying the actual layer-wise keep-ratios selected by Global Quantile (GQ) masking across different global budgets. This will visually prove if certain layers are indeed starved when $k \ge 0.7$.
3. **Shift Default Calibration Standard to $N_{\text{val}}=20$:** Since doubling the validation pool size from 10 to 20 samples immediately stabilizes the TV-Norm variance by more than 4x (cutting standard deviation from $\pm 4.56\%$ to $\pm 1.10\%$) and boosts accuracy to $63.73\%$, why not establish $N_{\text{val}}=20$ as the default standard for all baseline comparisons in Table 1? This would provide a much more robust and stable baseline.
4. **Code Release:** You claim to provide "reproducible code and detailed sweeps," but no code files were provided. Please include a repository link or provide PyTorch code scripts to ensure full reproducibility and immediate community adoption.
5. **Physical LLM Evaluation:** To resolve the generalizability concern, do you have any physical (non-simulated) evaluation of SG-TA on a real transformer-based NLP model (e.g., fine-tuning and merging on standard NLP datasets)? This would drastically increase the paper's impact and significance.
