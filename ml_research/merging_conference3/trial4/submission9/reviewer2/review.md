# Peer Review

**Title:** Exclusive Parameter Merging: Coherence-Preserved Multi-Task Model Fusion  
**Reviewer Recommendation:** 2: Reject

---

## 1. Summary of the Paper
This paper addresses the problem of spatial weight-space interference in weight-space model merging. When task-specific experts fine-tuned from a shared pre-trained base model are merged, standard linear averaging of their parameter updates (Task Arithmetic) often dilutes their distinctive representational features, causing catastrophic performance collapse under highly orthogonal task conflicts. To resolve this, the authors propose **Exclusive Parameter Merging (EPM)**, which introduces:
1. **Soft Exclusive Parameter Allocation (Soft-EPA):** A training-free, coordinate-level routing operator that routes parameter coordinates to a single "dominant" expert (identified via Task Vector Standardization) while attenuating non-dominant updates by a coherence retention factor $\gamma = 0.2$. To mitigate capacity starvation under sparsity, they apply **Dynamic Coherence Scheduling (DCS)**, which scales $\gamma$ with pruning target $p$.
2. **Task-Level Coefficient Tuning (TLC-Tune):** A gradient-free (1+1) Evolution Strategy designed to optimize $K$ global scaling factors on a 128-sample-per-task validation split to maximize a minimax multi-task accuracy metric.

The authors evaluate their method on a 5.7M parameter Vision Transformer (ViT-Tiny) backbone across four simple datasets: MNIST, FashionMNIST, CIFAR-10, and SVHN.

---

## 2. Strengths and Weaknesses

### Strengths
1. **Exceptional Writing and Structural Clarity:** The paper is extremely well-written, with a clear narrative, precise mathematical formulation, and high-quality figures. The transitions between theoretical motivations, methodological designs, and empirical validations are smooth and easy to follow.
2. **Highly Commendable Honesty and Transparency:** Unlike most submissions that gloss over their limitations, the authors are exceptionally rigorous and honest in identifying and detailing their method's weaknesses. Their deep discussions regarding the zero-sum trade-offs of the minimax objective, the capacity limitations of EPM under extreme sparsity, and the optimizer mismatch of the baselines are refreshing and exemplary.
3. **Extensive and Rigorous Sensitivity Analyses:** The paper features comprehensive empirical sensitivity sweeps of the validation calibration size $N_{\text{val}}$, the coherence retention factor $\gamma$, and the optimization budget steps $T$. These analyses provide valuable empirical insights into the stability of low-dimensional black-box search.
4. **Actionable Practical Guidance:** The inclusion of Algorithm 1, which outlines how to apply EPM to decoder-only autoregressive Large Language Models (LLMs), bridges the theoretical aspects of the paper with practical generative AI serving pipelines.

### Weaknesses
1. **Fundamental Technical Flaw: The Scale Mismatch Dilemma**  
   Soft-EPA utilizes a decoupled design where routing decisions are evaluated in a standardized space (divided by the task vector's standard deviation $\sigma_k$ to avoid "Rich Task" dominance), whereas the actual physical parameter updates applied to the model remain in the unstandardized space. This is a mathematically inconsistent formulation. If a simple task (like MNIST) has a very small global variance, its standardized score is artificially inflated, allowing it to win the coordinate-wise argmax even if its physical update is tiny. When integrated, this tiny update is applied at full strength, while the physically massive update of a complex task (like SVHN) is heavily attenuated (multiplied by $\gamma=0.2$). This scale mismatch actively destroys the representations of complex expert models, which is empirically confirmed by the catastrophic collapse of CIFAR-10 accuracy from **75.83%** down to **36.98%** under TLC-Tuned EPM.
2. **Destructive Coordinate-Wise Topography Scrambling**  
   Neural network weights are highly interdependent and operate as structured layers to transform activation manifolds. Soft-EPA performs routing independently at the individual $1 \times 1$ scalar coordinate level. Scrambling 13.8% of the model's 5.52 million parameters via coordinate-wise argmax destroys the joint covariance structure of the weight matrices, leading to representation drift in deeper layers. The authors' claim that $\gamma = 0.2$ acts as a "topological glue" is a post-hoc heuristic rather than a mathematically rigorous solution to representation scrambling.
3. **Methodological Injustice to Baselines (Optimizer Mismatch)**  
   To support their core claim regarding the "Overfitting-Optimizer Paradox," the authors evaluate SOTA layer-group-wise tuning methods (AdaMerging and ZipMerge) under a zero-order (1+1)-ES optimizer. However, these methods were specifically designed to optimize their 56- and 70-dimensional continuous spaces using first-order gradient descent on differentiable cross-entropy losses. Forcing them to use a non-differentiable minimax accuracy objective and a single-point zero-order search is an artificial setup where they are mathematically expected to fail. The "paradox" is not a fundamental limitation of these baselines, but an artifact of a crippled evaluation protocol. A fair comparison requires evaluating these baselines under their native, first-order continuous optimization pipelines.
4. **Unrepresentative and Highly Artificial Toy Evaluation**  
   The entire empirical evaluation is restricted to a compact ViT-Tiny model (5.7M parameters) on four simple, low-resolution toy datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) upsampled to 224x224. This is an extremely artificial and unrepresentative setting. Model merging is typically utilized on massive LLMs or CLIP backbones where overparameterization provides a highly redundant, high-dimensional weight space that can absorb conflicts. On ViT-Tiny, the absolute capacity bottleneck causes massive performance degradation, yielding a merged model with a joint mean of only **46.19%** (compared to a 94.91% joint expert ceiling). A model that gets only 48% on MNIST and 36% on CIFAR-10 is practically useless, which severely limits the practical significance of this work.
5. **Direct Empirical Refutation of the "Coordinate Exclusivity" Hypothesis**  
   The authors argue that "coordinate exclusivity" is the core driver of EPM's success. However, their own results in Table 2 ($p=0.5$) directly refute this. The "Standardized TA + Pruning" baseline (EPM with $\gamma=1.0$, which corresponds to standard Task Arithmetic with standardized pruning and zero coordinate exclusivity) achieves a joint mean accuracy of **44.87%**, outperforming EPM with TLC-Tune (**42.60%**). This proves that standard average blending actually outperforms coordinate exclusivity overall, and EPM's "balanced" performance is merely the result of the minimax objective forcing the parameters to favor MNIST at the expense of other tasks, rather than a superior routing mechanism.
6. **Lack of Expectation-Value Scale Preservation under Sparsity**  
   Under high target sparsity ($p=0.8$), EPM simply zeroes out pruned coordinates. Unlike DARE, which rescales remaining parameters by $1/(1-p) = 5.0$ to preserve the expectation value of the activation scales, EPM performs no scaling. This leads to severe activation magnitude decay, causing EPM to collapse to **26.41%** joint mean at $p=0.8$ compared to DARE's robust **40.90%**.

---

## 3. Detailed Ratings

### Soundness: Poor
The proposed Soft-EPA routing method suffers from a severe, mathematically inconsistent scale mismatch where standardized coordinates dictate routing but unstandardized weights are physically integrated. Furthermore, the coordinate-wise independent routing scrambles joint weight structures, and the paper's claims regarding baseline failure are based on a heavily crippled optimizer mismatch.

### Presentation: Excellent
The paper is exceptionally well-written, mathematically clear, and highly structured. The discussion of limitations and the detailed sensitivity sweeps are incredibly thorough, transparent, and exemplary.

### Significance: Poor
Because the empirical evaluation is restricted to a toy 5.7M parameter ViT on simple datasets, resulting in a merged model with a practically useless 46% accuracy, the immediate practical utility of this work is exceptionally weak. Practitioners are highly unlikely to adopt this method for real-world deployments.

### Originality: Fair
The proposed method represents an incremental combination of standard Task Arithmetic with a coordinate-wise argmax mask and standard hyperparameter tuning via (1+1)-ES. The conceptual framing is heavily inflated to mask what is fundamentally a very simple heuristic.

---

## 4. Questions and Comments for the Authors
1. **Baseline Optimization:** Why did the authors not evaluate AdaMerging and ZipMerge under their native first-order gradient descent pipelines on the validation split? This is essential to prove whether their failure is due to a genuine "Overfitting-Optimizer Paradox" or simply due to the zero-order optimizer mismatch.
2. **Scale Mismatch:** Can the authors provide a mathematical proof or theoretical justification for why routing based on standardized task vectors combined with physical integration of unstandardized task vectors does not violate representation scale consistency? How does this not explain the severe 31% collapse of CIFAR-10 performance?
3. **Scale Preservation under Sparsity:** Why does EPM not incorporate an expectation-value scaling factor (like DARE's $1/(1-p)$) under sparse merging to prevent the severe activation scale decay that leads to its collapse at $p=0.8$?
4. **Large-Scale Evaluation:** Given that model merging is primarily used on Large Language Models, why is there no empirical evaluation on actual generative LLMs or CLIP backbones?
