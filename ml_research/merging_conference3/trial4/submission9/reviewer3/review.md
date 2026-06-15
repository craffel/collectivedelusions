# Peer Review

## 1. Summary of the Paper
This paper addresses the problem of spatial weight-space interference in weight-space model merging. When merging task-specific expert models that share a pre-trained initialization, standard averaging (Task Arithmetic) or sign-consistent averaging (TIES-Merging) can cause conflicting updates from orthogonal tasks to cancel out, resulting in representational collapse. 

To mitigate this, the authors propose **Exclusive Parameter Merging (EPM)**, a training-free coordinate-level routing operator. EPM introduces:
1. **Soft Exclusive Parameter Allocation (Soft-EPA):** A coordinate-wise routing protocol that assigns updates to the dominant expert based on scaled magnitude, while attenuating non-dominant updates by a coherence factor $\gamma = 0.2$. The paper shows that Soft-EPA is mathematically equivalent to a convex combination of hard coordinate-wise exclusivity ($\gamma = 0$) and standard Task Arithmetic ($\gamma = 1$).
2. **Task Vector Standardization:** To prevent "rich" tasks with naturally larger gradients (e.g., SVHN or CIFAR-10) from dominating the routing, the task vectors are standardized globally (or layer-wise) by their standard deviation ($\sigma_k$).
3. **Dynamic Coherence Scheduling (DCS):** Under sparse merging, $\gamma$ is scaled dynamically using a quadratic schedule: $\gamma(p) = \gamma_0 + (1-\gamma_0) \cdot p^2$, where $p$ is target sparsity.
4. **Task-Level Coefficient Tuning (TLC-Tune):** A low-dimensional scale optimization that tunes only $K$ global scaling factors using a zero-order (1+1) Evolution Strategy on a tiny offline validation split of 128 samples per task.

Empirical evaluations are conducted on a Vision Transformer (ViT-Tiny, 5.7M parameters) across four conflicting image datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). The paper reports that EPM (TLC-Tune) achieves 46.19% joint mean accuracy under dense merging and 42.60% under 50% sparsity.

---

## 2. Strengths and Weaknesses

### Strengths
1. **Thorough Sensitivity and Ablation Studies:** The paper includes detailed sensitivity analyses for the coherence factor $\gamma$ (Table 4) and validation size scaling (Table 6). The 500-step optimization study trajectory (Figure 2) provides helpful visual insights into the convergence of different scale parameterizations.
2. **Transparent Discussion of Practical Trade-offs:** The authors provide a highly detailed and honest discussion (Section 4.3) comparing the operational trade-offs of Joint Multi-Task Fine-Tuning, Parameter-Efficient Fine-Tuning (PEFT/LoRA), and weight-space model merging. They do not hide the absolute accuracy degradation, framing it as a necessary trade-off for zero-oracle, zero-overhead edge deployment.
3. **Structured Algorithmic Blueprints:** The mathematical identity showing that Soft-EPA behaves as a convex combination is well-formulated, and Algorithm 1 provides a concrete, step-by-step implementation blueprint for decoder-only Large Language Models.

### Weaknesses
1. **Toy Experimental Scale & Lack of Domain Realism:** The experiments are restricted entirely to a toy-scale model (`vit_tiny`, 5.7M parameters) and four extremely basic image classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). Merging experts across these completely disjoint and artificial domains is highly unrealistic and unrepresentative of modern model merging tasks (which typically focus on Large Language Models or large multi-modal models). The paper provides zero empirical evidence that EPM scales or generalizes to modern foundation models.
2. **Unfair Baseline Evaluation (Deliberate Optimizer Mismatch):** To demonstrate the superiority of TLC-Tune, SOTA continuous adaptation methods (AdaMerging and ZipMerge) are evaluated under a zero-order (1+1) Evolution Strategy. However, both of these baselines were explicitly designed to be optimized using native first-order gradient descent on differentiable validation cross-entropy losses. Forcing these high-dimensional continuous parameter spaces into a greedy, single-point zero-order search and then citing their flat trajectories as "absolute optimization failure" is a highly contrived, unfair comparison. Under their native first-order gradient descent pipelines, these baselines would converge efficiently.
3. **Severe Performance-Utility Gap:** The individual expert models achieve test accuracies of 91.31% to 98.74% (joint ceiling of 94.91%). In contrast, EPM (TLC-Tune) peaks at a joint mean of only **46.19%** (dense) and **42.60%** (50% sparse). An absolute accuracy of ~46% on standard digit and fashion classification is practically useless. A practitioner serving these tasks on an edge device would easily achieve $>$90% accuracy with negligible latency or parameter overhead using simple task-specific LoRA adapters or MoE routing, rendering the proposed merging method functionally obsolete.
4. **Logical Contradiction of Scale Decoupling:** In Soft-EPA, decision routing is evaluated in a standardized space (divided by $\sigma_k$), but the unstandardized physical updates are integrated into the network. The authors report a scale override rate of 13.79% (761,836 parameters), meaning nearly 1 in 7 coordinates are routed based on an artificial standardized scale. This physical decoupling likely severs critical learned representation channels from the complex experts (like SVHN or CIFAR-10), causing severe representational fragmentation. This fragmentation explains why the merged model collapses so heavily compared to individual experts.
5. **EPM is heavily Outperformed by Averaging and Baselines:**
   * Under 50% sparsity (Table 2), standard averaging ("Standardized TA + Pruning", which is equivalent to EPM with $\gamma=1.0$) achieves a joint mean accuracy of **44.87%**, outperforming TLC-Tuned EPM's **42.60%**. This indicates that standard average blending performs better than coordinate exclusivity under moderate sparsity. The authors claim EPM is superior because it protects MNIST (the worst-performing task), but sacrificing SVHN and CIFAR-10 (which are much harder and more complex) to boost MNIST is a highly questionable trade-off.
   * Under 80% sparsity (Table 3), the standard **DARE** baseline achieves a joint mean of **40.90%**, completely dominating TLC-Tuned EPM (**26.41%**) by **14.49% absolute accuracy**. This highlights that EPM's coordinate exclusivity suffers from severe capacity starvation when parameters are heavily pruned.
6. **Empirical and Fragile Hyperparameter Heuristics:**
   * **Dynamic Coherence Scheduling (DCS):** The quadratic rule $\gamma(p) = \gamma_0 + (1 - \gamma_0) \cdot p^2$ is an empirical polynomial heuristic with no theoretical derivation or mathematical justification.
   * **Global Standardization:** Computing a single $\sigma_k$ across all $D$ parameters of the model (flattening attention projections, feed-forward weights, biases, layer-norm gains, and tokens) is mathematically questionable and ignores layer-specific scale dynamics.
   * **Fragility of TLC-Tune:** The authors report a localized performance dip for TLC-Tune at $N_{\text{val}}=512$ where joint mean accuracy drops to **35.36%**. They admit that (1+1)-ES gets trapped in suboptimal local minima, requiring "Multi-Start" or "CMA-ES" to recover. This directly contradicts their claims of "absolute stability" and "perfect generalization."

---

## 3. Soundness
* **Rating:** **Fair**
* **Justification:**
  While the mathematical formulations are formally written, several core methodologies are unsound. Computing a single global standard deviation across heterogeneous parameter tensors (e.g., weights vs biases, attention vs MLP) ignores localized layer physics. Furthermore, routing 13.8% of the parameter coordinates based on standard deviation scales while physically integrating unstandardized weights introduces a fundamental logical contradiction that likely fragments representations and causes the observed severe performance drops. Finally, the evaluation of baselines (AdaMerging and ZipMerge) relies on a contrived optimizer mismatch, disabling their native first-order pipelines to make EPM's zero-order search look superior.

---

## 4. Presentation
* **Rating:** **Good**
* **Justification:**
  The paper is well-structured, logical, and easy to follow. The mathematical notation is clean and consistent, and Figures 1 and 2 are highly polished and professional. However, the presentation is excessively verbose and preemptively defensive in Section 4.3, spending multiple pages trying to defend the method against obvious critiques regarding optimizer mismatch, performance gaps, and toy datasets. A more concise, objective presentation would be highly preferred.

---

## 5. Significance
* **Rating:** **Poor**
* **Justification:**
  The practical utility and significance of the proposed framework are extremely low. Sacrificing expert performance (collapsing from ~95% to ~46% joint mean accuracy) on standard digit and fashion benchmarks makes the merged model non-viable for any actual production deployment. Simple PEFT/LoRA alternatives or ensembles achieve far higher accuracies with negligible overhead. Furthermore, because the entire empirical validation is limited to a toy-scale model (ViT-Tiny, 5.7M parameters) and toy datasets, the theoretical claims regarding overparameterization in modern Large Language Models remain speculative and unsubstantiated.

---

## 6. Originality
* **Rating:** **Fair**
* **Justification:**
  The conceptual novelty is highly incremental. Coordinate-level selection and magnitude-based routing are well-explored in existing literature (e.g., TIES-Merging). Normalizing updates by standard deviation (Z-score normalization) and tuning task-level scaling factors via zero-order evolutionary strategies are direct applications of standard ML techniques. The proposed Soft-EPA routing is mathematically shown to be a simple linear interpolation between hard exclusivity and standard Task Arithmetic.

---

## 7. Overall Recommendation
* **Rating:** **2: Reject**
* **Justification:**
  This paper suffers from major weaknesses that outweigh its merits:
  1. The experimental scale is restricted entirely to toy settings (ViT-Tiny on MNIST/CIFAR-10) with no empirical validation on large-scale models.
  2. The absolute merged accuracy (~46%) represents a severe and non-viable performance collapse compared to the individual experts (~95%).
  3. The baseline evaluations are highly biased, relying on a contrived optimizer mismatch (disabling native first-order optimization for AdaMerging and ZipMerge).
  4. Standard averaging ("Standardized TA + Pruning") outperforms EPM at 50% sparsity, and DARE completely dominates EPM (by 14.49% absolute) at 80% sparsity, undermining the core claims regarding coordinate exclusivity and DCS.
  5. The scale decoupling and global standardization methods contain logical inconsistencies and ignore layer-specific parameter scales.

---

## 8. Questions for the Authors / Constructive Feedback

1. **Why was the baseline comparison conducted using an optimizer mismatch?** Please provide empirical results for AdaMerging and ZipMerge under their native, first-order gradient descent optimization pipelines using differentiable validation cross-entropy.
2. **Can you scale EPM to modern, large-scale architectures?** To verify your theoretical hypotheses regarding overparameterization, please provide empirical merging results on modern generative Large Language Models (e.g., Llama-3 or Mistral) or Large Vision-Language Models (e.g., CLIP).
3. **What is the mathematical justification for Dynamic Coherence Scheduling (DCS)?** Please provide a rigorous mathematical or probabilistic derivation for why the coherence retention factor should scale quadratically with target sparsity ($p^2$), rather than presenting it as an empirical polynomial heuristic.
4. **How do you address the global standardization of heterogeneous parameters?** Standardizing self-attention projection weights, feed-forward weights, biases, and layer-norm parameters with a single, global standard deviation $\sigma_k$ ignores layer physics. Please provide a comparison of Global vs. Layer-wise standardization across all tasks.
5. **Why should a practitioner accept a merged model with 46% accuracy?** Given that simple PEFT/LoRA adapters or dynamic routing networks preserve expert-level accuracies ($>$90%) on these basic datasets, please provide a concrete real-world vision deployment scenario where a merged model with 46% accuracy is preferred.
6. **Please provide the PyTorch code snippet of the core EPM routing protocol to verify reproducibility.**
