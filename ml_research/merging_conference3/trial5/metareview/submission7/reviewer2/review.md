# Peer Review: Pruned Gradient Merging (PG-Merge)

## Overall Recommendation
**Score:** 4: Weak Accept (Technically solid paper that advances test-time model merging with a minimalist contribution that others are likely to build on, but with some weaknesses in contextualization and evaluation scale that limit its immediate impact).

---

## Reviewer Ratings
- **Soundness:** Good (The mathematical formulation is rigorous, and the dynamic sparse masking is conceptually well-motivated, but strong speculative claims about SGD are left entirely unverified empirically).
- **Presentation:** Excellent (The paper is exceptionally well-structured, clear, and easy to follow. Visual aids and tables are high-signal).
- **Significance:** Good (The paper provides a valuable and refreshing conceptual "reality check" for the model merging community by deconstructing unnecessary complexity. Practical impact is currently bounded by the small evaluation scale).
- **Originality:** Good (Applying Top-$k$ gradient selection to active test-time model merging is an elegant, simple, and effective combination of existing optimization primitives).

---

## Paper Summary
This paper addresses the challenge of unsupervised **test-time model merging** (specifically in online test-time adaptation regimes under the AdaMerging framework). It targets the **Overfitting-Optimizer Paradox**, where unconstrained optimization of merging coefficients on tiny, unlabeled calibration streams minimizes prediction entropy by overfitting to local transductive noise, causing catastrophic multi-task representational collapse. 

To resolve this, the authors propose **Pruned Gradient Merging (PG-Merge)**. Guided by the principle of Occam's razor, PG-Merge applies a dynamic, non-parametric binary sparse gradient mask that freezes the vast majority ($85\%$) of merging coefficients and adapts only the most critical, high-sensitivity coordinates (the top-$15\%$) at each step. To prevent momentum-driven updates of inactive parameters under adaptive optimizers (like Adam), a post-update parameter projection is applied. Evaluations on a Vision Transformer (`vit_tiny`) across MNIST, FashionMNIST, CIFAR-10, and SVHN demonstrate that PG-Merge substantially stabilizes online adaptation, outperforming unconstrained AdaMerging and matching or exceeding more complex SOTA regularizers (such as RegCalMerge) with zero auxiliary loss terms or hyperparameter bloat.

---

## Major Strengths

1. **Elegant Conceptual Deconstruction:** The paper makes a highly compelling and refreshing case for simplicity in model merging. It demonstrates that the Overfitting-Optimizer Paradox is fundamentally a degrees-of-freedom problem, and shows that complex, hyperparameter-heavy SOTA regularizers (like RegCalMerge's elastic spatial penalties or PolyMerge's rigid trajectories) are largely redundant when compared to simple gradient sparsification.
2. **Clear Mathematical Formulation:** The proposed method is mathematically precise, self-contained, and clean. Sibling baselines are written in consistent notation, and Equations 1–12 outline the entire pipeline with exceptional clarity.
3. **Rigorous Optimization Trajectory Analysis:** Section 4.4 and Figure 3 provide deep, high-signal validation of the "paradox in action." The paper beautifully captures how unconstrained AdaMerging overfits (loss decreases while accuracy collapses) whereas PG-Merge successfully balances entropy minimization and multi-task generalization.
4. **Insightful Appendices:** The discussion on optimizer state decay/momentum mismatch (Appendix A) and active mask stability (Appendix B) addresses highly technical nuances of the proposed framework, showing that the authors are thinking deeply about the underlying optimization mechanics.

---

## Key Weaknesses and Areas for Improvement

### 1. Gaps in Scholarly Contextualization and Historical Attribution
While the paper is well-situated within recent model-merging literature, it exhibits significant gaps in connecting its core mechanics to the broader historical and concurrent literature of deep learning optimization and test-time adaptation (TTA):
- **Connection to Top-k Distributed Optimization:** The mathematical operation at the heart of PG-Merge—sorting absolute gradient coordinates and keeping only the top-$p\%$ components—is a direct, well-known descendant of **Top-$k$ gradient sparsification/compression** from the distributed deep learning literature (e.g., *Deep Gradient Compression* by Lin et al., 2017; *QSGD* by Alistarh et al., 2017). Framing this solely as a parameter-efficient fine-tuning (PEFT) analog misses a major historical connection. The paper should explicitly attribute Top-$k$ gradient selection to its distributed optimization origins.
- **Selective Test-Time Adaptation Lineage:** Restricting parameter updates to a specialized subset is a foundational stabilizing technique across the history of TTA. For instance, *Tent* (Wang et al., 2021) stabilizes entropy minimization by updating only Batch Normalization parameters, and *Parameter-Selective Mean Teacher (PSMT)* (2024) uses Fisher Information masks to update a sparse subset of weights to prevent catastrophic forgetting. PG-Merge does the exact same thing but at the level of merging coefficients instead of model weights. Integrating PG-Merge into this broader context of "selective updates for TTA stability" would make the paper feel much more academically grounded.
- **Missing Citations:** The authors mention "concurrent works like QWS-Merge" in Section 1 and Section 2.2, but fail to provide a formal bibliographic citation for it. Furthermore, they should explicitly clarify if the term "Overfitting-Optimizer Paradox" was introduced by *RegCalMerge/PolyMerge* or if they are formalizing it here.

### 2. Significant Empirical Gaps
- **"SGD Compatibility" Claim completely Unverified (Appendix A):** In Appendix A, the authors make a strong theoretical argument that standard SGD without momentum is the mathematically "ideal" optimizer for PG-Merge, as it naturally keeps masked parameters frozen without needing the post-update parameter projection (Equation 12) or causing optimizer state decay. **However, the paper contains absolutely zero empirical results comparing SGD against Adam.** To support the claims of Appendix A, parallel experiments evaluating PG-Merge + SGD must be provided.
- **Missing Standard Static Baseline (TIES-merging):** Despite citing and discussing **TIES-merging** (Yadav et al., NeurIPS 2023) in Section 2.1 as a prominent method that prunes redundant parameters and consensus-routes task vectors, TIES-merging is completely omitted from the quantitative scoreboard in Table 1. As the standard benchmark in model merging, its inclusion is crucial.
- **Initialization and Architectural Ambiguities:** The paper never states the initial values of the merging coefficients ($\alpha$) before TTA begins (e.g., are they initialized to $0.3$, $0.0$, or $1/K$?). Furthermore, since the four tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) have distinct output spaces, a shared classification head is impossible. The authors must explicitly clarify how the task-specific classification heads are routed, and whether they are kept frozen or adapted during test-time.

### 3. Scale and Generalizability Limits
The empirical validation is restricted to a highly compact `vit_tiny` model ($5.7$M parameters) trained on toy-scale datasets (MNIST, CIFAR-10, SVHN) with only $1,024$ images. Efficacy on modern, large-scale foundation networks (such as LLMs or large CLIP/Vision models), where parameter-space model merging is most vital and where optimization landscapes are significantly more complex, remains unverified. 

### 4. Nuanced Task Performance and Marginal Average Gains
While PG-Merge ($p=0.05$) achieves the highest average Joint Mean Accuracy ($62.70\%$), a task-by-task breakdown reveals a highly nuanced landscape:
- On **MNIST**, static **Uniform Merging achieves $65.04\%$**, outperforming PG-Merge ($p=0.05$) by **$1.76\%$**.
- On **SVHN**, standard **PolyMerge achieves $40.43\%$**, dramatically outperforming PG-Merge ($p=0.05$) by **$8.40\%$**.
PG-Merge is actually outperformed on half of the individual datasets by simpler or concurrent baselines. Furthermore, the absolute gain of PG-Merge ($62.70\%$) over standard static Uniform Merging ($62.16\%$) is just **$+0.54\%$**, which raises questions about the practical utility of backpropagating on test-time streams for such small gains. The authors should tone down claims of "dramatically outperforming" other baselines to reflect this reality.

---

## Actionable Questions and Suggestions for the Authors

1. **Contextualization:** Please enrich the Related Work and methodology sections by explicitly discussing PG-Merge's connection to:
   - **Top-$k$ gradient sparsification** from the distributed optimization literature (e.g., Deep Gradient Compression).
   - **Selective parameter update strategies in TTA** (e.g., Tent, PSMT), illustrating how selective updates of merging coefficients represent a natural extension of this line of thought.
   - Please provide a formal bibliographic reference for **QWS-Merge**.
2. **SGD Evaluation:** Please provide empirical results for PG-Merge when paired with standard SGD as advocated in Appendix A. Does PG-Merge + SGD achieve comparable or superior performance to PG-Merge + Adam? Does it successfully prevent the representation decay associated with Adam's decaying momentum states?
3. **Baselines:** Please add **TIES-merging** to the scoreboard in Table 1 to provide a more rigorous static baseline comparison.
4. **Architectural and Initialization Details:** 
   - How are the merging coefficients $\alpha_{k, l}$ initialized before test-time adaptation?
   - How are the task-specific classification heads handled and routed across the four distinct tasks, and are they optimized or kept frozen during TTA?
5. **Scale:** Do you have any preliminary results or observations on larger models (e.g., ViT-Base or RoBERTa)? How do you expect the "sparsity sweet spot" $p$ to scale with larger parameter counts?
