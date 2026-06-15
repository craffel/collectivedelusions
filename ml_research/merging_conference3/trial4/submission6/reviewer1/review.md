# Peer Review

## Summary of the Paper
This paper presents a deconstructive critique of modern sparse model merging techniques, specifically challenging the necessity of coordinate-wise sign voting, sign consensus, and stochastic scaling heuristics popularized by methods like TIES-Merging and DARE. Guided by Occam's razor, the authors propose a minimalist, training-free alternative: **Sparse Task Arithmetic (STA)**. STA consists of extracting task vectors, applying uniform layer-wise magnitude-based pruning (retaining the top-$s$\% largest absolute updates), and directly summing the sparse updates.

Crucially, the paper identifies a major methodological confounder in previous evaluations: **update under-scaling** (due to parameter magnitude attenuation after pruning). The authors introduce two scale-preserving variants to address this:
1. **Rescaled STA (R-STA)**: Scaling sparse vectors by the inverse survival density ($100/s$) to preserve vector energy.
2. **Tuned STA**: Keeping the sparse updates intact but dynamically optimizing the scaling coefficient $\lambda$ to align the sparse vector sum with the pre-trained feature space.

Evaluated on a 4-task classification benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN) using a ViT-B-32 backbone, Tuned STA ($s=20\%$) matches the performance of TIES-Merging and DARE under perfectly fair, symmetric hyperparameter tuning. The authors also provide a mathematical and empirical analysis demonstrating that coordinate-wise collisions are extremely rare (ranging from 3.1% to 4.3% across layers at $s=20\%$), which explains why sign consensus heuristics are largely redundant.

---

## Strengths and Weaknesses

### Strengths
1. **High Practical Utility and Engineering Simplicity**: The proposed STA algorithm is exceptionally clean, straightforward, and easy to deploy. Instead of complex coordinate-wise voting pipelines that require hundreds of lines of logic, STA is implemented in just three core lines of PyTorch code (as demonstrated in Appendix A). For real-world engineering and production systems, this extreme simplicity reduces engineering debt, serving latency, and implementation fragility.
2. **Exposing the Under-scaling Confounder**: The identification of the update under-scaling confounder is a high-signal contribution. It demonstrates that the previous failures of simple sparse merging were not due to parameter sign conflicts (which complex heuristics were designed to resolve), but rather due to simple magnitude attenuation. Correcting this scale imbalance solves the performance gap.
3. **Rigorous and Fair Evaluation Protocol**: The paper's experimental protocol is highly commendable. Instead of comparing a tuned STA to un-tuned baselines, the authors execute a complete symmetric sweep over the scaling coefficient $\lambda \in [0.1, 1.0]$ for all methods. This ensures that every baseline (including Task Arithmetic and TIES-Merging) is compared at its peak performance, making the empirical claims exceptionally solid.
4. **Insightful Mathematical and Empirical Analysis**: The theoretical analysis of mask overlap based on the $(s/100)^2$ independence bound and its empirical verification (3.1%–4.3% overlap rate) are elegant. It mathematically refutes the core assumption of TIES-Merging by proving that parameter collisions are rare and negligible in reasonable sparsity regimes.

### Weaknesses
1. **Critical Literature Gap and Terminology Collision**: The paper presents "Sparse Task Arithmetic" (STA) as a novel technique introduced in this work. However, the authors omit any citation to or discussion of the paper **"Localize-and-Stitch: Efficient Model Merging via Sparse Task Arithmetic"** by Yifei He et al. (arXiv:2408.13656), which was accepted at *Transactions on Machine Learning Research (TMLR)* in December 2024. He et al. (2024) already coined the exact term "Sparse Task Arithmetic" and proposed sparsifying task vectors to merge models. 
   While He et al. (2024) use a more complex, optimization-based localization process to identify a task-specific mask, this submission's focus on uniform layer-wise magnitude pruning and deconstructing sign-consensus is distinct. Nevertheless, the omission of this foundational paper must be addressed. The authors must cite He et al. (2024), contextualize their work, and rename or qualify their method (e.g., "Simple Magnitude-based Sparse Task Arithmetic") to avoid name collisions and maintain academic integrity.
2. **Lack of Scale and LLM Evaluation**: Model merging has gained immense traction in natural language processing (NLP) for merging instruction-tuned Large Language Models (LLMs) with billions of parameters (e.g., Llama, Mistral). The paper evaluates its method exclusively on a small Vision Transformer (ViT-B-32, ~86M parameters) on low-dimensional image classification benchmarks. While this setup is useful for reproducibility, it remains unproven whether the deconstruction of sign consensus holds at the scale of modern industrial LLMs, where deep causal attention and massive vocabularies introduce different representation dynamics.
3. **No Evaluation on Highly Similar Tasks**: The evaluated vision suite consists of diverse and unrelated domains, leading to independent and extremely low mask overlaps (3.1%–4.3%). In practice, model merging is often applied to merge models trained on highly similar domains or overlapping datasets (e.g., different coding or math reasoning LLMs). Under these conditions, the coordinate collision rate will be substantially higher than the independence bound. The authors theoretically argue that sign conflicts are self-resolving under overlap, but they provide no empirical validation under high task similarity.

---

## Detailed Ratings

### Soundness
* **Rating**: **Good**
* **Justification**: The methodology is theoretically well-grounded, and the identification of the under-scaling confounder is correct and crucial. The mathematical proof of collision rates is verified empirically. The experimental protocol is fair and rigorous due to the symmetric hyperparameter tuning. However, the soundness of the claims is slightly limited by the lack of evaluations on highly correlated/overlapping tasks and larger model architectures.

### Presentation
* **Rating**: **Good**
* **Justification**: The paper is exceptionally well-written, clear, and easy to follow. The figures and tables are highly informative and directly support the narrative. The inclusion of PyTorch code in Appendix A is outstanding for reproducibility and implementation. The rating is set to "Good" instead of "Excellent" solely due to the critical literature gap (the missing citation and discussion of the existing TMLR 2024 "Localize-and-Stitch: Efficient Model Merging via Sparse Task Arithmetic" paper).

### Significance
* **Rating**: **Good**
* **Justification**: The paper's contribution is highly significant because it challenges a major trend of over-engineering in the model-merging literature. Demonstrating that complex sign-consensus heuristics can be replaced by a simple 3-line magnitude-based pruning loop is a major win for practitioners. It drastically simplifies the development, hyperparameter tuning, and production serving of merged models. However, its overall significance is currently constrained by the lack of NLP and LLM benchmarks.

### Originality
* **Rating**: **Good**
* **Justification**: Although the general concept of using sparse task vectors for model merging is not entirely new (as proposed by He et al., 2024), the specific deconstructive focus of this work—showing that sign-consensus heuristics are redundant—is highly original and insightful. Shifting the focus from creating complex heuristics to uncovering the update under-scaling confounder provides a fresh and valuable perspective to the community.

---

## Overall Recommendation

* **Recommendation**: **4: Weak Accept**
* **Justification**: This is a technically solid and beautifully written paper that advances the model-merging literature by applying Occam's razor to over-engineered pipelines. It has high practical utility, proving that a simple magnitude-based pruning protocol with proper scaling matches or exceeds complex sign-voting heuristics while being extremely easy to deploy. 
  However, it has some key weaknesses that limit its immediate impact: the critical literature omission of He et al. (2024), the lack of evaluations on modern LLM architectures, and the absence of experiments on highly similar tasks. Addressing these limitations—specifically citing and differentiating from He et al. (2024), and discussing how the findings translate to large-scale NLP settings—would make this a very strong accept.
