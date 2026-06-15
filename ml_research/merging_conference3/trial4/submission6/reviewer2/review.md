# Peer Review: Sparse Task Arithmetic (STA)

## 1. Summary of the Paper
The paper presents a deconstructive critique of modern sparse model merging paradigms, specifically targeting methods like TIES-Merging and DARE. The authors challenge the necessity of complex coordinate-wise sign voting, dominant sign election, and stochastic scaling heuristics. Guided by Occam's razor, they propose **Sparse Task Arithmetic (STA)**, an extremely streamlined, training-free merging protocol consisting of only two steps: layer-wise magnitude pruning of task vectors followed by standard linear addition (direct summation). 

Crucially, the paper identifies a major methodological confounder in previous evaluations: **update under-scaling**, where magnitude pruning reduces the expected energy of the task updates, mimicking representation degradation. To resolve this, they introduce:
- **Rescaled STA (R-STA):** An analytical scaling variant that divides sparse updates by the survival density ($100/s$).
- **Tuned STA:** A hyperparameter-tuned variant that dynamically scales the direct sparse updates.

Evaluated on a Vision Transformer (ViT-B-32) benchmark across four diverse classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN), Tuned STA ($s=20\%$) matches the performance of Tuned TIES-Merging ($90.53\%$ vs $90.16\%$) and slightly outperforms Tuned DARE and full-density Task Arithmetic baselines. The paper provides mathematical and empirical arguments showing that:
1. Under sparsity, coordinate collisions are rare, rendering sign voting mathematically moot for over 96% of the parameters.
2. In rare overlap cases, direct addition naturally resolves sign conflicts through dominant signal alignment or local cancellation.
3. Magnitude pruning is fundamentally an absolute magnitude SGD noise filter rather than a sign-conflict resolver.

---

## 2. Strengths
- **Conceptual Simplicity and Practical Appeal:** The paper is a champion of simplicity. In an era of increasing "heuristic over-engineering," showing that a basic two-step pipeline (magnitude pruning + standard addition) matches SOTA performance is highly refreshing and practically valuable. It drastically reduces code complexity, execution latency, and storage overhead.
- **Symmetric and Rigorous Benchmarking:** The empirical evaluation standard is excellent. The authors perform a symmetric grid search over the scaling parameter $\lambda$ for *all* methods (Task Arithmetic, DARE, TIES-Merging, and STA), ensuring that each method is compared at its peak capacity. This corrects a common evaluation bias where only the proposed method is fully tuned.
- **Identification of the Under-Scaling Confounder:** The discovery of "update under-scaling" is a high-signal contribution. It explains why standard sparse linear addition historically underperformed and provides a simple, direct remedy for researchers and practitioners.
- **Strong Grounding and Explanatory Framework:** The paper provides highly intuitive and mathematically grounded explanations for its findings, supported by empirical mask overlap measurements (confirming that mask overlap conforms to the $(s/100)^2$ theoretical independence bound).

---

## 3. Weaknesses
- **Limited Scale of Experiments (Toy-Scale Setup):** From a practical deployment standpoint, the experimental validation is heavily constrained. Evaluating only on a small Vision Transformer backbone (ViT-B-32, 86M parameters) across standard, historic classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) represents a toy-scale setting. Modern model merging is heavily used to combine **Large Language Models (LLMs)** (LLaMA, Mistral, Gemma, T5) and large-scale **Vision-Language Models** (CLIP) on complex generative and downstream benchmarks. The lack of LLM and large-scale CLIP-ViT-L evaluations leaves a massive gap in generalizability.
- **The Task Similarity Assumption:** The paper's theoretical proof that coordinate collisions are rare (mask overlap bounded by $(s/100)^2 \approx 4\%$) assumes that the tasks being merged are independent. While this holds for highly disjoint domains (digits, apparel, natural images), it is highly unlikely to hold in practical settings where homologous models fine-tuned on highly related domains or instructions (e.g., merging multiple instruction-following LLMs) are combined. In such settings, task similarity is high, the mask overlap will significantly exceed the independence bound, and sign conflicts may be far more frequent. The paper's claims are untested in these critical high-similarity scenarios.
- **Practical Tuning Overhead of Tuned STA:** While Tuned STA achieves peak performance ($90.53\%$), it relies entirely on finding the optimal scaling factor $\lambda^* = 0.8$. In production settings, practitioners frequently merge models "in the wild" without access to the original training/validation datasets or the resources to run multi-task evaluation sweeps. When evaluated using a standard default scale ($\lambda = 0.3$), standard STA ($s=20\%$) drops to **$82.91\%$** average accuracy—lagging behind Tuned TIES-Merging ($90.16\%$) by **$-7.25\%$** and standard Task Arithmetic ($87.45\%$) by **$-4.54\%$**. This severe sensitivity to scaling makes Tuned STA less robust out-of-the-box compared to self-scaling methods like DARE.
- **Analytical Instability of Rescaled STA (R-STA):** The tuning-free analytical variant, R-STA, suffers from severe degradation at lower densities ($s \le 20\%$) due to "variance distortion." Because magnitude pruning deterministically selects extreme tail outliers, multiplying them by $1/s$ explodes their variance, pushing weights off the pre-trained manifold. This instability at high sparsity levels is a major practical drawback, as sparsity is precisely where model merging is most needed to combat parameter interference.

---

## 4. Detailed Evaluation of Criteria

### Soundness: Good
The technical claims and derivations are mathematically rigorous, and the symmetric benchmarking protocol is exemplary. The empirical verification of mask overlap rates ($3.1\%-4.3\%$) aligns perfectly with the theoretical model. However, the soundness of the general claim that "sign consensus is entirely redundant" is limited by the absence of evaluations on high-similarity task distributions and large-scale architectures (LLMs).

### Presentation: Excellent
The writing quality is top-tier. The overall narrative is exceptionally easy to follow, the notation is clean and precise, and the figures and tables are highly professional. The authors are commendably honest and transparent about the limitations of their work, explicitly discussing the "variance distortion" of R-STA and task similarity boundaries in Section 3.

### Significance: Good
The paper addresses an important problem and provides a much-needed push back against over-engineered heuristics, restoring simplicity as a core design principle in model merging. If these results generalize to LLMs, the practical impact will be highly significant. Currently, its significance is slightly moderated by the limited experimental scale and high hyperparameter tuning dependency.

### Originality: Good
While STA uses existing primitives (magnitude pruning and linear addition), the originality lies in its **deconstructive critique**. Conceptualizing magnitude pruning as an SGD noise filter rather than a sign-conflict resolver, and identifying the update under-scaling confounder, are highly creative and novel conceptual contributions that challenge established assumptions.

---

## 5. Overall Recommendation

**Rating: 4 (Weak Accept)**

**Justification:**
This is a technically solid, exceptionally well-written paper that applies Occam's razor to sparse model merging. The identification of the update under-scaling confounder and the conceptual framing of magnitude pruning as an SGD noise filter are highly valuable contributions that are likely to influence future model-merging research. The benchmarking methodology is rigorous and scientifically fair. 

However, its impact is limited by its experimental scale (ViT-B-32 on MNIST/CIFAR-10/SVHN/FashionMNIST). To be a strong candidate for acceptance at a major ML conference, the paper must validate its findings on modern, large-scale architectures—specifically **Large Language Models (LLMs)**—and high-similarity tasks where sign-consensus heuristics are widely applied in practice. Furthermore, the practical utility of the method is hampered by its reliance on hyperparameter tuning ($\lambda^*$) and the analytical instability of R-STA at high sparsity levels. It is a Weak Accept that would easily become a Strong Accept with the addition of LLM experiments and a more robust, tuning-free scaling mechanism.

---

## 6. Constructive Feedback and Questions for Authors

1. **LLM Validation:** Modern sparse merging methods (TIES, DARE) are standardly evaluated on encoder-decoder models (T5) and autoregressive LLMs (LLaMA, Mistral). Have you conducted any preliminary experiments on language models? Evaluating STA on a standard 7B instruction-merging suite (e.g., merging WizardLM, WizardMath, and WizardCoder) would dramatically strengthen the generalizability and practical significance of your claims.
2. **Empirical Overlap in High Task Similarity:** To test the boundaries of the independent mask assumption, can you evaluate STA on a set of tasks fine-tuned on highly related domains? For instance, merging multiple models fine-tuned on different subsets of CIFAR-10, or merging closely related translation checkpoints. In these scenarios, does the empirical mask overlap rate exceed $(s/100)^2$, and does STA's performance degrade compared to TIES-Merging?
3. **Addressing the Tuning-Free Gap:** Since Tuned STA is highly sensitive to the global scaling parameter $\lambda$, and R-STA suffers from variance distortion at low densities, how can a practitioner use STA "in the wild" without a validation set to tune $\lambda$? Is there a way to dynamically interpolate or estimate $\lambda^*$ using only weight-space statistics or base-model representations, similar to DARE's variance preservation but without stochasticity?
4. **Baselines:** Why was the hybrid **DARE-TIES** baseline omitted from the evaluations? Since DARE-TIES is widely regarded as a state-of-the-art sparse merging pipeline, including it in Table 1 would make your comparative analysis far more robust and convincing.
