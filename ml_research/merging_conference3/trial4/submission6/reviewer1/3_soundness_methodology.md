# 3. Soundness and Methodology

## Clarity of Description
The methodology is exceptionally well-described and structured:
* The paper clearly outlines each step: task vector extraction, layer-wise magnitude pruning, and the two correction techniques (R-STA and Tuned STA).
* The equations are mathematically clear and easy to follow.
* The authors provide a complete, self-contained PyTorch implementation in Appendix A. For practitioners, this is an outstanding addition that ensures the method is directly reproducible and easy to integrate into production codebases.

## Appropriateness of Methods
* **Magnitude-based Pruning**: Applying uniform layer-wise magnitude-based pruning is highly appropriate and robust. It relies on the absolute largest updates, which represent the most salient features of task-specific fine-tuning.
* **Under-scaling Correction**: The identification of update under-scaling is a crucial methodological contribution. Pruning inherently reduces vector magnitude and energy. Compensating for this via R-STA (dividing by $s/100$) or Tuned STA (tuning $\lambda$) is highly appropriate and theoretically sound.
* **Symmetric Tuning Protocol**: The experimental methodology of sweeping $\lambda \in [0.1, 1.0]$ for *all* methods and reporting peak performance $\lambda^*$ is extremely rigorous. It ensures that standard baselines (like Task Arithmetic) are not unfairly disadvantaged by sub-optimal static hyperparameters, a common issue in other model-merging papers.

## Reproducibility
* **Excellent**. The inclusion of the exact PyTorch script in the Appendix makes replication trivial. The authors also use a pre-trained ViT-B-32 backbone and standard datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) with a standardized validation split of 2,048 samples per dataset, making it accessible for CPU-level verification.

## Potential Technical Flaws and Limitations
While the methodology is sound, there are several key limitations that affect its real-world utility:

1. **Lack of Large Language Model (LLM) Evaluations**:
   In modern deep learning practice, weight-space model merging is most widely used in natural language processing (NLP) to merge large instruction-tuned models (e.g., Llama, Mistral, Qwen). The current evaluation is restricted entirely to a relatively small Vision Transformer (ViT-B-32, ~86M parameters) on simple classification tasks. It remains unproven whether the deconstruction of sign consensus holds on large models with billions of parameters, complex vocabulary heads, and deep causal attention layers.
   
2. **Limited Domain Overlap (Task Dissimilarity)**:
   The 4-task vision suite (digits, apparel, general objects) spans diverse and mostly unrelated domains. Under these conditions, the coordinate-wise mask overlap is independent and extremely small ($3.1\% - 4.3\%$), conforming to the $(s/100)^2 = 4\%$ theoretical bound.
   However, in industrial applications, practitioners often merge models that are fine-tuned on highly similar domains or share a significant portion of their instruction-tuning objectives (e.g., merging multiple LLMs trained on various code-generation or math reasoning tasks). In these scenarios, the parameter updates are likely to be highly correlated, and the empirical overlap rate could be substantially higher. While the authors theoretically discuss why sign conflicts are self-resolving under overlap (via dominant signal alignment or local cancellation), they provide **no empirical validation** of their method under high task similarity.
   
3. **Tail-Bias and Variance Distortion in R-STA**:
   As the authors correctly note in Section 4.3, Rescaled STA (R-STA) degrades at lower densities ($s \le 20\%$) due to "variance distortion." Because magnitude pruning deterministically selects the extreme tails of the update distribution, multiplying these values by $100/s$ inflates outliers, pushing weights off the pre-trained manifold. While Tuned STA avoids this, it requires sweeping and optimizing $\lambda$ dynamically, which adds some practical deployment overhead compared to DARE, which can preserve variance and scale dynamically without tuning due to its random selection.
