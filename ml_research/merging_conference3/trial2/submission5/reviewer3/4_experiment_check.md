# 4. Experimental Evaluation and Empirical Check

This section provides a critical, practitioner-focused analysis of the experimental setup, dataset choices, baseline comparisons, and the alignment between the empirical results and the paper's claims.

## Critical Critique of the Experimental Setup
While the experiments are conducted with rigorous statistical controls (using three independent random seeds and reporting standard deviations), there are several notable limitations that affect the practical generalizability of the findings:

1. **Academic Toy Datasets**:
   * The evaluation is restricted to four standard classification datasets: **MNIST**, **FashionMNIST**, **CIFAR-10**, and **SVHN**. 
   * For a powerful, modern vision-language model like CLIP (pretrained on 400 million image-text pairs), these low-resolution (28x28 or 32x32) datasets are extremely simple and represent "toy" problems.
   * In standard CLIP model merging literature (such as the original Task Arithmetic, TIES-Merging, or AdaMerging papers), evaluations are typically performed on a diverse suite of 8 larger, high-resolution downstream datasets (including ImageNet, Stanford Cars, FGVC Aircraft, DTD, EuroSAT, SUN397, Flowers102, and UCF101).
   * Restricting the evaluation to MNIST/FashionMNIST/CIFAR-10/SVHN makes it highly uncertain whether NETA's layer-wise norm-equalization generalizes to complex, high-resolution, real-world tasks. For instance, do the same pooling heuristics (Group 0) and isotropic assumptions hold when merging fine-tuned experts for specialized medical images, satellite data, or fine-grained vehicle classification?

2. **Single, Small Backbone Model**:
   * All experiments are conducted using **CLIP ViT-B/32**, which is a relatively small, early-generation vision transformer (86M parameters in the visual encoder).
   * Modern industrial practitioners deploy much larger models (ViT-L/14, ViT-H, LLMs, or multi-modal models). The lack of cross-architecture validation (e.g., on larger ViT backbones, text-based LLMs like Llama, or diffusion models) is a significant limitation that prevents drawing broader conclusions about NETA's scalability.

3. **Sub-Sampling of Test Sets**:
   * Evaluations are performed on a representative subset of **1024 test images** per dataset.
   * While the authors justify this sub-sampling to manage CPU/Slurm queue overhead under strict resource constraints, it remains a limitation. For simple datasets like MNIST or CIFAR-10, running full test-set evaluation (10,000 images) on a single GPU takes less than a minute. While 1024 images provide a stable statistical estimate, validating on full test sets is standard and would strengthen the empirical claims.

## Evaluation of Claims vs. Empirical Evidence
We evaluate whether the paper's central claims are fully backed by the presented data:

1. **Claim: NETA analytically resolves task vector magnitude imbalances, outperforming Task Arithmetic zero-shot.**
   * *Evidence*: Table 1 shows that NETA ($\alpha = 1.0$) improves over Task Arithmetic on MNIST ($96.29\%$ vs $96.03\%$) and FashionMNIST ($82.75\%$ vs $82.10\%$).
   * *Qualification*: On CIFAR-10, NETA's accuracy is slightly worse ($92.61\%$ vs $92.77\%$). On SVHN, NETA's performance drops significantly ($77.02\%$ vs $80.14\%$). Consequently, NETA's multi-task average accuracy ($87.17\%$) is actually **lower** than standard Task Arithmetic ($87.76\%$) and DARE ($87.78\%$).
   * *Practitioner's Take*: While NETA successfully prevents SVHN from dominating, the "isotropic regularizer" curtails peak performance, resulting in a net loss in average multi-task accuracy. If a practitioner's primary goal is to maximize average system capability, standard Task Arithmetic or DARE remains superior in this setup.

2. **Claim: The Overfitting-Optimizer Paradox collapses unsupervised test-time adaptation.**
   * *Evidence*: The catastrophic drop of **$-4.56\%$** on FashionMNIST and **$-3.07\%$** on CIFAR-10 under Task-Wise AdaMerging clearly supports this claim. The analysis showing that joint entropy minimization suppresses high-entropy, difficult tasks in favor of easy tasks is highly convincing and practically valuable.

3. **Claim: Layer-Wise AdaMerging's performance is overparameterized and transductive, making NETA a robust, training-free alternative.**
   * *Evidence*: Table 1 shows that **Layer-Wise AdaMerging** (optimizing 52 parameters) does *not* suffer from the Overfitting-Optimizer Paradox, achieving the highest performance on FashionMNIST ($84.04\%$), CIFAR-10 ($92.92\%$), SVHN ($88.14\%$), and the highest average multi-task accuracy ($90.89\%$).
   * *Practitioner's Take*: For a practical deployment with a GPU and 256 unlabeled calibration images, Layer-Wise AdaMerging outperforms NETA by a massive **$+3.72\%$** absolute accuracy margin ($90.89\%$ vs $87.17\%$). Even when comparing NETA's best scale-compensated config ($87.28\%$) or relaxed $\alpha=0.5$ config ($87.51\%$), Layer-Wise AdaMerging remains significantly superior. While the authors argue that 52 parameters are "overparameterized," a practitioner would likely accept this minor complexity to unlock $+3.7\%$ performance, since test-time optimization of 52 parameters over 256 images takes only a few seconds. The authors' claim that NETA is an advantageous compromise is true for highly resource-constrained or zero-calibration settings, but for standard deployments, the performance gap remains a major hurdle.

4. **Claim: $\alpha$-Relaxation and $\gamma^l$ Scale Compensation resolve the performance trade-offs.**
   * *Evidence*: Table 2 shows that NETA ($\alpha = 0.5$) recovers SVHN performance to $78.55\%$ while maintaining gains on MNIST and FashionMNIST, yielding an average of $87.51\%$. NETA + $\gamma^l$ Scale Compensation improves average performance to $87.28\%$ and boosts FashionMNIST to its peak of $82.85\%$.
   * *Practitioner's Take*: These are excellent, practical, training-free tools that make NETA much more flexible. However, they still do not close the gap with standard Task Arithmetic (87.76%) or Layer-Wise AdaMerging (90.89%) in terms of overall multi-task average utility.
