# Peer Review

## 1. Summary of the Paper
This paper introduces **GranMerge**, a systematic empirical framework designed to dissect the **Generalization-Granularity Trade-off** in adaptive multi-task model merging. In test-time adaptive merging, model parameters are combined using task-specific weights that are optimized dynamically on a small, unlabeled calibration stream (typically by minimizing an unsupervised surrogate objective like prediction entropy). 

The paper systematically partitions a 12-layer Vision Transformer (ViT-Tiny) into five nested levels of structural granularity for the merging coefficients, ranging from a single global scale per task (Level 1, 4 parameters) up to tensor-wise scaling of major projection modules (Level 5, 288 parameters). The coefficients are optimized using either first-order gradient descent (Adam) or zero-order derivative-free optimization (1+1 Evolution Strategies) on a calibration batch of $N=256$ samples per task across four visual tasks (MNIST, FashionMNIST, CIFAR-10, SVHN).

The main findings of the paper are:
1. **Transductive Overfitting at High Granularity:** Increasing structural granularity leads to severe overfitting on the small test-time calibration stream, degrading global multi-task generalization compared to simple uniform weight blending or coarser granularities.
2. **Optimization Trajectory Dynamics:** First-order Adam is highly vulnerable to rapid generalization collapse. Zero-order 1+1 ES maintains higher generalization accuracy, which the authors rigorously deconstruct as a combination of isotropic walk constraints and optimization sluggishness (underfitting) under high dimensions.
3. **Soft Regularization Limits:** The authors introduce joint Elastic Spatial Regularization (ESR) and depth-wise Total Variation (TV) smoothness penalties. While these L2 constraints successfully stabilize zero-order ES and improve Adam, they are insufficient to prevent first-order Adam from overfitting.
4. **Static Baseline Supremacy:** In this resource-constrained, low-fidelity regime, no adaptive merging configuration (even when regularized with ESR and TV) outperforms the zero-overhead, static Uniform Task Arithmetic baseline of 30.41%.

---

## 2. Strengths and Weaknesses

### Strengths
1. **Exceptional Empirical Honesty and Transparency:** The authors deserve significant praise for highlighting a "negative" result: that despite complex test-time optimization, no adaptive configuration beats the static, zero-overhead Uniform baseline of 30.41%. This level of empirical transparency is refreshing, highly credible, and prevents other researchers from chasing unproductive directions.
2. **Rigorous Deconstruction of Optimization Dynamics:** The deconstruction of the zero-order (1+1 ES) behavior is elegant. Instead of naively concluding that ES is a superior optimizer for representation learning, the authors formulate and support the "sluggishness hypothesis." They show that ES's apparent robustness in high-dimensional spaces is actually a form of underfitting, as it fails to optimize 288 parameters in 100 steps and remains near its robust, uniform initialization.
3. **Polished Structure and Presentation:** The paper is exceptionally well-written, logically organized, and visually clean. The mathematical formulation of the five granularities, the entropy loss, and the ESR/TV regularizers is precise and straightforward.
4. **Valuable Diagnostic Insight:** Identifying the fundamental misalignment between prediction entropy minimization and classification accuracy under compact calibration streams ($N=256$) is a key contribution that will help the community design better surrogate losses in the future.

### Weaknesses
1. **Highly Contrived and Toy Scale (Lack of Scalability):** The experiments are conducted on an extremely small scale. The authors use a `ViTTiny` backbone with a model dimension of $d_{\text{model}}=64$ and fine-tune experts on only 500 samples per task. Due to this constraint, the absolute performance of the individual experts is incredibly low (an average of 41.48%, with SVHN at an abysmal 17.50%—barely above the 10% random chance). High-frequency parameter noise is naturally amplified in poorly converged, under-trained experts, which likely worsens the transductive overfitting observed. This raises significant doubts about whether these granularity-generalization dynamics are representative of modern, high-fidelity foundation models (e.g., CLIP, LLaMA) where features are highly structured and robust.
2. **Limited Practical Utility of Proposed Techniques:** From an engineering and deployment standpoint, the findings heavily discourage the practical adoption of the proposed techniques. Since running complex test-time adaptation loops (requiring 60-100 forward/backward passes on edge devices, managing calibration queues, and storing optimizers) fails to outperform a simple static average, there is zero real-world incentive to deploy these adaptive methods. 
3. **Absence of Computational and Latency Analysis:** Since the paper positions itself as targeting "resource-constrained edge environments," the lack of any latency, memory, or energy overhead analysis is a major gap. Running 100 steps of zero-order ES on $N=256$ samples at test-time introduces a massive latency penalty, which should have been explicitly quantified against the zero-overhead static baseline.
4. **Missing Key Baselines:** The paper cites **RegCalMerge (Jin et al., 2026)** as a prior work addressing transductive overfitting in multi-task merging. Since RegCalMerge directly targets the exact problem studied in this paper, it was critical to include it as a baseline in Table 1 to see how the proposed ESR/TV regularizers compare against existing state-of-the-art solutions.
5. **Hyperparameter Tuning Feasibility:** The joint ESR/TV regularizers introduce regularization scale $\beta$ and depth balance $\gamma$. At test-time, since no labels are available, there is no way for a practitioner to tune these hyperparameters. The paper does not provide any sensitivity analysis of these parameters, leaving their robustness and stability in practical settings completely unexamined.

---

## 3. Soundness
**Rating: Good**

**Justification:** The paper's mathematical formulation of the structural granularities and L2 regularizers (ESR, TV) is technically sound. The comparison between first-order and zero-order optimizers is methodologically robust, and the deconstruction of the "sluggishness hypothesis" is exceptionally sound and insightful. 
However, there are minor description gaps that hinder full reproducibility. Specifically, the authors do not clarify how the task-specific classification heads are handled and routed in the merged model (whether they are merged, kept separate, or assumed known at test-time). Additionally, the status of non-projection parameters (such as LayerNorms, class tokens, and embeddings) at Level 5 is not specified.

---

## 4. Presentation
**Rating: Excellent**

**Justification:** The paper is exceptionally well-written, clear, and well-structured. The narrative flows logically from the introduction of the Generalization-Granularity Trade-off to the formalization of GranMerge, followed by quantitative tables and a comprehensive empirical analysis. The mathematical notations are precise, and the discussion of limitations and strategic future directions is thoughtful and mature.

---

## 5. Significance
**Rating: Fair**

**Justification:** The paper's conceptual significance is high because it serves as an excellent diagnostic "cautionary tale" that exposes the limits of prediction entropy minimization and high-granularity optimization at test-time. However, its significance for practical applications is limited. Because none of the proposed regularized configurations outperform the zero-overhead, static Uniform baseline, the proposed techniques do not offer immediate, deployable utility to machine learning practitioners. Furthermore, because the evaluation is confined to a tiny, under-trained ViT model on toy vision datasets (MNIST, FashionMNIST, CIFAR-10, SVHN), the generalizability of these trade-off dynamics to modern large-scale foundation models remains unproven.

---

## 6. Originality
**Rating: Good**

**Justification:** The paper does not introduce fundamentally new machine learning primitives (as entropy minimization, 1+1 ES, L2 smoothing, and depth-wise total variation are established methods). However, the systematic construction of the nested 5-level granularity hierarchy and the direct comparative analysis of optimizer trajectories (Adam vs. 1+1 ES) represent a highly original and valuable empirical contribution to the model merging literature.

---

## 7. Overall Recommendation
**Score: 4 (Weak Accept)**

**Justification:** This is a technically solid, exceptionally well-written paper that delivers a valuable, honest empirical mapping of the structural boundaries of adaptive weight blending. Its diagnostic insights—specifically exposing the transductive overfitting of high parameter resolution and deconstructing the robustness of zero-order optimization as optimization sluggishness—are highly compelling contributions that others in the model merging and test-time adaptation communities are likely to build on.

While the paper suffers from limitations in experimental scale (confined to toy models and under-trained experts) and lacks practical latency analyses, these weaknesses do not undermine the validity of its core diagnostic claims. Although the proposed methods do not outperform the static baseline, documenting this negative result is itself a highly significant contribution that saves researchers and engineers from wasting resources on over-parameterized test-time loops. The paper is recommended for acceptance as a strong diagnostic study, but the authors are highly encouraged to address the description gaps regarding classification heads and non-projection parameters, and to provide at least a preliminary discussion or latency table outlining the test-time computational costs of their methods.
