# Peer Review

## Paper Summary
The paper deconstructs the paradigm of adaptive test-time model merging (specifically targeting AdaMerging) and exposes two severe failure modes: (1) **the Overfitting-Optimizer Paradox (transductive overfitting)**, where fine-grained layer-wise coefficients overfit to the local statistics of small calibration batches instead of capturing true spatial interactions; and (2) **Sacrificial Task Bias**, where low-entropy, simple tasks dominate joint gradients, causing the optimizer to degrade performance on complex, high-entropy tasks (e.g., SVHN). 

To resolve these vulnerabilities, the authors propose **RegCalMerge**, which includes:
1. **CalMerge (Calibration Engine):** Combining Class-Capacity Normalization (CCN) and Scale-Normalized Entropy Weighting (SNEW) to normalize and weight prediction entropy by the inverse of its baseline uniform task arithmetic entropy at step 0, resolving sacrificial task bias.
2. **Elastic Spatial Regularization (ESR):** An optional structural stabilizer consisting of a Proximity Penalty ($\beta$) and a Spatial Deviation Penalty ($\gamma$) to constrain parameter drift and smooth layer-wise coefficient variance.

The authors evaluate their method on CLIP ViT-B/32 across MNIST, FashionMNIST, CIFAR-10, and SVHN, showing that CalMerge achieves a state-of-the-art Joint Mean accuracy of 61.82% and raises SVHN accuracy to 32.03% (the highest in the literature). They also present dense 2D grid sweeps over ESR hyperparameters, perform a dedicated heterogeneous label-space simulation, and introduce a novel spatial shuffling diagnostic to deconstruct optimization dynamics.

---

## Strengths and Weaknesses

### Strengths:
1. **High Practical Relevance & Training-Free Deployment:** The proposed framework operates entirely at test-time on unlabeled calibration streams, requiring zero access to original training data or retraining. From a deployment perspective, this makes the approach extremely cheap, lightweight, and easy to integrate into existing model-merging pipelines.
2. **Exposing Crucial Production Vulnerabilities:** Uncovering the Overfitting-Optimizer Paradox and Sacrificial Task Bias provides high-signal warnings to the community. In industrial or applied multi-task environments, sacrificing difficult domains or overfitting to small test streams can lead to catastrophic failures. Revealing these issues is a major service to both researchers and practitioners.
3. **Rigorous Empirical Standards:** The authors demonstrate commendable scientific rigor by running evaluations across multiple random seeds, utilizing extensive baseline configurations, and presenting smooth, continuous 2D hyperparameter sweeps ($\beta \times \gamma$) that map out a predictable generalization-regularization trade-off surface.
4. **Honest and Deep Limitations Analysis:** The paper is exceptionally transparent and intellectually honest. It explicitly addresses mathematical determinism across seeds under first-order gradient descent, discusses the representational conflicts of spatial smoothing across layers, and simulates unequal class capacities.

### Weaknesses:
1. **Evaluation Restrained to Academic "Toy" Datasets:** The primary weakness of this work is its exclusive evaluation on MNIST, FashionMNIST, CIFAR-10, and SVHN. These low-resolution, small-scale digit and object datasets are classic academic toy problems. They are not representative of modern, complex, high-resolution, or large-scale industrial computer vision applications. To prove genuine real-world utility, the method must be validated on challenging and diverse benchmarks like ImageNet-1K, VTAB, Stanford Cars, or Flowers102.
2. **Missing Real-Time Optimization Latency and Computational Metrics:** Test-time adaptation requires running active optimization (Adam gradient descent or evolutionary strategies) on the fly during the inference stream. In practical engineering systems, inference latency is a primary bottleneck. The paper completely omits runtime execution benchmarks (e.g., seconds per adaptation step, GPU memory overhead, or FLOPs). This makes it difficult to assess the real-world efficiency and viability of RegCalMerge.
3. **Marginal Performance-Complexity Trade-off:** The paper shows that Calibrated Spatial Mean (Cal-Mean, Method 9)—which optimizes only 1 scalar per task ($K=4$ variables) with CCN and SNEW calibration—achieves 61.13% Joint Mean accuracy. Our proposed flagship CalMerge (Method 8)—which optimizes layer-wise coefficients ($K \times L = 52$ variables)—achieves 61.82%. The absolute benefit of optimizing layer-wise parameters is only **0.69%**. In a practical engineering context, the simplicity, speed, and safety of optimizing 4 variables under Cal-Mean would likely outweigh the added architectural and optimization complexity of maintaining and regularizing 52 variables. The paper fails to analyze or justify this practical trade-off.
4. **Unexplained Optimization Anomaly:** In Table 1, under the evolutionary optimization paradigm, spatially shuffling the optimized coefficients (*Shuffled 1+1 ES*, Method 6) actually *improves* Joint Mean accuracy from 59.77% to 60.45%. The authors do not acknowledge or explain this anomaly. If shuffling improves performance, it raises questions about the optimization stability and soundness of derivative-free methods in this context.

---

## Detailed Ratings

### Soundness: Good
The underlying methodology is mathematically sound and described with high clarity. SNEW and CCN are logically designed to handle multi-task prediction entropy variance. The additional heterogeneous class capacity simulation provides strong empirical and analytical proof of the method's mathematical correctness. However, the unexplained evolutionary shuffling anomaly and the lack of a sensitivity analysis regarding calibration batch sizes prevent an "excellent" rating.

### Presentation: Excellent
The paper is exceptionally well-written, clearly structured, and easy to follow. Mathematical definitions and formulas are precise and elegant. The tables and graphs are clean and highly informative. The methodological discussion on limits (determinism, spatial smoothing representational conflicts) is of outstanding depth and intellectual honesty.

### Significance: Good
The paper addresses a highly important problem in multi-task learning and test-time model reuse. By deconstructing the vulnerabilities of adaptive merging and proposing training-free solutions (CalMerge/RegCalMerge), this work could influence future research and adaptive merging toolkits (such as `mergekit`). However, its significance is currently limited by evaluations on toy classification datasets rather than large-scale, industry-relevant benchmarks.

### Originality: Good
The "spatial shuffling diagnostic" is an exceptionally creative and original empirical contribution that exposes the overfitting-optimizer paradox. The identification of sacrificial task bias is also highly original. While SNEW, CCN, and ESR rely on relatively standard normalization and regularization techniques (inverse baseline weighting, L2 proximity, and variance penalization), their combination to resolve these specific vulnerabilities represents a highly valuable and practical adaptation.

---

## Overall Recommendation

**Rating: 4: Weak Accept**

### Justification:
The paper is a technically solid work that makes a highly valuable contribution to the test-time model merging literature. Exposing the Overfitting-Optimizer Paradox (via the spatial shuffling diagnostic) and Sacrificial Task Bias provides critical insights into weight-space optimization dynamics. The proposed calibration engine (CalMerge) and regularization stabilizers (ESR) are elegant, simple, and require zero training data. 

However, the paper is held back by some weaknesses that limit its immediate real-world impact. Specifically, evaluations are restricted exclusively to small academic toy datasets (MNIST/CIFAR/SVHN), critical latency/efficiency benchmarks for test-time optimization are missing, and the marginal 0.69% accuracy improvement of CalMerge over the simpler Cal-Mean baseline raises questions about the practical utility of layer-wise coefficient complexity.

---

## Constructive Questions and Feedback for Authors

1. **Inference Latency:** Can the authors provide runtime benchmarks (in milliseconds, FLOPs, or GPU memory) comparing the inference latency of naive Task Arithmetic, Cal-Mean, and CalMerge (under both Adam GD and 1+1 ES)? What is the latency overhead added by test-time optimization?
2. **Toy vs. Industrial Datasets:** How do RegCalMerge and CalMerge scale to complex, high-resolution datasets like ImageNet-1K or VTAB? Do the performance advantages of layer-wise optimization (CalMerge) over task-wise optimization (Cal-Mean) expand on larger models or more diverse heterogeneous benchmarks?
3. **The 1+1 ES Shuffling Anomaly:** In Table 1, why does spatially shuffling the optimized coefficients for 1+1 ES improve Joint Mean accuracy (59.77% to 60.45%)? Does this point to an optimization instability or convergence failure in evolutionary search?
4. **Calibration Stream Sensitivity:** SNEW depends on the baseline prediction entropy computed on a single calibration batch of size 16. Have the authors evaluated how sensitive SNEW is to sample noise or smaller calibration sizes (e.g., $N=4$ or $N=8$)?
