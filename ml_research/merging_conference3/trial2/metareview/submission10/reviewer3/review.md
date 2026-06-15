# Conference Peer Review

## Summary of the Paper
This paper presents a rigorous, deconstructive analysis of **AdaMerging**, a state-of-the-art unsupervised test-time model merging framework. Weight-space model merging is an important paradigm for multi-task learning because it integrates task-specific expert models (fine-tuned from a shared pre-trained base) without expensive retraining or simultaneous data access. AdaMerging parameterizes this combination with scaling coefficients (either task-wise or layer-wise) and optimizes them at test-time by minimizing the Shannon entropy of predictions on a small unlabeled calibration batch. 

The authors deconstruct this framework to expose two key optimization anomalies:
1. **The Overfitting-Optimizer Paradox**: The authors argue that while high-dimensional layer-wise AdaMerging (optimizing $L \times T$ parameters) captures vital architectural hierarchy and representational routing, it is prone to test-time overfitting. They show that shuffling coefficients across layers collapses performance, while taking their post-hoc spatial mean (Spatial Averaging) smooths out overfitting, achieving $84.96\%$ accuracy and beating the Task Arithmetic baseline.
2. **The Spatial Averaging Paradox**: Direct task-wise AdaMerging (optimizing a low-dimensional bottleneck of $T$ global parameters) fails spectacularly ($81.19\%$ accuracy, degrading below the baseline's $84.64\%$). The authors explain this through **multi-task gradient imbalance** under uncalibrated prediction entropy, where easy classification tasks (with sharp logit distributions) dominate joint weight updates, creating destructive parameter interference in shared early projection layers and collapsing representation manifolds on harder, heterogeneous tasks (e.g., SVHN).

To resolve this imbalance, the authors propose **Calibrated Prediction Entropy**, which normalizes task contributions at initialization. However, their experiments reveal that this remedy fails to restore performance ($80.59\%$ accuracy), demonstrating that the bottleneck itself, rather than just initial gradient magnitudes, causes destructive weight-space interference.

---

## Strengths and Weaknesses

### Major Strengths
1. **Critical, Much-Needed Perspective**: The paper adopts a highly refreshing and rigorous "Occam's razor" lens to deconstruct an over-parameterized test-time adaptive framework. Dissecting the underlying mechanics of weight-space optimization is highly valuable for the model-merging community.
2. **Elegant Diagnostic Treatments**: The design of the empirical controls is excellent. Specifically, *Intra-Task Layer Shuffling* provides a clean, elegant demonstration that layer-wise coefficients are structurally specialized to the neural network hierarchy and are not merely generalizable noise.
3. **Sound Theoretical Explanation**: Explaining the *Spatial Averaging Paradox* via uncalibrated prediction entropy and multi-task gradient imbalance is theoretically satisfying, clear, and mathematically robust. It explains why a low-dimensional bottleneck forces coarse trade-offs under joint unsupervised objectives.
4. **Disciplined Evaluation Protocol**: The authors evaluate their methods across three independent, seed-controlled runs on a total test scale of $56,032$ images. This extremely large evaluation split completely eliminates data selection bias and provides tight confidence intervals, which is highly appreciated.

### Major Weaknesses

#### 1. Severe Lack of Practical Utility
From a practical deployment and engineering standpoint, the paper does not offer an actionable or useful technique:
- **Redundancy of Spatial Averaging**: To compute the "Spatially Averaged" model (achieving $84.96\%$ accuracy), a practitioner must still execute the full, high-dimensional test-time optimization of layer-wise AdaMerging (optimizing $1,000$ parameters on the calibration batch). However, the unconstrained layer-wise model already achieves **88.05%** accuracy on the test set. Since there is zero additional computational or storage cost in deploying the layer-wise model once optimized, a practitioner would never choose to average the coefficients post-hoc and drop their accuracy by $3.09\%$. Taking the spatial mean throws away valuable task-specific architectural routing and degrades performance for no actual gain.
- **Failure of the Proposed Remedy**: The authors' constructive contribution—Calibrated Prediction Entropy—fails to improve performance. It achieves only $80.59\%$ accuracy, which is worse than baseline Task Arithmetic ($84.64\%$). Thus, the paper leaves the community with no functional or stable algorithm to directly perform low-dimensional task-wise test-time adaptation.

#### 2. Evaluation Restricted to Outdated, Toy Benchmarks
The empirical evaluation relies on **MNIST, FashionMNIST, CIFAR-10, and SVHN** using a small **CLIP ViT-B/32** visual backbone.
- In the year 2026, evaluating a visual foundation model on tiny, legacy datasets ($28 \times 28$ grayscale and $32 \times 32$ color images) is extremely outdated. These benchmarks do not represent the complexity, scale, or domain shifts of modern real-world vision pipelines.
- Modern weight-space model merging is most actively researched and applied in combining Large Language Models (LLMs) and large-scale multimodal models. The lack of any LLM or generative AI experiments (such as merging instruction-tuned models) makes the findings highly localized to toy visual classification setups and of limited significance to modern machine learning practice.
- The severe gradient imbalance observed is highly artificial, driven by mixing grayscale digits (MNIST) with real-world street numbers (SVHN). Real-world applications rarely merge such highly disparate and simple toy domains.

#### 3. Conceptual Framing Contradiction on "Overfitting"
The authors frame the layer-wise AdaMerging optimization as suffering from an "Overfitting-Optimizer Paradox" and describe it as a harmful pathology. 
- However, layer-wise AdaMerging (Adam GD) achieves **88.05%** average accuracy on the full, standard test splits (totaling $56,032$ images) after adapting on just 64 unlabeled images per task! 
- If a model optimized on a tiny batch generalizes substantially better to a massive, disjoint test set ($88.05\%$ vs. $84.96\%$ of Spatial Averaging and $84.64\%$ of Task Arithmetic), it cannot be described as suffering from a harmful "overfitting pathology" in a practical sense. To a practitioner, this is not a paradox; it is simply a superior, highly expressive local representational routing. Labeling this beneficial, specialized routing as "pathological overfitting" represents a significant conceptual stretch.

---

## Detailed Ratings and Justifications

### Soundness
- **Rating**: **Good**
- **Justification**: The mathematical derivations, experimental setups, and empirical validations are technically correct, reproducible, and robustly analyzed. The seed-controlled evaluations over a large test scale provide strong statistical significance. However, the conceptual framing of the "Overfitting-Optimizer Paradox" is contradictory, as the "overfitting" model actually generalizes much better than the "regularized" one. Additionally, the proposed remedy (Calibrated Prediction Entropy) is empirically shown to fail, limiting the soundness of the paper's constructive contributions.

### Presentation
- **Rating**: **Excellent**
- **Justification**: The paper is exceptionally well-written, clearly structured, and easy to follow. The mathematical notation is rigorous and consistent. The figures and tables are professional, informative, and beautifully support the main claims (particularly the layer-by-layer CKA representational similarity curve in Figure 2). The paper also does an outstanding job positioning itself relative to prior/concurrent weight merging literature.

### Significance
- **Rating**: **Fair**
- **Justification**: While the deconstructive analysis of AdaMerging and the gradient imbalance characterization are scientifically interesting, the overall practical significance of the paper is low. It does not introduce a functional new algorithm, the proposed remedy fails, and the post-hoc Spatial Averaging is practically redundant (as it performs worse than the already-optimized layer-wise model). Furthermore, evaluating exclusively on outdated toy image datasets (MNIST, CIFAR-10) without any Large Language Model (LLM) or large-scale foundation model experiments severely limits its impact for modern deep learning practitioners.

### Originality
- **Rating**: **Good**
- **Justification**: The introduction of *Intra-Task Layer Shuffling* and *Spatial Averaging* as diagnostic tools is highly creative and provides a clear, interpretable way to study the structural specialization of weights. Explaining the task-wise vs. layer-wise optimization trade-offs (the Spatial Averaging Paradox) through the lens of uncalibrated prediction entropy and multi-task gradient imbalance is also relatively novel, building upon and formalizing concepts from recent model merging literature.

---

## Overall Recommendation

- **Rating**: **3: Weak Reject**
- **Justification**: 
The paper has clear scientific merits, including an exceptionally well-written manuscript, rigorous mathematical formulations, and highly elegant empirical diagnostics (like layer shuffling) to analyze architectural specialization. However, its weaknesses overall outweigh its merits for publication at a premier machine learning conference. 

From a practical perspective, the paper offers very little actionable utility: its post-hoc Spatial Averaging is practically redundant (as it degrades accuracy compared to the unconstrained layer-wise model that must still be run), and its proposed Calibrated Prediction Entropy remedy fails to resolve the bottleneck pathology. Furthermore, the evaluation is heavily restricted to legacy toy datasets (MNIST, CIFAR-10) on a small visual backbone (ViT-B/32), with no Large Language Model (LLM) or generative AI experiments where weight-space merging is currently most relevant. Finally, there is a conceptual contradiction in framing the layer-wise adaptation as a harmful "overfitting pathology" when it actually generalizes much better than any other scheme. 

The paper would be significantly stronger if the authors (1) evaluated their findings on modern benchmarks or LLMs, (2) resolved the conceptual contradiction regarding beneficial local routing vs. "overfitting," and (3) developed a functional, constructive algorithm that actually stabilizes direct task-wise or layer-wise weight merging to outperform existing baselines in practical settings.

---

## Detailed Comments and Questions for the Authors

1. **Beneficial Routing vs. Overfitting**: Why do you frame layer-wise AdaMerging as suffering from an "Overfitting-Optimizer Paradox" when it achieves the highest generalization accuracy ($88.05\%$) on a massive, disjoint test set? If taking the spatial average drops performance by $3.09\%$, isn't Spatial Averaging just degrading a superior, structurally specialized model rather than "smoothing away harmful overfitting"? Please clarify this conceptual contradiction.
2. **Computational overhead of Spatial Averaging**: To compute the spatially averaged model, does the practitioner still have to perform the 1,000-parameter test-time optimization of layer-wise AdaMerging on the calibration batch? If so, once those layer-wise coefficients are learned, why should a practitioner deploy the spatially averaged model (getting $84.96\%$ accuracy) instead of the layer-wise model (getting $88.05\%$ accuracy)? Is there any practical benefit (e.g., storage, latency, or out-of-distribution robustness) to deploying the spatially averaged model that justifies this $3.09\%$ performance drop?
3. **Scaling to LLMs and Modern Benchmarks**: Model merging is most actively used in merging Large Language Models (LLMs) like Llama or Mistral. Why did you restrict your evaluation to legacy toy datasets like MNIST, FashionMNIST, CIFAR-10, and SVHN? Do you have any preliminary results showing if the Overfitting-Optimizer or Spatial Averaging Paradoxes scale to token-level perplexity objectives when merging instruction-tuned or domain-specialized LLMs?
4. **Pruning and Sign-Consensus Synergy**: You mention in the future directions that applying Spatial Averaging on top of pruned or sign-resolved base task vectors (like TIES or DARE) could be a promising synergy. Have you conducted any preliminary experiments on this? Given that TIES and DARE performed very poorly on SVHN, does Spatial Averaging on top of them help recover their representation collapse?
