# 4. Experimental Setup and Baseline Evaluation

This section provides a highly critical evaluation of the experimental setup, choice of datasets, baselines, and whether the empirical results actually support the paper's central claims.

---

## 1. Evaluation of the Experimental Setup and Datasets

### A. Toy-Like and Saturated Vision Benchmarks
The paper evaluates the model-merging paradigms on four standard image classification datasets: **MNIST, FashionMNIST, CIFAR-10, and SVHN**. 
* **Saturated performance:** These datasets are largely toy-like and saturated. For instance, the task experts trained by the author achieve very high test accuracies (MNIST: 96.94%, FashionMNIST: 88.67%, CIFAR-10: 88.93%, SVHN: 85.81%). 
* **Trivial Task-Vectors:** Because these datasets are simple, the task vectors extracted from the CLIP ViT-B/32 backbone lie extremely close to each other and to the pre-trained initialization. This creates an artificially flat loss landscape where almost any linear combination of weights performs reasonably well.
* **Lack of Generalizability:** Modern model merging is primarily utilized to combine large autoregressive language models (LLMs like LLaMA, Mistral) or highly complex diffusion models. A study restricted entirely to a 13-layer ViT-B/32 on MNIST/CIFAR-10 lacks the necessary complexity to make broad claims about model merging in modern, high-dimensional manifolds.

### B. Lightweight Architecture (ViT-B/32)
The choice of a CLIP ViT-B/32 backbone (13 parameter groups total) limits the strength of the paper's findings:
* **Shallow depth:** In a shallow 12-layer transformer, representational hierarchies are highly compressed. In larger models (e.g., 32-layer LLMs), layer-wise specialization (e.g., factual middle layers vs. semantic generation-focused later layers) is far more pronounced. 
* **False Generalization:** The conclusion that "layer-specificity is an illusion" is highly likely to be a direct artifact of ViT-B/32's shallow architecture. Merging diverse LLM experts is much more likely to exhibit genuine, physically critical layer-specific conflicts where spatial averaging would completely collapse the model.

---

## 2. Evaluation of Baselines

As noted in the soundness review, the paper's baseline comparison is flawed:
* **Uncalibrated Task Arithmetic Baseline:** The authors use a fixed $\lambda=0.3$ Task Arithmetic baseline. To claim that AdaMerging "fails to outperform" Task Arithmetic when the baseline itself has not been swept or tuned represents a weak empirical comparison.
* **No Tuned Single-Scalar Baseline:** The paper recommends that future works compare against a "properly tuned, single-scalar task-wise baseline." However, the authors do not present such a baseline in their primary comparison (Table 1), instead only showing the results of their own post-hoc "Spatial Mean" treatments. A true comparison should have included a "Tuned Task Arithmetic" baseline where a single scalar coefficient per task is optimized directly on the calibration split.

---

## 3. Do the Results Support the Claims?

While the author's analysis is highly detailed, a close reading of the results reveals that several of the core claims are overgeneralized or partially contradicted by the task-level data.

### A. Is "Layer-Specificity" Really an Illusion under 1+1 ES? (Claim 1)
* **The Claim:** The paper states that under 1+1 ES, layer-specificity behaves "entirely like a methodological illusion" and that Spatial Averaging "actually improves average accuracy from 85.07% to 85.21%."
* **The Contradiction:** Looking at Table 1, on **CIFAR-10**, the spatially averaged 1+1 ES model achieves **86.65%**, representing a **2.48% performance drop** compared to the optimized layer-wise 1+1 ES model (89.13%). CIFAR-10 is the most complex task in this suite. The "improvement" in average accuracy is driven entirely by the "SVHN rescue" (where SVHN accuracy rises from 75.13% to 77.67%). Therefore, for the most complex task (CIFAR-10), layer-specificity is **not** an illusion—it is functional and critical, even under zero-order search! Framing layer-specificity as a general "illusion" based on average accuracy is a misleading overgeneralization.

### B. Is Adam GD's Layer-Specificity "Only" an Overfitting Artifact? (Claim 2)
* **The Claim:** Under Adam GD, the delicate layer-wise configuration is "a delicate transductive overfitting artifact on the small calibration set."
* **The Contradiction:** While Adam GD fails to outperform the unoptimized baseline on average, the catastrophic performance collapse under Shuffling (average drops by 5.43%, CIFAR-10 drops by 15.69%) and Spatial Mean (CIFAR-10 drops by 10.35%) proves that the learned layer configuration is indeed **highly functional** and critical for keeping the merged model within the decision boundaries of complex tasks like CIFAR-10. While there is transductive overfitting, the delicate coordinate balance is *not* a meaningless artifact; it represents a highly specialized weight-space coordination.

### C. Extreme Landscape Flatness (Claim 3)
* **The Claim:** The model-merging landscape is exceptionally flat, tolerating up to 50% relative Gaussian noise with negligible performance decay.
* **The Caveat:** This "extreme flatness" is likely a symptom of the toy datasets and close task-vectors. In realistic scenarios where merged experts are highly divergent (e.g., merging a coding expert LLM and a medical expert LLM), the weight vectors are far apart, and the loss landscape around the coefficients is likely highly non-convex and sensitive to noise. The flatness claim is overgeneralized.

### D. CKA as a Poor Predictor of Accuracy (Claim 4)
* **The Claim:** Linear CKA is a poor predictor of downstream classification performance due to a decoupling between high-level activation correlation and fine-grained decision boundary integrity.
* **The Support:** This claim is **exceptionally well-supported** by the empirical results. Under Adam GD, the spatially averaged model has a higher CKA similarity to the CIFAR-10 expert ($0.9598$) than the optimized model ($0.9555$), yet its test accuracy collapses by over 10% (from 89.84% to 79.49%). This is a brilliant and rigorous contribution that warns the interpretability community against over-interpreting activation-space alignment.
