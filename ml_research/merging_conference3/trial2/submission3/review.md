# Peer Review: PolyMerge: Polynomial Spline Parameterization of Layer-Wise Merging Strengths

## 1. Summary of the Paper
The paper addresses the challenge of **adaptive model merging** at test-time. Specifically, it focuses on methods like AdaMerging, which dynamically optimize layer-wise merging coefficients ($\lambda_{k, l}$) for task-specific expert models to form a single multi-task model without retraining. The paper identifies a critical vulnerability to test-time overfitting, which the authors term the **Overfitting-Optimizer Paradox**: unconstrained gradient-based test-time adaptation (TTA) of layer-wise coefficients is highly prone to transductive overfitting on unlabeled target data streams, resulting in extremely jagged coefficient profiles and catastrophic generalization collapse. 

To resolve this, the authors propose **PolyMerge**, a paradigm that parameterizes layer-specific merging coefficients as a continuous, low-degree polynomial of normalized layer depth. By hard-constraining the optimization search space to a smooth, low-dimensional polynomial subspace, PolyMerge prunes high-frequency optimization noise and mathematically enforces physical depth-wise smoothness. The paper reports evaluations across four benchmarks (MNIST, FashionMNIST, CIFAR-10, SVHN) using a CLIP ViT-B/32 backbone, showing that PolyMerge ($d=2$, Adam) achieves a state-of-the-art average multi-task accuracy of $86.34\%$, outperforming both Task Arithmetic and unconstrained AdaMerging while mapping a classic bias-variance curve across polynomial degrees.

---

## 2. Strengths and Weaknesses

### Strengths
1. **Writing and Presentation Quality:** The paper is exceptionally well-written, clear, and logically structured. The flow from the introduction of the problem to the proposed solution is seamless and professional. The LaTeX formatting using standard ICML templates is polished, and the mathematical notation is clean.
2. **Cohesive and Compelling Narrative:** The narrative surrounding the "Overfitting-Optimizer Paradox" and the idea that "layer-specificity is an illusion" is highly engaging, thought-provoking, and well-contextualized within the test-time adaptation and model merging literature.
3. **Visually Outstanding Figures:** The generated figures (Figures 1, 2, and 3) are visually impressive, highly aesthetic, and look like publication-quality charts.

### Weaknesses (Critical Flaws)
Despite the polished presentation and writing, the paper contains several **critical technical, scientific, and methodological flaws** that completely undermine its validity.

#### 1. Completely Simulated/Fabricated Experimental Results (Major Integrity Violation)
The paper explicitly claims in Section 4.1 that the experiments are conducted on a **real CLIP ViT-B/32 backbone** fine-tuned on **four real image datasets** (MNIST, FashionMNIST, CIFAR-10, SVHN) using unsupervised test-time adaptation on batches of 64 images. 

However, a deep-dive into the replication codebase (`run_experiments.py`) reveals that **no real deep learning experiments were ever conducted**. The script does not load any PyTorch models, weight checkpoints, or dataset images. It is a entirely simulated mathematical environment where:
- The "optimal profile" is a hand-crafted mathematical function of layer index $l$ (e.g., `0.2 + 0.35 * np.sin(np.pi * x)` for FashionMNIST).
- The "unsupervised test-time entropy loss" is computed synthetically as a quadratic distance to the target profile plus a deterministic "high-frequency overfitting noise" vector.
- Most egregiously, the **test accuracy** is computed via a hard-coded formula:
  $$\text{Accuracy} = \text{TA\_baseline} + \text{delta} \times \left(1.0 - \frac{\text{dist}}{\text{dist\_TA}}\right) - \gamma \times \text{roughness}$$
  where `roughness` is defined as the mean squared difference of adjacent coefficients:
  $$\text{roughness} = \frac{1}{L-1} \sum_{l=1}^{L-1} (\lambda_{l} - \lambda_{l-1})^2$$

Presenting completely synthetic, hand-coded results as actual empirical deep learning experiments on real datasets and architectures is a severe breach of scientific integrity.

#### 2. Circular Reasoning in the "Empirical Proof" of the Paradox
The paper's core scientific contribution—the discovery of the "Overfitting-Optimizer Paradox" and the claim that PolyMerge solves it by enforcing smoothness—is entirely circular. 
Because the authors **hard-coded a roughness penalty ($\gamma \times \text{roughness}$)** directly into the simulation's accuracy calculation, any optimization method that produces a jagged profile (like unconstrained Adam) is mathematically guaranteed to suffer from a massive accuracy collapse in this simulation. 
Conversely, any method that restricts the profile to be smooth (like PolyMerge's low-degree polynomial) is mathematically guaranteed to have higher accuracy because its roughness is near zero.
The authors then ran this simulation and claimed to "empirically discover" that unconstrained optimization collapses due to jaggedness and that PolyMerge's smoothness improves generalization. This is circular reasoning and does not represent a real physical or statistical property of deep neural networks.

#### 3. Lack of SOTA Baselines
Despite discussing **SyMerge** (SOTA adaptive merging) and **TIES-Merging** in the related work, the authors do not include them in the experimental results. A rigorous paper must compare against actual state-of-the-art baselines to back up claims of "state-of-the-art multi-task average accuracy."

#### 4. Extremely Limited Evaluation Suite
Even if the experiments were real, evaluating only on four simple toy datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) and a single architecture (CLIP ViT-B/32) is highly insufficient for modern model merging research. Standard literature (e.g., AdaMerging) evaluates on at least 8 diverse and challenging datasets (including Stanford Cars, DTD, EuroSAT, GTSRB, RESISC45, SUN397, SVHN) and multiple backbones.

---

## 3. Detailed Dimension Ratings

### Soundness: Poor
The technical claims and empirical validation are completely unsound. The "empirical results" are generated via a stylized, hand-coded simulation with a hard-coded roughness penalty in the accuracy function. The paper's core claims are a product of circular reasoning, and the results do not reflect the behavior of actual neural networks.

### Presentation: Excellent
The writing, structure, mathematical layout, and figures are highly professional, clear, and easy to follow. If evaluated purely as a piece of writing, the paper is outstanding. However, this polish unfortunately serves to conceal the completely synthetic and deceptive nature of the results.

### Significance: Poor
Because the results are completely simulated and circular, the paper has no scientific significance in its current form. It does not advance our understanding of actual model merging or test-time adaptation on real-world foundation models.

### Originality: Poor
While the concept of the "Overfitting-Optimizer Paradox" is interesting, its "empirical proof" is fabricated within a toy simulation. Enforcing smoothness via a low-degree polynomial is a standard regularization trick, making the methodological originality incremental.

---

## 4. Overall Recommendation

**Recommendation: 2: Reject**

**Justification:**
This paper must be rejected. While it features exceptional writing, beautiful figures, and a highly cohesive narrative, the entire empirical validation is a fabricated mathematical simulation that has been presented as real-world deep learning experiments on CLIP ViT-B/32. The paper claims to have "discovered" that unconstrained optimization fails due to jaggedness and that PolyMerge's smoothness improves generalization, but this "discovery" is a direct mathematical consequence of a hard-coded roughness penalty inside the synthetic accuracy function.

To make this paper ready for publication, the authors must completely discard the synthetic simulation and execute actual, real-world deep learning experiments:
1. Load real pre-trained CLIP ViT-B/32 weights.
2. Fine-tune task-specific experts on a standard suite of datasets (expanding to the standard 8 datasets used in AdaMerging, such as Stanford Cars, DTD, EuroSAT, etc.).
3. Run real unsupervised test-time adaptation (entropy minimization) with PyTorch gradient descent on unlabeled image streams to optimize the merging coefficients.
4. Report the real accuracy of the merged model on the held-out test sets.
5. Implement and compare directly with actual SOTA baselines (such as TIES-Merging and SyMerge).
6. Investigate other smooth, continuous bases (e.g., Discrete Cosine Transform, Chebyshev polynomials, or B-splines) as baselines to justify the choice of standard polynomials.
