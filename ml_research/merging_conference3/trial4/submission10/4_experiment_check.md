# 4. Experimental Evaluation Check

## Deconstruction of the Experimental Design
The revised paper significantly addresses the critical flaws of the previous draft by retraining the individual task experts to high convergence and introducing a classical dynamic Linear Router baseline. However, several experimental limitations and vulnerabilities remain.

---

## Experimental Strengths
1. **Resolved "Fake Expert" Limitation:** In the previous version, the experts were trained for only 2 epochs on 512 images with a massive batch size, resulting in near-random performance (~11.0% accuracy). In this revision, the experts are trained for 15 epochs on 512 images using a standard batch size of 64 and AdamW. The specialized accuracies are: MNIST: 92.5%, FashionMNIST: 77.7%, CIFAR-10: 77.4%, SVHN: 34.5% (Joint mean 70.52%). This provides a mathematically valid specialized ceiling.
2. **Added Linear Router Baseline:** The inclusion of a classical soft routing baseline under comparable parameter-efficiency constraints (772 parameters vs 336 parameters, trained under identical optimization steps and validation data) provides a much-needed benchmark. This allows for isolating the effect of the non-monotonic cosine wave formulation.
3. **Rigorous Batch and Composition Analysis:** The heterogeneous evaluation across different batch sizes ($B \in \{1, 16, 256\}$) is highly thorough and transparently exposes the "heterogeneity collapse" that dynamic methods suffer from under task mixing.

---

## Remaining Experimental Flaws and Limitations

### 1. Data-Restricted Expert Training (Few-Shot Experts)
While the training epochs were increased to 15, the experts are still trained on only **512 samples per task**.
*   **The Problem:** Standard datasets like MNIST, FashionMNIST, CIFAR-10, and SVHN have tens of thousands of training images. Training on only 512 images represents a few-shot regime. This explains why the "converged" individual experts perform relatively poorly compared to standard models trained on the full datasets (e.g., SVHN expert at 34.50% vs typical >95% accuracy; CIFAR-10 expert at 77.40% vs typical >90% accuracy).
*   **The Impact:** Merging weak, few-shot experts is not representative of standard model merging scenarios, where practitioners attempt to merge large, fully converged, state-of-the-art models (e.g., LLaMA-7B or ViT-Base). Fully converged experts would have much deeper, highly specialized activation pathways, making the parameter interference and representational collapse even more severe, and potentially showing different relative behaviors between the dynamic routers.

### 2. Overfitting on the 64-Sample Validation Calibration Set
Both the Linear Router and QWS-Merge are calibrated on a tiny offline validation set containing only 16 samples per task (64 total).
*   **The Setup:** QWS-Merge optimizes 336 parameters for 100 steps of full gradient descent with a batch size of 16. In essence, the optimizer performs **100 full epochs of supervised training on the 64 calibration images**.
*   **The Impact:** With 336 parameters and 100 epochs, there is a very high risk of supervised overfitting to the 64 calibration images. This suggests that the performance gains over static baselines like OFS-Tune (which only optimizes 56 layer-wise static coefficients) are driven by the higher parameter capacity (6x more parameters) and intensive calibration rather than the "quantum superposition" phase dynamics.

### 3. Lack of Statistical Validity and Sensitivity Analysis
*   The entire experimental pipeline is evaluated on a single run with a fixed random seed (`seed=42`).
*   Given that the calibration set is extremely small (16 samples per task) and the experts are few-shot trained, the results are likely highly sensitive to the choice of the seed, the specific validation samples selected, and the random initialization of the projection matrix $P$.
*   Without reporting standard deviations, running across multiple random seeds, or performing a sensitivity analysis on the size of the calibration set, the empirical claims are statistically fragile.

### 4. Evaluation on a Test Subset
The models are evaluated on a randomly shuffled subset of 1,000 test images per task. While this is done to speed up evaluation, the paper should clearly state this and report performance on the full test sets to ensure that the reported metrics are not biased by the specific sub-sampling.
