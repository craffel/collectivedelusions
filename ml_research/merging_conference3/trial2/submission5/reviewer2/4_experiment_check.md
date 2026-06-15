# Experimental Setup and Results Evaluation

## Critical Evaluation of the Experimental Setup

### 1. Limited, Toy-Scale Dataset Suite
The paper evaluates NETA and its baselines across four visual classification datasets: **MNIST, FashionMNIST, CIFAR-10, and SVHN**. 
*   **Critique**: This is a highly limited, toy-scale evaluation suite. In the model merging literature, standard benchmarks using the CLIP ViT backbone typically evaluate on a diverse **8-dataset suite** (adding CIFAR-100, GTSRB, RESISC45, DTD, and EuroSAT) representing a wide variety of specialized domain shifts, high-resolution imagery, and diverse classification objectives. Evaluating on only four low-resolution, relatively simple visual datasets limits the generalizability of the empirical findings. The authors acknowledge this in Section 5.1, but from an empiricist perspective, this is a major experimental weakness.

### 2. Sub-Sampling of Test Sets to 1024 Images
The paper performs all evaluations on a "representative subset" of 1024 randomly sampled test images from each dataset.
*   **Critique**: This is highly non-standard and statistically weak. 1024 images represent a very small sample size. On 1024 images, a single classification decision changes the accuracy by $\approx 0.098\%$. 
    *   The reported improvement on MNIST of $+0.26\%$ is equivalent to **fewer than 3 images** out of 1024.
    *   The reported improvement of $+0.16\%$ on CIFAR-10 for DARE is equivalent to **fewer than 2 images**.
*   Such minuscule margins are well within the variance of random subset sampling. The justification that sub-sampling was necessary "to manage the computational overhead... under strict Slurm queue limits" is highly unconvincing. CLIP ViT-B/32 is an extremely lightweight model by modern standards; running zero-shot visual classification on 10,000 images takes less than 1-2 minutes on a single GPU or CPU. Evaluating on the full test sets would have dramatically enhanced the statistical power and credibility of the results.

### 3. Strength of Baselines
The baselines selected are standard and representative:
*   **Zero-Shot**: Task Arithmetic, TIES-Merging, and DARE.
*   **Test-Time Adaptation**: Task-Wise AdaMerging and Layer-Wise AdaMerging.

However, the authors evaluate TIES-Merging and DARE using a fixed default scaling factor of $\lambda_0 = 0.30$. While they acknowledge that tuning their global coefficients could impact performance, evaluating all methods under a single fixed hyperparameter budget does not represent a "well-tuned" baseline comparison. It is highly possible that TIES-Merging or DARE would significantly outperform NETA if their global scaling coefficients were optimized.

---

## Do the Results Support the Claims?

### 1. Isotropic Magnitude Balancing (Claim Supported with Qualifications)
The claim that NETA provides isotropic magnitude balancing to prevent dominant task updates (like SVHN) from hijacking representation space is supported. In Table 1, NETA improves MNIST and FashionMNIST accuracy over Task Arithmetic. However, the gains are extremely small ($+0.26\%$ and $+0.65\%$) and, as discussed, may suffer from sub-sampling noise.

On SVHN, NETA's accuracy drops significantly (from $80.14\%$ to $77.02\%$), causing a marginal drop in the multi-task average accuracy. The authors are commendable for being scientifically honest about this "peak performance vs. representation fairness trade-off" rather than trying to obscure it.

### 2. The Overfitting-Optimizer Paradox (Claim Highly Supported)
The claim that joint entropy minimization in test-time adaptation (AdaMerging) overfits to easy, low-entropy tasks and suppresses difficult ones is **strongly supported** by the results. Task-Wise AdaMerging suffers a catastrophic collapse on FashionMNIST (dropping from $82.10\%$ to $77.54\%$) and CIFAR-10 (dropping from $92.77\%$ to $89.70\%$). This is a major, highly convincing empirical finding that highlights a fundamental flaw in the popular prediction entropy objective for model merging.

The spatial degree-of-freedom explanation (why Layer-Wise AdaMerging survives while Task-Wise collapses) is also logical and well-reasoned, showing how high-dimensional parameter spaces can satisfy joint entropy minimization without global task suppression.

### 3. Continuous $\alpha$-Relaxation and Noise-Damping (Claims Supported)
The ablation studies in Table 2 strongly support the proposed extensions:
*   **$\alpha = 0.5$** recovers SVHN performance to $78.55\%$ while maintaining improvements on MNIST and FashionMNIST, demonstrating a practical trade-off mechanism.
*   The noise-damping stabilizer $\beta$ is shown to be highly robust; varying it from $10^{-6}$ to $10^{-2}$ has almost no negative impact on performance, confirming its safety.

### 4. Scale Compensation Factor $\gamma^l$ (Claim Strongly Supported)
The proposed closed-form scale-compensation factor $\gamma^l$ (Table 2) represents one of the strongest results in the paper. It achieves the highest zero-shot accuracy on MNIST ($96.32\%$) and FashionMNIST ($82.85\%$) while recovering SVHN accuracy to $77.34\%$, leading to an overall average of $87.28\%$. This shows that directional norm contraction can be resolved analytically and zero-shot without any manual grid search.

### 5. Grid Search over $\lambda_0$ (Claim Fully Supported)
The grid search in Table 3 is exceptionally rigorous and honest. It shows that:
*   Task Arithmetic is highly sensitive to the scaling coefficient, peaking at $\lambda_0 = 0.40$ with $89.16\%$ accuracy.
*   NETA also improves with scaling, peaking at $89.06\%$ for relaxed NETA ($\alpha=0.5$, $\lambda_0 = 0.40$).
*   When both methods are fully tuned, Task Arithmetic remains slightly superior in average accuracy ($89.16\%$ vs. $89.06\%$).
*   The authors honestly report this, clarifying that NETA's primary value is not in maximizing average accuracy, but in acting as an isotropic regularizer to guarantee representational fairness.
