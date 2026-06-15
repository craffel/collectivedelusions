# 3. Soundness and Methodology

## Clarity and Rigor of Description
The mathematical formulation and methodology of FIOSR are presented with exceptional clarity and high rigor. 
- **Notation Reference:** The authors provide a structured notation table in Appendix A.1 which significantly improves readability.
- **Underlying Theory:** The connection between diagonal empirical Fisher Information coordinates and inverse coordinate noise variances under conditional Gaussian distributions is formally derived.
- **Robustness under Non-Gaussianity:** The authors address the ReLU/GELU non-Gaussian activation limitation analytically in Appendix A.2, proving that even under rectified Gaussian activation distributions, the Fisher Information remains dominated by the inverse coordinate variance ($F_j \propto 1/\sigma_j^2$).
- **Classifier Weights as Proxies:** The potential mismatch between raw activation means and classifier weights is mathematically resolved in Appendix A.3, proving that under $L_2$-regularized softmax cross-entropy, the classifier weights converge directly to a scalar multiple of the true activation centroids, bounding the finite-sample misalignment error at $O_p(1/\sqrt{N_c})$.

## Appropriateness of Methods & Technical Limitations (Practitioner's Lens)

From a practical deployment and systems-engineering perspective, several methodological design choices and assumptions warrant critical evaluation:

### 1. The Independence Assumption of Diagonal FIM (Coordination-Alignment Bottleneck)
FIOSR utilizes a diagonal approximation of the Fisher Information Matrix (dFIM). This is computationally highly efficient—taking only **2.71 to 4.05 milliseconds** to estimate and smooth—which is incredibly attractive for real-world inference pipelines. However, diagonal FIM assumes that the features/coordinates are independent. In actual deep learning representation spaces, features are highly correlated and non-axis-aligned.
- **The Rotated Noise Collapse:** In Section 4.6, when evaluating the models under rotated, non-axis-aligned noise, the performance of diagonal Fisher (**FIOSR-Diag**) collapses below the flat baseline (67.38% vs 67.50%). This is because rotation spreads the noise variance evenly across coordinates, rendering the diagonal filter ineffective.
- **Mitigation Cost:** While the authors' online EVD shrinkage covariance alignment (**FIOSR-Online**) outclasses the flat baseline (67.68%), and the oracle block-diagonal K-FAC equivalent (**FIOSR-Rotated**) restores the gains (73.07%), these require estimating and decomposing a full covariance matrix. For higher-dimensional layers (e.g., $d \ge 1024$), on-the-fly eigenvalue decomposition is computationally prohibitive and fails to scale, representing a major systems bottleneck.

### 2. Micro-Batch Homogenization (MBH) Latency and Systems Overhead
Under highly heterogeneous streams, MBH partitions a mixed batch of size $B$ into $G \le K$ homogeneous micro-batches.
- **FLOPs-Equivalence Redundancy:** In the worst-case scenario where a stream contains samples from all $K$ tasks, MBH partitions the batch into $G = K$ micro-batches, requiring $K$ separate forward passes. From a systems perspective, running $K$ separate forward passes with $K$ distinct merged weights is computationally equivalent to simply executing the original, unmerged specialized expert models on their respective samples. This completely defeats the purpose of test-time weight merging (which aims to run a single merged model to save computation).
- **Gating Trade-off:** Setting the gating limit $M=1$ (hard Top-1 routing) restricts active forward passes to 1 and completely eliminates MBH overhead, but collapses parameter-space ensembling back to a hard task-routing selection mechanism. This is a crucial conceptual compromise: practitioners must sacrifice the actual benefits of weight-space ensembling for that sample in exchange for computational efficiency.

### 3. Calibration Split Dependency ($N_c$)
Although test-time optimization is bypassed, FIOSR still requires a microscopic calibration split ($N_c$ samples per task) to estimate the dFIM coordinates. 
- **Underdetermined Failure State:** As shown in Appendix B.3, for extreme few-shot regimes where $N_c \le 4$, the 48-dimensional variance estimation is highly underdetermined, causing FIOSR to overfit to calibration noise and perform worse than flat Cosine (-9.48% absolute loss at $N_c=2$).
- **Zero-Shot Limitation:** The framework requires at least $N_c \ge 8$ samples per task to stable-estimate coordinates and outperform the baseline. This dependency prevents immediate, zero-shot out-of-the-box deployment for new, uncalibrated tasks without first collecting and passing a calibration set.

## Reproducibility
The methodology is exceptionally reproducible. The authors explicitly lay out:
- Exact hyperparameter configurations (smoothing stabilizer $\beta=0.5$, power attenuation $\gamma=0.7$, scaling temperature $\tau=0.001$).
- Specific dimensions and parameters of the 192-dimensional Analytical Coordinate Sandbox and 64-dimensional LoRA simulations.
- Complete description of the physical ResNet-18 validation (Adam optimizer, 120 epochs, linear heads).
- Structured pseudocode, notation references, and proofs.
Given this level of exhaustive detail, reproducing the empirical results of this paper should be straightforward for any expert reader.
