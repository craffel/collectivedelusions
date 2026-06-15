# Evaluation of Soundness and Methodology

## Clarity of the Description
The description of the proposed framework, **Sparsity-Guided Task Arithmetic (SG-TA)**, and its variants is highly clear and mathematically precise.
- **Formulation:** The equations for task vector computation, binary mask generation under Global Quantile (GQ) and Layer-wise Quantile (LQ) scopes, weight fusion, and Task Vector Normalization (TV-Norm) are explicitly defined (Equations 1 through 10).
- **Parameters:** The variables involved, such as keep-ratios $k$ (or drop-probabilities $p$) and scaling coefficients $\alpha$, are clearly defined, making the algorithms easy to trace.
- **Calibration Methods:** The descriptions of Offline Few-Shot Validation Tuning (OFS-Tune) and its non-uniform extensions—Random Search (RS) and Coordinate Search (CS)—are conceptually and procedurally clear.

---

## Appropriateness of Methods
The methods utilized in the paper are generally appropriate and well-aligned with established practices in the model merging literature:
- **Baseline Selections:** The selection of baselines is comprehensive and highly rigorous. Comparing against Naive TA, Optimized TA, TIES-Merging, DARE-Merging, Decoupled Prune-then-Merge (P-then-M), Layer-Group Scaling (L-Scale), Fisher-Weighted Averaging, and Joint Multi-Task Learning (MTL) ensures a fair and challenging evaluation.
- **Calibration Framework:** Utilizing OFS-Tune (offline calibration on 10 validation samples per task) is a standard practice to find optimal merging parameters without incurring full retraining or test-time adaptation costs.
- **Controlled Sandbox:** The choice of a Vision Transformer (ViT-Tiny, 5.7M parameters) and low-resolution datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) is methodologically appropriate for isolating and analyzing the mechanics of weight-space masking. It allows the authors to conduct massive grid sweeps and run multiple random seeds (5 seeds per configuration), which is crucial for statistical rigor.

---

## Potential Technical Flaws and Limitations
While the experimental methodology is generally sound, several potential technical limitations and subtle flaws deserve critical attention:

1. **High Calibration Sensitivity and Volatility of Few-Shot Tuning:**
   The paper reveals a significant vulnerability in the OFS-Tune calibration framework under Task Vector Normalization (TV-Norm). For $N_{\text{val}}=10$, the standard deviation of SG-TA (GQ-Norm) is quite high (**$\pm 4.56\%$**). 
   - *Implication:* This indicates that the 10-sample validation split is highly sensitive to noise and random calibration seeds. While the authors conduct a control sweep showing that increasing $N_{\text{val}}$ to 20 or 100 stabilizes the standard deviation (reducing it to $\pm 1.10\%$ and $\pm 1.47\%$), the fact remains that under the default 10-sample setup, the calibration is highly volatile. This undermines the "training-free, zero-shot" appeal of the method, as it requires carefully curated or larger validation sets to achieve stable performance.
   
2. **Reliance on a Test-Time Task Routing Oracle:**
   The methodology assumes a "test-time task routing oracle" that routes input samples to their corresponding task-specific heads. While the authors correctly state that this is a standard convention in contemporary model merging literature, it remains a severe limitation. The shared backbone is consolidated, but the classification heads are kept separate. This means the model is not a single, fully unified multi-task classifier that can take any arbitrary image and output the correct class without knowing which task it belongs to beforehand. 
   
3. **The Absolute Performance Bottleneck:**
   There is a massive absolute performance drop between the merged models and both the Dense Expert Ceiling ($95.91\%$) and the Joint MTL Upper Bound ($95.55\%$). 
   - *Implication:* Even though SG-TA GQ (61.40%) represents an improvement over Naive TA (46.32%), a multi-task accuracy of 61% on extremely simple datasets (such as MNIST, which gets only $36.74\%$ accuracy under SG-TA GQ compared to $99.05\%$ under the expert) is practically unusable. This absolute degradation indicates that the proposed spatial regularization, while helpful in relative terms, does not solve the underlying representation collapse or capacity constraints in compact backbones.

---

## Reproducibility
The reproducibility of this work is rated as **excellent**:
- **Detailed Hyperparameters:** The training details (2 epochs, AdamW, learning rate $10^{-3}$, weight decay $0.01$, batch size 256) and backbone architecture details (\texttt{vit\_tiny\_patch16\_224} from the \texttt{timm} library) are fully disclosed.
- **Calibration Grids:** The authors specify the exact range of parameters swept (keep-ratio $k \in [0.1, 1.0]$, scaling factor $\alpha \in [0.1, 1.0]$) and the exact number of evaluations (60 for uniform grid search/random search, 100 for coordinate search).
- **Code Availability:** The authors state that they provide reproducible code and detailed sweeps to establish a verified baseline.
- **Statistical Rigor:** Reporting mean and standard deviations across 5 random calibration seeds is a high standard of experimental transparency that is often missing in machine learning publications.
