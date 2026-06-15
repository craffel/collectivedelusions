# Evaluation Component 4: Experiment Check

## Experimental Setup Evaluation

### 1. Model Backbone Limitations
The authors evaluate their framework on a **Vision Transformer (\texttt{vit\_tiny\_patch16\_224})** backbone with approximately **5.7M parameters**. 
- **Empiricist Critique:** While this is a reasonable model for proof-of-concept testing, a 5.7M parameter model is extremely small by modern standards. Model merging and post-training quantization are most critical and widely applied to large-scale models (e.g., ViT-Base with 86M parameters, ViT-Large with 307M parameters, or LLMs ranging from 1B to 7B parameters) where fine-tuning and running in full precision are computationally prohibitive. 
- Evaluating *only* on a toy-sized backbone (ViT-Tiny) severely limits the empirical strength of the paper. It is unclear if the task-vector norm scale pathology behaves the same way in larger models, or if the clipping threshold $\beta = 0.10$ would need completely different tuning. (Although the appendix discusses a percentile-based automated blueprint, no empirical scaling results on a larger backbone are presented).

### 2. Toy Dataset Selection
The evaluation is conducted on four small classification datasets: **MNIST, FashionMNIST, CIFAR-10, and SVHN**. 
- **Empiricist Critique:** These datasets are very simple, small, and low-resolution. 
- Furthermore, MNIST and FashionMNIST are single-channel (greyscale) datasets of size 28x28, while CIFAR-10 and SVHN are 3-channel datasets of size 32x32. Resizing and padding them to fit the 224x224 ViT input represents a highly artificial setup.
- Standard model merging benchmarks usually evaluate on more challenging, realistic datasets (e.g., DomainNet, ImageNet sub-populations, or VTAB benchmarks) where the transfer learning dynamics are more representative of real-world scenarios.

## Evaluation of Baselines

### 1. Missing Standard Static Merging Baselines
In the related work section, the authors discuss several prominent model merging methods, including **TIES-Merging** and **DARE**. 
- **Empiricist Critique:** Neither TIES-Merging nor DARE is evaluated as a baseline in Table 1! 
- TIES-Merging is one of the most widely used static model merging methods that explicitly resolves sign conflicts and parameter interference. DARE is another highly competitive static baseline. Their exclusion from Table 1 represents a significant gap in the evaluation.

### 2. "Strawman" Task Arithmetic Baseline
The authors compare against "Uniform Task Arithmetic (No TTA)", which blends weights with a fixed coefficient of $\lambda_k^l = 0.25$ across all tasks and layers.
- **Empiricist Critique:** Standard Task Arithmetic baseline in literature does *not* just use uniform coefficients. It involves searching for a single global scaling factor $\lambda \in [0, 1]$ (e.g., via a grid search on a validation set) to scale the sum of task vectors: $W_{\text{merged}} = W_{\text{pre}} + \lambda \sum_k \tau_k$. 
- By using a fixed $\lambda = 0.25$ without any tuning, the authors compare against an unnecessarily weak version of Task Arithmetic. A properly tuned global scaling factor would provide a much stronger and fairer static baseline.

## Statistical Soundness

### 1. Low Sample Size for Statistical Significance
The authors state that they run test-time optimization across **3 independent random seed splits** of the calibration subset.
- **Empiricist Critique:** Running only 3 seeds is a very small sample size for statistical analysis. It would be much more standard to use at least 5 or 10 independent splits to establish a robust estimate of variance.
- Crucially, with only 3 splits, a paired t-test has only 2 degrees of freedom ($df=2$). At $df=2$, the critical t-value for a two-tailed paired t-test at $p < 0.01$ is 9.925. Achieving statistical significance at $p < 0.01$ is extremely tight and highly sensitive to any minor variance in the seeds.
- More importantly, **Table 1 contains only single mean accuracy numbers and lacks standard deviation or confidence interval reporting.** Standard deviations should be directly integrated into Table 1 to allow readers to visually assess the overlap between methods.

## Do the Results Support the Claims?

### 1. Overstatement of "State-of-the-Art" Joint Multi-Task Accuracy
The authors claim that CR-PolySACM "consistently stabilizes model composition, achieving state-of-the-art joint multi-task accuracies." 
- **Empiricist Critique:** This claim is only supported in the aggressive **INT4 Symmetric Channel** format (19.07% vs. 18.10% for PolyMerge). 
- In all other evaluated formats (FP32, and all four INT8 formats), **standard PolyMerge consistently outperforms CR-PolySACM.** Therefore, CR-PolySACM is *not* the state-of-the-art method under standard precision scales—it actively degrades performance compared to the standard PolyMerge baseline.

### 2. Validation of Task-Vector Norm Scale Pathology
The results from Table 2 and Table 3 strongly support the theoretical claims regarding scale-blindness and clipping-regularization.
- **HessMerge Breakthrough:** The fact that HessMerge (Ours, incorporating CR-SACM) improves over AdaMerging (+1.36% in FP32) is a very clean empirical validation. Historically, unconstrained sharpness minimization failed in model merging, and showing that scale balancing resolves this is a strong result.
- **$\beta$ Ablation:** The ablation on $\beta$ (Table 3) shows a very clear non-monotonic trend, confirming both the gradient explosion regime ($\beta \le 0.01$) and the scale-blindness regime ($\beta \ge 0.25$) predicted by the theory. This is a very high-quality empirical confirmation.
