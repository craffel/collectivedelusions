# Impact and Presentation Evaluation

## Major Strengths
1. **Implementation Simplicity:** The 4-line vectorized PyTorch implementation is extremely simple, elegant, and highly parallelizable. It introduces virtually zero computational overhead compared to more complex merging pipelines.
2. **Elimination of Hyperparameters:** WTA-Sign bypasses the tedious hyperparameter tuning of TIES-Merging, such as finding optimal trimming ratios, sign-voting thresholds, and scaling coefficients.
3. **Writing Clarity:** The paper is structured well, and the explanation of the mathematical formulation of WTA-Sign is easy to follow.

## Critical Areas for Improvement

### 1. Theoretical Grounding (High Priority)
- The paper relies completely on a heuristic assumption ("magnitude is task confidence"). To make this work theoretically sound, the authors must provide formal mathematical proofs, error bounds, or convergence guarantees.
- For example, they should theoretically bound the functional degradation (or approximation error) introduced when an expert's parameter updates are discarded via conformity masking.
- They must ground the method in a rigorous optimization-theoretic or Bayesian framework that explains why the coordinate-wise maximum absolute change is an optimal or bounded-error estimator for multi-task parameter consolidation.

### 2. Resolving Scale Sensitivity
- The authors must address the scale sensitivity of the method. If one expert is trained with a larger learning rate, its task vector will dominate the sign election. 
- The authors should implement and evaluate normalization strategies (e.g., standardizing the task vectors to unit norm, or dividing by parameter-wise running variance) to ensure that different experts' updates are compared on a fair, scale-free basis.

### 3. Fixing the Fine-Tuning and Evaluation Pipeline
- The current empirical evaluation is catastrophic. Fine-tuned experts must be functional (achieving standard benchmark accuracies of >90% on MNIST, SVHN, and CIFAR-10) before any claims of model merging can be validated.
- The authors must diagnose and fix the bugs in their fine-tuning or evaluation code (e.g., classifier projection heads, label alignments, or preprocessing) so that they can show the performance of WTA-Sign on actual, functioning experts.
- A proper model merging evaluation must demonstrate that the merged model retains high accuracy *across all tasks simultaneously*, rather than simply reverting to the zero-shot base capability of 14.19%.

### 4. Scaling to Realistic Benchmarks
- Model merging is primarily used in LLMs (e.g., Llama-based models) and modern vision-language models. Evaluating exclusively on 1000-sample subsets of toy datasets (MNIST, SVHN, CIFAR-10) with near-random accuracies is insufficient. The authors must scale their evaluation to larger-scale benchmarks to prove the method's real-world utility.

## Overall Presentation Quality
The presentation of the paper is visually clean and mathematically explicit. However, the academic tone is compromised by the disingenuous framing in Section 4.3. The authors attempt to spin a failed experimental setup (broken experts and zero merging effect) as a "highly relevant, real-world adversarial negative knowledge regime." Academic writing must be honest about limitations rather than rebranding severe experimental bugs or failures as innovative features.

## Potential Impact and Significance
In its current form, the potential impact of this paper is **extremely low**. 
- Because the method is an incremental heuristic variant of TIES-Merging and lacks rigorous theoretical guarantees, it does not advance our theoretical understanding of model merging.
- Because the empirical validation is built on completely broken fine-tuned models and the merged results merely reflect the zero-shot base model, there is no empirical evidence that WTA-Sign actually works as a model merging method.
- Unless the theoretical foundation is built from scratch and the empirical results are validated with actual functioning experts, this submission is not ready for publication.
