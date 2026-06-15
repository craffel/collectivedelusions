# 3. Soundness and Methodology Evaluation

## Clarity of the Description
The methodology is formally structured with a sequence of mathematical formulations (Equations 1 to 15). However, several key implementation details are obscured, and the text employs highly verbose, defensive language to justify questionable architectural choices.

## Appropriateness of Methods & Potential Technical Flaws

A rigorous, adversarial inspection of the methodology reveals several significant technical flaws, logical inconsistencies, and methodological errors:

### 1. Global Standard Deviation Across Heterogeneous Parameters
* In **Global Task Vector Standardization** (Equations 2 & 3), the authors compute a single standard deviation $\sigma_k$ across *all* $D$ parameters of the model. 
* This is mathematically and physically questionable. A Vision Transformer (ViT-Tiny) contains a highly heterogeneous set of parameters operating under completely different gradient physics and activation ranges (e.g., self-attention query/key/value projection weights, MLP feed-forward weights, biases, layer-norm gains, and the class token). 
* Flattening these diverse parameters into a single vector and computing a monolithic standard deviation $\sigma_k$ ignores layer-specific scales and dynamics. 
* While the authors propose **Layer-wise Standardization** (Equations 4 & 5) as an alternative, they dismiss it in favor of Global Standardization, claiming that layer-wise normalization "ignores the global relative importance of layers." This justification is weak; ignoring layer-wise scales in favor of a monolithic standard deviation is a fundamental methodological shortcut that distorts representation hierarchies.

### 2. Logical Contradiction in "Scale Decoupling"
* The paper relies on a "deliberate and crucial design choice": **Scale Decoupling**. Standardization is used strictly as a decision-making filter for coordinate routing and saliency ranking, but the unstandardized physical updates are integrated into the network. 
* The authors analyze the frequency of "scale overrides" (where a task-distinctive coordinate with small unstandardized magnitude overrides a task with large magnitude) and report a scale override rate of exactly 13.79% (761,836 parameters) across the model.
* This decoupling exposes a major logical contradiction: if a task expert (like SVHN or CIFAR-10) has trained large unstandardized weights to represent a critical feature, routing that coordinate to a simpler task (like MNIST) because of standard deviation scaling physically severs that critical weight update from the complex expert. 
* Forcing 13.8% of the model's coordinates to be routed based on an artificial standardized metric rather than their actual physical magnitudes likely causes severe representational fragmentation. This fragmentation is the primary reason why the merged model's performance collapses so heavily compared to individual experts (retaining only $\sim$46% joint mean accuracy vs. a 94.91% expert ceiling).

### 3. Deliberate "Optimizer Mismatch" and Strawman Baseline Evaluation
* To demonstrate the superiority of EPM, the authors evaluate SOTA baselines (AdaMerging and ZipMerge) under a zero-order (1+1) Evolution Strategy optimizer. 
* **This represents a fundamental methodological error and a classic strawman setup.** AdaMerging and ZipMerge were explicitly designed and engineered to be optimized using native first-order gradient descent on differentiable validation cross-entropy losses. 
* The authors restrict their validation metric to a non-differentiable validation accuracy minimax score and force all methods to use (1+1)-ES, a greedy single-point random mutation search. Forcing a 56- or 70-dimensional non-convex continuous parameter space to be optimized under (1+1)-ES and then claiming their stagnation represents "absolute optimization failure" is highly unfair and contrived. 
* Under their native first-order gradient-descent optimization pipelines, AdaMerging and ZipMerge would converge efficiently and likely outperform TLC-Tune. The authors' claim of "unbiased conditions" is a cover for a highly biased optimizer mismatch.

### 4. Fragility of the (1+1)-ES Optimizer
* The authors claim TLC-Tune is "perfectly stable" and "immune to transductive noise." 
* However, in Section 4.4, they report a localized performance dip for EPM (TLC $p=0.5$) at $N_{\text{val}}=512$, where joint mean accuracy drops to **35.36%**. 
* They admit that (1+1)-ES is a "greedy local random search" that gets trapped in suboptimal local minima under intermediate sample sizes, and they must invoke a "Multi-Start" or "CMA-ES" strategy to recover. 
* This admission directly contradicts their core claims of "absolute stability" and reveals that their minimalist zero-order optimization setup is highly fragile and sensitive to random seeds and validation data distributions.

## Reproducibility
* **Code Availability:** The authors claim their core routing implementation requires "fewer than 10 lines of PyTorch," but they do not provide any repository link, code snippet, or supplemental code.
* **Data splits:** The specific samples selected for the 128-sample-per-task validation splits are not described. Given that 128 samples is a tiny fraction of standard datasets, different random splits will lead to significant validation variance.
* **Lack of hyperparameter details:** The specific hyperparameters of (1+1)-ES (e.g., initial mutation scale, step-size adaptation factors) are brushed over, limiting exact reproducibility.
