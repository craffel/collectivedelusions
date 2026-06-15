# 4. Experimental Evaluation and Claims Check

## Experimental Setup and Rigor
The experimental evaluation is highly rigorous and designed with exceptional fairness:
* **Symmetric Tuning Protocol**: Typical model merging papers compare a newly proposed method with un-tuned or statically-scaled baselines (such as fixing $\lambda = 0.3$). This paper exposes this as a major confounder and evaluates *all* baseline methods across a complete range of scaling coefficients $\lambda \in [0.1, 1.0]$. This symmetric tuning ensures that every method is compared at its peak performance, making the evaluation scientifically fair.
* **Curation for Reproducibility**: The selection of a ViT-B-32 backbone and the four image classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) allows for rapid, reproducible CPU-friendly evaluation sweeps. This enabled the authors to perform over forty complete sweeps of the multi-task pipeline to establish reliable performance curves.

## Evaluation of Claims with Empirical Evidence

### Claim 1: Sign-resolution heuristics in TIES-Merging and DARE are redundant.
* **Evidence**: Supported. Under the symmetric tuning protocol, **Tuned STA** (with simple magnitude pruning and no sign resolution) achieves an average multi-task accuracy of **90.53\%** at $s=20\%$. This is statistically equivalent to **Tuned TIES-Merging** (**90.16\%**) and outperforms **Tuned DARE** (**88.95\%**) and **Tuned Task Arithmetic** (**88.64\%**).
* **Significance**: The fact that a stripped-down, 3-line PyTorch loop matches the performance of the complex, multi-stage TIES-Merging pipeline strongly supports the authors' hypothesis. Sign consensus checks do not provide any meaningful performance benefit over simple direct summation once the scaling confounder is corrected.

### Claim 2: The coordinate-wise collision rate is extremely low, making sign voting moot.
* **Evidence**: Supported. The authors prove that the probability of coordinate-wise update overlap between two task masks is theoretically bounded by $(s/100)^2$. At a typical survival density of $s=20\%$, the expected collision rate is $4.0\%$.
* **Empirical Verification**: The authors measured the actual overlap rate across layers in their ViT-B-32 experiments and found it ranges from **3.1\% to 4.3\%**. This means that for over 96\% of the model parameters, there is no possibility of parameter interference or sign conflicts, directly proving that sign-voting is mathematically moot.

### Claim 3: TIES-Merging's sign-voting is structurally harmful on complex datasets.
* **Evidence**: Supported. On the SVHN dataset (which represents a difficult domain shift), Tuned STA achieves **87.60\%** accuracy, whereas Tuned TIES-Merging achieves **85.55\%** (+2.05\% absolute difference). This suggests that forcing updates to conform to a majority sign and zeroing out conflicting coordinates destroys valuable task-specific representations, over-regularizing the parameter space.

## Experimental Gaps and Limitations
Despite the rigor of the evaluation within its scope, several experimental gaps limit the generalizability of the findings:

1. **Restricted Model Size and Architecture**:
   The ViT-B-32 model (~86M parameters) is small compared to modern deep learning architectures. Large scale models (such as Llama-3-8B or larger) may behave differently under parameter pruning and merging due to the emergent properties of large parameter spaces and different layer structures (e.g., swiGLU activations, group-query attention).
2. **"Toy" Classification Suite**:
   The four datasets selected (MNIST, FashionMNIST, CIFAR-10, SVHN) are standard, classic benchmarks, but they do not represent the complexity of modern industrial workloads (e.g., instructions-following, logical reasoning, code generation, or large-scale multi-modal tasks). 
3. **No Evaluation on Highly Correlated Tasks**:
   The tasks in the suite are diverse and unrelated, which ensures that their parameter updates are highly independent. In practice, model merging is often applied to similar tasks (e.g., merging different translation or coding models). Under high task similarity, parameter updates will be highly correlated, and coordinate collision rates are expected to rise significantly above the $(s/100)^2$ independence bound. The paper lacks any empirical evaluation under these conditions to see if STA's performance degrades compared to TIES-Merging.
