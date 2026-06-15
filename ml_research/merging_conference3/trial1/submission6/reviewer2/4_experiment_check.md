# Evaluation Step 4: Experiment Check

## Critical Evaluation of the Experimental Setup
The experimental setup uses a standard model merging testbed:
- **Backbone:** OpenCLIP ViT-B-32 backbone.
- **Tasks:** MNIST, SVHN, CIFAR10.
- **Checkpoints:** Independently fine-tuned task-specific experts retrieved from a Hugging Face Hub repository (`kasurashan/checkpoints_tint`).
- **Evaluation Subset:** 1,000 validation samples per dataset.
- **Baselines:** Pretrained base (Zero-shot), Individual Experts (upper bound), Model Soup (direct average), Task Arithmetic (with scaling coefficient sweep), and TIES-Merging (with scaling coefficient sweep).

While these choices are standard in the model merging literature, the implementation and execution of this setup are completely flawed.

## Do the Results Support the Claims?
**Absolutely not.** The central claims of the paper are completely unsupported because the reported metrics are artifacts of a collapsed evaluation pipeline, not actual classification performance.

Let's address the key claims and why the experiments fail to support them:

### Claim 1: WTA-Sign maintains a "top average accuracy" of 14.19%, outperforming TIES-Merging (12.92%) and Task Arithmetic.
- **Reality:** 14.19% average accuracy is an extremely poor result, barely better than random guessing (10%) across these three simple datasets (MNIST, SVHN, CIFAR10). 
- **The Source of the Difference:** The 1.27% difference (14.19% vs 12.92%) is not due to superior weight-space conflict resolution. It is because the validation subset of 1,000 samples has slightly different class counts. 
  - 14.19% corresponds to the exact percentage of Class A in the validation subsets (12.70% in MNIST, 18.65% in SVHN, 11.23% in CIFAR10).
  - 12.92% corresponds to the exact percentage of Class E in MNIST (11.52%), Class C in SVHN (16.02%), and Class A in CIFAR10 (11.23%).
  - Both WTA-Sign and TIES-Merging are outputting a constant prediction for every single input. WTA-Sign's output happens to be stuck in the "pretrained base" constant prediction class (Class A), while TIES-Merging shifts to Class E/C/A. Neither model is actually classifying images.

### Claim 2: The expert checkpoints operate in a highly relevant, real-world adversarial "negative knowledge" regime.
- **Reality:** The authors are attempting to spin a severe bug in their checkpoint loading/evaluation code as a "negative knowledge regime." 
  - Fine-tuning a ViT-B-32 model on MNIST should yield near 100% accuracy. The reported MNIST expert accuracy is **8.69%** (worse than random guessing).
  - If a model is fine-tuned on a dataset, its accuracy on that dataset cannot realistically drop below random guessing unless the labels are flipped or the evaluation is completely broken. 
  - The most plausible explanation is that the evaluation script failed to load the fine-tuned parameters into the model's visual projection or linear heads, or failed to construct the correct zero-shot classification prompts for CLIP.
  - Operating under the assumption that these broken checkpoints represent a valid "negative knowledge" testbed is scientifically dishonest or highly negligent.

### Claim 3: WTA-Sign is exceptionally robust to noisy, underperforming, or adversarial expert vectors.
- **Reality:** Because WTA-Sign with scaling factors $\lambda \in \{0.1, 0.2, 0.3\}$ simply outputs the exact same accuracies as the Pretrained Base (12.70%, 18.65%, 11.23%), the authors claim it "neutralizes these negative signals." 
  - In truth, the task vector is scaled down so much (or WTA-Sign's mask has zeroed out enough of the updates) that the model's parameters remain virtually identical to the base model's parameters.
  - Thus, the model continues to output the exact same constant predictions as the base model. This is not "intelligent gatekeeping" or "robustness"—it is simply the mathematical consequence of the merged task vector having no effect on the model's collapsed output.

## Missing Baselines & Experimental Gaps
Even if the evaluation pipeline were not broken, the experimental section has major gaps:
1. **Lack of Scale:** Evaluating on only three simple datasets (MNIST, SVHN, CIFAR10) is insufficient for modern model merging papers. Leading methods are evaluated on larger 8-task benchmarks (including SUN397, Cars, RESISC45, EuroSAT, DTD, GTSRB) or on large language models (LLMs).
2. **Subset Evaluation:** Evaluating on a small subset of 1,000 samples per dataset is highly susceptible to noise and variance, especially if class distributions are not perfectly balanced.
3. **No Hyperparameter Ablation:** The paper argues that WTA-Sign is superior because it is hyperparameter-free. However, TIES-Merging's performance depends heavily on the trimming threshold $k\%$ and scaling factor. The paper uses a fixed $k=20$ and does not show if a tuned TIES-Merging would easily outperform WTA-Sign.
4. **Computational Analysis:** While the paper claims WTA-Sign is exceptionally fast and lightweight, it does not provide any actual execution time measurements or scaling curves relative to the number of experts $K$ or model size $D$.
