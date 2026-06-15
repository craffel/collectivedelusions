# 4. Experimental Setup and Critical Evaluation

An empirical evaluation must be held to the highest standards of statistical rigor, representative scale, and fair benchmarking. While the paper addresses a highly practical problem, several critical empirical limitations prevent the results from fully supporting the authors' claims.

## 1. Toy Scale of the Experimental Evaluation
The experimental testbed is extremely limited in scale, which severely constrains the generalizability of the findings:
- **Toy Model Backbone:** The authors employ a `ViT-Tiny` backbone containing only 5.7M parameters. Weight-space model merging and task arithmetic are primarily motivated by large-scale settings (e.g., LLMs or large-scale vision backbones) where storing and deploying separate models is computationally or financially prohibitive. A 5.7M parameter model is small enough that storing separate experts on modern edge devices is rarely a bottleneck. Evaluating on a toy backbone leaves it unclear if the proposed multi-schema co-optimization scales to larger, representative models where weight dynamics and quantization behaviors are vastly different.
- **Toy Datasets:** The evaluation is conducted on MNIST, FashionMNIST, CIFAR-10, and SVHN. MNIST is an extremely trivial dataset that can be solved with a tiny 2-layer CNN. Running model merging on a Vision Transformer for MNIST and FashionMNIST is highly non-standard and does not represent realistic edge multi-task ensembling. The paper would be significantly stronger if evaluated on challenging, modern multi-task benchmarks (e.g., ImageNet subsets, CUB-200, Flowers-102, or GLUE for NLP).
- **Weakly Trained Experts:** The task experts are fine-tuned on only **256 training images** for **3 epochs**. 
  - This extremely low-compute training regime results in very poor expert performance. 
  - Most notably, the SVHN expert achieves an individual validation accuracy of only **28.91%** (barely above random guessing for a 10-class dataset).
  - Merging experts that are barely trained and perform poorly on their respective domains makes the practical utility of the ensembling findings questionable. If the SVHN expert is non-functional, merging it into a multi-task model has little real-world value. The findings must be validated using fully-converged, high-performing experts.

## 2. Complete Absence of Statistical Rigor
The paper presents all experimental results (Table 1, Table 2, and discussions) as single-number accuracies. There are **no multiple random seeds, no standard deviations, and no confidence intervals** reported anywhere. This is a critical empirical flaw because:
1. **Sample Selection Sensitivity:** The experts are trained on an extremely small subset of 256 images, meaning the resulting weights are highly sensitive to the specific random subset of images selected.
2. **Calibration Stream Sensitivity:** The calibration stream consists of only 64 images per task, which introducing high variance depending on which 64 images are sampled.
3. **Compound Stochasticity in OmniMerge:** OmniMerge is highly stochastic by design. It relies on stochastically sampling operators (SOS) at each step and injecting Gaussian noise (SZNP) into the scales and zero-points.
- Given these multiple layers of stochasticity and the extremely small evaluation size (1024 images total), reporting single-run accuracies is highly unscientific. 
- The minor performance differences reported (e.g., OmniMerge's 50.10% vs 50.29%, or the 0.12% drop in the ablation study) could easily be due to random noise. 
- The authors **must** run all experiments across at least 3 to 5 random seeds and report the results as **mean $\pm$ standard deviation** with corresponding significance tests (e.g., p-values) to prove that their improvements are statistically meaningful.

## 3. Weak Statistical Grounding for the "Weight Denoising" Hypothesis
- In Section 4.4, the authors present a speculative hypothesis: that weight-space discretization (quantization rounding) can act as a "beneficial noise filter" or "weight denoising" regularizer, occasionally allowing quantized models to outperform their unquantized ceilings (e.g., quantized OmniMerge achieving 50.78% vs unquantized achieving 50.39%).
- The authors acknowledge that this difference is statistically modest (+0.39% absolute, representing exactly 4 correct predictions out of 1024 images) and well within the binomial standard error ($\approx 1.56\%$).
- Drawing scientific hypotheses or claiming "structural regularization" from an effect that is completely within the noise margin of a single run lacks empirical rigor. To validate such a highly speculative claim, the authors must conduct a rigorous, large-scale statistical analysis across multiple seeds and diverse model architectures. Without such evidence, this hypothesis remains unsubstantiated.

## 4. Optimization Bias and Baseline Fairness
- The authors limit the test-time adaptation to exactly 15 steps. They evaluate OmniMerge with a learning rate of $\eta = 2 \times 10^{-2}$ and the baselines with $\eta = 10^{-2}$, claiming that larger rates cause the baselines to oscillate.
- Restricting the baselines to a lower learning rate under an extremely tight 15-step budget introduces an optimization bias. A lower learning rate prevents the baselines from fully converging within the step limit.
- To ensure a fair comparison, the authors should evaluate the baselines over a wider range of step budgets (e.g., 30, 50, or 100 steps) or present learning rate convergence curves. This would verify that the baselines' lower performance is not merely an artifact of the 15-step constraint combined with a restricted learning rate.
