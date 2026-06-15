# Intermediate Evaluation 4: Experimental Evaluation Check

## Critique of the Experimental Setup
The experimental setup is cleanly designed but represents a highly scaled-down, toy-like setting compared to modern model merging research:
*   **Model Backbone:** The authors use **ViT-B-32** (approximately 86M parameters). While suitable for rapid local prototyping and CPU-level verification, this model is tiny compared to the billions-of-parameters scale (e.g., LLaMA-7B, Mistral-7B, or large-scale encoder-decoder networks) where modern sparse merging methods like TIES-Merging and DARE are standardly applied.
*   **Task Domains:** The 4-task suite consists of classic, relatively simple image classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). TIES-Merging and DARE are fundamentally text-centric and are evaluated on complex instructions, mathematical reasoning (GSM8k), and code generation (HumanEval) in large-scale LLMs. It remains unproven whether the authors' findings translate from simple 2D vision classifiers to the high-dimensional representation space of generative autoregressive LLMs.

---

## Evaluation of Baselines
The paper compares against the most relevant paradigms:
1.  **Task Arithmetic (TA)** (Ilharco et al., 2022)
2.  **DARE** (Yu et al., 2024)
3.  **TIES-Merging** (Yadav et al., 2023)

### Major Strengths of Baseline Comparison
*   **Symmetric Hyperparameter Tuning:** This is the most outstanding scientific contribution of the experimental section. Typical model merging papers suffer from "tuning bias," where the proposed method's hyperparameters (like scaling coefficient $\lambda$) are heavily optimized while baselines are run with sub-optimal default configurations (e.g., a fixed $\lambda=0.3$). By sweeping $\lambda \in [0.1, 1.0]$ with a step size of $0.1$ across **all** methods and reporting the peak performance at each method's optimal $\lambda^*$, the authors establish a perfectly fair and rigorous baseline comparison.

### Major Weaknesses in Baseline Selection
*   **Omission of DARE-TIES:** The authors evaluate the core "DARE-Linear" (DARE-TA) baseline which performs stochastic drop-and-rescale followed by standard linear addition. However, the preeminent and strongest variant proposed in the DARE paper is **DARE-TIES**, which combines stochastic sparsification with TIES-style sign consensus and disjoint merging. By omitting DARE-TIES from the evaluation, the authors have omitted the actual state-of-the-art DARE configuration.
*   **No Multi-Task Scale:** The benchmark evaluates merging only up to **4 tasks**. The severe parameter interference that motivates methods like TIES-Merging and DARE typically becomes catastrophic only when scaling to a much larger number of tasks (e.g., 8, 16, or 32 tasks). In a 4-task setup, coordinate collisions are naturally rare (as shown in Section 3.2.1), which favors the simpler STA. The authors' thesis would be much stronger if evaluated on a larger suite of 8+ tasks where interference is more severe.

---

## Do the Results Support the Claims?
Yes, for the most part, the results in Table 1 support the authors' primary thesis that sign-consensus heuristics are redundant when scaling is properly tuned:

1.  **Tuned STA vs. Tuned TIES-Merging:** Tuned STA ($s=20\%$) at $\lambda=0.8$ achieves **90.53\%** average accuracy, matching Tuned TIES-Merging ($90.16\%$) within the margin of statistical error. This demonstrates that a stripped-down magnitude-pruning baseline without any sign voting matches a complex, multi-stage pipeline, establishing sign consensus as redundant in this vision setting.
2.  **The SVHN Result:** Tuned STA's outstanding performance on SVHN (**87.60\%** vs. **85.55\%** for Tuned TIES) strongly supports the claim that TIES's hard sign consensus is overly regularizing and destructive to task-specific representations under domain shifts.
3.  **Confounder Correction:** The stark improvement of Tuned STA ($\lambda = 0.8$, **90.53\%**) over un-tuned STA ($\lambda = 0.3$, **82.91\%**) confirms that previous poor results for simple pruning were artifacts of update attenuation (under-scaling), not representing a fundamental failure of the merging logic itself.

### Flaws in Experimental Explanations
*   **Mischaracterization of R-STA Performance:** The authors assert that the performance drop of Rescaled STA (R-STA) at low densities is due to "variance distortion" and "tail-bias" inherent to magnitude pruning. As analyzed in the Soundness report, this drop is actually a direct consequence of their **incorrect mathematical scaling formula** ($\mathbb{E}[\|v^{\text{sparse}}_{k, l}\|_2^2] \approx \frac{s}{100} \mathbb{E}[\|v_{k, l}\|_2^2]$), which severely over-scales the pruned vector. The experimental comparison in Section 4.3 lacks a correct control (i.e., R-STA scaled by the actual energy fraction retained), undermining their explanation of the results.
*   **Lack of Statistical Error Bars:** Since the validation subsets are restricted to $2{,}048$ samples, and the average accuracy difference between Tuned STA (90.53%) and Tuned TIES (90.16%) is only $+0.37\%$, the authors must report standard deviations over multiple trials/seeds. Without error bars, it is impossible to verify if the minor performance improvements are statistically robust or merely random fluctuations.
