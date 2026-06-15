# 4. Experimental and Empirical Check

## Assessment of Experimental Design
The experimental and empirical evaluation is **fair to good**. The authors have constructed a direct, multi-task benchmark using standard datasets and a well-known Vision Transformer backbone (`ViT-Tiny`). They compare OmniMerge against relevant baselines across five realistic target quantization schemas under robust 8-bit post-training quantization. However, there are critical empirical anomalies in the ablation study, massive overclaims regarding statistical significance, and severe limitations in the experimental setup that weaken the paper's scientific rigor.

---

## Key Empirical Flaws and Critiques

### 1. The Ablation Study Anomaly (Performance Regression)
The ablation study in Table 2 reveals a major contradiction that the authors completely ignore in their text:
- **Baseline + TCR + SZNP (No SOS):** **50.45%** average accuracy.
- **Full OmniMerge (TCR + SOS + SZNP):** **50.33%** average accuracy.

#### Critique:
- The sub-component combination that uses *only* Scale/Zero-Point Noise Perturbation (SZNP) under a static Symmetric Per-Channel operator actually **outperforms** the full proposed OmniMerge framework (which adds Stochastic Operator Sampling - SOS) by **0.12%**.
- This indicates that adding SOS to the SZNP baseline causes a **performance regression**, suggesting that the two core contributions are sub-additive or that SOS introduces excess gradient variance during test-time adaptation that slightly degrades the final learned coefficients.
- Despite this, the authors write: *"Finally, the full OmniMerge framework (SOS + SZNP + TCR) maintains a highly robust and balanced average accuracy of 50.33% across all five operators, validating our unified formulation."*
- This is a classic example of glossing over negative or inconsistent ablation results. A rigorous reviewer must point out that, according to the paper's own data, **Stochastic Operator Sampling (SOS) is not only redundant but actually slightly harmful to average cross-schema performance when combined with SZNP.** The authors must address this discrepancy.

---

### 2. Overclaiming Statistical Significance on the "Denoising" Hypothesis
In Section 4.4, the authors propose a novel "discrete weight denoising" hypothesis, claiming that the discrete rounding operator ($\lfloor \cdot \rceil$) acts as a beneficial non-linear noise filter:
- They report that quantized OmniMerge achieves **50.78%** accuracy under the Symmetric Per-Channel operator, while its unquantized FP16 counterpart achieves **50.39%** accuracy.
- They state: *"This direct control experiment provides definitive empirical proof that weight-space discretization via rounding acts as a beneficial non-linear noise filter..."*

#### Critique:
- Given that the evaluation stream consists of exactly $N_{\text{eval}} = 1024$ total images, let's look at the absolute numbers:
  - 50.39% of 1024 = **516.0** correct predictions.
  - 50.78% of 1024 = **520.0** correct predictions.
- The difference is exactly **4 images** out of 1024!
- A difference of 4 correct predictions is completely statistically insignificant and well within the standard error of a binomial distribution for $N=1024$ and $p \approx 0.5$ (which is $\sqrt{p(1-p)/N} \approx 1.56\%$).
- Claiming that a 0.39% accuracy change on a tiny validation sample is "definitive empirical proof" of a fundamental mathematical ensembling phenomenon ("discrete weight denoising") is a massive overclaim. Without statistical error bars, standard deviations across multiple random seeds, or a formal statistical significance test (e.g., McNemar's test), this claim lacks scientific validity and should be rejected as a statistical fluke.

---

### 3. Artificial Benchmark and Weak Task Experts
The paper evaluates the ensembling on four highly disparate image datasets: MNIST, FashionMNIST, CIFAR-10, and SVHN.

#### Critique:
- **Low Expert Performance:** Due to severe training restrictions (256 samples, 3 epochs), the task experts are extremely weak. In particular, the SVHN expert achieves only **28.91%** accuracy (barely above the 10% random guessing baseline). 
- **Severe Parameter Interference:** Merging completely disjoint task domains (digits vs. fashion vs. natural objects) into a tiny, 5.7M parameter `ViT-Tiny` backbone is an highly artificial scenario. It triggers severe parameter interference, which explains why uniform ensembling (Task Arithmetic) collapses to a miserable **38.67%** average accuracy (far below the individual experts' average accuracy of $66.60\%$).
- **Lack of Realistic Merging Scenario:** In real-world applications, model merging is typically performed on experts fine-tuned on related tasks or similar domains (e.g., multi-lingual translation, instruction following, or medical imaging sub-specialties) using much larger backbones (which naturally resist interference). Merging four unrelated toy datasets into a tiny ViT backbone is a toy setup that does not reflect realistic model merging practices.

---

### 4. Learning Rate Mismatch and Step Budget
The authors use a learning rate of $\eta = 10^{-2}$ for the baselines (AdaMerging and Q-Merge) and $\eta = 2 \times 10^{-2}$ for OmniMerge, claiming that baselines oscillate at the larger learning rate. All methods are restricted to exactly 15 steps.

#### Critique:
- If standard AdaMerging and Q-Merge are restricted to a lower learning rate, they may require more than 15 steps to fully converge.
- Forcing all methods to adapt for exactly 15 steps while giving OmniMerge a 2x larger learning rate introduces a confounding variable. If the baselines were allowed to run for 30 or 50 steps until convergence, would they close the gap to OmniMerge? To ensure a fair comparison, the authors should either show convergence curves or evaluate all methods at their respective fully converged states, rather than imposing a strict 15-step limit that favors the method with the larger learning rate.
