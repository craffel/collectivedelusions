# Peer Review of "Winner-Take-All Sign Election: A Minimalist Approach to Model Merging"

## Paper Summary
The paper introduces **Winner-Take-All Sign Election (WTA-Sign)**, a training-free and hyperparameter-free method for merging specialized task experts (adapted from a shared pre-trained backbone) into a single multi-task network. WTA-Sign replaces the democratic sign-voting consensus mechanism in TIES-Merging (Yadav et al., NeurIPS 2023) with an oligarchic, magnitude-driven "Winner-Take-All" approach. Specifically, at each parameter index, the expert with the largest absolute update is elected as the "winner," and its sign determines the direction of the merged parameter. Non-conforming updates from other experts (those with opposing signs) are masked out (set to zero), and the conforming updates are averaged. The paper evaluates WTA-Sign against Model Soups, Task Arithmetic, and TIES-Merging using an OpenCLIP ViT-B-32 backbone on MNIST, SVHN, and CIFAR-10 datasets.

---

## Strengths and Weaknesses

### 1. Strengths
- **Implementation Simplicity:** The method is extremely elegant and straightforward, requiring only four lines of vectorized PyTorch code. It runs entirely in parallel with virtually zero computational overhead.
- **No Hyperparameters:** WTA-Sign bypasses the tedious hyperparameter tuning of TIES-Merging, such as finding optimal trimming ratios, sign-voting consensus thresholds, and energy-based scaling coefficients.
- **Clarity of Presentation:** The paper is well-structured, and the mathematical steps of the algorithm are explained clearly.

### 2. Weaknesses

#### A. Lack of Rigorous Theoretical Grounding
- **Unsubstantiated Heuristic Foundation:** The foundational premise of WTA-Sign is that the absolute update magnitude of a parameter is a direct proxy for "task confidence." This is a purely heuristic assumption that lacks any formal mathematical proof or optimization-theoretic justification.
- **Scale Sensitivity and Lack of Normalization:** Let $\Delta w_k$ represent the task vector of expert $k$. The absolute magnitude $|T_{k,j}|$ is highly sensitive to training-time hyperparameters. If expert $A$ was trained with a larger learning rate $\eta_A$, or for more epochs $E_A$, or with weaker regularization than expert $B$, its parameter updates will be systematically larger in magnitude ($|T_{A,j}| \gg |T_{B,j}|$ in expectation). Under WTA-Sign, expert $A$ will "win" the coordinate-wise sign election across almost the entire network, completely drowning out the signal from expert $B$ regardless of task-specific "confidence." The paper does not propose or evaluate any task-vector normalization (such as unit $L_2$ normalization or variance-scaling) to address this scale-sensitivity flaw.
- **Hand-Waving Gradient Justification:** The authors' "Gradient-Space Justification" ($\Delta w \propto \sum_t \eta_t \nabla_{w} \mathcal{L}_t$) is a crude, physical-style analogy rather than a rigorous derivation. In modern deep learning, optimizers like Adam use first and second moments to scale gradients ($\Delta w \approx -\eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$). The denominator (second-moment running variance) heavily decouples update magnitude from cumulative raw gradient pressure. Furthermore, in overparameterized regimes, large parameter changes often reflect high local curvature or optimization noise rather than "task confidence."
- **Functional Degradation from Masking:** In Step 3, if an expert's update sign disagrees with the elected sign, it is set to zero by the mask:
  $$M_{k, j} = \mathbb{I}(\text{sign}(T_{k, j}) == s_j)$$
  The paper offers no mathematical analysis, convergence guarantees, or error bounds to quantify the functional approximation error introduced when an expert's task-critical updates are completely discarded. Masking out disagreeing parameters can severely degrade the representations of non-winning experts, and this is completely unaddressed from a theoretical standpoint.

#### B. Catastrophic Empirical Flaws
- **Completely Broken Fine-Tuned Experts:** The foundational premise of model merging is to combine high-performing task experts. However, the fine-tuned experts evaluated in this paper are completely broken, achieving accuracies that are worse than or near random guessing on simple 10-class classification tasks:
  - **MNIST Expert:** **8.69%** (Worse than the 10% random guessing threshold. A standard fine-tuned ViT-B-32 on MNIST should easily exceed 98-99% accuracy).
  - **CIFAR-10 Expert:** **10.16%** (Exactly at the random guessing threshold of 10%. Standard fine-tuning should exceed 90%).
  - **SVHN Expert:** **16.02%** (Extremely poor, where standard fine-tuning exceeds 90%).
  
  This suggests a severe bug in the fine-tuning or evaluation pipelines (such as mismatched class index mappings, incorrect classification heads, or broken data loaders). Evaluating a model merging method on garbage inputs (non-expert models) invalidates any empirical claims made in this paper.
- **The "Winner-Take-All" Illusion:** The paper claims WTA-Sign achieves a "stellar average accuracy" of **14.19%**, outperforming TIES-Merging and Task Arithmetic. However, **14.19% is the exact accuracy of the Pretrained (Zero-shot Base) model** (MNIST 12.70%, SVHN 18.65%, CIFAR-10 11.23%). For scaling factors $\lambda \in \{0.1, 0.2, 0.3\}$, WTA-Sign's accuracies are *identical* to the pretrained zero-shot model. This indicates that WTA-Sign is **not merging any task knowledge at all**; it is simply scaling down or masking out the updates so aggressively that the merged model remains functionally identical to the pre-trained base model ($\lambda = 0$). When $\lambda$ is increased to 0.4 and 0.5 (forcing the model to actually incorporate the expert updates), WTA-Sign's performance immediately drops to 12.92% and 11.07%.
- **Disingenuous Rebranding of Bugs as Features:** The authors attempt to frame this severe failure as a "highly relevant, real-world adversarial negative knowledge regime" and praise WTA-Sign as an "intelligent gatekeeper" for maintaining the zero-shot baseline of 14.19%. This framing is mathematically and scientifically disingenuous. If a practitioner wanted to maintain the zero-shot baseline and ignore downstream experts, they would simply use the pretrained model directly ($\lambda = 0$) rather than applying a model merging script. A legitimate model merging method must show that it successfully integrates functional expert representations to improve downstream multi-task capabilities, which is completely absent here.
- **Extremely Low Zero-Shot Baselines:** A zero-shot average of 14.19% across MNIST, SVHN, and CIFAR-10 is incredibly low for an OpenCLIP ViT-B-32 backbone. This indicates that even the zero-shot evaluation code is bugged or misconfigured (e.g., lack of proper text prompting templates), casting further doubt on the validity of all reported metrics.

---

## Ratings

- **Soundness:** **poor**
  *Justification:* The method relies on an unsubstantiated, scale-sensitive magnitude heuristic with no mathematical proofs, convergence analysis, or error bounds. More critically, the empirical evaluation is completely invalid: the individual "experts" perform worse than random guessing, and the claimed "superiority" of WTA-Sign is an illusion resulting from the method effectively scaling down the updates to zero, thereby merely preserving the pre-trained zero-shot baseline.
  
- **Presentation:** **fair**
  *Justification:* While the algorithm is presented clearly and mathematically explicitly, the overall narrative is heavily compromised by the unscientific framing of a broken experimental setup (failed training of downstream experts) as an "adversarial negative knowledge feature."
  
- **Significance:** **poor**
  *Justification:* Since the method lacks theoretical guarantees and the experiments are conducted on broken models with near-random accuracies, the contribution does not advance our understanding or capabilities in model merging. It has virtually no potential impact on future research or applications in its current state.
  
- **Originality:** **fair**
  *Justification:* The work is an incremental variation of TIES-Merging, replacing majority sign voting with an oligarchic, coordinate-wise maximum magnitude sign selection. It does not introduce any novel theoretical paradigm or optimization-theoretic perspectives.

---

## Overall Recommendation

**Recommendation: 2: Reject**

*Detailed Justification:* 
This paper is not ready for publication due to critical theoretical and empirical shortcomings. 
Theoretically, the proposed "Winner-Take-All" sign election is a heuristic modification of TIES-Merging that suffers from extreme scale sensitivity due to a lack of task-vector normalization. It lacks any mathematical grounding, error bounds on discarded parameters, or optimization-theoretic justifications. 
Empirically, the evaluation is catastrophically flawed. The task "experts" are completely broken, achieving accuracies around or below random guessing on toy datasets. The claimed success of WTA-Sign is merely an artifact of the method failing to apply any meaningful parameter updates for small $\lambda$, thereby preserving the pre-trained model's performance. The framing of this pipeline failure as a "negative knowledge robustness feature" is academically disingenuous. The paper must be rejected, the fine-tuning/evaluation pipeline must be thoroughly debugged, and the method must be grounded in rigorous mathematical theory.

---

## Detailed Feedback & Suggestions for Improvement

1. **Fix the Training and Evaluation Pipeline:** This is the most urgent priority. You must debug your fine-tuning scripts and evaluation heads. A ViT-B-32 fine-tuned on MNIST, SVHN, and CIFAR-10 must achieve accuracies of at least 95%, 90%, and 90% respectively. You must show that WTA-Sign can merge these *successful* experts to achieve high multi-task performance across all tasks simultaneously (e.g., an average accuracy of >85%).
2. **Address Scale Sensitivity:** Introduce and evaluate task-vector normalization techniques. Compare raw task vectors against variance-standardized task vectors and unit $L_2$-normalized task vectors before running the sign election. Show theoretically and empirically how normalization prevents an expert trained with a large learning rate from dominating the merge.
3. **Provide Theoretical Bounds and Guarantees:** 
   - Formally analyze the error introduced by conformity masking. Can you provide a bound on the representation change or loss increase for expert $k$ when a fraction of its parameters are masked to zero?
   - Connect the magnitude heuristic to a rigorous mathematical formulation (e.g., optimization paths, parameter sensitivities, or Bayesian posterior mixtures).
4. **Remove Disingenuous Spin:** Do not attempt to frame a broken experimental pipeline as an "adversarial negative knowledge regime." Ground your paper in standard, honest machine learning practices by showing your method's actual strengths and limitations on properly trained models.
5. **Scale to Real-World Benchmarks:** Standard toy datasets like MNIST are insufficient for modern model merging papers. Please evaluate your method on realistic benchmarks, such as merging instruction-tuned Large Language Models (LLMs) on standard academic benchmarks (MMLU, GSM8K, etc.) or larger multi-task vision benchmarks.
