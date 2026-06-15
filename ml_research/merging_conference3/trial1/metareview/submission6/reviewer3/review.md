# Peer Review

**Paper Title:** Winner-Take-All Sign Election: A Minimalist Approach to Model Merging

---

## Overall Recommendation

**Rating:** `2: Reject`

**Justification:**
This submission proposes **Winner-Take-All Sign Election (WTA-Sign)**, a minimalist, hyperparameter-free, and training-free method to resolve sign conflicts when merging specialized task vectors. 

While the paper is exceptionally well-written, mathematically clear, and conceptually appealing, it suffers from two critical flaws that prevent its acceptance in its current form:
1. **Severe Experimental/Methodological Flaw:** The task-specific expert checkpoints evaluated in the paper are completely broken, achieving accuracies worse than random guessing (e.g., MNIST expert achieves 8.69% vs. 12.70% base). The authors' attempt to spin this as a "highly relevant, real-world adversarial 'negative knowledge' regime" is unconvincing. In this regime, the expert task vectors contain only corrupting noise, meaning the "optimal" merging strategy is to apply a zero-magnitude update (doing nothing). WTA-Sign's apparent "empirical superiority" at low scaling factors ($\lambda \in \{0.1, 0.2, 0.3\}$) is a trivial artifact of its default to the zero-shot base model performance (14.19%). It does not demonstrate successful knowledge consolidation.
2. **Critical Literature Omission and Overstated Novelty:** The submission completely omits and fails to compare against **MagMax (ECCV 2024)**, which is the foundational "Winner-Take-All" magnitude-based model merging method. This scholarly gap leads the authors to make several false claims of novelty regarding the "magnitude-as-confidence" proxy and "Winner-Take-All" philosophy in model merging, both of which were pioneered by MagMax.

To be suitable for publication, the authors must fix their evaluation pipeline to show performance on high-functioning experts (which should achieve >95% accuracy), re-run the experiments, and properly situates their work relative to MagMax.

---

## Strengths and Weaknesses

### Strengths:
- **Exemplary Presentation and Writing:** The paper is exceptionally clear, logically structured, and professional. The writing is highly engaging and persuasive.
- **Compelling Narrative:** The philosophical appeal to Occam's razor to strip away "heuristic bloat" (such as trimming and rescaling in TIES-Merging) is a refreshing and highly valuable perspective for the community.
- **Implementation Elegance:** The inclusion of a 4-line, fully vectorized PyTorch code snippet is highly commendable and guarantees immediate reproducibility and ease of integration.
- **Rigorous Complexity Analysis:** The detailed time and memory complexity comparison in Appendix B is mathematically sound and clearly demonstrates the computational advantage of avoiding sorting operations.

### Weaknesses:
- **Broken Expert Checkpoints:** The specialized expert models on MNIST (8.69%), SVHN (16.02%), and CIFAR10 (10.16%) perform worse than random chance. This indicates a severe bug in the evaluation pipeline (likely a misconfigured classification head for CLIP).
- **Illusory Empirical Claims:** Because the experts are corrupted, the task vectors represent destructive noise. WTA-Sign's "superiority" is simply because its winner-take-all masking and low scaling factor ($\lambda \le 0.3$) completely mask out the task vectors, preserving the base zero-shot performance (14.19%). Thus, the method's "success" is its ability to do nothing, which has no practical utility in a standard model merging workflow.
- **Omission of MagMax (ECCV 2024):** MagMax is the direct predecessor of WTA magnitude-based merging. Neglecting to cite or compare against MagMax is a major literature oversight that compromises the scholarly integrity of the submission.
- **Lack of Baseline Variety:** Key baselines mentioned in the related work, such as **DARE** and **SyMerge**, are absent from the empirical comparison.
- **No Ablation Studies:** The paper does not isolate the effects of trimming and rescaling. To prove these are "needless complexity," the authors should ablate TIES-Merging without trimming/rescaling, and WTA-Sign with rescaling.

---

## Detail Ratings

### Soundness
**Rating:** `Poor`

**Justification:**
While the mathematical formulation is clean and logical, the empirical validation is technically unsound. Merging models that perform worse than random guessing is a conceptually flawed experimental setup. The "negative knowledge" spin is post-hoc and unscientific. The resulting empirical results do not prove that WTA-Sign can successfully consolidate specialized downstream knowledge. Furthermore, the absence of a comparison to the foundational WTA baseline (MagMax) and the lack of isolated ablation studies undermine the technical soundness of the claims.

### Presentation
**Rating:** `Excellent`

**Justification:**
The paper is outstandingly well-written, well-structured, and easy to follow. The mathematical notation is clean and rigorous, and the transition from the philosophy of Occam's razor to a practical 4-line PyTorch implementation is exceptionally elegant. The appendix provides a highly clear and complete complexity analysis.

### Significance
**Rating:** `Poor`

**Justification:**
In its current state, the practical significance of the work is extremely low. Because the experiments are conducted on broken expert models, there is no evidence that WTA-Sign actually works or provides any utility in a standard, functioning multi-task model merging workflow. If the evaluation pipeline is fixed and the method is shown to outperform or match TIES-Merging on high-performing experts, the significance would be high.

### Originality
**Rating:** `Fair`

**Justification:**
The algorithmic delta from TIES-Merging is highly incremental (replacing a weighted consensus vote with a single-expert argmax-absolute selection, and removing trimming/rescaling). More importantly, the core concept of utilizing magnitude as a proxy for confidence in a winner-take-all model merging framework was already introduced by MagMax (ECCV 2024). The submission fails to acknowledge this, leading to false claims of novelty.

---

## Detailed Constructive Feedback for the Authors

1. **Fix the Evaluation Pipeline:** 
   Please debug your CLIP evaluation code. Fine-tuned CLIP ViT-B-32 experts should achieve high accuracies (typically >95% on MNIST, >90% on SVHN and CIFAR10). The near-random performance indicates that the zero-shot classification head (text embeddings of class names) is likely not being constructed or loaded correctly for the fine-tuned checkpoints, or there is an offset in the label indices.
2. **Re-run the Merging Experiments:**
   Once the experts are achieving high accuracies, re-run all the sweeps. A successful merge should show the merged model achieving high accuracy across all three tasks simultaneously, significantly outperforming the zero-shot base model on those tasks.
3. **Properly Cite and Compare with MagMax:**
   In your Related Work and Experiments sections, you must cite **MagMax (ECCV 2024)**. You should explain the difference between MagMax (which selects the winner's *value*) and WTA-Sign (which elects the winner's *sign* and averages conforming updates). Crucially, you must include MagMax as an empirical baseline in Table 1 to demonstrate that your "conformity averaging" strategy is superior to pure winner-take-all value selection.
4. **Include Missing Baselines:**
   Please add **DARE** to Table 1 to strengthen the empirical comparison.
5. **Add Structured Ablation Studies:**
   To back up your claims that trimming and rescaling are "needless complexity," please provide an ablation table:
   - WTA-Sign (No Trimming, No Rescaling)
   - WTA-Sign + TIES Rescaling
   - TIES-Merging (Trimming, Rescaling)
   - TIES-Merging without Trimming
   - TIES-Merging without Rescaling
   This will scientifically isolate which components contribute to performance or degradation.
6. **Scale Beyond Toy Datasets:**
   Evaluating on MNIST, SVHN, and CIFAR10 is very limited. Please consider evaluating on standard CLIP model merging benchmarks (e.g., ImageNet, Stanford Cars, RESISC45, DTD) or modern LLM merging benchmarks to prove the scalability of WTA-Sign.
