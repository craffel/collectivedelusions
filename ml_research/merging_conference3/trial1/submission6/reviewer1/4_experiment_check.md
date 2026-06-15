# Experimental Evaluation Check

A critical and rigorous audit of the experimental results in Section 4 and Table 1 reveals catastrophic empirical flaws that undermine the validity of the entire paper.

## 1. Catastrophic Performance of Fine-Tuned "Experts"
The foundational premise of model merging is to combine *high-performing, task-specific expert models* to create a multi-task network. However, the fine-tuned experts in this paper are completely broken:
- **MNIST Expert:** Achieves an accuracy of **8.69%**. For a 10-class classification task (MNIST), random guessing yields 10%. A standard OpenCLIP ViT-B-32 model fine-tuned on MNIST should easily achieve over **98% to 99%** accuracy. The expert in this paper performs *worse than random guessing*.
- **CIFAR-10 Expert:** Achieves an accuracy of **10.16%**. This is exactly at the random guessing threshold (10%). Fine-tuning a ViT-B-32 on CIFAR-10 should yield **90%+** accuracy.
- **SVHN Expert:** Achieves an accuracy of **16.02%**. This is extremely poor for SVHN, where standard fine-tuning yields **90%+** accuracy.

These individual expert accuracies (averaging 11.62%) demonstrate that the fine-tuning process completely failed, or there is a severe bug in the evaluation pipeline (such as mismatched class index mappings, incorrect logit projection heads, or wrong input normalization). Operating on broken, non-expert models invalidates any claims about "expert merging."

## 2. The Illusion of "Winner-Take-All" Superiority
The paper claims that WTA-Sign achieves a "stellar average accuracy" of **14.19%** and outperforms TIES-Merging and Task Arithmetic. However, a closer look reveals that this "superiority" is an absolute illusion:
- The **Pretrained (Zero-shot Base)** model achieves exactly **14.19%** average accuracy (MNIST 12.70%, SVHN 18.65%, CIFAR-10 11.23%).
- WTA-Sign at $\lambda \in \{0.1, 0.2, 0.3\}$ achieves **exactly the same** results: MNIST 12.70%, SVHN 18.65%, CIFAR-10 11.23%, Average 14.19%.
- This perfect equivalence indicates that WTA-Sign at small scaling coefficients is **not incorporating any task knowledge at all**. It is simply scaling down or masking out the updates so aggressively that the model remains functionally identical to the pre-trained base model ($\lambda = 0$).
- When $\lambda$ is increased to 0.4 and 0.5 (which forces the method to actually incorporate the expert updates), WTA-Sign's performance immediately drops to 12.92% and 11.07%.
- In other words, because the fine-tuned experts are broken, their updates act as destructive noise. The only reason WTA-Sign "outperforms" other methods at small $\lambda$ is because it is the most effective at *ignoring* the expert updates and preserving the pre-trained model. Celebrating this as a "merging success" is fundamentally incorrect.

## 3. Disingenuous "Negative Knowledge" Framing
The authors attempt to frame this severe empirical failure as a "highly relevant, real-world adversarial 'negative knowledge' regime" where WTA-Sign acts as an "intelligent gatekeeper" (Section 4.3).
- This is a highly disingenuous framing of a failed experimental setup. Model merging benchmarks are designed to show how well a method integrates *positive* knowledge from successful experts.
- A merging method that merely "ignores updates" is trivial and does not solve the model merging problem. If the goal is to preserve zero-shot performance and ignore the experts, one should simply use the pre-trained base model directly ($\lambda = 0$), which requires zero code and achieves the exact same 14.19% average.

## 4. Extremely Low Zero-Shot Baselines
An average zero-shot accuracy of 14.19% across MNIST, SVHN, and CIFAR-10 is incredibly low for an OpenCLIP ViT-B-32 backbone. Standard zero-shot evaluation of OpenCLIP on these datasets with appropriate text templates (e.g. "a photo of a [class]") yields significantly higher accuracy. This suggests that the zero-shot evaluation pipeline itself is misconfigured or bugged, casting further doubt on the validity of any numbers reported in this submission.
