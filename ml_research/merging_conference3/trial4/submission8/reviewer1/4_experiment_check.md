# Intermediate Review: 4. Experiment Check

## Critical Evaluation of the Experimental Setup
The experimental setup is standard and well-designed:
- **Architecture:** The authors use a Vision Transformer (\texttt{vit\_tiny\_patch16\_224}) with 5.7M parameters. This is a reasonable lightweight backbone for a proof-of-concept.
- **Datasets:** MNIST, FashionMNIST, CIFAR-10, and SVHN. These represent four diverse 10-class image classification tasks.
- **Calibration Stream:** An extremely small calibration set of $N=64$ unlabeled samples ($B=16$ per task) is used for test-time adaptation, which represents a realistic low-data test-time scenario.
- **Baselines:** The paper compares against a comprehensive set of baselines, including static blending (Uniform TA), unregularized TTA (AdaMerging), smooth spatial regularized TTA (RegCalMerge), quantization-simulating TTA (Q-Merge), unconstrained sharpness-aware TTA (HessMerge), and subspace-constrained TTA (PolyMerge).
- **Quantization Schemas:** Six diverse target precision formats are evaluated, covering FP32, INT8 symmetric/asymmetric tensor/channel-wise, and INT4 symmetric channel-wise formats, which covers a highly thorough range of edge-deployment scenarios.

---

## Do the Results Support the Claims?
While the authors claim that CR-PolySACM is a "unified framework that successfully stabilizes test-time adaptive model composition under downstream post-training quantization," a critical analysis of the empirical results reveals that **the results do not support the practical utility of the proposed CR-PolySACM method**.

### 1. In Practical Settings, the Proposed Method is Detrimental
If we look at any precision format that is actually functional and usable in real-world deployment (FP32 and all four INT8 formats), the proposed **CR-PolySACM consistently degrades performance** compared to the simpler **PolyMerge** baseline:
- **FP32:** PolyMerge (**57.40%**) vs. CR-PolySACM (**57.00%**) [hurt by -0.40%]
- **INT8 Sym (Tensor):** PolyMerge (**57.62%**) vs. CR-PolySACM (**56.62%**) [hurt by -1.00%]
- **INT8 Sym (Channel):** PolyMerge (**58.15%**) vs. CR-PolySACM (**57.23%**) [hurt by -0.92%]
- **INT8 Asym (Tensor):** PolyMerge (**56.57%**) vs. CR-PolySACM (**56.48%**) [hurt by -0.09%]
- **INT8 Asym (Channel):** PolyMerge (**57.43%**) vs. CR-PolySACM (**56.93%**) [hurt by -0.50%]

In other words, in 100% of the cases where the model is actually functioning at a usable accuracy level, a practitioner is far better off using the **simpler, faster, and more elegant PolyMerge baseline** which has no sharpness-aware optimization, no norm-measuring overhead, and no hyperparameters to tune.

### 2. The proposed method only succeeds in a "broken" regime
The only setting where the proposed CR-PolySACM outperforms PolyMerge is the highly aggressive **INT4 Symmetric (Channel)** quantization format, where CR-PolySACM achieves **19.07%** vs. PolyMerge's **18.10%** (a gain of +0.97%).
However, on these 10-class classification tasks:
- Random guessing yields **10.00%** accuracy.
- The average single-task expert performance is **88.67%**.
- An accuracy of **19.07%** is completely non-functional, broken, and practically useless.

The authors are commendable for their scientific honesty in admitting this in the text: "While CR-PolySACM achieves a clear and statistically significant relative improvement (+0.97% over PolyMerge), an absolute joint mean accuracy of 19.07% remains extremely low and practically unusable for production systems... The primary value of our INT4 results is therefore scientific."

However, this scientific defense means that the entire complexity of the proposed CR-SACM framework (norm-scaling, clipping parameters, double forward-backward passes, boundary clamping, etc.) is introduced **solely to achieve a sub-1.0% relative improvement in a regime that is completely non-functional and unusable anyway**. 

### 3. The "Expert-to-Merge Drop" (Domain Disconnect)
The individual experts achieve an average accuracy of **88.67%** (MNIST: 96.30%, FashionMNIST: 86.90%, CIFAR-10: 90.20%, SVHN: 81.30%).
The best merged model in continuous space (PolyMerge) achieves only **57.40%** FP32 accuracy.
This represents a catastrophic performance drop of **-31.27%** compared to the task-specific expert ceilings.
This demonstrates that weight-space post-hoc model merging on highly disjoint, disparate domains (like MNIST and CIFAR-10) is fundamentally limited by a "domain disconnect." While the authors discuss this honestly in Section 4.4, this massive performance gap suggests that the entire post-hoc model merging approach is highly suboptimal compared to simply keeping separate experts or using routing networks, undermining the practical significance of the entire study.

---

## Summary of Empirical Critique
- The proposed CR-PolySACM method is a highly complex, over-engineered wrapper that actually **degrades** model performance in all practical deployment precisions (FP32 and INT8) compared to standard PolyMerge.
- The only setting where CR-PolySACM improves over PolyMerge is in 4-bit (INT4) precision, but the absolute accuracy there is 19.07% (barely above the 10% random guessing baseline), meaning the model is completely broken and unusable anyway.
- The simpler baseline, **PolyMerge**, is a highly elegant, effective, and parameter-efficient method that represents the true, high-value contribution of this paper, but it is obscured by the unjustified complexity of the CR-SACM extension.
