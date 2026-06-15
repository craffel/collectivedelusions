# 3. Soundness and Methodology

An evaluation of the clarity of description, appropriateness of methods, potential technical flaws, and reproducibility of the paper.

## Clarity of Description
The methodology in Section 3 is **exceptionally clear, well-structured, and easy to follow**. 
- The mathematical formulation is simple and precise. 
- Equations for dynamic parameter blending (Eq 1–3) and the regularized calibration loss (Eq 5) are mathematically standard and use standard notation.
- The description of where routing representations are extracted (Block 11 for the late layers, patch embedding for early layers) is clear.
- The explanation of the failure mechanism of classical routing (representation shift leading to high-variance logits, causing saturated softmax gates and catastrophic discrete switching) is highly intuitive and well-articulated.

## Appropriateness of Methods
The methods chosen are highly appropriate for the goals of the paper:
1. **$L_2$ Weight Decay and Softmax Temperature Scaling**: These are highly standard, computationally trivial, and scientifically sound techniques for bounding logit magnitudes and controlling entropy in gating networks. Using them to regularize the router is a direct and logical way to address overfitting and representation shift on tiny calibration datasets.
2. **Experimental Setup**: The choice of a compact Vision Transformer (ViT-Tiny) and the four-task dataset (MNIST, FashionMNIST, CIFAR-10, SVHN) is highly appropriate. It directly replicates the controlled benchmark environment of Vance et al. (2025) to perform a direct, apples-to-apples deconstruction.
3. **Statistical Evaluation**: Performing a multi-seed sweep over 5 random calibration seeds is appropriate and necessary to establish statistical significance, showing that the results are robust to random few-shot draws.

## Potential Technical Flaws and Limitations
While the methodology is generally sound, several potential limitations and areas of concern should be highlighted:

1. **Failure to Resolve Heterogeneity Collapse**: 
   The paper shows that both classical linear routing and RLR degrade severely under mixed-task heterogeneous serving as the batch size increases (Table 4). At $B=256$, RLR drops to $75.03\%$ accuracy and classical routing drops to $73.15\%$. While RLR maintains a minor accuracy buffer, neither method structurally resolves the core issue. In fact, static supervised OFS-Tune outperforms both by a massive margin at $B=256$ ($86.23\%$ vs. $75.03\%$). This indicates that dynamic model merging via linear gating remains fundamentally limited in heterogeneous settings, and the proposed regularization is only a partial, defensive bandage rather than a robust solution.

2. **Lack of Empirical Scaling to Large Architectures**:
   The authors claim that RLR's "simplicity and classical nature are structurally designed to scale seamlessly to modern high-dimensional architectures like Large Language Models (LLMs)." However, this claim is entirely theoretical. No empirical validation is provided on LLMs or larger models (e.g., LLaMA, Mistral, or ViT-Base/Large). High-dimensional weight spaces and representation spaces in LLMs may introduce unique challenges (such as token-level routing variance or different linear mode connectivity characteristics) that are not captured by a ViT-Tiny on 28x28 or 224x224 image classification tasks.

3. **Overfitting on Tiny Calibration Data**:
   The router is optimized using only 64 total calibration samples (16 per task). While the multi-seed evaluation shows that performance is stable on average, calibrating a continuous router on such an extremely small set makes it highly sensitive to the representational variance of those specific 64 samples. If the calibration samples contain outliers or are slightly biased, the router's generalization under distribution shifts could degrade.

## Reproducibility
The reproducibility of the work is **excellent**:
- The paper uses standard, publicly available datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) and a standard model backbone from the popular `timm` library (`vit_tiny_patch16_224`).
- The optimization parameters (100 steps of Adam with a learning rate of 0.01) and hyperparameter values ($\alpha=0.005$, $T=2.0$) are explicitly specified.
- The mathematical simplicity of the method (768 parameters, standard linear projection, and temperature-scaled softmax) ensures that any deep learning practitioner can easily reproduce the code and results within a few dozen lines of standard PyTorch.
- The use of standard functional calling APIs (`torch.func.functional_call`) is standard and easy to implement.
