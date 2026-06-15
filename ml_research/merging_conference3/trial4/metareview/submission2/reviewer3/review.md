# Peer Review

## 1. Summary of the Paper
This paper addresses a critical and highly practical bottleneck in deploying multi-task model ensembling on resource-constrained edge hardware. Weight-space model merging (e.g., Task Arithmetic, Model Soups, and AdaMerging) combines multiple specialized task experts into a single backbone with zero computational or memory overhead at inference time. However, to meet edge constraints, these models must undergo post-training quantization (PTQ). 

The paper identifies **Cross-Schema Performance Degradation**: optimizing layer-wise merging coefficients under a single simulated quantization operator (as done by Q-Merge) causes the continuous coefficients to overfit to that operator's specific, discrete rounding grid. Consequently, performance degrades when the merged model is deployed on heterogeneous edge accelerators running mismatched compilers or different PTQ standards (e.g., Symmetric Per-Tensor or asymmetric schemas).

To resolve this, the authors propose **OmniMerge**, a training-free and metadata-free test-time co-optimization framework. It optimizes merging coefficients over a small, unlabeled calibration stream using Shannon prediction entropy and Task-Consensus Regularization (TCR). It incorporates two main techniques:
1. **Stochastic Operator Sampling (SOS):** Stochastically sampling the active simulated quantization operator from a discrete pool of four schemas at each optimization step.
2. **Scale and Zero-Point Noise Perturbation (SZNP):** Injecting Gaussian noise into scale factors and zero-point offsets during the forward pass of coefficient search to smooth the non-differentiable loss landscape.

The authors evaluate OmniMerge on a `ViT-Tiny` backbone across four tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) under robust 8-bit quantization and report that it closes the cross-schema generalization gap, achieving up to 50.78% average accuracy and outperforming standard baselines.

---

## 2. Strengths
- **Pragmatic and Highly Relevant Motivation:** Addressing post-training quantization heterogeneity across edge compilers and accelerators is a crucial, real-world deployment challenge that is often neglected in standard academic merging literature.
- **Zero-Overhead Inference:** The proposed framework is training-free, requires no hardware metadata, and adds absolutely zero inference-time latency, memory, or computational overhead, as the learned coefficients are used to compile standard quantized models.
- **Empirical Accuracy Improvements:** The introduction of scale and zero-point noise perturbation during test-time adaptation successfully drives clear accuracy improvements over the naive and standard Q-Merge baselines across all five target accelerators.

---

## 3. Weaknesses

### 3.1. Lack of Parsimony and Redundant Complexity in the Framework
From a design and implementation standpoint, a method should be as simple and elegant as possible, introducing complexity only when strictly necessary and performatively justified. A critical examination of the paper's ablation study (Table 2) reveals that the full OmniMerge framework violates this principle:
- **Redundant Complexity:** Optimizing with Scale and Zero-Point Noise Perturbation alone under a single static operator (**SZNP Only**) achieves an average accuracy of **50.45%** across all five target schemas.
- **Detrimental Combined Effect:** Combining SZNP with Stochastic Operator Sampling (SOS) in the **Full OmniMerge** framework actually **degrades** average performance to **50.33%**.
- **Implication:** The simpler, more elegant variant of simply adding scale/zero-point noise perturbation during standard optimization is performatively superior to the full, complex co-optimization framework. Constructing, maintaining, and sampling from a discrete pool of multiple simulated operators (SOS) adds substantial implementation complexity without any performance benefit.

### 3.2. Asymmetric Hyperparameter Tuning and Unfair Evaluation
The experimental evaluation suffers from a significant methodological flaw regarding hyperparameter controls:
- Under a highly restricted test-time adaptation budget of only 15 steps, OmniMerge is optimized using a learning rate of $\eta = 2 \times 10^{-2}$.
- In contrast, standard AdaMerging and Q-Merge are restricted to $\eta = 10^{-2}$ because the authors state a larger learning rate causes their optimization to oscillate.
- **Consequence:** Comparing OmniMerge using a learning rate that is **twice as large** as the baselines under a strict, very short 15-step limit means that OmniMerge's performance advantage may be largely an artifact of faster optimization convergence rather than the intrinsic superiority of its learned solution. To isolate the true performance gains, all methods must be compared under symmetric learning rates or run to full convergence (e.g., 50 or 100 steps). The paper lacks this critical control experiment.

### 3.3. Over-Engineered and Toy-Like Benchmark Suite
The choice of datasets and model backbone represents a highly over-engineered and non-elegant setup:
- The authors upsample tiny, low-resolution grayscale images from MNIST ($28 \times 28$), FashionMNIST ($28 \times 28$), and SVHN ($32 \times 32$) to $224 \times 224$ pixels to feed them into a multi-million parameter pre-trained Vision Transformer (`ViT-Tiny`).
- Using a full Vision Transformer for simple digit or clothing classification tasks is computationally wasteful and conceptually inappropriate. A more elegant and realistic setup would use a standard convolutional backbone (e.g., ResNet) on realistic transfer learning or domain adaptation benchmarks (e.g., Office-Home, DomainNet) where a transformer is standard and necessary.

### 3.4. Weak and Incomplete Expert Training
The practical utility of the entire ensembling benchmark is severely constrained by the poor training quality of the "genuine task experts":
- The task experts are fine-tuned on only 256 training images for 3 epochs.
- This leads to extremely weak, under-adapted models. Most notably, **the SVHN expert validation accuracy is restricted to only 28.91%** (which is barely above a random guess of 10% on a 10-class problem).
- Merging models that are barely functional makes the entire ensembling setup highly unrealistic. It is doubtful whether practitioners will adopt a weight-space model merging framework where the underlying specialized models are so poorly trained.

### 3.5. Over-Interpretation of Statistical Noise
The authors claim that 8-bit post-training quantization can act as a beneficial regularizer or "weight denoising" filter because quantized OmniMerge (50.78% under Symmetric Per-Channel) outperforms the unquantized optimized FP16 ceiling of AdaMerging (46.68%). This comparison is highly misleading:
- The unquantized FP16 optimized ceiling of 46.68% is achieved under *AdaMerging* coefficients.
- When the exact continuous coefficients optimized by OmniMerge are evaluated in unquantized FP16, the model achieves **50.39%** accuracy.
- The quantized model under Symmetric Per-Channel achieves **50.78%** accuracy.
- The difference between the quantized model (50.78%) and its unquantized counterpart (50.39%) under the exact same coefficients is exactly **0.39% absolute** (representing an increase of just 4 correct predictions out of 1024 total test images).
- This tiny difference is well within the binomial standard error of the evaluation stream ($\approx 1.56\%$) and is statistically insignificant. Promoting this negligible random fluctuation into a major scientific finding that weight discretization serves as a beneficial noise filter is an over-interpretation of experimental noise.

### 3.6. Mathematical Over-Specification and Lack of Statistical Rigor
- **Mathematical Padding:** Section 4.3 presents a highly verbose sequence of standard, textbook equations for asymmetric, symmetric, and double quantization (Equations 5 through 18). Devoting over half of the methodology section to repeating standard formulas pads the paper with redundant complexity.
- **No Multi-Seed Verification:** Because OmniMerge incorporates two layers of stochasticity (stochastic operator sampling and Gaussian noise perturbation) under a very tight 15-step optimization window, the optimization path is highly stochastic. The authors do not report standard deviations or run multiple random seeds, which is a major omission for validating a highly stochastic test-time adaptation framework.

---

## 4. Questions for the Authors
1. **On Parsimony:** Given that the "SZNP Only" variant achieves better average cross-schema accuracy (50.45%) than the full OmniMerge framework (50.33%), why should practitioners adopt the added complexity of stochastically sampling different operators (SOS)?
2. **On Fair Hyperparameter Controls:** What are the average accuracies of the baselines (AdaMerging and Q-Merge) and OmniMerge when evaluated at full convergence (e.g., 100 steps) or under a symmetric learning rate of $\eta = 10^{-2}$?
3. **On Statistical Insignificance:** Can you provide the results of a multi-run seed evaluation (including standard deviations) for the main results (Table 1) and the ablation study (Table 2)? Is the 0.39% performance difference between quantized and unquantized OmniMerge statistically significant across multiple random seeds?
4. **On Task Expert Convergence:** Why were the task experts not trained to full convergence (especially SVHN, which sits at an extremely poor 28.91% accuracy)? How does OmniMerge perform when merging fully converged, highly accurate task experts?

---

## 5. Ratings and Recommendation

### 5.1. Soundness: **Fair**
*Justification:* The paper introduces an asymmetric hyperparameter tuning comparison (learning rate twice as large for the proposed method under a strict 15-step budget), lacks multi-seed statistical verification for its highly stochastic framework, and over-interprets statistically insignificant performance differences as a "weight denoising" phenomenon.

### 5.2. Presentation: **Fair**
*Justification:* While the overall narrative flow is clear, the methodology contains significant mathematical over-specification, dedicating excessive space to standard, textbook quantization formulas (Equations 5-18). It would be much more elegant and concise to replace these standard definitions with brief references.

### 5.3. Significance: **Fair**
*Justification:* While the problem of cross-schema PTQ mismatch is highly practical, the significance of the empirical validation is severely constrained by the extremely weak and under-trained task experts (especially SVHN at 28.91% validation accuracy) and the over-engineered, toy-like datasets (upsampled MNIST/FashionMNIST on a Vision Transformer).

### 5.4. Originality: **Fair**
*Justification:* The core components—randomized operator sampling (SOS) and scale/zero-point noise perturbation (SZNP)—are highly incremental applications of standard parameter-space data augmentation and classic optimization noise injection. The ablation study shows that the primary driver of performance is simple noise injection, and combining both components in a "co-optimization framework" is performatively redundant.

### 5.5. Overall Recommendation: **3: Weak Reject**
*Justification:* This paper addresses an important, highly practical, and interesting problem in edge-AI deployment with a training-free, zero-inference-overhead approach. However, the proposed "OmniMerge" framework suffers from redundant complexity, as a simpler noise-perturbation variant (SZNP Only) is performatively superior on average while being much simpler. Additionally, the experimental evaluation suffers from asymmetric hyperparameter tuning under strict adaptation limits, extremely under-trained task experts, and over-engineered toy benchmarks. The paper requires a major revision to simplify the proposed method, ensure fair baseline comparisons, and validate the findings on realistic, fully converged model experts.
