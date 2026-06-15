# Peer Review of SuiteMerge

## 1. Summary of the Submission
The paper presents **SuiteMerge**, a systematic and independent methodological audit of modern adaptive model-merging paradigms. It exposes a severe and previously unreported **"Task Suite Bias"** in the literature, where standard adaptive model-merging algorithms (e.g., Online AdaMerging and Online PolyMerge) are validated on exactly one combination of four simple classification tasks (MNIST, FashionMNIST, CIFAR-10, and SVHN), which masks critical local failures under different task relationships. 

By partitioning these datasets into five distinct multi-task evaluation suites of varying domain distances and representational conflict, the paper demonstrates that:
1. Online Test-Time Adaptation (TTA) methods suffer from transductive overfitting to stream-level noise in high-conflict regimes.
2. Unconstrained online TTA requires oracle task labels at test time (the "privilege trap") to direct gradients correctly; otherwise, joint unsupervised prediction entropy minimization causes severe representation collapse.
3. A simple, regularized offline alternative—**Offline Few-Shot Validation Tuning (OFS-Tune)** using continuous low-degree polynomial trajectory constraints ($d=1$ and $d=2$)—consistently matches or exceeds online TTA methods in simulation, while completely bypassing test-time compute, backpropagation latency, and the privilege trap.
4. Independent physical weight-space validation on a custom DeepCNN confirms these predictions, where OFS-Tune outperforms online PolyMerge by up to $3.70\%$ and online AdaMerging by up to $4.20\%$, acting as a highly robust, "safe-by-default" paradigm.

---

## 2. Strengths and Weaknesses

### Strengths
- **Immense Practical Relevance and Actionability:** 
  The paper directly addresses critical engineering constraints that dictate real-world deployment viability. In practical production settings, online TTA is a massive liability due to (a) the high compute/energy overhead of executing backpropagation on live edge-devices, (b) the "privilege trap" of needing oracle task labels to route gradients, and (c) the risk of catastrophic representation collapse on rugged unsupervised entropy surfaces. OFS-Tune resolves all three bottlenecks by shifting calibration to a highly regularized offline pre-deployment phase using a tiny calibration set ($M=10$ samples), making it exceptionally attractive for practical deployment.
- **Exceptional Methodological Rigor:**
  The evaluation is remarkably thorough, reporting average accuracies and standard deviations across **30 independent random seeds** and five evaluation suites. The authors include multiple crucial baselines (Uniform, unconstrained AdaMerging, constrained PolyMerge, offline unconstrained validation tuning, and temporal smoothing parameter EMA) to isolate the exact sources of generalizability.
- **Intellectual Honesty and Thorough Ablations:**
  The authors proactively address and neutralize potential critiques of their work:
  - *Optimization Asymmetry:* They deconstruct this empirically by standardizing Adam and L-BFGS-B across online/offline setting, proving that OFS-Tune's gains are driven by structural constraints rather than optimizer choice.
  - *Simulator Circularity:* They test their framework under a highly challenging "non-smooth zig-zag" optimal profile, demonstrating that localized Piecewise Splines and Block-wise Parameter Sharing can natively capture sensitivity spikes while filtering noise.
  - *Temporal Smoothing:* They implement Parameter EMA on the online baselines to prove that temporal smoothing cannot rescue TTA from the misalignment of unsupervised entropy.
- **High Presentation Quality and Structured Scaling Roadmap:**
  The manuscript is exceptionally well-structured, easy to follow, and features clean, informative figures and tables. Section 5 provides three concrete, easily implementable engineering strategies (representative validation subsets, coordinate gradient backpropagation via OFS-Adam, and CPU expert parameter offloading) using standard libraries, making the transition to large foundation models highly actionable.

### Weaknesses
- **Toy-Scale Physical Weight-Space Validation:**
  While the calibrated Model II simulation study is highly rigorous, the physical weight-space validation is restricted to a custom 5-layer Convolutional Neural Network (CNN) trained on CPU for simple MNIST and FashionMNIST subsets. This is a toy-scale model on highly saturated datasets. Standard model merging in production is applied to massive architectures (e.g., 86M ViT, 7B LLMs, Stable Diffusion). As the model width, depth, and parameter space scale up, the nature of weight-space connectivity, representational clashing, and layer-wise sensitivity changes. It remains unproven whether the exact numerical advantages (e.g., OFS-Tune outperforming PolyMerge by 3.70% and AdaMerging by 4.20%) hold in physical weight-space model merging on massive, high-dimensional parameter spaces. 
- **Missed Opportunity on Empirical ViT weights:**
  In Section 3.2, the authors describe empirically calibrating their Model II sensitivity landscape by perturbing layer-wise coefficients of a 12-layer Vision Transformer (ViT-B/32) backbone. Given that fine-tuned ViT-B/32 weights on MNIST, FashionMNIST, CIFAR-10, and SVHN were already generated and evaluated to perform this calibration regression, it is a missed opportunity not to run physical weight-space merging experiments directly on these ViT-B/32 models. Doing so would have closed the "toy-scale" physical gap and provided a highly convincing medium-scale validation.
- **Practical Computational Cost of Nelder-Mead Solver:**
  The proposed OFS-Tune uses the Nelder-Mead simplex algorithm to search the low-dimensional polynomial parameter space offline. While Nelder-Mead is exceptionally fast when evaluating the validation loss takes under a millisecond, it requires up to 200 function evaluations. For a massive foundation model (like a 7B LLM), running 200 forward evaluations of the model on 100 validation samples would introduce a major, expensive compute bottleneck during the pre-deployment phase. Although the authors propose coordinate gradients (OFS-Adam) and expert offloading as future directions to address this scaling challenge, these strategies are not empirically validated in the paper.
- **No Physical Proof for Localized Trajectory constraints:**
  The paper introduces Piecewise Linear Splines and Block-wise Parameter Sharing to capture localized sensitivity spikes in Transformer architectures. However, these localized formulations are only evaluated within the simulated Model II landscape (Section 4.3). A physical evaluation of these localized formulations on an actual pre-trained Transformer backbone is missing.

---

## 3. Detailed Feedback and Questions for the Authors
1. **Physical ViT Weight-Space Merging:** 
   Since the authors already fine-tuned the 12-layer Vision Transformer (ViT-B/32) backbones to calibrate the Model II landscape parameters ($A_k^{(l)}$ and $B_k^{(l)}$), why did they not run physical weight-space merging experiments on these ViT backbones directly? Evaluating physical merging on ViT-B/32 on CIFAR-10 / SVHN (Suite B) would completely validate the findings on a realistic, medium-scale architecture. Are there plans to include this physical ViT validation?
2. **Computational Scaling of Nelder-Mead:**
   How long does the 200-evaluation Nelder-Mead optimization take in practice on physical networks? When transitioning to large-scale LLMs (e.g., LLaMA-3), running 200 full forward-pass evaluations across multiple tasks is computationally expensive. Have the authors empirically tested the proposed first-order OFS-Adam coordinate gradient descent (Eq. 12) on physical networks, and does it converge as robustly as Nelder-Mead with a restricted step budget?
3. **Physical Validation of Block-wise Parameter Sharing:**
   Do the authors plan to validate the proposed Piecewise Splines and Block-wise Parameter Sharing (which group attention and MLP layers separately) in physical weight-space experiments on Transformers? This would empirically confirm whether localized parameterizations can capture the distinct block-level sensitivity spikes of physical attention projections.
4. **Stratified Sampling Variance:**
   In highly imbalanced or extremely noisy few-shot scenarios, does the stratified sampling budget of $M=10$ samples per task introduce any risk of selection bias where the learned offline trajectory fails to generalize to a highly diverse test stream? Have the authors swept different validation sample sizes (e.g., $M \in [5, 50]$) to analyze the sample-size sensitivity of OFS-Tune?

---

## 4. Ratings
- **Soundness:** **Excellent**
  The methodology is mathematically grounded, calibrated against empirical statistics of a 12-layer ViT, and verified across 30 seeds. The deconstruction of optimization budgets and standardizing optimization algorithms across frameworks represents an exceptional level of scientific rigor.
- **Presentation:** **Excellent**
  The paper is beautifully structured and incredibly transparent about its assumptions, limitations, and potential critiques. Table 2 provides a highly practical summary of architectural/deployment assumptions, and the appendix is exceptionally detailed.
- **Significance:** **Good**
  The paper provides a vital and highly practical "reality check" for the model-merging community. It highlights that online TTA methods—which appear elegant in academic simulations—are highly fragile and computationally expensive in production, and offers a robust, zero-test-time-compute, "safe-by-default" alternative that is highly appealing to industry practitioners.
- **Originality:** **Good**
  While the individual concepts (offline calibration and polynomial constraints) are known, the systematic audit, the formulation of "transductive overfitting" to correlated stream noise, exposing the "privilege trap" of online TTA routing, and the deconstruction of simulator circularity represent highly original and valuable contributions.

---

## 5. Overall Recommendation
- **Overall Recommendation:** **5: Accept**
  This is a technically solid, exceptionally rigorous, and outstandingly well-written paper that addresses a critical deployment dilemma in model merging. It exposes a major confounding variable ("Task Suite Bias") and provides a highly stable, zero-test-time-compute alternative to fragile online adaptation. While the physical weight-space validation is conducted on a small-scale CNN, the overall methodological rigor, extensive ablations, and clear engineering roadmap for foundation models make this a highly valuable and high-impact paper for both researchers and practitioners in the machine learning community.
