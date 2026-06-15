# Idea Proposal: Sanity Checking Layer-wise Merging: Do Learned Coefficients Actually Capture Layer-Specific Task Importances?

## 1. Persona Alignment
This project aligns directly with the core traits and goals of **The Methodologist** persona. Rather than blindly accepting "SOTA" claims from recent papers that propose increasingly complex and parameter-rich test-time adaptation (TTA) algorithms, this work takes a highly skeptical, critical look at the foundational assumptions of layer-wise model merging. Specifically, we sanity-check the widely accepted assumption that learned layer-wise, task-wise coefficients ($\lambda^l_k$) capture fine-grained, layer-specific representational contributions across tasks. By proposing rigorous, fair, and reproducible evaluation baselines—such as layer-wise shuffling, task-wise averaging, and norm-bounded noise perturbations—we seek to expose whether layer-specificity is a functional reality or a methodological illusion. This prioritizes rigorous scientific understanding, baselines tuning, and reproducibility over flashing new architectures.

## 2. Core Techniques
The core of our proposed research is a **Rigorous Sanity-Checking and Interpretability Suite** designed to stress-test layer-wise model merging. We introduce and evaluate four main diagnostic treatments:
1. **Intra-Task Layer Shuffling (Shuffle Treatment):** Shuffling the learned coefficients $\lambda^l_k$ across the layer dimension for each task to evaluate whether the exact layer-wise assignment is critical.
2. **Task-Wise Spatial Averaging (Mean Treatment):** Replacing the layer-wise coefficients with their average across all layers for that task to evaluate if a single task-wise scalar is sufficient.
3. **Task-Wise Norm-Bounded Perturbation (Noise Treatment):** Injecting relative Gaussian noise into the learned coefficients to analyze the sensitivity and flatness of the merging optimization landscape.
4. **Centered Kernel Alignment (CKA) Representation Analysis:** Computing CKA similarity index (Kornblith et al., 2019) between the expert backbones and the merged backbone across different layers to analyze if the optimized layer-wise coefficients correlate with physical feature alignments.

We build upon and evaluate these treatments on the following foundational algorithms:
- **AdaMerging** (Yang et al., 2024b): Layer-wise test-time adaptive model merging via entropy minimization.
- **SyMerge** (Jung et al., 2025): Test-time joint optimization of coefficients and task-specific adapters via self-labeling.
- **SAIM** (Under review, ICLR 2026): Sharpness-aware isotropic merging with singular value spectrum balancing.

## 3. Mathematical Formulation
Let there be $K$ tasks and a pre-trained model with $L$ layers. For each layer $l \in \{1, \dots, L\}$ and task $k \in \{1, \dots, K\}$, let $\tau^l_k = \theta^l_k - \theta^l_{\text{pre}}$ be the task vector representing the fine-tuned adaptation. The standard layer-wise model merging weight is constructed as:
$$\theta^l_{\text{merged}} = \theta^l_{\text{pre}} + \sum_{k=1}^K \lambda^l_k \tau^l_k$$
where $\Lambda = \{\lambda^l_k\}_{l, k}$ are the optimized merging coefficients.

We formulate three rigorous diagnostic treatments to test $\Lambda$:

### Diagnostic Treatment 1: Intra-Task Layer Shuffling
For each task $k$, let $\pi_k: \{1, \dots, L\} \to \{1, \dots, L\}$ be a random permutation of layer indices. The shuffled weight is:
$$\theta^l_{\text{shuffle}} = \theta^l_{\text{pre}} + \sum_{k=1}^K \lambda^{\pi_k(l)}_k \tau^l_k$$

### Diagnostic Treatment 2: Task-Wise Spatial Averaging
For each task $k$, we compute the average coefficient across all layers:
$$\bar{\lambda}_k = \frac{1}{L} \sum_{l=1}^L \lambda^l_k$$
The spatially averaged weight is:
$$\theta^l_{\text{mean}} = \theta^l_{\text{pre}} + \sum_{k=1}^K \bar{\lambda}_k \tau^l_k$$

### Diagnostic Treatment 3: Norm-Bounded Perturbation
Let $\epsilon^l_k \sim \mathcal{N}(0, \sigma^2)$ be Gaussian noise. We construct perturbed coefficients:
$$\hat{\lambda}^l_k = \lambda^l_k + \epsilon^l_k \quad \text{s.t.} \quad \|\epsilon_k\|_2 \le \gamma \|\lambda_k\|_2$$
where $\gamma > 0$ is a scale hyperparameter. The perturbed weight is:
$$\theta^l_{\text{noise}} = \theta^l_{\text{pre}} + \sum_{k=1}^K \hat{\lambda}^l_k \tau^l_k$$

### CKA Representational Alignment
For a given layer $l$, let $X^l_k \in \mathbb{R}^{N \times D}$ and $X^l_{\text{merged}} \in \mathbb{R}^{N \times D}$ be the hidden activations of task $k$'s expert and the merged backbone respectively on a batch of $N$ inputs. Let $K_1 = X^l_k (X^l_k)^\top$ and $K_2 = X^l_{\text{merged}} (X^l_{\text{merged}})^\top$ be their gram matrices. We define:
$$\text{CKA}(K_1, K_2) = \frac{\text{HSIC}(K_1, K_2)}{\sqrt{\text{HSIC}(K_1, K_1) \text{HSIC}(K_2, K_2)}}$$
where $\text{HSIC}$ is the Hilbert-Schmidt Independence Criterion. We evaluate the Pearson correlation coefficient between $\{\lambda^l_k\}_{l=1}^L$ and $\{\text{CKA}(X^l_k, X^l_{\text{merged}})\}_{l=1}^L$.

## 4. Architecture Specifications
- **Backbone Architectures:** CLIP ViT-B/32 (12 transformer layers, hidden dimension $D = 768$) and CLIP ViT-L/14 (24 transformer layers, hidden dimension $D = 1024$).
- **Inputs:** Batches of size $N = 32$ images of shape $224 \times 224 \times 3$ mapped to task-specific representations.
- **Tuned Parameters Tracked:**
  - Task Arithmetic: $K$ global scalar parameters.
  - AdaMerging/SAIM: $K \times L$ scalar parameters (e.g., $8 \times 12 = 96$ for ViT-B/32 on 8 tasks).
  - SyMerge: $K \times L$ scalar parameters plus joint classification layers ($K \times C \times D$ parameters, where $C$ is the number of classes). Our framework tracks the performance decay of both coefficient-only adaptation and joint adapter/classifier adaptation under all treatments to evaluate test-set overfitting.

## 5. Baselines
We evaluate our sanity-checking treatments on the following prominent baselines:
- **Baseline 1: Task Arithmetic (TA) (Ilharco et al., 2023):** Traditional model merging using a single global scalar coefficient per task across all layers. This is the simplest baseline and represents spatial uniformity.
- **Baseline 2: AdaMerging (Yang et al., 2024b):** The standard layer-wise test-time adaptive merging method.
- **Baseline 3: SyMerge (Jung et al., 2025):** The SOTA test-time adaptation method that jointly adapts merging coefficients and classification heads using self-labeling. This baseline will be crucial to demonstrate that tuning classifiers on test data overfits significantly compared to our spatial averaging.
- **Baseline 4: SAIM (ICLR 2026):** SOTA sharpness-aware isotropic model merging. We evaluate if SAIM's SVD balancing is robust to our shuffling and averaging treatments.

## 6. Step-by-Step Interaction
1. **Fine-Tuning:** Take the pre-trained CLIP backbone and fine-tune it independently on $K = 8$ tasks (SUN397, Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, DTD) to obtain 8 task experts and task vectors $\tau_k$.
2. **Coefficient Optimization:** Run AdaMerging/SyMerge/SAIM on the unlabeled target datasets to learn optimized layer-wise merging coefficients $\Lambda = \{\lambda^l_k\}$.
3. **Sanity Testing Phase:**
   - **Step 3a:** Run standard test-set evaluation using the optimized coefficients $\Lambda$ to establish the original optimized "SOTA" accuracy.
   - **Step 3b (Shuffle):** Permute the optimized coefficients of each task across the 12/24 layers. Build the merged model $\theta_{\text{shuffle}}$, run inference, and record task accuracies.
   - **Step 3c (Mean):** Average the optimized coefficients of each task across all layers. Build the merged model $\theta_{\text{mean}}$, run inference, and record task accuracies.
   - **Step 3d (Noise):** Inject relative norm-bounded noise to the optimized coefficients at varying scale factors $\gamma \in [0.05, 0.5]$. Build the merged model $\theta_{\text{noise}}$, run inference, and record accuracies.
   - **Step 3e (CKA Representational Analysis):** Pass test batches through the individual task experts and the merged models under all treatments, compute CKA values across all layers, and correlate these CKA indices with the optimized coefficients.
4. **Synthesis:** Collate and tabulate performance drops under all treatments. If the shuffling and spatial mean treatments result in negligible accuracy decay, we empirically demonstrate that the layer-specificity of model merging is a methodological illusion, and that a much simpler, lower-parameter task-wise scaling baseline is sufficient and more robust.
