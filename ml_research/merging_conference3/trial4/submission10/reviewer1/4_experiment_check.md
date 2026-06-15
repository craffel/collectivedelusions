# Intermediate Evaluation 4: Experimental Evaluation Check

## 1. Experimental Setup & Datasets
The experimental setup is exceptionally well-conceived and rigorous:
- **Model Backbone**: Using the compact $\mathtt{vit\_tiny\_patch16\_224}$ (5.7M parameters) is an outstanding choice. Many model merging papers evaluate exclusively on highly over-parameterized models (e.g., 7B parameter LLMs) where weight conflicts can be easily absorbed by redundant capacity. Testing on a compact Vision Transformer highlights the severe, real-world issue of representational collapse.
- **Multi-task Benchmark**: The chosen suite of datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) is highly diverse, spanning simple handwritten strokes, clothing features, natural objects, and street-view digits. This creates a challenging test environment with high parameter sign and gradient conflicts.
- **Low-Data Budget**: Restricting the calibration validation set to exactly $16$ samples per task ($64$ total) represents a highly realistic, few-shot, low-compute deployment scenario.

## 2. Sufficiency of Baselines
The paper includes a robust set of modern and competitive baselines:
- **Individual Experts (Ceiling)**: Correctly establishes the empirical performance upper bound.
- **Uniform Merging (Task Arithmetic)**: Represents the standard static merging baseline.
- **AdaMerging**: Represents unsupervised test-time adaptation, which optimizes coefficients at test-time using entropy minimization.
- **OFS-Tune**: Represents supervised static merging coefficient optimization on the calibration set.
- **Linear Router (Classical Baseline)**: A crucial and highly relevant baseline that directly maps inputs to merging coefficients via a classical linear layer. This directly isolates the benefit of the wave phase-interference formulation against a standard neural gating network.

## 3. Results Supporting the Claims
The empirical results provide compelling support for the paper's key claims:

### Claim A: QWS-Merge resolves catastrophic representational collapse on compact backbones
- *Evidence*: Standard Uniform Merging exhibits severe performance degradation, achieving only $49.35\%$ average accuracy. QWS-Merge elevates this to $59.32\%$, an absolute gain of $+9.97\%$. This demonstrates that dynamic weight assembly successfully routes representations through specialized activation pathways on-the-fly, avoiding destructive spatial interference.

### Claim B: Wave-like cosine projections provide robust subspace regularization
- *Evidence*: Under the high-conflict SVHN task, the unconstrained Linear Router collapses completely, achieving only $15.30\%$ accuracy (near-random guess). In stark contrast, QWS-Merge maintains a high performance of $31.60\%$, which preserves $91.5\%$ of the expert ceiling ($34.50\%$). This provides convincing proof that the cosine phase-interference projections act as a highly constrained, bounded subspace, filtering out the optimization noise and preventing the parameter-space collapse that plagues unconstrained classical routing.

### Claim C: Systematic evaluation of batch size and task heterogeneity
- *Evidence*: Table 2 and Figure 2 show that as the test batch size increases from $B=1$ to $B=256$ under mixed-task streams, both dynamic methods suffer from "heterogeneity collapse" due to batch averaging of coefficients. However, QWS-Merge exhibits superior resilience compared to the Linear Router ($48.70\%$ vs. $47.70\%$ at $B=256$). This is a highly honest, scientifically transparent contribution that clarifies the boundaries of dynamic model merging.

## 4. Strengths & Opportunities for Experimental Expansion
### Strengths
- **Converged Experts**: The specialized expert ceiling of $70.52\%$ joint average accuracy indicates that the experts were trained to true convergence, bypassing the "fake expert" issue where poorly-trained models are easily merged.
- **Honest Disclosures**: Rather than hiding the "heterogeneity collapse" at larger batch sizes, the authors transparently present and analyze this phenomenon, which elevates the scientific value of the work.

### Opportunities for Future Exploration
- **Scaling to Large Models**: While evaluating on compact models is the ideal testbed for representational collapse, evaluating on medium-sized backbones (e.g., ViT-Base or Swin-Transformer) or small LLMs (e.g., LLaMA-1B/3B) would demonstrate the scalability of the quantum phase formulation.
- **EMA or Rolling Queues for Single-Sample Inference**: Since batch dependency is a key limitation, implementing and testing an Exponential Moving Average (EMA) or rolling queue of routing coefficients for larger batch sizes under mixed streams would offer a practical engineering solution to the I.I.D. violation.
