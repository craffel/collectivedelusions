# Peer Review: PAC-ZCA

## Section 1: Summary of the Submission
This paper introduces **PAC-ZCA**, a learning-theoretic framework for dynamic model-merging routing on the edge. In multi-tenant environments, serving multiple task-specific Parameter-Efficient Fine-Tuning (PEFT) experts (e.g., LoRA) concurrently is challenging due to the linear latency scaling $O(K)$ of sequential routing and the "heterogeneity collapse" of static weight-space merging. PAC-ZCA resolves these limitations by executing Single-Pass Activation Blending (SPS) to restore a constant $O(1)$ backbone latency, and performing Softmax routing parameterized by task-specific log-temperatures $\mathbf{w} \in \mathbb{R}^K$.

To avoid heuristic grid sweeps, the authors derive a parameter-space PAC-Bayesian bound (Catoni's formulation) over the log-temperatures and directly minimize it using a tiny calibration set. The framework extracts routing coordinates via an unsupervised Subspace Energy Projection (SEP). To satisfy the strict data-independence requirement of McAllester's theorem, the authors introduce a disjoint calibration split partitioning protocol (separating subspace extraction from temperature optimization). 

The authors evaluate PAC-ZCA inside a 14-layer, 192-dimensional analytical Coordinate Sandbox with extreme heteroscedastic noise and on real vision datasets (MNIST, Fashion-MNIST, CIFAR-10) using a pre-trained ResNet-18 backbone. Under Block features, PAC-ZCA achieves **64.16% $\pm$ 2.23%** joint accuracy, outperforming raw-coordinate SABLE (+23.70%) and matching unregularized Empirical Risk Minimization while reducing ensembling variance. They identify that SVD/PCA overfits to noise directions under low-data regimes ($N_c \ll D$) causing train-test scale mismatch, and resolve this by introducing Unit-Norm PCA Subspace Projection (UN-PCA-SEP). On real images, PAC-ZCA achieves **70.87% $\pm$ 2.20%** joint accuracy, outperforming SABLE (65.67%) and unregularized ERM (69.47% $\pm$ 2.21%).

---

## Section 2: Strengths and Weaknesses

### Strengths
1. **Outstanding Empirical and Statistical Rigor:** The experimental design is exceptionally thorough and statistically sound. All results across all tables are evaluated over **5 random seeds**, and the authors systematically report both **means and standard deviations (margins of error)**. This allows for clear evaluation of ensembling stability and variance reduction.
2. **Well-Tuned, Strong Baselines:** The authors compare PAC-ZCA against **eight competitive and representative baselines** (including SABLE, PFSR, QWS-Merge, Linear Router, and Temp-Only ERM), isolating the exact benefits of both task-specific temperature calibration and parameter-space KL complexity regularization.
3. **Comprehensive Ablations and Sensitivity Analysis:** The paper includes excellent, systematic empirical sweeps:
   - *Sample Complexity Sweep:* Evaluating performance across different calibration budgets ($N_c \in \{8, 16, 32, 64, 128\}$), showing that the disjoint split penalty vanishes asymptotically.
   - *Prior Variance Sensitivity Sweep:* Sweeping $\sigma_0^2 \in \{0.1, 0.5, 1.0, 5.0, 10.0\}$ to map out the optimization landscape.
   - *Manifold Structures:* Evaluating under both orthogonal and overlapping representation manifolds.
4. **Scientific Honesty and Post-Mortem Analysis:** The paper demonstrates exceptional intellectual maturity. The authors do not hide the SVD overfitting collapse under uncentered PCA-SEP. Instead, they perform a deep mathematical and empirical post-mortem, diagnose the train-test scale mismatch (SVHN projected norms collapsing from 17.29 to 5.40 at test time), and propose a sound solution (UN-PCA-SEP) which they validate empirically.
5. **Real-World Served Image Experiment:** The vision served experiment using a pre-trained ResNet-18 backbone and real datasets (MNIST, Fashion-MNIST, CIFAR-10) provides complete, rock-solid empirical proof of the framework's practical applicability.
6. **Rigorous Theoretical Foundation:** The authors provide solid, elegant mathematical proofs (Lemma 3.1 establishing the localized Lipschitz bound; Theorem 3.2 proving Lipschitz-Entropy duality; Section 3.5.1 bounding the continuous activation blending gap).

### Weaknesses
1. **Low-Dimensional Task Registries ($K \le 4$):** The experiments are conducted on relatively small registries ($K=4$ in the Sandbox, $K=3$ on real images). Bounding the log-temperature parameters is less critical when $K=4$ (as unregularized ERM performs similarly in mean accuracy), but becomes practically indispensable as $K$ increases because the parameter-to-sample ratio scales up rapidly. Evaluating on a higher-dimensional task registry (e.g., $K \ge 10$) would make the necessity of the PAC-Bayesian complexity penalty even more striking.
2. **Meta-Heuristic Center for the Prior:** The current Gaussian prior is centered at an empirical meta-heuristic scale $\mathbf{w}_0 = \ln(0.05) \cdot \mathbf{1}$. While the authors propose an automated data-free prior based on early-layer representation statistics in Section 5.1.2, they do not implement or test it.
3. **Lack of Natural Language Processing (NLP) Evaluation:** Although modular PEFT serving is highly prominent in LLM serving, the paper only evaluates on visual and synthetic benchmarks. Running experiments on an autoregressive language model (e.g. Llama-3-8B fine-tuned on GLUE tasks) as outlined in the roadmap would significantly elevate the paper's systems-level significance.

---

## Section 3: Detailed Evaluation

### Soundness: Excellent
The submission is technically and empirically flawless. All central claims are backed up by rock-solid empirical evidence and sound mathematical proofs. The authors use appropriate methods, resolve data-dependency flaws via decoupled calibration splits, and demonstrate outstanding scientific honesty in diagnosing the SVD overfitting bottleneck and the "rigor-vs-accuracy" trade-off.

### Presentation: Excellent
The paper is exceptionally well-written, clearly structured, and easy to follow. The mathematical notation is highly precise, all variables are explicitly defined, and the tables/figures are detailed, properly captioned, and informative.

### Significance: Excellent
The paper addresses an important and highly relevant problem in modular deep learning serving on the edge. By restoring constant $O(1)$ backbone latency, resolving heterogeneity collapse, and providing provable, mathematically certified generalization bounds, this work has high practical and theoretical significance, especially for safety-critical edge serving where certified bounds are mandatory.

### Originality: Excellent
Connecting statistical learning theory (PAC-Bayes) to model merging and dynamic routing is highly original and creative. The authors establish a beautiful Lipschitz-entropy duality, derive the first continuous activation blending discrepancy bounds, and develop novel regularized projection protocols (UN-PCA-SEP).

---

## Section 4: Questions and Constructive Suggestions

1. **High-Dimensional Expert Registries:** Have the authors run, or do they plan to run, experiments with a larger number of tasks (e.g. $K \ge 10$)? Showing how the gap between unregularized ERM and PAC-ZCA behaves as the task parameter dimension scales up would be highly valuable and would demonstrate the linear scaling necessity of the PAC-Bayesian bound.
2. **Automated Prior Initialization:** Implementing the proposed automated, data-free prior based on early-layer activation statistics would be a great addition. It would eliminate the Meta-Heuristic center $\mathbf{w}_0 = \ln(0.05) \cdot \mathbf{1}$ and make the framework fully self-contained.
3. **NLP Served Evaluation:** Evaluating PAC-ZCA on standard text benchmarks (such as RoBERTa-Large or Llama-3-8B on the GLUE-LoRA benchmark) would significantly broaden the paper's appeal to the natural language processing community.

---

## Section 5: Overall Recommendation

**Overall Recommendation: 5: Accept**

**Justification:** This is an exceptionally strong, technically sound, and beautifully written paper. From an empirical perspective, the rigor of evaluating over 5 random seeds with standard deviations, comparing against eight comprehensive baselines, and executing thorough sensitivity and sample complexity sweeps is exemplary. The theoretical bridge between PAC-Bayes theory and active model merging parameters is highly creative and original. The authors show outstanding scientific integrity in discussing the "rigor-vs-accuracy" trade-off and unmasking the SVD overfitting collapse. The minor limitations (small $K$ registries, meta-heuristic prior center, and lack of NLP experiments) do not detract from the paper's overall exceptional quality. I strongly recommend accepting this submission.
