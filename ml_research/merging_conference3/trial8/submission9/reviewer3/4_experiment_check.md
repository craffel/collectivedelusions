# Intermediate Review Evaluation 4: Experimental Evaluation

## Critical Evaluation of the Experimental Setup
The empirical evaluation of the paper is split into a synthetic 192-dimensional sandbox and real-world 512-dimensional ResNet-18 embeddings. While the experiments are extensive, they contain several key design flaws, simplifying assumptions, and contradictions that undermine the strength of the claims:

1. **Highly Artificial and Unrealistic Synthetic Sandbox:**
   The synthetic sandbox is constructed using extreme simplifying assumptions:
   - **Perfect Subspace Orthogonality:** The 192-dimensional space is divided into four strictly orthogonal 48-dimensional subspaces.
   - **Perfect Class Orthogonality:** Within each subspace, the 10 class prototypes are generated via QR decomposition to be strictly mutually orthogonal.
   - **Isotropic Gaussian Noise:** Noise is injected as a spherical Gaussian.
   
   These geometric assumptions (perfect orthogonality and spherical noise) do not exist in real representation manifolds (which exhibit high correlation, topological overlaps, and highly non-isotropic, directional noise). By constructing a sandbox that perfectly satisfies their assumptions, the authors artificially amplify the success of EER while magnifying the "Representational Sparsity Paradox" of centroid-based methods.

2. **The "Overlapping Namespace" Evaluation Flaw:**
   In both synthetic and real-world experiments, all $K=4$ tasks share the exact same class labels $\{0, \dots, 9\}$. Because sample-level accuracy is evaluated via `pred == y`, any incorrect routing can still lead to a "false correct" prediction if the wrong expert predicts the same index.
   The authors admit this introduces a background chance probability of $\approx 10\%$ ($1/C$ under uniform predictions), creating an optimistic bias in absolute accuracy scores across all ensembling models. This is a notable experimental design flaw. The authors could have easily avoided this evaluation bias by using disjoint label spaces (e.g., mapping class labels to $[10k, 10k+9]$ for task $k$). Failing to do so makes the reported absolute accuracies mathematically unreliable and artificially inflated.

3. **Catastrophic Failure of the Main Proposed Paradigm on Real Features:**
   The paper's core contribution is proposing calibration-free dynamic ensembling. Yet, the empirical results on real ResNet-18 embeddings demonstrate a **total collapse** of the proposed unsupervised paradigms:
   - **EER (Direct Routing)** drops to **35.38%** (compared to 71.38% in the sandbox) due to OOD overconfidence (Entropy Calibration Discrepancy).
   - **EPL-OCA Hard** collapses to **27.45%**.
   - **EPL-OCA Soft** collapses to **31.52%**, failing to statistically outperform static **Uniform Weight Merging (31.66 ± 0.91%)**.
   
   Essentially, the experimental section reveals that the proposed calibration-free paradigms (EER, EPL-OCA) are **completely non-viable on real-world features**. The only functional method on real features is **CG-EER (61.50%)**, which is a hybrid semi-supervised method that relies on the exact same offline labeled calibration data as the SOTA SPS-ZCA baseline. This empirical outcome directly contradicts and refutes the main calibration-free claims of the paper.

4. **SVHN Noise Manipulation:**
   The noise scale for SVHN is set to an extremely high value ($0.56$), reducing the expert ceiling of SVHN to a poor $39.44\%$. While the authors claim this is a stress-test for noise rejection, it heavily drags down and distorts the overall 4-task Joint Mean accuracy for all models, making it difficult to assess ensembling performance on standard benchmarks.

5. **Lack of Rigorous Statistical Significance Testing:**
   The paper reports standard deviations over 5 seeds but does not conduct any formal statistical tests (e.g., t-tests or Wilcoxon signed-rank tests) to confirm the significance of the results. For example, on real embeddings, CG-EER achieves **61.50 ± 0.18%** while SPS-ZCA achieves **60.80 ± 0.17%**. While the standard deviations are small, a formal p-value calculation is required to mathematically assert that CG-EER's $+0.70\%$ gain is statistically significant rather than a minor artifact of seed selection.

## Summary Rating: Experiments
- **Experimental Rating: Fair**
- **Justification:** The synthetic sandbox is highly artificial and biased. The overlapping class namespace introduces an optimistic evaluation bias. Crucially, the experiments on real embeddings show that the proposed unsupervised, calibration-free methods are completely non-functional, and the only successful model (CG-EER) relies on offline calibration data, refuting the paper's primary thesis.
