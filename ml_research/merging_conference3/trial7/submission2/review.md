# Mock Review

## Paper Title
**Fisher-Information Optimal Subspace Routing (FIOSR): A Provably Stable Parameter-Free Framework for Test-Time Model Merging**

---

## 1. Summary of the Paper
The paper addresses open and critical challenges in **test-time model merging** (dynamic weight ensembling of domain-specialized experts/adapters without retraining). Specifically, it identifies and resolves three severe vulnerabilities in existing dynamic routing paradigms:
1. **The Dynamic Routing Paradox:** Trainable parametric routers easily overfit on microscopic calibration support sets (e.g., $N=64$), collapsing to uniform performance.
2. **Vectorization Collapse:** Trainable routers rely on batch-averaging smoothing and exhibit extreme instability under single-sample sequential streams (batch size $B=1$).
3. **Flat Euclidean Geometrical Misspecification:** Parameter-free projection methods (like standard Cosine similarity) assume flat, isotropic representation and parameter spaces, failing to account for local coordinate sensitivity warped by specialized training.

To solve these, the authors propose **Fisher-Information Optimal Subspace Routing (FIOSR)**:
- It treats the parameter space as a Riemannian manifold, constructing a local metric tensor defined by a smoothed, power-scaled representation-space **diagonal Fisher Information Matrix (dFIM)** estimated over tiny calibration splits ($N_c = 16$ per task). In a Gaussian regime, dFIM corresponds to the inverse coordinate noise variance ($1/\sigma^2$), automatically suppressing noisy dimensions.
- It introduces **Class-Size Scaling Calibration (CSC)** to correct maximum-selection statistical bias across asymmetric task vocabularies.
- It employs **Micro-Batch Homogenization (MBH)** to partition heterogeneous stream batches into homogeneous task-specific micro-batches, preventing **heterogeneity collapse** and safeguarding sequential streams ($B=1$).

The authors evaluate FIOSR in a 192-dimensional synthetic **Analytical Coordinate Sandbox** across 10 random seeds and validate it on a simulated 64-dimensional LoRA activation space. FIOSR achieves **76.86%** homogeneous joint accuracy, outperforming unweighted flat Cosine similarity (**68.30%**) by **+8.56%** absolute accuracy, while learned parametric models collapse to near-uniform accuracy ($\sim 36-39\%$). Under mixed heterogeneous streams, FIOSR + MBH maintains flat-line stability from batch sizes $B=1$ to $512$.

---

## 2. Strengths and Weaknesses

### Strengths
- **Theoretical Rigor and Elegance:** Framing weight ensembling through information geometry and Riemannian manifold theory is highly principled and refreshing. The mathematical linking of dFIM to inverse coordinate noise variance ($F_j \propto 1/\sigma_j^2$) is elegant, and the **Dual-Space Alignment** proof in Appendix A.3 formally grounds classifier weight-to-centroid correspondence.
- **Practical Engineering Relevance:** By identifying and resolving the **Dynamic Routing Paradox** and **Vectorization Collapse**, the paper provides deep, practical guidelines for deploying dynamic adapter ensembling in low-latency sequential streams.
- **Parameter-Free Stability:** Bypassing test-time optimization ensures that the framework has zero trainable parameters, making it completely immune to overfitting and highly sample-efficient ($N_c \ge 8$ per task).
- **Comprehensive Ablations:** The authors meticulously ablated the role of smoothing parameters ($\beta, \gamma$), class-size calibration (CSC), task vs class-conditional variance, rotated/non-axis-aligned noise, and computational gating.
- **Outstanding Presentation:** The paper is exceptionally well-written, with high-quality figures, clean tables, and highly consistent mathematical notation.

### Weaknesses
- **Extreme Reliance on Simulated/Synthetic Environments:** Almost all primary evaluations, stress-tests, and ablations are conducted within the synthetic "Analytical Coordinate Sandbox." The LoRA validation is also based on simulated activation covariance statistics rather than an end-to-end physical execution.
- **Significant Systems Overhead under Heterogeneous Batches:** Micro-Batch Homogenization (MBH) requires up to $G \le K$ separate sequential forward passes with different merged weights. If $G=K$, the computational and memory-bandwidth overhead is equivalent to running all individual experts separately, eliminating the latency advantage of weight-space merging.
- **Vulnerability of Diagonal dFIM to Rotated Noise:** Diagonal Fisher coordinate warping collapses under rotated, non-axis-aligned noise, requiring complex on-the-fly covariance EVD shrinkage estimators that undermine the simple, computationally negligible nature of the diagonal model.

---

## 3. Detailed Assessment of Key Dimensions

### Soundness
**Rating: Good**  
The theoretical proofs and derivations are mathematically sound, and the connection of dFIM to inverse coordinate noise variance is category-error-free. However, two methodological vulnerabilities limit a rating of "Excellent":
1. **Consistency Gaps in Pre-Calibration Mean-Centering:** The dual-space alignment proof in Appendix A.3 assumes a zero global centroid ($\mathbb{E}[z]=0$). While the authors correctly implement the pre-calibration mean-centering step ($z' = z - \bar{z}_{\text{cal}}$) in the primary script `run_experiments.py`, this step is completely omitted in **all eight auxiliary/ablation scripts**. Under uncentered activations typical of physical models, this omission would lead to severe translation bias and degrade alignment in the ablation sweeps.
2. **Isotropic Collapse under Rotated Noise:** Under rotated noise, diagonal dFIM collapses, requiring much more complex block-diagonal (K-FAC) or online shrinkage EVD alignments.

### Presentation
**Rating: Excellent**  
The paper is exceptionally clear, structured, and easy to follow. Figure 1 beautifully conveys stream-batch robustness, and the notation is rigorous. The inclusion of a comprehensive Mathematical Notation table in Appendix A.1 demonstrates an impressive attention to detail.

### Significance
**Rating: Good**  
The paper addresses a highly relevant problem (test-time model merging) and introduces robust, training-free information-geometric tools. However, the lack of end-to-end evaluation on real physical models (e.g., GLUE on LLaMA-3 or ImageNet on ViT) limits its immediate practical significance to developers, as the performance on actual high-dimensional activation spaces remains simulated.

### Originality
**Rating: Excellent**  
The work is highly original. Applying local sensitivity metrics (dFIM) as an on-the-fly coordinate-warping metric tensor for test-time model merging represents a substantial conceptual advancement over flat Euclidean ensembling.

---

## 4. Overall Recommendation
**Score: 4: Weak Accept**  
*Justification:* The paper is a technically solid, mathematically beautiful contribution that advances information-geometric weight ensembling. However, its primary weaknesses—heavy reliance on synthetic coordinate sandboxes, lack of true end-to-end physical model validation on standard PEFT/LLM benchmarks, and the sequential execution latency overhead of MBH—limit its immediate practical impact. If the authors can address these concerns and provide physical evaluations, this paper has Strong Accept potential.

---

## 5. Identify Critical Weaknesses / Flaws (Up to 3)

### Flaw 1: Lack of End-to-End Evaluation on Physical Models and Real Datasets (External Validity Gap)
Almost all quantitative results and stress-tests are evaluated inside the synthetic 192-dimensional sandbox or simulated LoRA activation covariance statistics. No actual vision models (e.g., ViT-Base) or language models (e.g., LLaMA-3-8B) are loaded or run end-to-end on real images or texts. This massive gap between theoretical simulation and physical execution makes it hard to confirm if representation-space dFIM remains robust under complex real-world activation drift, token distribution shifts, and massive multi-layer transformer dynamics.

### Flaw 2: High Latency and Memory-Bandwidth Overhead of Micro-Batch Homogenization (MBH)
Under mixed heterogeneous streams, MBH partitions stream batches into $G \le K$ homogeneous micro-batches. While this prevents heterogeneity collapse, executing up to $G$ separate sequential forward passes on $G$ dynamically merged models introduces substantial systems-level latency and memory bandwidth overhead (due to on-the-fly weight reconstruction and parameter swapping). If all experts are active ($G=K$), dynamic merging is computationally equivalent to running all individual experts separately, completely negating the computational benefits of merging. While Top-1 expert gating ($M=1$) resolves this, it collapses weight-space ensembling back to a hard task selector, representing a fundamental trade-off that is bypassed rather than solved.

### Flaw 3: Susceptibility of Diagonal Fisher to Isotropic Collapse under Rotated/Non-Axis-Aligned Noise
The core diagonal Fisher formulation relies on the coordinate-aligned noise assumption. In Section 4.6, stress-testing under rotated noise causes the diagonal Fisher model (FIOSR-Diag) to collapse below the flat Cosine baseline. Capturing off-diagonal correlations requires the much more complex K-FAC block-diagonal approximation or on-the-fly covariance EVD shrinkage alignment, which undermines the core "computationally negligible" and "simple" claims of the diagonal model.

---

## 6. Actionable Suggestions for Improvement

1. **Conduct End-to-End Evaluation on Physical Models:**  
   The authors MUST evaluate FIOSR on real-world multi-task benchmarks (e.g., GLUE tasks on LLaMA-3-8B LoRA adapters or multi-domain classification on ViT-Base adapters). Measuring end-to-end classification accuracy, routing accuracy, and wall-clock time on actual models is crucial to bridge the external validity gap and justify the mathematical assumptions.
2. **Quantify and Benchmark MBH Systems Overhead:**  
   Provide a detailed systems-level benchmark (using wall-clock latency and GPU memory bandwidth metrics) comparing MBH against:
   - Running all specialized experts separately in parallel.
   - Standard parametric dynamic routing.
   This will help clarify the exact latency-accuracy curve of the MBH gating configurations ($M=1$ vs. $M \ge 2$).
3. **Resolve Codebase Omission of Pre-Calibration Mean-Centering:**  
   To guarantee consistency with the dual-space alignment proof under non-zero global centroids, integrate the pre-calibration mean-centering step ($z' = z - \bar{z}_{\text{cal}}$) into all eight auxiliary/ablation test scripts. Currently, this step is only executed in `run_experiments.py`, leaving the other test scripts vulnerable to translation bias under uncentered representation spaces.
4. **Encapsulate Primary Loop inside `if __name__ == "__main__":` block:**  
   Resolve the significant module-level execution side-effect in `run_experiments.py`. Currently, the entire 10-seed simulation suite is written globally, causing scripts like `test_fiosr.py` (which import `run_experiments` to reuse data-generation and evaluation routines) to globally trigger the full, slow 10-seed experiment before doing their own work.
