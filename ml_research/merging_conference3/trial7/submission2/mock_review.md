# Mock Review

## Paper Title
**Information-Geometric Subspace Routing: A Provably Stable Parameter-Free Framework for Test-Time Model Merging**

---

## 1. Summary of the Paper
The paper addresses open and critical challenges in **test-time model merging** (dynamic weight ensembling of domain-specialized experts/adapters without retraining). Specifically, it identifies and resolves three severe vulnerabilities in existing dynamic routing paradigms:
1. **The Dynamic Routing Paradox:** Trainable parametric routers easily overfit on microscopic calibration support sets (e.g., $N=64$ samples), collapsing to uniform performance.
2. **Vectorization Collapse:** Trainable routers rely on batch-averaging smoothing and exhibit extreme instability under single-sample sequential streams ($B=1$).
3. **Flat Euclidean Geometrical Misspecification:** Parameter-free projection methods (like standard Cosine similarity) assume flat, isotropic representation and parameter spaces, failing to account for local coordinate sensitivity warped by specialized training.

To solve these, the authors propose **Fisher-Information Optimal Subspace Routing (FIOSR)**:
- It treats the parameter space as a Riemannian manifold, constructing a local metric tensor defined by a smoothed, power-scaled representation-space **diagonal Fisher Information Matrix (dFIM)** estimated over tiny calibration splits ($N_c = 16$ per task). In a Gaussian regime, dFIM corresponds to the inverse coordinate noise variance ($1/\sigma^2$), automatically suppressing noisy dimensions.
- It introduces **Class-Size Scaling Calibration (CSC)** to correct maximum-selection statistical bias across asymmetric task vocabularies.
- It employs **Micro-Batch Homogenization (MBH)** to partition heterogeneous stream batches into homogeneous task-specific micro-batches, preventing **heterogeneity collapse** and safeguarding sequential streams ($B=1$).

The authors evaluate FIOSR in a 192-dimensional synthetic **Analytical Coordinate Sandbox** across 10 random seeds, validate it on a simulated 64-dimensional LoRA activation space, and—crucially—evaluate it in an end-to-end physical deployment setting using a pre-trained **ResNet-18 backbone** on real image datasets (MNIST, FashionMNIST, and SVHN). Under mixed heterogeneous streams, FIOSR + MBH maintains flat-line stability from batch sizes $B=1$ to $512$, outperforming complex parametric ensembling baselines by up to $40.7\%$.

---

## 2. Strengths and Weaknesses

### Strengths
- **Theoretical Rigor and Elegance:** Framing weight ensembling through information geometry and Riemannian manifold theory is highly principled and refreshing. The mathematical linking of dFIM to inverse coordinate noise variance ($F_j \propto 1/\sigma_j^2$) is elegant, and the **Dual-Space Alignment** proof in Appendix A.3 formally grounds classifier weight-to-centroid correspondence.
- **Decisive End-to-End Physical Validation:** The addition of Section 4.8 (physical ResNet-18 validation on MNIST, FashionMNIST, and SVHN) is a major strength that directly bridges the external validity gap. It proves that FIOSR can be deployed on physical feature activations and handle dead or near-zero variance dimensions (common in ReLU-activated layers) using scale-regularization.
- **Practical Engineering Relevance:** By identifying and resolving the **Dynamic Routing Paradox** and **Vectorization Collapse**, the paper provides deep, practical guidelines for deploying dynamic adapter ensembling in low-latency sequential streams.
- **Parameter-Free Stability:** Bypassing test-time optimization ensures that the framework has zero trainable parameters, making it completely immune to overfitting and highly sample-efficient ($N_c \ge 8$ per task).
- **Comprehensive Ablations:** The authors meticulously ablated the role of smoothing parameters ($\beta, \gamma$), class-size calibration (CSC), task vs class-conditional variance, rotated/non-axis-aligned noise, and computational gating.
- **Outstanding Software Hygiene:** A rigorous audit of the code reveals exceptional engineering standards. The pre-calibration mean-centering step ($z' = z - \bar{z}_{\text{cal}}$) is consistently implemented across the entire codebase—including both the primary script `run_experiments.py` and all auxiliary/ablation scripts (such as `test_csc_ablation.py`, `test_rotated_noise.py`, `test_fiosr.py`, etc.). Furthermore, the main execution suite in `run_experiments.py` is safely encapsulated within a proper `if __name__ == "__main__":` guard block, enabling clean modular importing without execution side-effects.

### Weaknesses (Minor Areas for Improvement)
- **Significant Systems Overhead under Heterogeneous Batches:** Micro-Batch Homogenization (MBH) requires up to $G \le K$ separate sequential forward passes with different merged weights. If $G=K$, the computational and memory-bandwidth overhead is equivalent to running all individual experts separately, eliminating the latency advantage of weight-space merging.
- **Vocabulary Scaling for Large Language Models (LLMs):** While the authors propose "Class-Grouped Pooling" and "Low-Rank FIM Factorization" in Section 4.5, these strategies are merely discussed and not empirically validated. Autoregressive LLMs with massive vocabularies ($C \approx 32\text{K}$ to $128\text{K}$) and high representation dimensions present substantial storage and memory challenges for storing $K \times C \times d$ Fisher coefficients.

---

## 3. Detailed Assessment of Key Dimensions

### Soundness
**Rating: Excellent**  
The theoretical proofs and derivations are mathematically sound, and the connection of dFIM to inverse coordinate noise variance is category-error-free. The potential conceptual gap of applying a representation-derived metric tensor to warp classifier parameters is formally resolved via the Dual-Space Alignment proof in Appendix A.3. Additionally, the software engineering implementation is highly robust, featuring consistent pre-calibration mean-centering across all test files and clean module encapsulation.

### Presentation
**Rating: Excellent**  
The paper is exceptionally clear, structured, and easy to follow. Figure 1 beautifully conveys stream-batch robustness, and the notation is rigorous. The inclusion of a comprehensive Mathematical Notation table in Appendix A.1 demonstrates an impressive attention to detail.

### Significance
**Rating: Excellent**  
The paper addresses a highly relevant problem (test-time model merging) and introduces robust, training-free information-geometric tools. The successful end-to-end evaluation on real physical models (ResNet-18) on MNIST, FashionMNIST, and SVHN benchmarks decisively resolves the external validity gap, demonstrating high practical utility for developers deploying dynamic ensembling systems.

### Originality
**Rating: Excellent**  
The work is highly original. Applying local sensitivity metrics (dFIM) as an on-the-fly coordinate-warping metric tensor for test-time model merging represents a substantial conceptual advancement over flat Euclidean ensembling.

---

## 4. Overall Recommendation
**Score: 5: Accept**  
*Justification:* This is an exceptionally polished, mathematically beautiful, and empirically rigorous paper. The authors have successfully resolved previous critiques regarding "synthetic over-reliance" by adding Section 4.7 (realistic LoRA activation space validation) and Section 4.8 (end-to-end physical validation on a pre-trained ResNet-18). The theoretical framework is sound, the presentation is excellent, and the codebase exhibits outstanding engineering standards.

---

## 5. Minor Suggestions & Questions for the Authors

1. **Empirical Validation of LLM Vocabulary Compression:**  
   In Section 4.5, the authors discuss "Class-Grouped Pooling" and "Low-Rank FIM Factorization" to scale the storage of $K \times C \times d$ Fisher coefficients to massive vocabularies. Providing a small, proof-of-concept empirical validation (e.g., using a pre-trained GPT-2 model or a tiny LLaMA model) of these compression techniques would significantly strengthen the paper's appeal to LLM practitioners.
2. **Benchmarking MBH Wall-Clock Latency and Swapping Cost:**  
   Provide a detailed systems-level benchmark (measuring wall-clock latency, parameter swapping cost, and GPU memory bandwidth) comparing MBH against:
   - Running all specialized experts separately in parallel.
   - Standard parametric dynamic routing.
   This will help clarify the exact latency-accuracy curve of the MBH gating configurations ($M=1$ vs. $M \ge 2$) in a practical execution environment.
3. **Exploration of Rotation-Invariant Local Information Metrics:**  
   Since diagonal Fisher is highly sensitive to rotated, non-axis-aligned noise (requiring complex EVD shrinkage covariance online estimation as validated in Section 4.6), can the authors comment on whether a simplified, rotation-invariant local metric could be derived? Exploring coordinate-independent scaling layers or low-rank approximation techniques for the FIM could preserve the "parameter-free" simplicity under arbitrary noise rotations.
