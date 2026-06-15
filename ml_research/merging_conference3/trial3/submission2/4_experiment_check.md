# Experimental Check and Evaluation

## 1. Completeness and Rigor of Evaluation
The experimental evaluation in this paper is exceptionally thorough, comprehensive, and statistically rigorous. Rather than relying on a single lucky configuration, the authors conduct extensive multi-seed evaluations across both simulated and physical deep learning environments.

Key aspects of the evaluation's rigor are detailed below:

### A. Robustness of the Evaluation Protocol
- **High Statistical Significance:** 30 independent random seeds (42 to 71 inclusive) are swept for the simulation, and 5 independent random seeds (42 to 46 inclusive) are swept for the physical neural network validation. This is far more rigorous than most model-merging publications, which typically report results from a single seed.
- **Sterile Replication Controls:** The authors include a "sterile" noiseless control (noise scale = 0.0, cosine weight = 0.0) where they successfully replicate the published SOTA online TTA claims (AdaMerging: 87.81%, RegCalMerge: 87.32% vs Uniform: 84.44%). This ensures that their critique is not due to a broken baseline, but rather because they evaluate under realistic deployment conditions.
- **Hyperparameter Sweeps:** The authors performed extensive sweeps for the online TTA baselines (Adam learning rates, spatial elastic regularization strength $\lambda$), ensuring that their competitors are optimally tuned.

---

## 2. Evaluation of Baselines
The selected baselines are highly appropriate and cover all critical paradigms:
1. **Model-Merging Baselines:** Task Arithmetic (Uniform) represents the standard, static, unoptimized model-merging method.
2. **Online Test-Time Adaptation SOTA:** Online AdaMerging (layer-wise), Online RegCalMerge, and Online PolyMerge ($d=2$).
3. **Supervised Few-Shot Baselines (Physical CNN):** 
   - **Few-Shot Joint Fine-Tuning (FT-Val):** Optimizing all 100,000+ weights on the validation set.
   - **Few-Shot Head Tuning (Head-Val):** Optimizing only the linear projection head (1,290 weights).
   This comparison provides vital architectural and practical insights, showing how weight-space coefficient tuning is superior to standard post-hoc calibration under scarce, noisy validation data.

---

## 3. Multi-Dimensional Sensitivity Sweeps
The paper contains an impressive suite of sensitivity analyses, including:
- **Sample Complexity Sweeps:** Sweeping the validation sample size $M \in \{5, 10, 20, 50\}$ per task to map the boundaries of the Overfitting-Optimizer Paradox (visualized in Figure 3).
- **Adversarial Stream Stress-Tests:** Stress-testing online methods under Extreme Label Shift, Temporal Task Clustering (bursty streams), and Small Batch Sizes (gradient noise $\sigma = 0.5$).
- **Validation Selection Bias Sweeps:** Sweeping systematic validation shift $\sigma_{bias} \in [0\%, 30\%]$ under both isotropic Gaussian noise and structured late-layer semantic shift (visualized in Figure 5).
- **Task Scalability Sweeps:** Sweeping the number of merged tasks $K \in \{4, 8, 16, 32, 64\}$, which successfully maps the dimensionality limitations of Nelder-Mead search and proves the necessity of PyTorch Adam gradient search (visualized in Figure 4).
- **Domain Diversity / Task Interference Sweeps:** Sweeping task interference levels $D \in [0\%, 20\%]$ to show that weight-space optimization becomes absolutely indispensable as experts diverge (visualized in Figure 2A).
- **Landscape Roughness Sweeps:** Sweeping the frequency of non-convex cosine oscillations $F \in [1, 20]$ to analyze online parameter trapping (visualized in Figure 2B).
- **Computational Overhead and Convergence Budgets:** Reporting execution runtimes (in ms) and tracking convergence step-by-step (proving that OFS-Tune GT-Merge achieves peak performance in just 2 gradient steps).

---

## 4. Experimental Strengths and Weaknesses
### Strengths:
- **Flawless empirical design:** The experiments directly support the paper's main theoretical and methodological claims (the "no-data" strawman, the Overfitting-Optimizer Paradox, online fragility).
- **Extremely rich Appendix:** The inclusion of Figures 2, 3, 4, 5 and 6 in the Appendix provides an incredible wealth of supplementary data, addressing almost any question a reviewer might raise.
- **Physical Validation:** The physical CNN results (Table 5) provide essential, empirical confirmation of the Overfitting-Optimizer Paradox, proving that high-capacity adaptation (Joint FT, Head Tuning) suffers under few-shot validation label noise ($30\%$) while OFS-Tune Poly-Val is immune.

### Weaknesses:
- **Physical CNN Datasets are Simple:** MNIST and FashionMNIST are relatively simple, flat datasets. Extending the physical evaluation to more complex image classification benchmarks (e.g., CIFAR-10, CIFAR-100, or SVHN) or using a pre-trained Vision Transformer backbone would make the physical experiments much more representative of real-world model-merging environments.
- **Lack of Multi-Modal/Heterogeneous Tests:** The physical experiments focus entirely on homogeneous classification tasks. While the authors discuss disjoint heterogeneous output spaces in Section 4.5.3, they do not provide experimental results for merging cross-modal models (e.g., merging text and vision models, or classification and generation models).

---

## 5. Summary of Experimental Rating
**Rating: Excellent**
The experimental section is exemplary. The sweeps are thorough, the baselines are well-tuned and representative, and the physical neural network validation bridges the simulation-empirical gap with strong, reproducible results.
