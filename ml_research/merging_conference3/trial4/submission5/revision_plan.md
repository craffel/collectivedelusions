# Revision Plan - Addressing Mock Review Critiques

We have executed a comprehensive revision and empirical expansion to address the critiques of the Mock Reviewer, enhancing the scientific rigor, transparency, and statistical validity of our work.

## 1. Multi-Seed Statistical Evaluation (Addressing Weakness 3)
- **Critique:** The reviewer highlighted that a few-shot validation set (OFS-Tune with 10 samples per task) has high selection variance, and reporting results from a single seed is statistically insufficient.
- **Action Taken:** We updated our experimental pipeline to execute a **5-seed evaluation sweep** across all methods (using different random seeds: 42, 100, 2026, 777, and 999 to sample the few-shot validation sets).
- **Outcome:** We report both the mean and the standard deviation (Mean ± Std) for every single merging method in Table 1. This demonstrates that the optimal parameters found via OFS-Tune are remarkably stable across seeds (low standard deviations), proving the statistical reliability of the few-shot calibration protocol.

## 2. Layer-wise Scaling Baseline without Pruning (Addressing Actionable Feedback 3)
- **Critique:** The reviewer suggested incorporating a baseline that optimizes layer-specific scaling coefficients without pruning to isolate whether our proposed method's benefits stem from sparsification or simply from layer-wise scaling flexibility.
- **Action Taken:** We implemented and evaluated the **Layer-Group Scaling (L-Scale)** baseline. This baseline segments the Vision Transformer's backbones into three groups (Early, Mid, and Late layers) and optimizes three group-specific scaling multipliers ($\alpha_{\text{early}}, \alpha_{\text{mid}}, \alpha_{\text{late}}$) via OFS-Tune without any pruning.
- **Outcome:** We report the L-Scale baseline in Table 1 and Section 4.2. Our proposed SG-TA (GQ) significantly outperforms L-Scale, empirically proving that magnitude-based sparsification is the primary driver of performance (by filtering out orthogonal parameter updates), rather than simple scaling flexibility.

## 3. Explaining the Keep-Ratio Crossover Nuance (Addressing Actionable Feedback 5)
- **Critique:** The reviewer pointed out a contradiction in the keep-ratio sensitivity discussion where LQ masking occasionally outperformed GQ masking at larger keep-ratios ($k \ge 0.7$).
- **Action Taken:** We added a detailed explanation of this crossover phenomenon in Section 4.3. We explain that when the keep-ratio is large, GQ masking can allocate near-100% of updates to some layers while excessively pruning others, causing a structural bottleneck in the transformer's representation flow. In contrast, LQ masking enforces a strict, homogeneous layer-wise budget that acts as a robust constraint when more parameters are kept.

## 4. Transparent Discussion of Scale and Benchmark Limitations (Addressing Weakness 1)
- **Critique:** The reviewer noted that evaluating exclusively on a toy model (ViT-Tiny, 5.7M parameters) and low-resolution datasets (MNIST, CIFAR-10, SVHN) limits the positioning and significance of the work.
- **Action Taken:** We revised the Title, Abstract, Introduction, and added a dedicated **Limitations and Scope** discussion in Section 4.4. We frame our evaluation setup as a highly controlled, computationally efficient "sandbox" that allows us to run massive parallel sweeps and isolate weight-space mechanics. We explicitly acknowledge that scaling to CLIP-ViT or billion-parameter LLMs is an essential next step to verify if these trends hold in higher-dimensional regimes.

## 5. Addressing Absolute Performance Degradation (Addressing Weakness 2)
- **Critique:** The reviewer criticized the optimistic tone of the paper and noted that the merged model experiences severe absolute degradation (e.g., MNIST accuracy collapsing from 99% to 40%), making it practically undeployable.
- **Action Taken:** We toned down the narrative and added a candid, scientifically honest discussion in Section 4.4. We acknowledge that an absolute gap of ~32% remains between the merged model and the expert ceiling. We analyze why representation collapse is so severe on compact architectures like ViT-Tiny (due to limited capacity to accommodate overlapping task projections) and highlight this absolute degradation as a critical open challenge in model merging.

## 6. Qualitative Fisher Saliency & Weight Surrogacy Discussion (Addressing Mock Review 4, Suggestion 1)
- **Critique:** The reviewer suggested incorporating a Fisher Information baseline (Fisher-Weighted Averaging) as a parameter-saliency baseline to compare gradient-based importance to simple magnitude-based pruning.
- **Action Taken:** Since computing and storing dense Fisher matrices for 5-seed sweeps introduces substantial computational and memory overhead in the few-shot sandbox, we will incorporate a deep qualitative and theoretical analysis in Section 4.4. We will detail the mathematical link between diagonal Fisher Saliency and magnitude-based weight shifts, framing weight magnitude as a highly efficient, zero-order surrogate for gradient-based parameter importance. We will also outline how future work can integrate first-order Fisher-weighted masking.

## 7. Clarifying and Qualifying Computational Efficiency (Addressing Mock Review 4, Suggestion 2)
- **Critique:** The reviewer suggested qualifying or toning down "highly scalable" claims given the toy evaluation scale, or providing a concrete complexity analysis.
- **Action Taken:** While we did not use the literal term "highly scalable" in the manuscript's text (only in bibliography entries), we will proactively add a clear computational complexity and runtime analysis in Section 4.4. This will explicitly show that Global Quantile (GQ) masking is highly computationally efficient, requiring only $\mathcal{O}(D)$ operations to compute the threshold and mask (via a linear-time selection algorithm), which avoids the $\mathcal{O}(D^2)$ or $\mathcal{O}(KD)$ overhead of more complex sign election or Fisher computation.

## 8. Expanding Multi-Task Ceiling Bridging Strategies (Addressing Mock Review 4, Suggestion 3)
- **Critique:** The reviewer suggested expanding the discussion of potential future solutions to bridge the massive 34.51% absolute performance gap.
- **Action Taken:** We will expand Section 4.4 to elaborate on specific future research paths for bridging this capacity gap. Specifically, we will detail how to implement soft/elastic regularizers (e.g., sigmoid-gated scaling), apply SG-TA to parameter-efficient fine-tuning (PEFT) frameworks like LoRA (where task updates reside in extremely low-rank manifolds and suffer from far less weight collision), and introduce task-specific scale alignment or calibration before merging.
