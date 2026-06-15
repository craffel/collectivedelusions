# Final Peer Review

## Summary of the Paper
This paper presents a rigorous empirical and statistical deconstruction of test-time dynamic model merging, an emerging paradigm that combines task-specific expert neural networks into a single multi-task model at inference time. While dynamic routing networks have been theoretically proposed to adapt merging coefficients on the fly, this work identifies a severe deployment vulnerability: standard unregularized dynamic routers (including global linear routers, random-initialized L3-Softmax, and advanced quantum-inspired activation models like QWS-Merge) suffer from catastrophic **Vectorization Collapse** when deployed in real-time, sample-wise vectorized pipelines where the batch size $B=1$.

The authors reveal that standard large-batch evaluations suffer from a **Batch-Average Smoothing Confounder**, where batch averaging of predicted coefficients acts as an implicit smoothing operator that masks the severe overfitting of the router on small calibration splits ($|D_{\text{cal}}| = 64$ samples). When evaluated under a batch-independent stream ($B=1$), this smoothing mask is removed, causing accuracy to plummet (e.g., standard L3-Softmax drops by nearly $17\%$ below static Uniform Merging). 

To resolve this, the authors propose the **Prior-Driven Classical Routing** framework, showing that simple, elegant architectural priors—specifically, zero-initialized Softmax routing layers regularized with $L_2$ weight decay—completely eliminate Vectorization Collapse and maintain flatline, highly generalizing joint accuracy across all batch sizes ($B=1$ to $B=512$). Furthermore, they expose the **Dynamic Routing Paradox**: under data scarcity, a dynamic router must be regularized so heavily (restricting its coefficients to stay within a Mean Absolute Deviation of only $2.36\%$ from the uniform baseline) that its learned benefits are marginal, yielding a tiny $+1.16\%$ gain over naive, training-free **Uniform Merging**. Given the massive $O(B \cdot M)$ memory expansion and memory-bandwidth bottlenecks of dynamic full-parameter assembly, the authors conclude that naive, static Uniform Merging represents an exceptionally strong and practical default for production environments.

---

## Main Strengths
1. **Exemplary Adherence to Simplicity**: The paper provides a highly refreshing and refreshing perspective on model merging by systematically challenging unnecessary architectural complexity. It successfully demonstrates that highly convoluted, non-monotonic wave-interference activations (like QWS-Merge) and explicit task-variance regularization training losses ($\mathcal{L}_{VR}$) are either brittle or redundant. A simple classical linear projection combined with a zero-initialized Softmax prior and standard weight decay completely resolves the collapse, proving that proper basic regularization is the true, necessary driver of generalization.
2. **Identification of Widespread Evaluation Flaws**: The formal characterization of the **Batch-Average Smoothing Confounder** and **Vectorization Collapse** is an outstanding contribution. It exposes a major scientific blindspot in previous literature, demonstrating that earlier models appeared competitive only because their severe overfitting was masked by large-batch evaluations.
3. **Outstanding Systems-Level Sincerity**: The paper is exceptionally honest and transparent about the physical latency and memory overheads of dynamic parameter routing. Rather than hyping up a minor $+1.16\%$ improvement, it quantitatively deconstructs the **Dynamic Routing Paradox** and actively advocates for static Uniform Merging as a zero-cost, zero-overhead baseline.
4. **Comprehensive Experimental Rigor**: The evaluation is exhaustive, evaluating all models across 10 independent random seeds. The authors go above and beyond by analyzing physical CPU latency benchmarks, validation on real convolutional image experts (MNIST + FashionMNIST), task feature overlap sensitivity sweeps, alternative multi-layer MLP routing depths, projection dimension sweeps, calibration data scaling, and sequential smoothness regularizers.

---

## Main Weaknesses
The submission is exceptionally strong and well-polished, leaving very few avenues for criticism. However, some minor constructive feedback can further enhance the work:
1. **More Forceful Early Warnings**: Given the severe hardware latency slowdowns ($110.06\times$ slowdown for large batches on CPU) and VRAM footprint expansions of dynamic full-parameter assembly, the abstract and introduction could emphasize the massive systems-level benefits of static merging even more forcefully. This would immediately alert practitioners to the hidden costs of dynamic routing.
2. **Additional Static Baselines**: Comparing against other prominent training-free static merging methods (such as TIES-Merging or DARE) directly in the main results table (Table 1) would further enrich the comparison and solidify the paper's recommendation of static merging as a superior and highly practical default.

---

## Detailed Evaluation of Dimensions

### Soundness
**Rating: Excellent**  
The paper is technically flawless and mathematically rigorous. The proposed Prior-Driven Classical Routing framework is simple, elegant, and highly effective. The authors perform comprehensive statistical significance audits across 10 random seeds, map regularization sensitivity frontiers, conduct mixed-task heterogeneity stress tests, and run exhaustive ablation studies. All potential technical flaws are anticipated and mitigated: they address the systems latency bottleneck of full-parameter assembly by evaluating Low-Rank Parameter Assembly (Dynamic LoRA), showing it recovers full-parameter accuracy at $r \ge 10$ with only a $1.01\times$ latency slowdown; and they mitigate sequential multi-layer routing jitter by formulating and validating a Sequential Smoothness Regularizer ($\mathcal{L}_{\text{smooth}}$).

### Presentation
**Rating: Excellent**  
The manuscript is beautifully written, exceptionally well-structured, and highly polished. The narrative flows seamlessly from identifying real-world vulnerabilities to presenting mathematical frameworks, rigorous sandbox experiments, systems-level physical latency profiles, and real-world image expert merging. The equations are clean, precise, and easy to follow. The appendices are of outstanding quality, providing comprehensive analyses of computational complexity, hardware constraints, PCA vs random projections, a theoretical extension to non-linear model merging, and a detailed CLIP ViT-B/16 replication roadmap.

### Significance
**Rating: Excellent**  
The paper addresses a highly relevant problem in model merging and Mixture-of-Experts (MoE) routing, offering immediate utility for production deployment by preventing catastrophic vectorization collapse in low-latency streams ($B=1$). Most importantly, its conceptual impact is profound: it forces the community to rethink the necessity of overly complex, convoluted architectures and establishes naive Uniform Merging as a mandatory, formidable baseline.

### Originality
**Rating: Excellent**  
The work is highly original. It formally identifies, names, and deconstructs two major phenomena (**Vectorization Collapse** and the **Batch-Average Smoothing Confounder**) and introduces the **Dynamic Routing Paradox**. Its originality lies in its deconstructive and honest approach: rather than proposing yet another hyper-engineered behemoth, it deconstructs existing complexity to reveal that simple classical priors are the true drivers of generalization under data scarcity.

---

## Overall Recommendation
**Rating: 6: Strong Accept**  
This is a technically flawless, highly refreshing, and exceptionally impactful paper. It exposes and resolves a critical, widespread bug in test-time model merging, deconstructs complex routing architectures to prove that simple classical priors are superior, and establishes a highly valuable, honest systems perspective. Its empirical and statistical rigor across 10 random seeds is exemplary. This work is a model of high-signal, transparent, and elegant machine learning research, and it deserves a strong accept.
