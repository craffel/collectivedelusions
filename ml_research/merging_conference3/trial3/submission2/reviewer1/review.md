# Peer Review

**Title:** The "No-Data" Strawman: Demystifying Test-Time Adaptation vs. Offline Few-Shot Validation Tuning  
**Authors:** Marcus Thorne (University of Oxford)  

---

## Overall Recommendation
**Recommendation:** **5: Accept**  

This paper presents an exceptional, highly rigorous, and much-needed methodological course correction in the model-merging literature. From a practical engineering and real-world deployment perspective, this work is of immense value. In production settings, deploying active, backpropagation-heavy online Test-Time Adaptation (TTA) models is highly undesirable due to substantial computational overhead, latency, and the extreme operational risk of representational collapse on noisy, non-i.i.d. deployment streams. By demonstrating that **Offline Few-Shot Validation Tuning (OFS-Tune)**—using as few as 5 to 10 validation samples per task—can produce a static, zero-compute merged model that matches or exceeds online TTA performance under standard streams while being perfectly robust to adversarial distribution shifts, the authors provide a highly reliable, simple, and practical alternative. 

The paper is technically solid, exceptionally well-written, and thoroughly validated via both a highly calibrated continuous simulation across 30 random seeds and a physical PyTorch deep CNN validation on real image datasets. I strongly recommend its acceptance.

---

## Strengths and Weaknesses

### Strengths
1. **Outstanding Practical Utility & Ease of Deployment:** OFS-Tune delivers a static merged model requiring **zero test-time compute** and zero runtime modifications. Bypassing test-time backpropagation and forward adaptation passes removes a major barrier to deploying merged expert models in low-resource edge devices or high-throughput real-time systems.
2. **Robustness to Real-World Stream Corruptions:** The systematic stress-tests under Extreme Label Shift, Bursty Temporal Streams, and Small Batch Sizes represent safety-critical deployment conditions. Showing that online TTA collapses under these shifts (e.g., AdaMerging drops to 77.99% under label shift) while OFS-Tune maintains perfect, deterministic stability (85.89%) is a crucial, high-signal finding for practitioners.
3. **Formalization of the Overfitting-Optimizer Paradox:** The paper does a brilliant job of disentangling optimization and generalization failure under low-data regimes. Proving that unconstrained high-dimensional search spaces overfit to sample noise (Table 4) when optimized with highly capable optimizers like PyTorch Adam is a foundational insight.
4. **Physical Neural Network Validation:** Implementing and evaluating a physical 5-layer CNN on MNIST/FashionMNIST on real image streams (and under 30% validation label noise) provides an outstanding real-world proof-of-concept that bridges the gap between simulated landscapes and actual deep weights. It empirically verifies that high-capacity baselines (Joint Fine-Tuning and Head Tuning) overfit catastrophically, while OFS-Tune Poly-Val act as structural filters to ensure stable generalization.
5. **Rigorous Scalability and Domain Diversity Sweeps:** Sweeping the task scale up to $K=64$ and domain diversity up to $20\%$ addresses critical practical bottlenecks, proving that differentiable validation optimization (Adam Poly-Val) scales flawlessly to high-dimensional multi-task setups.

### Weaknesses (Constructive Suggestions)
1. **Practical Hardware/Memory I/O Bottleneck during Offline Tuning:**
   - Because prediction loss is non-linear with respect to the merging coefficients (due to activation layers), evaluating the validation loss $\mathcal{L}_{val}(\theta)$ at each step of the offline optimization loop requires reconstructing the merged weights $W_{merged}(\theta)$ and running a validation forward pass.
   - For massive models (e.g., 7B to 70B parameter LLMs), repeatedly performing weight additions and transferring parameters in VRAM/RAM can represent a major memory bandwidth and I/O bottleneck. 
   - While this is an *offline* cost (which is much easier to manage than runtime overhead), the paper would benefit from a brief paragraph in the "Limitations and Future Work" section discussing this practical hardware scaling bottleneck and proposing possible mitigations (e.g., caching intermediate activations, low-rank parameter updates, or coordinate descent to rebuild only subset layers).
2. **Physical Evaluation on Pre-Trained Backbones:**
   - The physical deep CNN experiments are trained starting from a shared random base weight initialization. While this serves as an elegant, clean "laboratory setup" that isolates task-vector weight optimization from pre-training representational leakage, modern model merging is almost exclusively applied to pre-trained backbones (e.g., CLIP-ViT, LLaMA).
   - Although the authors argue that pre-trained weights exhibit stronger linear alignment, verifying OFS-Tune on top of pre-trained expert task vectors is an essential future milestone to fully solidify its industrial applicability.

---

## Soundness
**Rating:** **Excellent**  

The submission is methodologically and technically sound. The claims are fully supported by massive empirical evidence in both continuous simulation and physical PyTorch deep CNN weight-space experiments. The use of exact Adam controls to isolate the Overfitting-Optimizer Paradox, the extensive hyperparameter sweeps to ensure baseline fairness, and the evaluation of standard TTA mitigations (learning rate decay and EMA smoothing) showcase a flawless methodological approach. The authors are transparent about their simulation calibration and assumptions.

---

## Presentation
**Rating:** **Excellent**  

The paper is exceptionally clearly written, logically structured, and easy to follow. The mathematical notation is precise and consistent. The visualizations are outstanding: the robustness plot (Figure 1) and the physical 2D contour plot of the prediction entropy landscape (Figure 2) are highly polished, informative, and provide deep qualitative support to the quantitative results.

---

## Significance
**Rating:** **Excellent**  

The significance of this contribution is exceptionally high. It deconstructs an overly complex academic trend (online test-time adaptation for model merging) and replaces it with a simple, static, and zero-compute alternative that utilizes small validation sets. This is a vital methodological course correction that will influence future academic evaluation protocols and provide machine learning engineers with a robust, reliable, and easily deployable tool for modular weight-space multi-tasking.

---

## Originality
**Rating:** **Excellent**  

The paper demonstrates high originality in shifting the research perspective from active test-time optimization to structured offline few-shot validation tuning. The formulation of the Overfitting-Optimizer Paradox, the introduction of low-dimensional trajectories as analytical filters against validation noise and selection bias, and the comprehensive stress-testing under non-i.i.d. stream conditions are highly novel and provide deep conceptual insights.

---

## Questions for the Authors

1. **Practical Offline Tuning Scaling:** For large-scale transformer models (e.g., 7B+ LLMs), reconstructing the merged model weights $W_{merged}(\theta)$ at each step of the Adam or Nelder-Mead optimization loop can be extremely slow and memory-intensive. Do you have plans to evaluate or optimize this weight-reconstruction step, perhaps by optimizing coefficients layer-by-layer or using cache-sharing mechanisms during validation?
2. **Back-porting to Strict Zero-Shot Regimes:** In highly restrictive deployment scenarios where target labels are strictly impossible to acquire, online TTA remains the only option. Based on your findings regarding low-dimensional Poly-Val trajectories, could we restrict active online TTA optimizers to these same low-degree polynomial search spaces to stabilize online updates against stream noise?
3. **Advanced Weight Sparsification:** You mathematically formalize how OFS-Tune's low-dimensional trajectories integrate on top of weight-sparsification methods like TIES-Merging and DARE. Do you plan to release a library implementing this unified pipeline, and have you seen any synergistic benefits where weight sparsification and trajectory optimization combined yield even higher performance?
