# Intermediate Review File 4: Experimental Evaluation and Rigor

## 1. Evaluation of the Experimental Setup and Datasets
The authors design a highly controlled visual classification benchmark featuring four domains (MNIST, FashionMNIST, CIFAR-10, SVHN) and a Vision Transformer backbone (`vit_tiny_patch16_224`, 5.7M parameters). 
* **Strengths:** 
  * The selection of datasets simulates a high-conflict scenario with diverse and conflicting task boundaries (e.g., handwritten digits vs. real-world house numbers vs. general objects).
  * The setup is computationally efficient, which enables the authors to run extensive sweeps, multiple seeds, and diverse ablations that would be prohibitive on larger models.
* **Critiques (Empiricist Scrutiny):**
  * **Severe Scale and Generalizability Limitations:**
    Modern model merging literature is overwhelmingly focused on Large Language Models (LLMs) with billions of parameters or larger Vision-Language models like CLIP-ViT-B/16 (86M to 300M+ parameters). The chosen backbone (`vit_tiny`) is extremely small (5.7M parameters), and the datasets are low-resolution toy benchmarks ($28 \times 28$ and $32 \times 32$ pixels).
    In large-scale foundation models, the high-dimensional parameter redundancy may naturally prevent representational collisions, or conversely, introduce different layer-wise specialization dynamics. It is highly questionable whether the empirical findings on a tiny ViT on toy datasets will generalize to modern foundation models. A simulated NLP study on 72 simulated tensors is a start, but it does not replace a physical evaluation on real NLP models (e.g., Llama or RoBERTa).

## 2. Evaluation of Baselines
The authors compare SG-TA against a very strong and comprehensive set of baselines:
1. **Naive Uniform TA:** Standard linear addition.
2. **Optimized TA:** Globally scaled addition.
3. **TIES-Merging:** Trim, Elect Sign, and Merge (representative state-of-the-art).
4. **DARE-Merging:** Stochastic drop and rescale (representative state-of-the-art).
5. **Decoupled Prune-then-Merge (P-then-M):** Magnitude pruning per expert before linear merging.
6. **Layer-Group Scaling (L-Scale):** Fine-grained layer multipliers without pruning.
7. **Fisher-Weighted Averaging:** Diagonal Fisher Information weighting.
8. **Joint Multi-Task Learning (MTL):** Simultaneous training on all datasets, establishing the ultimate multi-task training upper bound.
* **Rigor:** All baselines are fully optimized on the exact same few-shot validation set across 5 random calibration seeds. This represents an exceptionally fair and rigorous comparison, far exceeding the typical standard of tuning only the proposed method.

## 3. Do the Results Statistically Support the Claims?
We evaluate the main claims of the paper against the reported data:

* **Claim: SG-TA "outperforms" state-of-the-art baselines like TIES-Merging and DARE-Merging.**
  * *Verdict:* **Unsupported / Overstated.**
  * *Data:* In Table 1, SG-TA (GQ) achieves **61.40% $\pm$ 1.39%**, TIES-Merging achieves **60.64% $\pm$ 1.30%**, and DARE-Merging achieves **58.44% $\pm$ 3.02%**.
  * *Analysis:* The difference between SG-TA (GQ) and TIES-Merging is only **$0.76\%$**. Because their standard deviations overlap significantly ($\pm 1.39\%$ and $\pm 1.30\%$), this difference is **not statistically significant**. While the authors honestly acknowledge this in the discussion text, they still assert that SG-TA "outperforms" TIES in the Abstract, Introduction, and Conclusion. An empiricist must demand that the authors tone down this claim of superiority and instead frame SG-TA as achieving *comparable performance to TIES while being conceptually and computationally simpler* (which is still a highly valuable contribution).
* **Claim: Spatial Regularization via magnitude masking prevents representational collapse.**
  * *Verdict:* **Fully Supported.**
  * *Data:* SG-TA (GQ) achieves $61.40\%$ vs. Naive Uniform TA's $46.32\%$ ($+15.08\%$ absolute improvement).
* **Claim: Global Budget Flexibility (GQ) is vital and superior to layer-wise homogeneous budgets (LQ).**
  * *Verdict:* **Fully Supported.**
  * *Data:* GQ masking achieves $61.40\% \pm 1.39\%$ compared to LQ's $57.81\% \pm 2.52\%$. The keep-ratio sensitivity analysis (Table 2) further confirms that at the optimal keep-ratio $k=0.3$, GQ achieves $60.11\%$ vs. LQ's $55.06\%$.
* **Claim: TV-Norm successfully resolves task vector magnitude imbalance.**
  * *Verdict:* **Fully Supported.**
  * *Data:* Table 1 shows that incorporating TV-Norm (SG-TA GQ-Norm) boosts MNIST accuracy from $36.74\%$ to $53.70\%$ (a massive $+16.96\%$ absolute increase). However, this comes at a cost to SVHN performance ($85.35\%$ down to $70.18\%$), which is an honest and expected trade-off.
* **Claim: Sigmoid-gating (SG-TA-Soft) stabilizes calibration.**
  * *Verdict:* **Fully Supported.**
  * *Data:* Table 1 shows that SG-TA (GQ-Soft) achieves a standard deviation of only **$\pm$ 0.75%**, nearly cutting the variance of the hard GQ variant ($\pm 1.39\%$) in half while maintaining a comparable Joint Mean Accuracy ($61.06\%$).
* **Claim: Coordinate Search (CS) is a scalable and effective non-uniform calibration strategy.**
  * *Verdict:* **Fully Supported.**
  * *Data:* Table 3 shows Coordinate Search achieves $58.40\% \pm 2.32\%$ Joint Mean Accuracy in linear time $\mathcal{O}(T)$ (100 evaluations, 43.61s), while successfully rebalancing joint performance (boosting MNIST from $36.74\%$ to $50.38\%$ and CIFAR-10 from $67.82\%$ to $75.04\%$).
