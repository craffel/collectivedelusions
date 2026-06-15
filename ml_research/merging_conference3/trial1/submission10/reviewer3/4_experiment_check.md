# 4. Experimental Evaluation Check

## Evaluation of Experimental Setup and Datasets
The experimental setup is standard and well-designed:
* **Model Backbone:** The choice of the widely used CLIP ViT-B/32 backbone and target layer (`model.visual.proj` of shape $768 \times 512$) is standard and challenging.
* **Datasets:** The 8 classification datasets (SUN397, Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, DTD) span a broad range of visual domains, ensuring a rigorous multi-task evaluation.
* **Baselines:** The paper compares FoldMerge against standard static baselines (Task Arithmetic, TIES-Merging) and top-performing adaptive baselines (AdaMerging, Representation Surgery, SyMerge). This is a comprehensive list of competitors.

## Critical Analysis of Results and Claims
While the experimental setup is sound, a closer analysis of the results reveals that **the empirical claims do not justify the extreme complexity of the proposed method.**

### 1. The Classifier Head Confound (The "Elephant in the Room")
The most critical result in the paper is found in **Table 6 (Frozen Classifier Head Ablation)**.
* When classifier head adaptation is disabled (\texttt{args.classifier\_train = False}), the average accuracy of both SyMerge and FoldMerge drops from $\sim 89.75\%$ to **83.56%** (83.5597% vs 83.5572% respectively).
* This reveals that the entire $2.6\text{M}$ parameter normalizing flow network contributes virtually **zero functional improvement** over the much simpler, linear SyMerge baseline when isolated from classifier head training.
* The $6.2\%$ performance gain (from $83.56\%$ to $89.76\%$) is almost entirely driven by the direct optimization of the 388K parameters of the task classification heads on test pseudo-labels.
* This strongly suggests that FoldMerge's complex coordinate-bending machinery is a highly convoluted, redundant regularizer. A minimalist approach—such as simply training the classification heads or applying linear scaling—is much simpler, more elegant, and achieves identical functional performance.

### 2. Statistical Insignificance of Gains
* The default FoldMerge achieves an average accuracy of **89.76%**, which is a microscopic **0.02%** improvement over the SyMerge baseline (**89.74%**).
* Even with their best scale-preserving "Latent Task Vector Warping" formulation, the average accuracy is **89.77%** (a **0.03%** improvement over SyMerge).
* While the authors argue that their Test-Time Adaptation protocol is completely deterministic (yielding zero run-to-run variance under a fixed sequential stream), a 0.02% to 0.03% difference is well within standard statistical noise in deep learning.
* This microscopic performance "gain" is completely disproportionate to the massive architectural and computational complexity introduced.

### 3. Practical and Computational Infeasibility
* FoldMerge requires **10.6 minutes** of Adam optimization ($500$ steps at $1.28$ seconds per step) on an **NVIDIA H100 GPU** to adapt a **single visual projection layer** ($393,216$ parameters) of a small ViT-B/32 backbone.
* Scaling this approach to larger models (e.g., ViT-L, ViT-H, or LLMs with billions of parameters) or attempting to warp multiple layers would be computationally catastrophic, requiring hours or days of high-end GPU time for a test-time adaptation setting.
* Although the authors argue there is "zero inference overhead" because the merged model is decoded once, the setup/adaptation time itself represents an extremely high practical barrier that makes the method virtually unusable in real-world test-time adaptation scenarios.
* A simpler method like SyMerge or direct task arithmetic is vastly superior in speed and resource efficiency.
