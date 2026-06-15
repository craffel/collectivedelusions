# 4. Experiment Check

A critical evaluation of the experimental setup, datasets, baselines, and whether the results actually support the claims.

## Evaluation of Experimental Setup & Datasets
The experimental setup is exceptionally thorough and robust.
1. **The Sandbox Setup:** The authors simulate the representation space of a pre-trained Vision Transformer (ViT-Tiny) with $L=14$ layer groups, generating synthetic $192$-dimensional feature vectors for $K=4$ tasks (MNIST, FashionMNIST, CIFAR-10, SVHN). Class-specific prototypes are placed in orthogonal subspaces, and realistic noise is added to match actual expert accuracies (e.g., MNIST expert achieves 100.0%, SVHN expert achieves 32.0% due to noise).
2. **Tiny Calibration Split:** The routers are trained on an extremely challenging and realistic **low-data regime** of exactly 64 samples (16 samples per task), which is highly representative of test-time adaptation data constraints. Testing is done on a separate split of 1000 samples.
3. **Robustness Audits:** The authors thoroughly evaluate performance under:
   - Sequential task arrival (Section 3.2).
   - Non-orthogonal task overlap (Appendix G).
   - True layer-by-layer merging without averaging (Appendix H).
   - Variation in projection dimension $d \in \{2, 4, 8\}$ (Appendix I).

## Evaluation of Baselines
The paper includes an exceptionally strong set of baselines:
- **Expert Ceiling:** The individual specialized expert networks, establishing the performance upper-bound.
- **Uniform Merging (Static):** The standard model-merging baseline that does not adapt dynamically.
- **QWS-Merge SOTA:** The state-of-the-art "quantum-inspired" dynamic merging method, with exact initialization and optimizer controls matching the original work.
- **Global Classical Linear Router:** A simple, global classical Linear Router baseline that bypasses layer-wise specialization and low-dimensional projection.
- **L3-Router Variants:** Multiple classical alternatives under identical low-dimensional projections, evaluated both with and without classical $L_2$ regularization (weight decay).

The baseline selection is comprehensive and intellectually honest. The inclusion of the unregularized and regularized versions of each router is highly rigorous and directly exposes the role of classical regularization.

## Do the Results Support the Claims?
Yes, the empirical results provide overwhelming support for the paper's central claims:
1. **Wave formulation collapse:** Under the sandbox setup, QWS-Merge completely collapses, achieving **36.10%** Joint Mean (worse than uniform merging's **43.40%**), and near-random **2.00%** on OOD SVHN. This is shown to be a structural failure across all five seeds (Appendix F) and across all swept learning rates (Appendix E).
2. **L3-Router superiority:** L3-Linear achieves a Joint Mean of **63.10%** (+27.00% absolute over QWS-Merge).
3. **The Global Linear Router confounder:** The simplest global classical Linear Router achieves **67.20%** Joint Mean, outperforming all multi-layer models. This strongly validates the authors' proof of layer-averaging collapse.
4. **Heterogeneity collapse:** Mixed-task batches cause the Linear Router to drop from **67.20%** to **51.10%** and QWS-Merge to drop to **10.80%**, verifying the severe vulnerability of dynamic routing under realistic deployment streams. L3-Linear (L2 Reg) achieves **52.30%**, representing the most robust dynamic performance.
5. **Robustness-Accuracy Illusion:** L3-Softmax drops by only **4.10%** under heterogeneous streams, but its absolute accuracy (**50.30%**) is inferior to the Linear Router's absolute performance (**51.10%**), demonstrating that relative stability can mask absolute inferiority due to simplex-constraint uniform-compromise averaging.
6. **Real-Scale Generalization:** In the CLIP-ViT-B/16 pilot, QWS-Merge collapses to **41.20%**, while the simple L3-Linear achieves **84.80%** (+43.60% absolute over QWS-Merge), demonstrating that the findings scale directly to actual vision-language weight manifolds.

Overall, the empirical evidence is exceptionally complete, rigorous, and completely supports the paper's claims.
