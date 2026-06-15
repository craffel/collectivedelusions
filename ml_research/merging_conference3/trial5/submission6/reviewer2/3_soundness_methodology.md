# 3. Soundness and Methodology Evaluation

## Clarity of Description
The methodology is exceptionally clear, precise, and logically structured. 
* **Phase 1 (Offline SVD):** The equations (1) through (5) clearly define the extraction of task vectors and their projection into low-rank matrices $B_k$ and $A_k$. 
* **Phase 2 (Cosine-Similarity Router):** Equation (6) and (7) mathematically define the global token average pooling and the bounded cosine-similarity routing score.
* **Phase 3 (Top-1 Gating & Parallel Pass):** Equations (8) through (11) show how Top-1 gating is applied element-wise and parallelized across the batch in a vectorized forward pass.
* **Activation-Space Mean Calibration & Head Selection:** Section 3.4 and 3.5 clearly detail the empirical activation mean computation and the layer-averaged classification head selection rule.

## Appropriateness of Methods
The choice of methods is highly appropriate and tailored for edge and resource-constrained settings:
* **SVD on Task Vectors:** SVD is the mathematically optimal low-rank approximation under the Frobenius norm. It is highly appropriate for compressing redundant parameter shifts of fine-tuned experts.
* **Top-1 Hard Gating:** Soft merging of experts requires executing all $K$ expert paths and weighted-averaging their activations, which is computationally expensive. Hard gating ensures only one low-rank adapter path is executed, keeping computational overhead at a minimal **+8.3% FLOPs**.
* **Bounded Cosine Similarity:** Restricting scores to $[-1, 1]$ prevents representation collapse and noise scaling, which is a common failure mode in unconstrained linear routers under domain shift.
* **Activation-Space Mean Calibration:** Computing activation centroids is computationally free and robust, bypassing unstable and complex reinforcement learning or Gumbel-Softmax relaxations.

## Potential Technical Flaws and Questions
1. **Routing Jitter across Layers:** 
   The paper applies independent routers at each specialized layer (blocks 9, 10, and 11). While the authors report that routers achieve **96.48% agreement** across layers on the evaluation samples, what happens when they disagree? If layer 9 selects expert A and layer 10 selects expert B, does the representation become garbled? It would be valuable to analyze the impact of routing disagreement on downstream classification accuracy.
2. **Representation Shift during Calibration:** 
   The Activation-Space Mean Initialization computes activation centroids $\Phi_k^{(l)}$ using a standard forward pass under a **uniform merging configuration** on the calibration set. However, at inference time, the model executes using **sparse low-rank weights**. This represents a mathematical representation shift. While the authors state that early-layer freezing (blocks 0--8) and late-layer separability minimize the negative impact of this shift, a more rigorous analysis of this representation shift would strengthen the theoretical soundness of the calibration phase.
3. **Scaling to Full-Network Merging:**
   The paper evaluates a specialized late-layer setup (specializing blocks 9--11 while freezing blocks 0--8). If a practitioner needs to merge experts that have been fully fine-tuned (updating all 12 blocks), how does SLD-Merge scale? The authors state in Section 4.6 that parameter overhead would scale linearly from 0.295M to 1.18M parameters. However, there are no empirical results showing how routing accuracy and representation shift hold up when routing is applied across all 12 layers. Representation shift is likely to compound dramatically across a deeper stack of active routers.
4. **Under-Trained Experts:**
   The SVHN standalone expert ceiling is extremely low (**29.30%**), indicating that the expert is heavily under-trained or severely data-constrained (using only 256 training samples). While the authors frame this as a realistic low-shot stress-test, using un-converged, noisy representations might artificially inflate the benefits of SVD low-rank truncation (which acts as an implicit regularizer, as shown in Section 4.4). It is crucial to verify if the SVD regularizing benefit still holds when experts are fully converged and have saturated representations.

## Reproducibility
The reproducibility of this paper is **excellent**. 
* The mathematical formulations are complete and standard.
* The model architecture (`vit_tiny_patch16_224`), hyperparameters (such as rank $r \in \{4, 8, 16\}$, calibration set size of 128 samples, training set size of 256 samples), and datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) are explicitly documented.
* The training and evaluation protocols are detailed, and standard PyTorch functions are sufficient to implement the described vectorized parallel forward pass.
