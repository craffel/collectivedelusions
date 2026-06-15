# 4. Experimental Check

## Critique of the Experimental Setup

### 1. The "Toy" Dataset Domain Setup (Unrealistic Evaluation)
The paper evaluates SLD-Merge on a joint stream composed of **MNIST, FashionMNIST, CIFAR-10, and SVHN**. 
- These represent four completely disjoint and highly distinct visual domains (handwritten digits, clothing types, natural objects, and street numbers).
- In any modern Vision Transformer (even a ViT-Tiny), the representations for such highly distinct domains are highly separated and orthogonal in the latent space. Distinguishing between them with a simple cosine-similarity router is trivial.
- In real-world multi-task model merging, the tasks are typically related or fine-grained (e.g., different NLP tasks on a language model, or different fine-grained styles/views in vision) where activation-based routing is highly challenging due to overlapping representation distributions.
- By selecting a "toy" setup of completely disjoint domains, the authors artificially inflate their router's classification accuracy (reported as $93.26\%$ to $\approx 100\%$) and hide the potential routing jitter and gating fragility that would occur in realistic, overlapping multi-task deployments.

### 2. Severely Under-Trained and Low-Shot Experts
The entire empirical evaluation is restricted to a subset of **256 training, 128 validation, and 256 test samples per dataset**.
- This is an extremely low-resource setup. Consequently, the independent experts are severely under-trained, as shown by the SVHN expert's extremely low standalone accuracy of **$29.30\%$** (whereas a standard model trained on the full SVHN dataset easily achieves $>95\%$).
- Model merging is designed to combine fully converged, high-performing experts. Merging severely under-trained and overfitted experts trained on a tiny 256-sample set is highly non-standard.
- The under-trained nature of the experts introduces massive representation noise and instability, making the entire experimental evaluation a poor representation of actual model merging dynamics.

## Evaluation of the Baselines (Strawman Comparisons)
The dynamic merging baselines evaluated are the **Linear Router** and **QWS-Merge**.
- The authors demonstrate that these baselines suffer from "soft collapse" as the batch size increases because they average routing coefficients across the batch dimension to reconstruct a single set of merged weights.
- However, this comparison is a classic **strawman**:
  - The baselines average routing coefficients across the batch because they are *actual weight-space model merging methods*. Reconstructing a single set of merged weights $W_{merged}$ is the only mathematically viable way to process a batch under a single shared weight matrix.
  - SLD-Merge avoids batch-averaging simply because it is **not doing weight-space merging**. It keeps $K$ separate low-rank adapters in memory and executes a parallel forward pass over them. This is an activation-space Mixture-of-Experts (MoE) design.
  - Comparing SLD-Merge to weight-merging baselines and claiming SLD-Merge "resolves" their batch dependency is conceptually unfair. It bypasses the problem by adopting a completely different architecture (MoE) that scales compute linearly with $K$.
- Furthermore, the authors fail to compare SLD-Merge against actual, relevant baselines:
  - There are no comparisons to existing **multi-LoRA/multi-adapter routing frameworks** (such as *LoRA-Hub*, *LoRA-MoE*, or *LLaVA-MoE*).
  - There are no comparisons to standard **Sparse Mixture-of-Experts (MoE)** baselines.
  - Without these comparisons, it is impossible to determine whether the proposed "SLD-Merge" offers any actual benefits or novelty over standard, existing multi-adapter and MoE routing methods.

## Do the Results Support the Claims?

### 1. Overclaiming on "SVD Regularization"
The authors claim that: "rank-16 SLD-Merge achieves $66.50\%$ joint accuracy, which actually outperforms the full-rank baseline by $+1.38\%$," and call this a "profound scholarly insight" where SVD acts as a "heavy implicit regularizer."
- This is highly suspicious and represents a classic case of framing a training bug as a feature.
- Because the experts were trained on a tiny 256-sample dataset, they are severely overfitted. SVD truncation "outperforms" the full-rank model simply by discarding the noisy, overfitted components of the weights (low-singular-value noise).
- If the experts were properly trained to convergence with standard regularization (e.g., weight decay, dropout, or early stopping) on a standard-size dataset, SVD truncation would almost certainly *hurt* performance compared to full-rank. 
- Presenting SVD truncation as a superior alternative to full-rank models based on this artificial toy setup is highly misleading and overstates the regularizing benefits of SVD.

### 2. Disingenuous Computational Complexity Claims
The authors claim that SLD-Merge is "computationally lean" and "adds only $8.3\%$ computational overhead (FLOPs)."
- This claim is only true for the evaluated case of $K=4$ tasks.
- Because their PyTorch parallel forward pass executes all $K$ low-rank adapters in parallel, the computational overhead scales linearly with $K$.
- For $K=50$ or $K=100$ tasks, the computational overhead would be massive, making the model far from "computationally lean." The experimental results on a tiny $K=4$ setup are used to make broad, unscalable claims about edge-deployment efficiency.
