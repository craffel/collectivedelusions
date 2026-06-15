# 4. Experimental Design and Validation Check

## Experimental Framework Critique
The experimental deconstruction in this paper is highly thorough, systematic, and well-designed. The authors sweep multiple dimensions (representation entanglement $\rho$, data budgets $N_{\text{cal}}$, weight decays $\lambda$), ablate the gating function (Softmax vs. Sigmoid) and initialization (Zero-Init vs. Random), and compare layer-invariant vs. layer-wise architectures.

The final version of the paper features several major improvements that dramatically strengthen the experimental validity:

### 1. Robust and Statistically Significant Evaluation Split
In previous drafts, the real-world validation used extremely compact test samples, leading to high potential statistical variance. 
- The authors have completely addressed this by upgrading `real_world_validation.py` to evaluate on a substantial, real-world test stream of **1,000 samples total** (500 samples per task: SST-2 and QQP) drawn directly from the actual GLUE benchmark validation sets.
- By scaling the evaluation to 1,000 natural language sequences, the authors have reduced statistical variance to near zero, ensuring that the reported serving accuracies (e.g., parametric models outperforming training-free models by $+2.50\%$) are highly reliable and robust.

### 2. Under-representative Architecture Scale (BERT-Tiny)
The real-world validation is performed using **BERT-Tiny** (which has only 4 encoder layers, 2 attention heads, and a hidden dimension of 128).
- **The Impact:** Modern Parameter-Efficient Fine-Tuning (PEFT) serving and dynamic model merging are designed for and deployed on large foundation models (e.g., LLaMA, Mistral, ViT-L) with billions of parameters and deep layer structures (32+ layers).
- **The Critique:** A 4-layer model with 128 hidden units does not reflect the complex geometric representations, activation routing dynamics, or memory/compute bottlenecks of deep foundation models. The authors have explicitly and transparently disclosed this in Section 5.3 as a key limitation, clarifying that BERT-Tiny serves as a compact, computationally accessible proof-of-concept. This transparency is highly commendable, but the limitation remains an active caveat for real-world production scaling.

### 3. Poor Baseline Standalone Adapter Accuracy
As reported in Section 4.9 and Table 5, the individual task adapters achieve standalone accuracies of only **58.80%** on SST-2 and **65.60%** on QQP.
- **The Impact:** SST-2 and QQP are relatively straightforward binary classification tasks where standard fine-tuned models typically achieve >85% and >90% accuracy respectively. An accuracy of 58.80% is extremely low, barely outperforming a simple majority class or random baseline.
- **The Critique:** This poor performance implies that the underlying LoRA adapters were poorly trained or severely under-fitted (due to the tiny model capacity of BERT-Tiny and the highly data-constrained fine-tuning regime). The authors honestly disclose this as a limitation. Evaluating model merging frameworks on top of such weak, under-performing expert adapters makes it difficult to draw definitive conclusions about how these methods behave in high-performance production settings where adapters are highly specialized and accurate.
- **The Spurious Correlation Risk:** If the experts themselves are weak and under-fitted, the features they extract are highly noisy. In this scenario, the "success" of the parametric router over SABLE/ChemMerge in the real-world validation might not represent high-quality semantic routing. Instead, the router could merely be learning simple spurious correlations or fitting to local class-imbalance priors rather than true task semantics. Evaluating on fully converged, high-performance expert adapters represents an essential future requirement.

### 4. Embedding-level Stateless Gating
In the real-world validation, the parametric router is implemented as a simple linear classifier (`BertRouter`) trained on the mean-pooled Layer 0 (embedding-level) token representations: `test_reps_emb = get_pooler_embeddings(test_ids, test_att)`.
- **The Critique:** This means the router only sees raw token embeddings before they have passed through any Transformer encoder layers. This is fundamentally a **stateless, layer-invariant gating** mechanism. SABLE and ChemMerge, on the other hand, are designed to dynamically adjust ensembling weights layer-by-layer using intermediate activations. Comparing an embedding-level static router against layer-wise dynamic ensembling is slightly asymmetric. While the authors' layer-wise classical routing ablation in Section 4.8 successfully debunks the routing jitter myth and shows that layer-wise classical routing is viable, the BERT-Tiny experiments themselves remain static and embedding-bound, which should be explicitly noted as a structural limitation of the real-world validation.

### 5. Unrealistic SVHN Expert Performance in Sandbox
In the Analytical Coordinate Sandbox (ICS), the SVHN expert classifier is calibrated to achieve a very poor standalone accuracy of **22.80%** under a massive task-specific noise variance ($\sigma_3 = 1.20$).
- **The Impact:** On a 10-class task like SVHN, 22.80% accuracy is barely better than random guessing. In practice, no serving engineer would deploy an expert adapter with such low standalone performance. 
- **The Critique:** This massive, asymmetric noise level ($1.20$ vs. MNIST's $0.05$) creates an artificial scenario that heavily penalizes open-loop parametric routers and artificially favors ChemMerge's stateful kinetics due to its temporal low-pass noise-dampening. The authors discuss this in Section 5.2, explaining how a balanced noise profile narrows this stabilization premium, which helps put the results into a highly realistic and practical perspective.
