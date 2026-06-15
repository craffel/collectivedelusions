# 4. Experimental Check and Critical Evaluation

## Experimental Setup
- **Model Architecture:** ViT-Tiny ($d_{\text{model}}=64, H=2, L=12$ blocks).
- **Pre-training Base:** Pre-trained on a joint pool of all four downstream datasets (500 samples per task, 2000 samples overall) for 15 epochs.
- **Fine-tuning:** Four task-specific experts fine-tuned for 25 epochs (MNIST, FashionMNIST, CIFAR-10, SVHN).
- **Calibration Batch:** $N=256$ samples per task.
- **Evaluation:** Evaluated across 3 independent random seeds on 1000 test samples per task (4000 test samples overall).

---

## Critical Evaluation and Weaknesses (Practitioner's Perspective)

As a **Practitioner** focused on real-world utility, efficiency, scalability, and broader impact, I identify several critical experimental weaknesses:

1. **Highly Contrived and Toy Setting (Lack of Scale):**
   - **Model Scale:** The authors use a `ViTTiny` backbone with $d_{\text{model}}=64$. In practical settings, a model with a dimension of 64 is extremely tiny—almost a toy model. Real-world Vision Transformers (e.g., ViT-Base/Large, CLIP) use dimensions of 768 or 1024.
   - **Dataset Scale:** The downstream experts are fine-tuned on only 500 samples per task. This is an extremely low-data regime.
   - **Performance Level:** Due to these extreme constraints, the expert performance is extremely poor. The individual experts' "upper bound" average accuracy is only **41.48%**, with SVHN at an abysmal **17.50%** (barely above the 10% random chance!).
   - **Practitioner Critique:** Merging experts that are barely trained and exhibit extremely low performance is of very limited practical interest. High-frequency parameter noise in poorly converged, under-trained experts is naturally much higher, which heavily amplifies vulnerability to transductive overfitting. A practitioner cannot confidently apply these findings to fully converged, high-fidelity foundation models (e.g., CLIP, LLaMA) deployed in the wild. The paper's claim of a "low-resource edge warm-start setting" feels like an excuse for running cheap, CPU-friendly toy experiments rather than a representative real-world scenario.

2. **The Demotivating Practical Reality (No Benefit Over Static Baseline):**
   - The primary result of the paper (Table 1) shows that **no adaptive configuration (even regularized L5 ES at 30.17%) outperforms the simple, static, zero-overhead Uniform Task Arithmetic baseline of 30.41%**.
   - **Practitioner Critique:** From an engineering and deployment perspective, this completely invalidates the practical utility of adaptive merging at test-time. Why would a practitioner deploy a complex, resource-heavy test-time optimization loop (running 60-100 forward/backward passes on edge devices, managing calibration queues, and storing optimizers) if they can simply take a static weight average and achieve *better* generalization with *zero* computational overhead and *zero* latency? The proposed ESR and TV regularizers, while academically interesting, only succeed in recovering performance back towards the static baseline, never surpassing it.

3. **Complete Absence of Resource and Latency Analysis:**
   - The paper targets "resource-constrained edge environments." However, test-time adaptation (especially zero-order ES which requires 100 forward passes on $N=256$ samples, or Adam which requires backpropagation) introduces massive computational latency and energy overhead.
   - **Practitioner Critique:** The paper does not report any metrics on inference latency, optimization runtime, memory footprint, or energy consumption on edge devices. This is a critical gap. For a practitioner, running a 100-step ES loop on a calibration batch at deployment time is incredibly slow and highly impractical.

4. **Missing Critical Baselines:**
   - The paper cites **RegCalMerge (Jin et al., 2026)** as a concurrent or prior work that specifically targets transductive overfitting in multi-task merging via Class-Capacity Normalization and spatial regularization.
   - **Practitioner Critique:** Since RegCalMerge directly addresses the exact problem this paper studies, it was absolutely mandatory to include it as a baseline in Table 1. Excluding a direct competitor that claims to solve transductive overfitting weakens the comparative value of the experimental section.

5. **Lack of Sensitivity Analysis for Regularizers:**
   - The authors introduce ESR and TV regularizers with hyperparameters $\beta = 1.0$ and depth balance $\gamma = 0.2$.
   - **Practitioner Critique:** There is no ablation or sensitivity analysis showing how varying $\beta$ and $\gamma$ affects generalization. At test-time, since there are no labels, a practitioner cannot perform cross-validation to select these hyperparameters. If the method's performance is highly sensitive to the exact choice of $\beta$ and $\gamma$, it becomes extremely fragile and unusable in practical settings.
