# 4. Experimental Setup and Evaluation Check

This section critically analyzes the empirical design, datasets, baselines, and the alignment between the paper's claims and its actual experimental results.

## 1. Scale and Generalizability Concerns (Toy Setup)
The most prominent weakness of the empirical evaluation is its extremely small scale. 
- **Backbone Model:** The paper uses `vit_tiny_patch16_224` (only $5.7$M parameters), which is a tiny toy model. Standard model-merging literature (including the cited AdaMerging, Task Arithmetic, and TIES-Merging papers) evaluates on large foundation models such as **CLIP-ViT-B/32** ($86$M parameters) or **CLIP-ViT-L/14** ($307$M parameters), as well as large language models (LLMs) like **LLaMA-7B** or **Mistral-7B**.
- **Toy Datasets and Small Subsets:** The experts are trained on highly restricted subsets of size **1,024 images** per task for simple, low-resolution vision datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).
- **Tiny Calibration/Evaluation Sets:** The calibration stream consists of only **16 samples per task** (64 total images), and the test evaluation sets consist of only 512 images.

Because deep models behave very differently at scale, findings on a $5.7$M parameter `vit_tiny` trained on tiny image subsets cannot be assumed to generalize to large-scale foundation models. The severe "transductive overfitting" and performance collapse of AdaMerging seen here might be an artifact of the very low capacity of `vit_tiny` combined with the minuscule calibration set, rather than a general property of active model merging. 

## 2. Marginal Practical Gains vs. High Computational Overhead
The paper heavily emphasizes the "superior generalization" and SOTA performance of PG-Merge ($p=0.05$). However, a close inspection of Table 1 reveals that the actual practical improvement is marginal:
- **Static Uniform Merging (Task Arithmetic):** Average joint accuracy is **$62.16\%$** (requires **zero** optimization steps, **zero** backpropagation, and has **zero** runtime overhead).
- **PG-Merge (Ours, $p=0.05$):** Average joint accuracy is **$62.70\%$**.

This represents a tiny improvement of only **$0.54\%$** over the static uniform baseline.
To achieve this $0.54\%$ gain, PG-Merge requires:
1. Running **100 gradient steps** of backpropagation on the 64-image calibration set.
2. Computing and backpropagating gradients through the entire $5.7$M parameter network at each step.
3. Sorting and masking gradients at each step.

For most real-world test-time adaptation applications, running 100 backward passes through a transformer network is computationally prohibitive and introduces severe latency. The paper completely fails to discuss this trade-off, presenting PG-Merge as having "zero computational overhead." While it has zero *parameter* overhead, its *computational* overhead is massive compared to the static baseline, and a $0.54\%$ accuracy gain does not justify this cost.

## 3. Persistent Failure to Close the Expert Ceiling Gap
The average individual expert performance (Expert Ceiling) is **$78.08\%$**.
- Static Uniform Merging achieves $62.16\%$, leaving a massive gap of **$15.92\%$** due to parameter-space interference.
- PG-Merge ($p=0.05$) improves this to $62.70\%$.

This means PG-Merge closes only **$0.54\%$ of the $15.92\%$ gap**—which is a minuscule **$3.4\%$ of the total degradation**. In other words, $96.6\%$ of the performance degradation from parameter interference remains unresolved, despite running 100 steps of active test-time optimization. This demonstrates that active test-time adaptation of merging coefficients on tiny calibration sets is highly ineffective at actually resolving task conflicts in this setting, a critical detail that the authors' positive narrative glosses over.

## 4. Under-performing and Collapsed Baselines
The authors compare PG-Merge against several baselines, but some seem poorly tuned or exhibit extreme behavior:
- **PolyMerge ($d=2$):** PolyMerge's MNIST performance collapses catastrophically to **$13.48\%$** (near-random guessing on a 10-class task), leading to a terrible joint average of $46.97\%$. While PolyMerge is highly constrained, such an extreme collapse suggests either an improper learning rate, incorrect implementation, or lack of proper tuning for this baseline. Using a collapsed baseline makes PG-Merge look artificially strong.
- **Unregularized AdaMerging:** AdaMerging drops performance to $61.08\%$ ($1.08\%$ below static uniform), which again indicates a highly fragile and unstable optimization environment where active tuning does more harm than good.
