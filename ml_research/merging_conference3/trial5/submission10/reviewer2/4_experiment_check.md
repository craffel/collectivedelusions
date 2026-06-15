# Experimental Setup and Results Evaluation: ChaosMerge

## 1. Evaluation of the Experimental Setup & Datasets
From a practical and real-world engineering perspective, the experimental setup is highly restricted, synthetic, and outdated:
- **Toy-Scale Datasets:** The evaluation relies entirely on classic, small-scale computer vision datasets: MNIST, FashionMNIST, CIFAR-10, and SVHN. 
  - MNIST and FashionMNIST consist of $28 \times 28$ grayscale images, which are widely considered trivial "toy" problems in modern machine learning. 
  - Projecting 1-channel grayscale images to 3-channels to fit a 5.7M parameter Vision Transformer backbone is a highly synthetic, contrived task.
  - These datasets do not reflect any modern, real-world deployment constraints or industry use-cases (such as multi-task language processing, high-resolution object detection, or multimodal understanding).
- **Failure to Generalize to Modern Scales:** In contemporary machine learning, model merging is predominantly applied to massive Large Language Models (LLMs, e.g., 7B to 70B parameters) or large vision-language models. Evaluating exclusively on ViT-Tiny with 5.7M parameters on $32 \times 32$ images fails to prove whether the complex Coupled Map Lattice dynamics can generalize to or scale up to industry-relevant architectures and data workloads.

## 2. Do the Results Support the Claims?
The empirical results in Table 1 directly contradict several of the paper's central claims:

### A. The Underperformance of ChaosMerge
The authors claim that ChaosMerge "achieves highly competitive performance compared to over-parameterized dynamic routers." However, a direct comparison of the numbers shows that ChaosMerge is consistently the **worst-performing optimized method** across both settings:
1. **Task-Averaged Setting (Single Merged Model):**
   - **OFS-Tune (Supervised Static):** $73.55\%$
   - **QWS-Merge (Quantum Wave):** $73.55\%$
   - **Linear Router (Classical):** $73.50\%$
   - **ChaosMerge (G-CML):** **$71.20\%$** (Underperforming the static OFS-Tune by $-2.35\%$ and the Linear Router by $-2.30\%$)
2. **Task-Specific Setting (Dynamic / Task-Conditional):**
   - **OFS-Tune Task-Specific (Static, Supervised):** $82.90\%$
   - **Linear Router (Classical):** $77.10\%$
   - **QWS-Merge (Quantum Wave):** $77.05\%$
   - **ChaosMerge (G-CML):** **$73.80\%$** (Underperforming the Linear Router by $-3.30\%$ and the simple static task-conditional baseline by a massive **$-9.10\%$ absolute**)

If a simple, unconstrained static task-conditional baseline (OFS-Tune Task-Specific) outperforms the complex G-CML by $9.10\%$ absolute, and a standard Linear Router outperforms it by $3.30\%$ absolute, the claims of ChaosMerge's "exceptional dynamic separation" and "outstanding empirical results" are not supported by the data. 

### B. Unsupervised Baseline Comparison
Even **AdaMerging**, which is completely unsupervised and performs test-time adaptation on unlabelled test streams, achieves **$70.85\%$** average accuracy. G-CML's Task-Averaged accuracy of **$71.20\%$** is only $+0.35\%$ higher, but G-CML requires a supervised training phase on the 64-sample calibration labels, low-dimensional phase projections, and complex recurrent gating. This marginal improvement does not justify the massive increase in engineering and mathematical complexity.

### C. The "Annealed Chaos-to-Order" Illusion
While the authors introduce "Annealed Chaos-to-Order Merging" in Section 4.5 and claim it achieves $78.12\%$ average accuracy, this result is achieved by **dampening and eventually removing the chaotic Logistic Map altogether** in favor of a contractive Tanh-gated map at the end of training. This empirically proves that the chaotic dynamics are a major liability for final performance, and the model only becomes competitive by transforming into a standard non-chaotic gated recurrent model. Furthermore, even this complex annealed hybrid model still underperforms the simple static OFS-Tune Task-Specific baseline ($82.90\%$) by **$-4.78\%$ absolute**, while requiring a complex training-time interpolation schedule.
