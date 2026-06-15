# 4. Experimental Evaluation Check

## Evaluation Setup and Datasets
The experimental evaluation is conducted on an **extremely small, toy-scale setup** that lacks modern relevance and statistical significance:
1. **Obsolete and Tiny Backbone:** The authors use the `vit_tiny_patch16_224` architecture, which has only ~5 million parameters. In the current era of model merging, standard evaluations are performed on large-scale models such as LLaMA-7B/13B, Mistral-7B, or at least ViT-Base/Large.
2. **Highly Simplified Toy Datasets:** The downstream tasks are MNIST, FashionMNIST, CIFAR-10, and SVHN. MNIST and FashionMNIST are heavily outdated and extremely simple datasets that can be easily solved by tiny networks.
3. **Severe Data Subsampling:** 
   - Fine-tuning experts is restricted to a mere 500 training samples per task.
   - Most critically, the evaluation test sets are subsampled to only **100 samples per task**. Such a tiny test set introduces immense statistical noise. A single correct or incorrect prediction changes the accuracy by $1\%$. This explain why the reported standard deviations are so high (up to $4.03\%$ for individual experts and $3.65\%$ for optimized configurations). It severely undermines the statistical confidence of the empirical claims.

## Baselines
The selection of baselines is reasonable, including Uniform Task Arithmetic (TA), FREE-Merging, AdaMerging, and PolyMerge. However, the comparative results reveal a devastating outcome for the proposed method.

## Comparative Performance and the PolyMerge Superiority
The most critical empirical finding is that **PolyMerge completely dominates PhaseMerge and U-PhaseMerge across all settings**:
- **FP32 (Unquantized):** PolyMerge achieves **$48.00 \pm 1.62\%$**, whereas U-PhaseMerge gets $42.83 \pm 1.76\%$ and PhaseMerge ($r=2$) gets $40.75 \pm 1.43\%$. PolyMerge is $+5.17\%$ better than U-PhaseMerge and $+7.25\%$ better than PhaseMerge.
- **8-bit PTQ:** PolyMerge gets **$48.00 \pm 1.47\%$**, while U-PhaseMerge gets $42.33 \pm 1.76\%$ (gap of $5.67\%$).
- **4-bit PTQ:** PolyMerge gets **$43.42 \pm 1.30\%$**, while U-PhaseMerge gets $37.42 \pm 1.94\%$ (gap of $6.00\%$).

These are massive, statistically significant performance gaps. Despite the immense mathematical complexity, Fourier projections, Straight-Through Estimators, and symmetry-preserving masks, the proposed frequency-domain method is substantially inferior to a simple real-space polynomial baseline that uses only 12 parameters. The authors attempt to explain this in Section 4.2 by pointing out the lack of depth-wise coordination in PhaseMerge and suggesting "PolyPhaseMerge" as a future direction. However, as submitted, the paper presents a method that is empirically obsolete compared to existing work.

## Catastrophic Multi-Task Performance Collapse
An unaddressed issue in the experimental results is the **catastrophic drop in absolute performance** after merging:
- Individual expert diagonal performances are:
  - MNIST: $81.00\%$
  - FashionMNIST: $74.67\%$
  - CIFAR-10: $71.67\%$
  - SVHN: $85.33\%$
  - Average Expert Performance: **$78.17\%$**
- When merged using PhaseMerge ($r=2$), the multi-task average accuracy collapses to **$40.75\%$**—a catastrophic loss of **$37.42\%$** absolute accuracy.
- Even with the best baseline (PolyMerge), the merged model only reaches **$48.00\%$**, which is a $30.17\%$ absolute performance drop.

These results indicate that the merged models are practically useless, as they perform barely better than random guessing on several tasks. In a realistic model-merging scenario, merged models are expected to maintain a significant fraction of the expert capabilities (often within $5-10\%$ of the experts). Merging models in a setup where they lose more than half of their original performance is highly artificial and raises serious doubts about the practical validity of the entire experimental setting.

## Core Proposed Method Underperforms Its Simplest Variant
In Table 4 (Ablation Study), the authors evaluate the grid dimensions. The primary proposed method, **PhaseMerge ($r=2$)** which uses the $2\times 2$ bilinear upsampling grid, is consistently **outperformed by the simpler U-PhaseMerge ($r=1$)** which uses a single scalar per layer:
- In FP32, U-PhaseMerge is $+2.08\%$ better ($42.83\%$ vs $40.75\%$).
- In 8-bit, U-PhaseMerge is $+1.50\%$ better ($42.33\%$ vs $40.83\%$).

This result proves that the spatially-continuous frequency-smoothing grid ($r=2$), which is a major part of the paper's novelty and complexity, is actually **detrimental** to performance on dense layers. The authors acknowledge that this is because dense layer weight coordinates have no native spatial correlation. This finding directly supports our theoretical critique: the Fourier-domain 2D representation is a mismatch for dense layers, and enforcing spatial frequency constraints on unordered coordinates hurts performance.
