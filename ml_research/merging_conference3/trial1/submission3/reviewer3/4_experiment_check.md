# 4. Experimental Check

## Rigor of Experimental Setup and Datasets
The paper evaluates the proposed **ThermoMerge** framework on:
1. A simulated 1D non-convex loss landscape.
2. Multi-dataset image classification tasks on **MNIST**, **FashionMNIST**, and **KMNIST**.

While the inclusion of three distinct image domains is commendable, the actual experimental scale is extremely limited:
* The deep learning experiments are performed on very lightweight **Multi-Layer Perceptrons (MLPs)** consisting of only two hidden layers (128 and 64 units) and classification heads with just 640 parameters.
* Modern model merging literature (as of 2026) evaluates frameworks on massive foundation models (e.g., Vision-Language models like CLIP ViT-B/ViT-L, or Large Language Models like LLaMA-2/3, Mistral, Gemma) across complex multi-task benchmarks (such as ImageNet, GLUE, MMLU, GSM8k).
* Evaluating exclusively on MLPs and MNIST-level tasks constitutes a **major scale limitation**. The paper's claim that the framework "scales to deep neural landscapes" is not convincingly demonstrated, as the tested models are toy-scale and do not reflect the complexity of modern foundation models.

## Baseline Comparisons
The paper compares against a reasonable set of baselines:
* **Training-free**: Task Arithmetic, Ties-Merging, DARE.
* **Test-time adaptive**: AdaMerging, SyMerge.
* **Active flat-minima optimizers**: SWA, SAM.

However, several critical baseline validations are missing:
* **SAIM** (Sharpness-Aware Isotropic Merging), a concurrent adaptive merging method mentioned in the Related Work, is not included in the empirical tables.
* Given the small scale of the MLP tasks, a **hyperparameter-tuned SyMerge** or **Multi-Start SyMerge** should have been evaluated on the neural network benchmarks. In the 1D toy simulation, Multi-Start GD solved the landscape perfectly; evaluating whether multiple random restarts of SyMerge can match ThermoMerge on deep networks would have provided a stronger baseline comparison.

## Evaluation of Empirical Claims: Do the Results Support the Claims?
This is the most critical weakness of the paper. The authors claim that deterministic optimizers (like SyMerge) are trapped in sub-optimal local minima, while ThermoMerge's physical global exploration consistently escapes these traps to settle in superior, flat global minima, yielding highly robust and competitive accuracies.

However, a close examination of the empirical results on actual neural networks (Tables 6, 7, 8, and 9) reveals that **this central claim is not supported by the deep learning data**:

1. **Clean MLP Joint Adaptation (Table 6)**:
   * **MNIST**: SyMerge ($89.97\% \pm 0.19\%$) outperforms ThermoMerge ($89.94\% \pm 0.16\%$).
   * **FashionMNIST**: SyMerge ($84.61\% \pm 0.49\%$) outperforms ThermoMerge ($84.46\% \pm 0.59\%$).
   * **KMNIST**: ThermoMerge ($80.37\% \pm 0.24\%$) is virtually identical to SyMerge ($80.35\% \pm 0.19\%$).
2. **Out-of-Distribution MLP Joint Adaptation (Table 7)**:
   * **MNIST**: SyMerge ($87.51\% \pm 0.53\%$) outperforms ThermoMerge ($87.49\% \pm 0.54\%$).
   * **FashionMNIST**: SyMerge ($76.91\% \pm 0.83\%$) outperforms ThermoMerge ($76.83\% \pm 0.84\%$).
   * **KMNIST**: SyMerge ($72.25\% \pm 0.37\%$) outperforms ThermoMerge ($72.20\% \pm 0.19\%$).
3. **Clean PEFT/LoRA Joint Adaptation (Table 8)**:
   * **MNIST**: SyMerge ($88.68\% \pm 0.86\%$) outperforms ThermoMerge ($88.65\% \pm 0.63\%$).
   * **FashionMNIST**: ThermoMerge ($78.41\% \pm 1.67\%$) achieves a higher mean than SyMerge ($77.42\% \pm 1.52\%$), but their standard deviations overlap heavily ($[75.9, 78.9]$ vs. $[76.7, 80.1]$), making this improvement statistically fragile.
   * **KMNIST**: ThermoMerge ($76.62\% \pm 0.28\%$) is virtually identical to SyMerge ($76.57\% \pm 0.18\%$).
4. **Out-of-Distribution PEFT/LoRA Joint Adaptation (Table 9)**:
   * **MNIST**: SyMerge ($81.62\% \pm 1.72\%$) outperforms ThermoMerge ($81.27\% \pm 1.57\%$).
   * **FashionMNIST**: ThermoMerge ($65.68\% \pm 2.14\%$) has a higher mean than SyMerge ($64.57\% \pm 1.38\%$), with overlapping standard deviations.
   * **KMNIST**: SyMerge ($67.43\% \pm 0.34\%$) outperforms ThermoMerge ($67.41\% \pm 0.47\%$).

### Key Empirical Discrepancy:
* Across almost all deep learning configurations, **ThermoMerge does not provide statistically significant improvements over deterministic joint adaptation (SyMerge)**. 
* In **8 out of 12 evaluations, deterministic SyMerge actually achieves a higher mean accuracy than ThermoMerge (Ours)**. 
* On KMNIST, the results are virtually identical.
* The only cases where ThermoMerge shows a higher mean accuracy are on the FashionMNIST LoRA tasks (clean and OOD), but even there, the standard deviations overlap heavily, which indicates the gains are statistically fragile.
* Therefore, the paper's claims of escaping inescapable traps and finding superior flat global minima on deep networks are **unsupported**. The massive "56.7% reduction in final loss" is **strictly observed on the toy 1D synthetic simulation** which was specifically engineered to exhibit a sharp trap. On real neural networks, the benefit of physical exploration is subtle or non-existent.

## Statistical Significance
* The deep learning experiments are run over only 5 independent random seeds.
* Given the tiny differences in performance (mostly within $0.05\% - 0.2\%$) and the relatively large standard deviations (often $0.5\% - 1.5\%$), **the paper is missing any rigorous statistical significance testing (such as t-tests)**.
* Reporting overlapping confidence intervals and claiming superior performance is methodologically weak and does not meet the standards of a top-tier machine learning publication.
