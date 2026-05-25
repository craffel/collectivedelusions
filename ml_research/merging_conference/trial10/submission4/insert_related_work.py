import os

file_path = "template/submission.tex"
with open(file_path, "r") as f:
    content = f.read()

related_work_text = r"""
\section{Related Work}
\label{submission-related-work}

\subsection{Test-Time Adaptation (TTA)}
Test-Time Adaptation (TTA) aims to adapt pre-trained model parameters to a shifted target domain on-the-fly during deployment, using only unlabeled streaming test inputs. The foundational work in this field, Tent \cite{wang2021tent}, adapts the scale and shift parameters of batch normalization layers by minimizing the entropy of model predictions. Since then, various approaches have been developed to handle more realistic and challenging scenarios, such as continuous non-stationary shifts and extreme domain shifts. For instance, ECoTTA \cite{song2023ecotta} uses an ensemble of models and weight regularization to mitigate error accumulation and catastrophic forgetting under continuous shift. Adaptive and contrastive methods have also been proposed to extract more stable target representations under noise \cite{gong2025adaptive, neff2023realtime}. Other efforts focus on Bayesian formulations \cite{baek2026imse}, sequence-independent adaptations \cite{xu2025sequenceindependent}, and dynamic scenarios \cite{han2025dynamic, wang2024adaptive}. While these methods have demonstrated success, adapting full networks or even subnetworks via backpropagation on edge devices is highly computationally expensive and introduces non-trivial latencies \cite{lu2025cnngruattention, lv2025potential, shah2021content}.

\subsection{Model Merging and Weight Interpolation}
To address the high latency and parameter instability of gradient-based adaptation, model merging has emerged as a lightweight, training-free alternative. Model merging operates by interpolating the parameters of specialized pre-trained expert networks directly in the weight space. Initial works on model soups showed that averaging the weights of models fine-tuned with different hyperparameters can improve generalization and robustness without additional training costs \cite{park2022representations}. This has been extended to merge models fine-tuned on entirely different tasks or modalities, such as Vision-Language Models (VLMs) \cite{li2023modular, amador2022predicting, boudiaf2022parameterfree}. Standard weight interpolation can be viewed as finding flat regions in the loss landscape where specialized networks can consensus-match without destructive interference \cite{chaves2025weight, panariello2025accurate, lv2023deepmerge}. Advanced methods like SLER-IR \cite{zhang2025graphshaper} and MINGLE \cite{qiu2025mingle} utilize sophisticated routing priors and spherical interpolations to optimize merging trajectories on-the-fly. However, standard model merging approaches typically assume that the target domain is stationary, or rely on static validation sets to determine interpolation weights. FL-AHR overcomes this limitation by dynamically updating the merging coefficients based on real-time stream statistics, ensuring robust performance under non-stationary streams.

\subsection{Sparsity-Aware Gating and Mixture of Experts}
Our work is also closely related to Sparsity-Aware Gating and Mixture-of-Experts (MoE) architectures, which activate different parameters or pathways depending on the input properties. Dynamic routing is commonly used in MoEs to dispatch inputs to specialized expert sub-networks \cite{du2021efficient, brants1996better}. For test-time scenarios, research has explored test-time MoE pruning and dynamic routing to reduce computation and interference \cite{koikeakino2025μmoe, lei2024adaptedmoe, duan2019sparse}. To measure and control input or representation sparsity, researchers frequently employ scale-invariant metrics like Hoyer's sparsity measure \cite{yang2019deephoyer}. Hoyer sparsity has been successfully applied to control neural network sparsity during autoencoder training \cite{zhang2026decomposing} and streaming image processing \cite{qu2022generalized}. However, previous methods like BK-AHR rely on input pixel-space Hoyer's sparsity, which collapses under environmental noise as noise masks the physical sparsity structure. FL-AHR addresses this by computing sparsity on denoised convolutional activation maps, bridging the gap between sparsity estimation and noise-robust gating.

\subsection{Sharpness-Aware Minimization}
Sharpness-Aware Minimization (SAM) \cite{foret2020sharpnessaware} seeks to improve model generalization by finding flat regions in the loss landscape. It accomplishes this by perturbing model parameters in the worst-case direction during training. Flatness in the loss landscape has been shown to correlate strongly with robustness to out-of-distribution shifts and noise \cite{moradi2023learning, kwon2021asam}. SAM has been widely adapted for diverse architectures and tasks, including time-series forecasting \cite{ilbert2024samformer, cheng2025whoever} and fine-tuning defenses \cite{wang2023identification}. In the context of model merging, minimizing sharpness during the weight interpolation process prevents the merged model from falling into sharp local minima that suffer from representation collapse under noisy streams \cite{oikonomou2025sharpnessaware, li2025focalsam}. While SAM-TTMM provides sharpness-regularized interpolation, it requires a computationally heavy second forward-backward pass. FL-AHR complements sharpness-aware objectives by providing a highly efficient, training-free gating and routing mechanism that operates robustly under extreme noise without doubling the adaptation latency.
"""

old_target = r"\section{Background and Preliminaries}"
new_target = related_work_text + "\n" + old_target

if old_target in content:
    content = content.replace(old_target, new_target)
    with open(file_path, "w") as f:
        f.write(content)
    print("Successfully inserted Related Work section.")
else:
    print("Failed to find target string in LaTeX file.")
