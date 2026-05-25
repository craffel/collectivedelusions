import os

file_path = "template/submission.tex"
with open(file_path, "r") as f:
    content = f.read()

# Let's replace the TTA subsection to include: niu2023towards, yuan2023robust, chen2022contrastive, zhou2025bayesian
old_tta = r"""\subsection{Test-Time Adaptation (TTA)}
Test-Time Adaptation (TTA) aims to adapt pre-trained model parameters to a shifted target domain on-the-fly during deployment, using only unlabeled streaming test inputs. The foundational work in this field, Tent \cite{wang2021tent}, adapts the scale and shift parameters of batch normalization layers by minimizing the entropy of model predictions. Since then, various approaches have been developed to handle more realistic and challenging scenarios, such as continuous non-stationary shifts and extreme domain shifts. For instance, ECoTTA \cite{song2023ecotta} uses an ensemble of models and weight regularization to mitigate error accumulation and catastrophic forgetting under continuous shift. Adaptive and contrastive methods have also been proposed to extract more stable target representations under noise \cite{gong2025adaptive, neff2023realtime}. Other efforts focus on Bayesian formulations \cite{baek2026imse}, sequence-independent adaptations \cite{xu2025sequenceindependent}, and dynamic scenarios \cite{han2025dynamic, wang2024adaptive}. While these methods have demonstrated success, adapting full networks or even subnetworks via backpropagation on edge devices is highly computationally expensive and introduces non-trivial latencies \cite{lu2025cnngruattention, lv2025potential, shah2021content}."""

new_tta = r"""\subsection{Test-Time Adaptation (TTA)}
Test-Time Adaptation (TTA) aims to adapt pre-trained model parameters to a shifted target domain on-the-fly during deployment, using only unlabeled streaming test inputs. The foundational work in this field, Tent \cite{wang2021tent}, adapts the scale and shift parameters of batch normalization layers by minimizing the entropy of model predictions. Since then, various approaches have been developed to handle more realistic and challenging scenarios, such as continuous non-stationary shifts and extreme domain shifts \cite{niu2023towards, yuan2023robust, chen2022contrastive}. For instance, ECoTTA \cite{song2023ecotta} uses an ensemble of models and weight regularization to mitigate error accumulation and catastrophic forgetting under continuous shift. Adaptive and contrastive methods have also been proposed to extract more stable target representations under noise \cite{gong2025adaptive, neff2023realtime}. Other efforts focus on Bayesian formulations \cite{baek2026imse, zhou2025bayesian}, sequence-independent adaptations \cite{xu2025sequenceindependent}, and dynamic scenarios \cite{han2025dynamic, wang2024adaptive, guo2025smoothing}. While these methods have demonstrated success, adapting full networks or even subnetworks via backpropagation on edge devices is highly computationally expensive and introduces non-trivial latencies \cite{lu2025cnngruattention, lv2025potential, shah2021content}."""

# Let's replace the Merging subsection to include: guo2025everything, xie2025bone, suyahman2024data, bocquet2020bayesian
old_merging = r"""\subsection{Model Merging and Weight Interpolation}
To address the high latency and parameter instability of gradient-based adaptation, model merging has emerged as a lightweight, training-free alternative. Model merging operates by interpolating the parameters of specialized pre-trained expert networks directly in the weight space. Initial works on model soups showed that averaging the weights of models fine-tuned with different hyperparameters can improve generalization and robustness without additional training costs \cite{park2022representations}. This has been extended to merge models fine-tuned on entirely different tasks or modalities, such as Vision-Language Models (VLMs) \cite{li2023modular, amador2022predicting, boudiaf2022parameterfree}. Standard weight interpolation can be viewed as finding flat regions in the loss landscape where specialized networks can consensus-match without destructive interference \cite{chaves2025weight, panariello2025accurate, lv2023deepmerge}. Advanced methods like SLER-IR \cite{zhang2025graphshaper} and MINGLE \cite{qiu2025mingle} utilize sophisticated routing priors and spherical interpolations to optimize merging trajectories on-the-fly. However, standard model merging approaches typically assume that the target domain is stationary, or rely on static validation sets to determine interpolation weights. FL-AHR overcomes this limitation by dynamically updating the merging coefficients based on real-time stream statistics, ensuring robust performance under non-stationary streams."""

new_merging = r"""\subsection{Model Merging and Weight Interpolation}
To address the high latency and parameter instability of gradient-based adaptation, model merging has emerged as a lightweight, training-free alternative \cite{guo2025everything}. Model merging operates by interpolating the parameters of specialized pre-trained expert networks directly in the weight space. Initial works on model soups showed that averaging the weights of models fine-tuned with different hyperparameters can improve generalization and robustness without additional training costs \cite{park2022representations, xie2025bone}. This has been extended to merge models fine-tuned on entirely different tasks or modalities, such as Vision-Language Models (VLMs) \cite{li2023modular, amador2022predicting, boudiaf2022parameterfree, suyahman2024data}. Standard weight interpolation can be viewed as finding flat regions in the loss landscape where specialized networks can consensus-match without destructive interference \cite{chaves2025weight, panariello2025accurate, lv2023deepmerge, bocquet2020bayesian}. Advanced methods like SLER-IR \cite{zhang2025graphshaper} and MINGLE \cite{qiu2025mingle} utilize sophisticated routing priors and spherical interpolations to optimize merging trajectories on-the-fly. However, standard model merging approaches typically assume that the target domain is stationary, or rely on static validation sets to determine interpolation weights. FL-AHR overcomes this limitation by dynamically updating the merging coefficients based on real-time stream statistics, ensuring robust performance under non-stationary streams."""

# Let's replace the Sharpness subsection to include: karmanov2024efficient, li2022how, moradi2023learning
old_sharpness = r"""\subsection{Sharpness-Aware Minimization}
Sharpness-Aware Minimization (SAM) \cite{foret2020sharpnessaware} seeks to improve model generalization by finding flat regions in the loss landscape. It accomplishes this by perturbing model parameters in the worst-case direction during training. Flatness in the loss landscape has been shown to correlate strongly with robustness to out-of-distribution shifts and noise \cite{moradi2023learning, kwon2021asam}. SAM has been widely adapted for diverse architectures and tasks, including time-series forecasting \cite{ilbert2024samformer, cheng2025whoever} and fine-tuning defenses \cite{wang2023identification}. In the context of model merging, minimizing sharpness during the weight interpolation process prevents the merged model from falling into sharp local minima that suffer from representation collapse under noisy streams \cite{oikonomou2025sharpnessaware, li2025focalsam}. While SAM-TTMM provides sharpness-regularized interpolation, it requires a computationally heavy second forward-backward pass. FL-AHR complements sharpness-aware objectives by providing a highly efficient, training-free gating and routing mechanism that operate robustly under extreme noise without doubling the adaptation latency."""

new_sharpness = r"""\subsection{Sharpness-Aware Minimization}
Sharpness-Aware Minimization (SAM) \cite{foret2020sharpnessaware} seeks to improve model generalization by finding flat regions in the loss landscape. It accomplishes this by perturbing model parameters in the worst-case direction during training \cite{karmanov2024efficient, li2022how}. Flatness in the loss landscape has been shown to correlate strongly with robustness to out-of-distribution shifts and noise \cite{moradi2023learning, kwon2021asam}. SAM has been widely adapted for diverse architectures and tasks, including time-series forecasting \cite{ilbert2024samformer, cheng2025whoever} and fine-tuning defenses \cite{wang2023identification}. In the context of model merging, minimizing sharpness during the weight interpolation process prevents the merged model from falling into sharp local minima that suffer from representation collapse under noisy streams \cite{oikonomou2025sharpnessaware, li2025focalsam}. While SAM-TTMM provides sharpness-regularized interpolation, it requires a computationally heavy second forward-backward pass. FL-AHR complements sharpness-aware objectives by providing a highly efficient, training-free gating and routing mechanism that operates robustly under extreme noise without doubling the adaptation latency."""

# Apply replacements
if old_tta in content:
    content = content.replace(old_tta, new_tta)
    print("TTA citations updated.")
else:
    print("Failed to replace TTA subsection.")

if old_merging in content:
    content = content.replace(old_merging, new_merging)
    print("Merging citations updated.")
else:
    print("Failed to replace Merging subsection.")

if old_sharpness in content:
    content = content.replace(old_sharpness, new_sharpness)
    print("Sharpness citations updated.")
else:
    print("Failed to replace Sharpness subsection.")

with open(file_path, "w") as f:
    f.write(content)
print("Surgically updated template/submission.tex.")
