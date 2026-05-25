import os

with open("submission.tex", "r") as f:
    tex = f.read()

# Define the new expanded Related Work section
new_related_work = r"""\section{Related Work}
\label{sec:related}

Our work operates at the intersection of weight-space model merging, test-time adaptation, and Fisher-based parameter preconditioning. In this section, we situate FDF-DPA within the broader context of these established domains.

\subsection{Weight-Space Model Merging}
Weight-space model merging represents a parameter-efficient alternative to prediction-level Mixtures-of-Experts, enabling direct fusion of multiple fine-tuned models without additional inference-time costs. Foundational static merging methods, such as Model Soups~\cite{Wortsman2022} and basic parameter averaging~\cite{Matena2022}, demonstrate that linear combinations of weight parameters can yield significant multi-task generalization. 

To mitigate interference when merging models fine-tuned on different objectives, Task Arithmetic~\cite{Ilharco2023} and TIES-Merging~\cite{Yadav2023, Yadav2023TIESMergingRI} propose resolving parameter conflicts by masking insignificant parameter updates and resolving sign disagreements. Similarly, RegMean~\cite{Jin2023, Nguyen2025RegMeanEE} uses linear regression to preserve expected layer activations across distinct tasks. Robust Task Arithmetic~\cite{Yuan2023RobustTA} and Protected Task Arithmetic~\cite{Bar2024ProtectedTA} further stabilize this paradigm by safeguarding shared foundations.

Beyond simple linear interpolation, weight alignment techniques like Git Re-Basin~\cite{Ainsworth2022GitRM} and ZipIt!~\cite{Stoica2023ZipItMM} resolve permutation symmetries to merge models with different initialization paths. Recent developments have expanded model merging to equivariant deep weight spaces~\cite{Navon2023EquivariantDW}, geometric model merging~\cite{Cao2026GeometricMW}, layer-wise parameter interpolation~\cite{Xiong2025LayerwisePR, Du2025AdaMMSMM}, weight flow estimation~\cite{Gupta2026DeepWeightFlowRF}, evolutionary-based merging pathways~\cite{Akiba2024EvolutionaryOO, Cao2026EvolutionaryNM}, and parameter-efficient adapters~\cite{Chen2023ParameterEfficientFD, Nallabollu2026ParameterEF, Touko2026LightweightTA, Sangamnerkar2025LightweightAD}. Diverse applications explore multi-task fusion across distinct domains and scales~\cite{Crisostomi2026ModelMF, Chaves2025WeightWP, Pala2024DELLAMergingRI, Wang2025InModelMF, Verma2024MergingTT, Kinderman2024FoldableSS, Zhu2024DPPAPM, Ye2025DynamicMM}.

\subsection{Test-Time Adaptation \& OOD Generalization}
Test-Time Adaptation (TTA) aims to adjust pre-trained models to unlabeled non-stationary test streams during inference~\cite{Sun2019TestTimeTF}. Fully test-time adaptation methods, most notably TENT~\cite{Wang2020, Wang2021TentFT}, optimize Batch Normalization scale and shift parameters by minimizing the prediction entropy. Continual test-time adaptation (CTTA) frameworks, such as CoTTA~\cite{Wang2022ContinualTD, Jiang2024PCoTTACT}, expand this to long-term drift scenarios by regularizing predictions against a teacher model.

To prevent catastrophic representational collapse under extreme domain shifts, several methods focus on efficient and robust test-time update mechanisms~\cite{Cheng2025FullyTA, Zhou2025TesttimeTW}, test-time prediction alignment~\cite{You2021TesttimeBS, Peng2025TestTimeAF}, and out-of-distribution (OOD) generalization~\cite{Scalbert2022TesttimeIT, Ouellette2025OutofDistributionGI, Wang2025OutofDistributionGM, Zhou2024OnTE, Zhou2024CrossTaskLE}. Under highly non-stationary streams, however, parameter backpropagation remains computationally prohibitive and susceptible to representation degradation~\cite{Xiong2024TestTimeAW}. Test-Time Model Merging (TTMM)~\cite{Yang2024} addresses this by dynamically optimizing blending coefficients instead of backpropagating weight updates.

\subsection{Fisher Information and Curvature Preconditioning}
The Fisher Information Matrix (FIM) plays a foundational role in estimating parameter sensitivities in deep neural networks~\cite{Jha2025FishersIM, Weimar2025FisherIF}, as popularized in Elastic Weight Consolidation (EWC)~\cite{Yang2021ElasticWC} to avoid catastrophic forgetting. To model layer-wise parameter dependencies efficiently, Kronecker-Factored Approximate Curvature (K-FAC)~\cite{Eschenhagen2023KroneckerFactoredAC, Zhang2023KroneckerfactoredAC} factorizes the Hessian using Kronecker products of activation and pre-activation gradient covariance matrices. While diagonal Fisher preconditioning often requires clean offline validation data~\cite{Gul2024FisherMaskEN}, KT-Fisher~\cite{submission9} leverages the Kronecker Trace metric to achieve data-free, single-backward layer-wise sensitivity estimation during test-time. FDF-DPA advances this line of work by introducing a unified framework that couples Kronecker trace sensitivities with early-layer feature anchoring, preventing the feedback trap and representational collapse without offline source calibration.
"""

# We find the related work block in the original text and replace it
# The section starts with \section{Related Work} and ends before \section{Proposed Method: FDF-DPA}
start_marker = "\\section{Related Work}"
end_marker = "\\section{Proposed Method: FDF-DPA}"

start_idx = tex.find(start_marker)
end_idx = tex.find(end_marker)

if start_idx != -1 and end_idx != -1:
    before = tex[:start_idx]
    after = tex[end_idx:]
    updated_tex = before + new_related_work + "\n" + after
    
    with open("submission.tex", "w") as f:
        f.write(updated_tex)
    print("Successfully expanded Related Work section in submission.tex!")
else:
    print("Error: Could not find markers in submission.tex")
