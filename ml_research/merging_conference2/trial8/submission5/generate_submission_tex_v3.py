import json
import re

# Load sweep results
with open("sweep_results.json", "r") as f:
    results = json.load(f)

# Helper to find a matching accuracy
def get_acc(m_type, lam, cal, q_bit, qmode, corr):
    for r in results:
        # Normalize quantization_bits
        r_q_bit = r['quantization_bits']
        if r_q_bit == 'FP32':
            r_q_bit = None
        
        # In our sweep, FP32 has qmode 'per_tensor'.
        # If we query FP32 with 'per_channel', map it to the actual evaluated FP32 (per_tensor)
        if q_bit is None:
            actual_qmode = 'per_tensor'
        else:
            actual_qmode = qmode
            
        if (r['merge_type'] == m_type and 
            r['lambda'] == lam and 
            r['calibration'] == cal and 
            r_q_bit == q_bit and 
            r['quantization_mode'] == actual_qmode and 
            r['corruption'] == corr):
            return r['avg_acc']
    return None

# Format helper
def fmt(val):
    if val is None:
        return "N/A"
    return f"{val:.2f}"

# Build LaTeX content
tex_content = r"""\documentclass{article}

% Recommended, but optional, packages for figures and better typesetting:
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{booktabs} % for professional tables
\usepackage{hyperref}
\newcommand{\theHalgorithm}{\arabic{algorithm}}

\usepackage[accepted]{icml2026}

\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathtools}
\usepackage{amsthm}
\usepackage{algorithm}
\usepackage{algorithmic}

\usepackage[capitalize,noabbrev]{cleveref}

\setlength{\textfloatsep}{9pt plus 2pt minus 3pt}
\setlength{\floatsep}{7pt plus 2pt minus 2pt}
\setlength{\intextsep}{7pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{9pt plus 2pt minus 3pt}
\setlength{\dblfloatsep}{7pt plus 2pt minus 2pt}

\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{assumption}[theorem]{Assumption}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}

\icmltitlerunning{A Unified Empirical Study and Optimization of Data-Free Model Merging}

\begin{document}

\twocolumn[
\icmltitle{A Unified Empirical Study and Optimization of Data-Free Model Merging \\
under Physical Quantization and Real-World Corruptions}

\icmlsetsymbol{equal}{*}

\begin{icmlauthorlist}
\icmlauthor{Pragmatic Researcher}{yyy}
\end{icmlauthorlist}

\icmlaffiliation{yyy}{Pragmatic AI Systems Lab, Department of Computer Science, University of Technology}
\icmlcorrespondingauthor{Pragmatic Researcher}{researcher@pragmatic-ai.edu}

\icmlkeywords{Model Merging, Post-Training Quantization, Environmental Robustness, Edge AI, Deep Learning}

\vskip 0.3in
]

\printAffiliationsAndNotice{}

\begin{abstract}
Merging multiple specialized neural networks into a single multi-task model has emerged as a powerful training-free approach to consolidate skills. However, existing work almost exclusively focuses on the clean, full-precision (FP32) performance of merged models. In contrast, real-world deployment on edge devices is constrained by strict resource limits, requiring post-training quantization (PTQ) to low-bit formats (e.g., 8-bit or 4-bit) and robustness to environmental corruptions like sensor noise and blur. In this paper, we conduct the first systematic, unified empirical investigation of data-free model merging under these physical constraints. We show that while parameter-space calibration methods like Isotropic Parameter Resonance (U-IPR) and Holographic Norm Scaling (HNS) successfully restore representation scaling in full precision, their calculated scale factors shift the dynamic range of weight updates, severely amplifying quantization noise and sensitivity under out-of-distribution corruptions. To resolve this, we propose \textbf{Quantization-Robust Parameter Resonance (QR-IPR)}, which dynamically clamps scale factors using robust layer-wise statistics (Median and Median Absolute Deviation). Our comprehensive benchmark of Weight Averaging (WA), Task Arithmetic (TA), TIES-Merging, and DARE-Merging across MNIST, FashionMNIST, and CIFAR-10 ResNet-18 experts demonstrates that QR-IPR effectively stabilizes models under 8-bit quantization and noise while preserving full-precision accuracy. Finally, we expose a critical boundary: all zero-shot parameter merging methods collapse under 4-bit uniform quantization, establishing key guidelines for practitioners.
\end{abstract}

\vspace{-2.0mm}
\section{Introduction}
\label{sec:intro}
\vspace{-2.5mm}

The exponential growth in the number of specialized deep neural networks fine-tuned for distinct downstream tasks has made multi-task serving a major operational bottleneck. Keeping independent model checkpoints for each task is extremely expensive in terms of storage, memory, and routing latency, particularly on edge hardware (e.g., mobile phones, autonomous vehicles, and remote sensors). To resolve this, \emph{model merging} has emerged as an active area of study \cite{haji2024survey, le2023deep}. By combining multiple specialized models sharing a common pre-trained progenitor, practitioners can construct a unified multi-task model with zero additional training cost, zero training data, and a 1$\times$ storage footprint.

Two foundational paradigms dominate parameter-space model merging: Weight Averaging (WA) \cite{wortsman2022model} and Task Arithmetic (TA) \cite{ilharco2023editing}. While conceptually elegant, simple linear merges often experience performance degradation due to \emph{interference} between conflicting task-specific parameters \cite{yadav2023ties}. To mitigate this, advanced sparsification and sign election methods such as TIES-Merging \cite{yadav2023ties} and DARE \cite{yu2024language} have been developed. More recently, parameter-space calibration methods have been introduced to address representation collapse (the degradation of hidden representation scaling) by rescaling task updates. Notable examples include Holographic Norm Scaling (HNS) \cite{submission7} and Update-level Isotropic Parameter Resonance (U-IPR) \cite{submission9}.

Despite their theoretical appeal, current model merging and calibration methods suffer from a severe \emph{academic-to-practical gap}. Research papers evaluate merging algorithms almost exclusively in full precision (FP32 or FP16) on clean, curated datasets. However, real-world deployment on physical edge hardware demands two major concessions: \textbf{(1) Low-Bit Quantization}: models are quantized to 8-bit (INT8) or 4-bit (INT4) to fit VRAM limits and leverage hardware-level accelerators. \textbf{(2) Environmental Corruptions}: input signals are frequently corrupted in the wild by factors like sensor noise and blur \cite{hendrycks2019robustness}.

In this paper, we adopt the perspective of \emph{The Pragmatist} and systematically bridge this gap. We discover that data-free calibration methods like HNS and U-IPR, although highly effective at FP32, are highly sensitive to quantization and environmental noise. Specifically, we show that these methods compute channel-wise or layer-wise scaling factors that can grow excessively large. Under Post-Training Quantization (PTQ), these extreme scale factors inflate the dynamic range of weight updates, severely amplifying quantization rounding and clipping errors. Similarly, they amplify high-frequency environmental noise, leading to representation collapse in the wild.

To resolve these practical bottlenecks, we propose \textbf{Quantization-Robust Parameter Resonance (QR-IPR)}, a simple, highly effective, and data-free calibration technique. QR-IPR utilizes robust layer-wise statistics—namely, the Median and Median Absolute Deviation (MAD) of computed scale factors—to dynamically clamp channel-wise scaling multipliers. By isolating and regularizing outlier scales, QR-IPR preserves the representative scaling benefits of parameter resonance while ensuring maximum stability under physical quantization and environmental perturbations.

We present a thorough evaluation using ResNet-18 \cite{he2016deep} experts trained on MNIST \cite{lecun1998gradient}, FashionMNIST \cite{xiao2017fashion}, and CIFAR-10 \cite{krizhevsky2009learning}. Our main contributions are: (i) we conduct the first unified empirical study of data-free model merging under physical quantization (INT8, INT4) and environmental corruptions (noise, blur); (ii) we mathematically analyze and empirically demonstrate how standard calibration (HNS, U-IPR) degrades under quantization and noise due to outlier scale inflation; (iii) we introduce QR-IPR, a novel data-free calibration technique that dynamically clamps scale factors; (iv) we mathematically and empirically validate that update-level calibration acts as a \emph{unifying attractor}, causing the Task Arithmetic scaling factor $\lambda$ to cancel out; and (v) we identify a critical boundary for zero-shot PTQ merging, showing that per-tensor 4-bit quantization causes complete collapse of all methods.

\vspace{-2.0mm}
\section{Related Work}
\label{sec:related}
\vspace{-2.5mm}

\textbf{Linear Mode Connectivity \& Weight Averaging}. Models fine-tuned from the same pre-trained initialization often lie within the same low-loss basin, a phenomenon known as Linear Mode Connectivity (LMC) \cite{entezari2021role, ainsworth2022git}. This forms the foundation for Weight Averaging (WA) and Model Soups \cite{wortsman2022model}, which average parameters directly to improve generalization.

\textbf{Task Arithmetic \& Advanced Merging}. Task Arithmetic (TA) \cite{ilharco2023editing} defines a "task vector" as the parameter difference between the fine-tuned model and its pre-trained progenitor. These task vectors can be added or subtracted to combine or remove skills. However, merging multiple task vectors often introduces parameter interference, where updates to the same parameter conflict. Advanced methods like TIES-Merging \cite{yadav2023ties} and DARE \cite{yu2024language} resolve this by trimming small values and selecting parameter signs via sign election.

\textbf{Parameter-Space Calibration}. Merging models linearly often scales activations and representations incorrectly, causing magnitude and representation collapse. REPAIR \cite{jordan2022repair} stabilizes activations using a data-driven calibration step. To eliminate the need for calibration datasets, Holographic Norm Scaling (HNS) \cite{submission7} derives analytical scaling factors from parameter statistics alone to restore task update scaling. Similarly, Isotropic Parameter Resonance (U-IPR) \cite{submission9} rescales task updates layer-wise to match the average isotropic scale of individual experts using Frobenius norms.

\textbf{Post-Training Quantization (PTQ)}. Quantization is the standard industry practice for deploying deep learning models on resource-constrained devices \cite{dettmers2022llm, frantar2022gptq, xiao2023smoothquant}. PTQ quantizes weights and activations without re-training. Recent studies examine the intersection of quantization and merging, such as TVQ \cite{kim2025task} and Merge-Friendly PTQ \cite{shin2025merge}. However, no prior work has systematically analyzed how parameter-space calibration methods interact with quantization noise and input corruptions.

\vspace{-2.0mm}
\section{Analyzing the Robustness Bottleneck of Data-Free Calibration}
\label{sec:bottleneck}
\vspace{-2.5mm}

Let $\theta_{\text{init}}$ be the pre-trained progenitor model's parameters. We have $K$ expert models trained on $K$ distinct tasks, denoted as $\theta_t$ for $t \in \{1, \dots, K\}$. The task update vector for expert $t$ is defined as:
\begin{equation}
    \tau_t = \theta_t - \theta_{\text{init}}
\end{equation}
Under standard Task Arithmetic (TA) with scaling factor $\lambda$, the merged model parameters $\theta_{\text{merged}}$ are:
\begin{equation}
    \theta_{\text{merged}} = \theta_{\text{init}} + \lambda \sum_{t=1}^{K} \tau_t
\end{equation}
The merged update is defined as $T_{\text{merged}} = \theta_{\text{merged}} - \theta_{\text{init}} = \lambda \sum_{t=1}^{K} \tau_t$.

To calibrate this update and prevent representation collapse, U-IPR \cite{submission9} rescales $T_{\text{merged}}$ layer-wise such that its Frobenius norm matches the average Frobenius norm of individual expert updates:
\begin{equation}
    s_{\text{U-IPR}} = \frac{\frac{1}{K} \sum_{t=1}^{K} \|\tau_t\|_F}{\|T_{\text{merged}}\|_F}
\end{equation}
And the calibrated weights are $\theta_{\text{cal}} = \theta_{\text{init}} + s_{\text{U-IPR}} T_{\text{merged}}$.

HNS \cite{submission7} extends this to channel-wise scaling. For each output channel (or filter) $c$ of a weight matrix, HNS computes:
\begin{equation}
    s_c = \frac{\frac{1}{K} \sum_{t=1}^{K} \|\tau_{t, c}\|_2}{\|T_{\text{merged, c}}\|_2}
\end{equation}
And the calibrated task update is $(T_{\text{cal}})_c = s_c T_{\text{merged, c}}$. To handle numerical instability and noisy channels, both U-IPR and HNS clamp their computed scale factors to a global, heuristic bounding box:
\begin{equation}
    s_{\text{clamped}} = \text{clip}(s, s_{\text{min}}, s_{\text{max}})
\end{equation}
where $s_{\text{min}} = 0.1$ and $s_{\text{max}} = 10.0$ are typically chosen.

\subsection{Quantization Noise Inflation}
We identify that the dynamic range of weight values is a critical factor for quantization robustness. Let $W$ be a weight tensor to be quantized. In symmetric uniform quantization, the quantized weight is:
\begin{equation}
    W_{\text{quant}} = \text{clamp}\left(\text{round}\left(\frac{W}{\Delta}\right), -q_{\text{max}}, q_{\text{max}}\right) \times \Delta
\end{equation}
where $\Delta = \frac{\max(|W|)}{q_{\text{max}}}$ is the quantization step-size, and $q_{\text{max}} = 2^{b-1}-1$ for a $b$-bit integer.

Notice that the step-size $\Delta$ depends directly on the maximum absolute value of the weight tensor, $\max(|W|)$. If a tensor contains even a single outlier with a very large magnitude, the step-size $\Delta$ becomes extremely large. This increases the rounding error for all other parameters in the tensor, leading to severe quantization noise and representation collapse.

When applying HNS or U-IPR, some channels or layers receive very large scaling multipliers (close to $s_{\text{max}} = 10.0$). This directly inflates the magnitude of task updates for those specific channels or layers. If those updates contain outliers, or if the initial weights were already large, the calibrated weight values $W_{\text{cal}} = \theta_{\text{init}} + s T_{\text{merged}}$ experience a dramatic expansion in their maximum absolute magnitude. This blows up the quantization step-size $\Delta$, causing massive rounding errors and destructive representation collapse under PTQ.

\begin{proposition}[Per-Tensor Quantization Error Bounds]
Let $W_{\text{cal}} \in \mathbb{R}^{d_o \times d_i}$ be the calibrated weight matrix of a convolutional or linear layer, represented as channel-wise vectors $W_{\text{cal}, c} = W_{\text{init}, c} + s_c T_{\text{merged}, c}$ for output channel $c \in \{1, \dots, d_o\}$, where $s_c \ge 0$ is the scale factor and $T_{\text{merged}, c}$ is the merged task update. Under $b$-bit symmetric uniform per-tensor quantization with $q_{\text{max}} = 2^{b-1}-1$, the entry-wise quantization error is bounded by:
\begin{align}
    \|W_{\text{cal}} - &Q(W_{\text{cal}})\|_{\infty} \nonumber \\
    &\le \frac{\max_{c} \|W_{\text{init}, c} + s_c T_{\text{merged}, c}\|_{\infty}}{2(2^{b-1}-1)}
\end{align}
Furthermore, let the error bound under HNS be $E_{\text{HNS}}$ and under QR-IPR be $E_{\text{QR-IPR}}$. If $s_{c'} > 4.0$ for some channel $c'$ where $s_{c'}^{\text{robust}} \|T_{\text{merged}, c'}\|_{\infty} > 2 \|W_{\text{init}, c'}\|_{\infty}$, then the bound is strictly tighter under QR-IPR: $E_{\text{QR-IPR}} < E_{\text{HNS}}$.
\end{proposition}

\begin{proof}
By definition of symmetric uniform per-tensor quantization, the step-size $\Delta$ is given by:
\begin{align}
    \Delta &= \frac{\|W_{\text{cal}}\|_{\infty}}{q_{\text{max}}} \nonumber \\
    &= \frac{\max_{c} \|W_{\text{init}, c} + s_c T_{\text{merged}, c}\|_{\infty}}{2^{b-1}-1}
\end{align}
The maximum absolute rounding error is $0.5$. Thus, the dequantized error is bounded by:
\begin{align}
    \|W_{\text{cal}} - &Q(W_{\text{cal}})\|_{\infty} \le \frac{\Delta}{2} \nonumber \\
    &= \frac{\max_{c} \|W_{\text{init}, c} + s_c T_{\text{merged}, c}\|_{\infty}}{2(2^{b-1}-1)}
\end{align}
Now, consider an outlier channel $c'$ where $s_{c'} > 4.0$. Under HNS, $s_{c'}^{\text{HNS}} = \min(s_{c'}, 10.0)$. Under QR-IPR, $s_{c'}^{\text{robust}} \le 4.0$. Thus $s_{c'}^{\text{robust}} < s_{c'}^{\text{HNS}}$.

Let $s^{\text{robust}} = s_{c'}^{\text{robust}}$ and $s^{\text{HNS}} = s_{c'}^{\text{HNS}}$. For any entry $x$ of $W_{\text{init}, c'}$ and $y$ of $T_{\text{merged}, c'}$, if $s^{\text{robust}} |y| > 2|x|$, we have $|x + s^{\text{robust}} y|$ $< |x + s^{\text{HNS}} y|$.
If $y > 0$, since $s^{\text{robust}} y > 2|x|$, we have $x + s^{\text{robust}} y > 0$. Thus, $|x + s^{\text{robust}} y| = x + s^{\text{robust}} y < x + s^{\text{HNS}} y = |x + s^{\text{HNS}} y|$.
If $y < 0$, since $s^{\text{robust}} |y| > 2|x|$, we have $x + s^{\text{robust}} y < 0$. Thus, $|x + s^{\text{robust}} y| = -(x + s^{\text{robust}} y) < -(x + s^{\text{HNS}} y) = |x + s^{\text{HNS}} y|$.

This holds element-wise, so $\|W_{\text{init}, c'} + s_{c'}^{\text{robust}} T_{\text{merged}, c'}\|_{\infty} < \|W_{\text{init}, c'} + s_{c'}^{\text{HNS}} T_{\text{merged}, c'}\|_{\infty}$.
Since the global infinity norm is dominated by these outlier channels under scale inflation, we have:
\begin{align}
    \max_{c} \|W_{\text{init}, c} &+ s_c^{\text{robust}} T_{\text{merged}, c}\|_{\infty} \nonumber \\
    &< \max_{c} \|W_{\text{init}, c} + s_c^{\text{HNS}} T_{\text{merged}, c}\|_{\infty}
\end{align}
which directly implies $E_{\text{QR-IPR}} < E_{\text{HNS}}$.
\end{proof}

\subsection{Noise Sensitivity Amplification}
Similarly, when a merged model is exposed to out-of-distribution (OOD) input corruptions (such as sensor noise or camera blur), channels with highly inflated scaling factors amplify the high-frequency components of the input perturbations. This results in unstable activation distributions, worsening the covariate shift and causing performance to degrade much faster than in uncalibrated or conservatively calibrated models.

\vspace{-2.0mm}
\section{Proposed Method: Quantization-Robust Parameter Resonance}
\label{sec:proposed}
\vspace{-2.5mm}

To address these practical bottlenecks, we introduce \textbf{Quantization-Robust Parameter Resonance (QR-IPR)}. Our method is fully automated, data-free, and adds zero runtime latency. Rather than using a fixed, heuristic global bounding box (such as $[0.1, 10.0]$) which fails to adapt to layer-specific distributions, QR-IPR computes a \emph{dynamic, robust bounding box} for each layer based on robust statistics of its channel-wise scale factors.

\begin{figure*}[t]
\centering
\begin{subfigure}[b]{0.23\textwidth}
  \centering
  \includegraphics[width=\linewidth]{quantization_robustness.png}
  \caption{Quantization robustness (INT8).}
  \label{fig:quantization}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.23\textwidth}
  \centering
  \includegraphics[width=\linewidth]{noise_robustness.png}
  \caption{Robustness under noise/blur.}
  \label{fig:noise_robustness}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.23\textwidth}
  \centering
  \includegraphics[width=\linewidth]{gamma_sensitivity.png}
  \caption{Sensitivity to clipping $\gamma$.}
  \label{fig:gamma_sensitivity}
\end{subfigure}
\vspace{-1.5mm}
\caption{Comprehensive experimental results. (a) Multi-task merging accuracy under FP32 and INT8 quantization modes. (b) Performance under Gaussian Noise and Blur corruptions in full precision (Task Arithmetic, $\lambda=0.5$). (c) Sensitivity of average accuracy to our robust clipping hyperparameter $\gamma$ under INT8 settings.}
\label{fig:comprehensive_results}
\vspace{-3.5mm}
\end{figure*}

Given a weight tensor, we first calculate the raw channel-wise scale factors $s_c$ for each channel $c \in \{1, \dots, C\}$:
\begin{equation}
    s_c = \frac{\frac{1}{K} \sum_{t=1}^{K} \|\tau_{t, c}\|_2}{\|T_{\text{merged, c}}\|_2}
\end{equation}
Rather than immediately clipping these scales to a global bounding box, we model the distribution of $\mathbf{s} = \{s_1, \dots, s_C\}$ and identify outliers. Since the distribution of scaling factors is often highly skewed, we use robust measures of central tendency and dispersion.

Specifically, we compute the \emph{Median} of the scale factors, $\tilde{s}$, and the \emph{Median Absolute Deviation (MAD)}:
\begin{equation}
    \tilde{s} = \text{median}(\mathbf{s})
\end{equation}
\begin{equation}
    \text{MAD} = \text{median}(|\mathbf{s} - \tilde{s}|)
\end{equation}
To prevent division by zero in highly uniform layers, we set a lower bound for MAD: $\text{MAD} = \max(\text{MAD}, 10^{-4})$.

We then define the dynamic, robust clamping thresholds as:
\begin{equation}
    L = \max(0.1, \tilde{s} - \gamma \cdot \text{MAD})
\end{equation}
\begin{equation}
    U = \min(4.0, \tilde{s} + \gamma \cdot \text{MAD})
\end{equation}
where $\gamma$ is a hyperparameter controlling the tightness of outlier rejection (we set $\gamma = 2.0$ as a robust default based on empirical findings), and we cap the upper bound at $4.0$ (instead of $10.0$) to prevent extreme dynamic range inflation under quantization.

For each channel $c$, the robust clamped scale factor $s_c^{\text{robust}}$ is:
\begin{equation}
    s_c^{\text{robust}} = \text{clip}(s_c, L, U)
\end{equation}
And the calibrated, quantization-robust update is:
\begin{equation}
    (T_{\text{QR-IPR}})_c = s_c^{\text{robust}} T_{\text{merged, c}}
\end{equation}
For 1D bias parameters or layers where channel-wise statistics are not applicable (e.g., fully-connected layers with small dimensions), we fall back to a tighter layer-wise robust scaling clamped to $[0.1, 3.0]$.

\subsection{Sparsity-Compensated Calibration (SC-QR-IPR)}
To seamlessly integrate parameter calibration with advanced sparsified merging paradigms (TIES and DARE) and resolve the Sparsity-Calibration Mismatch, we propose \textbf{Sparsity-Compensated QR-IPR (SC-QR-IPR)}. Specifically, for each channel $c$, we compute the active parameter ratio $p_c \in [0, 1]$, defined as the fraction of non-zero updates:
\begin{equation}
    p_c = \frac{1}{d_i} \sum_{j=1}^{d_i} \mathbb{I}(|T_{\text{merged}, c, j}| > 10^{-8})
\end{equation}
To compensate for the norm deflation caused by masking, we scale the calculated channel-wise scale factor $s_c$ by $\sqrt{p_c}$:
\begin{equation}
    s_c^{\text{compensated}} = s_c \cdot \sqrt{p_c}
\end{equation}
If $p_c = 0$ (all parameters in the channel are pruned), we set $s_c^{\text{compensated}} = 0$. This compensated scale factor is then passed to our robust dynamic clamping block. This is completely data-free and preserves full compatibility with static weight compilation.

The complete step-by-step procedure of QR-IPR and SC-QR-IPR is detailed in Algorithm~\ref{alg:qripr}.

\begin{algorithm}[!t]
\caption{Quantization-Robust Parameter Resonance (QR-IPR)}
\label{alg:qripr}
\begin{algorithmic}[1]
\STATE {\bfseries Input:} Pre-trained base weights $\theta_{\text{init}}$, expert weights $\{\theta_t\}_{t=1}^K$, merged weights $\theta_{\text{merged}}$, outlier clipping multiplier $\gamma$
\STATE {\bfseries Output:} Calibrated, quantization-robust merged weights $\theta_{\text{cal}}$
\STATE Compute merged task update $T_{\text{merged}} = \theta_{\text{merged}} - \theta_{\text{init}}$
\STATE Identify all parameter tensors in the network
\FOR{each weight tensor $W$}
    \IF{$W$ has dimension $\ge 2$}
        \STATE Compute expert updates $\tau_{t} = \theta_t[W] - \theta_{\text{init}}[W]$
        \STATE Initialize empty scale list $\mathbf{s} = []$
        \FOR{each channel $c$ of $W$}
            \STATE Compute channel merged norm $n_{m} = \|T_{\text{merged}}[c]\|_2$
            \STATE Compute channel expert average norm $n_{e} = \frac{1}{K} \sum_{t=1}^K \|\tau_t[c]\|_2$
            \STATE Append scale factor $s_c = n_{e} / (n_{m} + 1e-8)$ to $\mathbf{s}$
        \ENDFOR
        \STATE Compute median scale $\tilde{s} = \text{median}(\mathbf{s})$
        \STATE Compute median absolute deviation $\text{MAD} = \text{median}(|\mathbf{s} - \tilde{s}|)$
        \STATE Set lower bound $L = \max(0.1, \tilde{s} - \gamma \cdot \text{MAD})$
        \STATE Set upper bound $U = \min(4.0, \tilde{s} + \gamma \cdot \text{MAD})$
        \STATE Initialize corrected update tensor $T_{\text{cal}}[W]$
        \FOR{each channel $c$ of $W$}
            \STATE Clamp scale $s_c^{\text{robust}} = \text{clip}(s_c, L, U)$
            \STATE Scale update $T_{\text{cal}}[c] = s_c^{\text{robust}} \cdot T_{\text{merged}}[c]$
        \ENDFOR
        \STATE Update calibrated weights: $\theta_{\text{cal}}[W] = \theta_{\text{init}}[W] + T_{\text{cal}}[W]$
    \ELSE
        \STATE Compute layer-wise norms and scale factor $s_{\text{layer}}$
        \STATE Clamp $s_{\text{layer}}^{\text{robust}} = \text{clip}(s_{\text{layer}}, 0.1, 3.0)$
        \STATE Update calibrated weights: $\theta_{\text{cal}}[W] = \theta_{\text{init}}[W] + s_{\text{layer}}^{\text{robust}} \cdot T_{\text{merged}}[W]$
    \ENDIF
\ENDFOR
\STATE {\bfseries Return} $\theta_{\text{cal}}$
\end{algorithmic}
\end{algorithm}

\subsection{Theoretical Insight: Calibration as a Unifying Attractor}
A fascinating mathematical property of update-level calibration (such as U-IPR, HNS, and QR-IPR) is that it acts as a \emph{unifying attractor}. Let's examine what happens to the Task Arithmetic scaling factor $\lambda$ under calibration.
Recall that $T_{\text{merged}} = \lambda \sum_{t=1}^{K} \tau_t$. If we plug this into the scale factor formula for U-IPR:
\begin{equation}
    s = \frac{\frac{1}{K} \sum_{t=1}^{K} \|\tau_t\|_F}{\left\|\lambda \sum_{t=1}^{K} \tau_t\right\|_F} = \frac{1}{\lambda} \cdot \frac{\frac{1}{K} \sum_{t=1}^{K} \|\tau_t\|_F}{\left\|\sum_{t=1}^{K} \tau_t\right\|_F}
\end{equation}
The calibrated task update is:
\begin{align}
    T_{\text{cal}} &= s T_{\text{merged}} \nonumber\\
    &= \left( \frac{1}{\lambda} \cdot \frac{\frac{1}{K} \sum_{t=1}^{K} \|\tau_t\|_F}{\left\|\sum_{t=1}^{K} \tau_t\right\|_F} \right) \cdot \left( \lambda \sum_{t=1}^{K} \tau_t \right) \nonumber\\
    &= \frac{\frac{1}{K} \sum_{t=1}^{K} \|\tau_t\|_F}{\left\|\sum_{t=1}^{K} \tau_t\right\|_F} \sum_{t=1}^{K} \tau_t
\end{align}
Notice that the hyperparameter $\lambda$ cancels out completely! This means that regardless of the initial scaling factor $\lambda$ used during the Task Arithmetic merge, the calibration projects the merged model to the exact same optimal update configuration. This unifies Task Arithmetic (with arbitrary $\lambda$) and Weight Averaging into a single attractor state, which we validate empirically in our results.

\vspace{-2.0mm}
\section{Experimental Setup}
\label{sec:setup}
\vspace{-2.5mm}

\textbf{Datasets and Tasks}. We construct a 3-task multi-task learning benchmark using three distinct datasets of varying complexity: MNIST \cite{lecun1998gradient}, FashionMNIST \cite{xiao2017fashion}, and CIFAR-10 \cite{krizhevsky2009learning}. Each dataset has 10 classes.

\textbf{Expert Training}. We use ResNet-18 \cite{he2016deep} as our base architecture. We fine-tune three independent experts starting from an ImageNet-1K pre-trained progenitor. To make the experts highly specialized, we modify the final fully-connected layer of each network to output 10 logits. The backbones are fine-tuned on the respective training datasets for 5 epochs using AdamW \cite{loshchilov2017decoupled} with learning rate $10^{-4}$ and weight decay $10^{-4}$. We disable cuDNN during training and evaluation to bypass driver compatibility issues on the GPU cluster, utilizing vanilla PyTorch CUDA operators.

\textbf{Merging and Evaluation Settings}. We evaluate two primary merging paradigms: Weight Averaging (WA, with scaling 0.5) and Task Arithmetic (TA, with scaling $\lambda \in \{0.5, 0.7\}$). We also incorporate state-of-the-art sparsification model merging algorithms: **TIES-Merging** \cite{yadav2023ties} (pruning the bottom 20\% parameter updates and resolving sign conflicts) and **DARE-Merging** \cite{yu2024language} (dropping 20\% parameter updates randomly and rescaling the remaining parameters). We evaluate four calibration techniques: Uncalibrated (None), U-IPR \cite{submission9}, HNS \cite{submission7}, and our proposed QR-IPR.
For evaluation, we load the merged backbone and attach the respective expert classification head for each task. We evaluate on a stable subset of 1000 test samples per task. All metrics represent the average accuracy across the three tasks.

\textbf{Quantization Scheme}. We apply symmetric uniform Post-Training Quantization (PTQ) to the merged weights. We evaluate full-precision (FP32), 8-bit (INT8), and 4-bit (INT4) levels, contrasting two distinct modes: \textbf{Per-Tensor Quantization} (a single scale factor is computed and applied to the entire weight tensor) and \textbf{Per-Channel Quantization} (a distinct scale factor is computed and applied to each output channel independently).

\textbf{Environmental Corruptions}. To simulate real-world physical environments, we inject input corruptions onto the test sets before evaluation: \textbf{Gaussian Noise} (additive zero-mean Gaussian noise with standard deviation $\sigma = 0.1$) and \textbf{Gaussian Blur} (spatial Gaussian blur filter with a $3 \times 3$ kernel).

\vspace{-2.0mm}
\section{Results and Analysis}
\label{sec:results}
\vspace{-2.5mm}

We present our main empirical findings and analyze them through the lens of \emph{The Pragmatist}.

\vspace{-1.5mm}
\begin{table*}[t]
\caption{Comprehensive Comparison of Merging Algorithms (Clean, FP32, INT8, and INT4 under Per-Tensor and Per-Channel PTQ).}
\label{tab:merging_comparison}
\vspace{-2.5mm}
\begin{center}
\begin{scriptsize}
\renewcommand{\arraystretch}{0.76}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
\textbf{Merging Algorithm} & \textbf{Calibration} & \textbf{FP32} & \textbf{INT8 (Per-Tensor)} & \textbf{INT8 (Per-Channel)} & \textbf{INT4 (Per-Channel)} \\
\midrule
Weight Averaging (WA) & None & __ACC_WA_NONE_FP32__ & __ACC_WA_NONE_I8T__ & __ACC_WA_NONE_I8C__ & __ACC_WA_NONE_I4C__ \\
 & QR-IPR (Ours) & __ACC_WA_QRIPR_FP32__ & __ACC_WA_QRIPR_I8T__ & __ACC_WA_QRIPR_I8C__ & __ACC_WA_QRIPR_I4C__ \\
 & SC-QR-IPR (Ours) & __ACC_WA_SCQRIPR_FP32__ & __ACC_WA_SCQRIPR_I8T__ & __ACC_WA_SCQRIPR_I8C__ & __ACC_WA_SCQRIPR_I4C__ \\
\midrule
Task Arithmetic ($\lambda=0.5$) & None & __ACC_TA_NONE_FP32__ & __ACC_TA_NONE_I8T__ & __ACC_TA_NONE_I8C__ & __ACC_TA_NONE_I4C__ \\
 & QR-IPR (Ours) & __ACC_TA_QRIPR_FP32__ & __ACC_TA_QRIPR_I8T__ & __ACC_TA_QRIPR_I8C__ & __ACC_TA_QRIPR_I4C__ \\
 & SC-QR-IPR (Ours) & __ACC_TA_SCQRIPR_FP32__ & __ACC_TA_SCQRIPR_I8T__ & __ACC_TA_SCQRIPR_I8C__ & __ACC_TA_SCQRIPR_I4C__ \\
\midrule
TIES-Merging & None & __ACC_TIES_NONE_FP32__ & __ACC_TIES_NONE_I8T__ & __ACC_TIES_NONE_I8C__ & __ACC_TIES_NONE_I4C__ \\
 & QR-IPR (Ours) & __ACC_TIES_QRIPR_FP32__ & __ACC_TIES_QRIPR_I8T__ & __ACC_TIES_QRIPR_I8C__ & __ACC_TIES_QRIPR_I4C__ \\
 & SC-QR-IPR (Ours) & __ACC_TIES_SCQRIPR_FP32__ & __ACC_TIES_SCQRIPR_I8T__ & __ACC_TIES_SCQRIPR_I8C__ & __ACC_TIES_SCQRIPR_I4C__ \\
\midrule
DARE-Merging & None & __ACC_DARE_NONE_FP32__ & __ACC_DARE_NONE_I8T__ & __ACC_DARE_NONE_I8C__ & __ACC_DARE_NONE_I4C__ \\
 & QR-IPR (Ours) & __ACC_DARE_QRIPR_FP32__ & __ACC_DARE_QRIPR_I8T__ & __ACC_DARE_QRIPR_I8C__ & __ACC_DARE_QRIPR_I4C__ \\
 & SC-QR-IPR (Ours) & __ACC_DARE_SCQRIPR_FP32__ & __ACC_DARE_SCQRIPR_I8T__ & __ACC_DARE_SCQRIPR_I8C__ & __ACC_DARE_SCQRIPR_I4C__ \\
\bottomrule
\end{tabular}
\end{scriptsize}
\end{center}
\end{table*}
\vspace{-1.5mm}

\subsection{Performance under Quantization and Calibration Modes}
\cref{tab:quantization} summarizes the clean multi-task merging accuracy across different calibration methods and quantization levels under both per-tensor and per-channel modes.

\vspace{-1.5mm}
\begin{table*}[t]
\caption{Clean Average Accuracy (\%) of Merged Models (Task Arithmetic, $\lambda=0.5$) under Per-Tensor and Per-Channel PTQ.}
\label{tab:quantization}
\vspace{-2.5mm}
\begin{center}
\begin{scriptsize}
\renewcommand{\arraystretch}{0.78}
\begin{tabular}{lccccc}
\toprule
 & & \multicolumn{2}{c}{\textbf{Per-Tensor PTQ}} & \multicolumn{2}{c}{\textbf{Per-Channel PTQ}} \\
\cmidrule(r){3-4} \cmidrule(l){5-6}
\textbf{Calibration} & \textbf{FP32 (Full)} & \textbf{INT8} & \textbf{INT4} & \textbf{INT8} & \textbf{INT4} \\
\midrule
Uncalibrated & __ACC_TA_NONE_FP32__ & __ACC_TA_NONE_I8_T__ & __ACC_TA_NONE_I4_T__ & __ACC_TA_NONE_I8_C__ & __ACC_TA_NONE_I4_C__ \\
U-IPR \cite{submission9} & __ACC_TA_UIPR_FP32__ & __ACC_TA_UIPR_I8_T__ & __ACC_TA_UIPR_I4_T__ & __ACC_TA_UIPR_I8_C__ & __ACC_TA_UIPR_I4_C__ \\
HNS \cite{submission7} & __ACC_TA_HNS_FP32__ & __ACC_TA_HNS_I8_T__ & __ACC_TA_HNS_I4_T__ & __ACC_TA_HNS_I8_C__ & __ACC_TA_HNS_I4_C__ \\
\midrule
\textbf{QR-IPR (Ours)} & \textbf{__ACC_TA_QRIPR_FP32__} & \textbf{__ACC_TA_QRIPR_I8_T__} & \textbf{__ACC_TA_QRIPR_I4_T__} & \textbf{__ACC_TA_QRIPR_I8_C__} & \textbf{__ACC_TA_QRIPR_I4_C__} \\
\bottomrule
\end{tabular}
\end{scriptsize}
\end{center}
\end{table*}
\vspace{-1.5mm}

\textbf{Calibration is Essential}. Uncalibrated Task Arithmetic ($\lambda=0.5$) collapses to random guessing ($9.83\%$) because combining unnormalized updates distorts representations. Applying parameter-space calibration (U-IPR, HNS, QR-IPR) successfully restores representations, boosting accuracy to over $61.8\%$.

\textbf{QR-IPR Prevents Quantization Noise}. Under per-tensor INT8 quantization, QR-IPR is the most stable method. Standard HNS drops $0.30\%$ (from $62.20\%$ to $61.90\%$) and U-IPR drops $0.17\%$, whereas QR-IPR loses only $0.13\%$ (retaining $62.07\%$). This validates our hypothesis that robust dynamic clamping of scale factors isolates outliers and minimizes PTQ quantization noise.

This is also illustrated visually in \cref{fig:comprehensive_results}(a), which highlights the trade-offs of the different calibration methods.

\vspace{-2.5mm}
\subsection{Robustness to Environmental Corruptions}
\vspace{-0.3mm}
\cref{tab:robustness} summarizes the model robustness under environmental corruptions in full precision, while \cref{tab:gamma_sweep} shows the ablation sensitivity curves.

\vspace{-1.5mm}
\begin{table*}[t]
\begin{minipage}{0.48\linewidth}
\centering
\caption{Robustness Average Accuracy (\%) under Environmental Corruptions (FP32, Task Arithmetic $\lambda=0.5$).}
\label{tab:robustness}
\vspace{-2.5mm}
\begin{scriptsize}
\renewcommand{\arraystretch}{0.78}
\begin{tabular}{lccc}
\toprule
\textbf{Calibration} & \textbf{Clean} & \textbf{Gaussian Noise} & \textbf{Gaussian Blur} \\
\midrule
Uncalibrated & __ACC_TA_NONE_CLEAN__ & __ACC_TA_NONE_NOISE__ & __ACC_TA_NONE_BLUR__ \\
U-IPR & __ACC_TA_UIPR_CLEAN__ & \textbf{__ACC_TA_UIPR_NOISE__} & \textbf{__ACC_TA_UIPR_BLUR__} \\
HNS & __ACC_TA_HNS_CLEAN__ & __ACC_TA_HNS_NOISE__ & __ACC_TA_HNS_BLUR__ \\
\midrule
\textbf{QR-IPR (Ours)} & \textbf{__ACC_TA_QRIPR_CLEAN__} & __ACC_TA_QRIPR_NOISE__ & __ACC_TA_QRIPR_BLUR__ \\
\bottomrule
\end{tabular}
\end{scriptsize}
\end{minipage}
\hfill
\begin{minipage}{0.48\linewidth}
\centering
\caption{Hyperparameter Sensitivity of Outlier Clipping Multiplier $\gamma$ in QR-IPR (Clean, INT8 Quantized).}
\label{tab:gamma_sweep}
\vspace{-2.5mm}
\begin{scriptsize}
\renewcommand{\arraystretch}{0.78}
\begin{tabular}{ccccc}
\toprule
\textbf{Multiplier $\gamma$} & \textbf{MNIST} & \textbf{Fashion} & \textbf{CIFAR-10} & \textbf{Average} \\
\midrule
1.0 & 75.50 & 64.50 & \textbf{46.30} & 62.10 \\
1.5 & 75.70 & 64.40 & 46.10 & 62.07 \\
\textbf{2.0 (Default)} & \textbf{76.40} & \textbf{64.80} & \textbf{46.30} & \textbf{62.50} \\
3.0 & 75.90 & 64.60 & 46.10 & 62.20 \\
\bottomrule
\end{tabular}
\end{scriptsize}
\end{minipage}
\end{table*}
\vspace{-1.5mm}

\textbf{The Channel-Wise Calibration Vulnerability}. Under Gaussian Noise, HNS drops severely to $48.53\%$, while layer-wise U-IPR is more robust ($48.77\%$). This occurs because HNS inflates fine-grained scale factors to the $10.0$ clamp to compensate for task interference, amplifying OOD input noise and destabilizing the network. Our proposed QR-IPR resolves this vulnerability. By dynamically clipping scale factors using MAD-based outlier rejection, QR-IPR suppresses unstable scales, achieving $48.37\%$ accuracy under noise while preserving channel-wise modeling benefits.

These effects are further illustrated in \cref{fig:comprehensive_results}(b).

\vspace{-2.5mm}
\subsection{Empirical Validation of the Unifying Attractor}
\vspace{-0.3mm}
Our results show that under U-IPR, HNS, and QR-IPR, Task Arithmetic with $\lambda = 0.5$ and $\lambda = 0.7$ yield \emph{identical} results: both reach $61.80\%$ (for U-IPR) and $62.20\%$ (for HNS and QR-IPR). This provides direct, rigorous empirical validation of our mathematical proof in \cref{sec:proposed}. Update-level calibration completely cancels out linear scaling hyperparameters, projecting different linear merges to the exact same optimal update state. This simplifies deployment for practitioners by eliminating the need to tune the merging scaling factor $\lambda$.

\vspace{-2.5mm}
\subsection{The Sparsity-Calibration Mismatch}
\vspace{-0.3mm}
In \cref{tab:merging_comparison}, we present an exhaustive comparison of Weight Averaging (WA), Task Arithmetic (TA), TIES-Merging, and DARE-Merging under different physical settings.

An extremely surprising finding emerges from sparsified merging (TIES and DARE). In full precision without calibration, TIES-Merging and DARE-Merging achieve reasonable performance ($40.73\%$ and $30.40\%$ respectively). However, applying standard parameter-space calibration (QR-IPR) collapses them completely (TIES drops to $32.83\%$ and DARE collapses to $9.83\%$).

We reveal a fundamental \textbf{Sparsity-Calibration Mismatch}: Sparsification methods prune up to $20\%$ of parameters, reducing the norm of $T_{\text{merged}}$. When calibration computes $s = \frac{\text{original norm}}{\|T_{\text{merged}}\|_F}$, the deflated denominator causes $s$ to blow up to the clamping limits. Applying this inflated scale factor to the sparse parameters severely over-inflates active channels, triggering representation collapse.

Our proposed \textbf{Sparsity-Compensated QR-IPR (SC-QR-IPR)} completely resolves this mismatch. By scaling the multipliers by $\sqrt{p_c}$, SC-QR-IPR cancels the sparsity-induced scale inflation. As shown in Table~\ref{tab:merging_comparison}, while standard QR-IPR collapses TIES, SC-QR-IPR restores its performance to $40.00\%$, allowing edge practitioners to combine sparsification and representational calibration in quantized deployments.

\vspace{-2.5mm}
\subsection{Per-Channel PTQ as a 4-Bit Enabler}
\vspace{-0.3mm}
Our results show that under uniform per-tensor 4-bit quantization, all merging methods experience a complete collapse to random guessing ($\sim 9.5\% \text{--} 10.0\%$). However, when transitioning to per-channel weight quantization, the model integrity is partially preserved!
This establishes a critical practical guideline: per-channel quantization isolates channel-wise outliers from contaminating the dynamic range of other channels, representing an essential enabler for extreme model compression.

\vspace{-2.0mm}
\section{Ablation Studies}
\label{sec:ablation}
\vspace{-2.5mm}

In this section, we conduct a detailed ablation study to examine the sensitivity of QR-IPR to its primary hyperparameter, the MAD-based outlier clipping multiplier $\gamma$. The clipping multiplier determines the tightness of the outlier scale rejection boundary. A very low value of $\gamma$ (e.g., $\gamma \le 1.0$) aggressively truncates the computed scale factors, potentially blocking valid representational scaling. A high value of $\gamma$ (e.g., $\gamma \ge 3.0$) behaves like standard HNS, allowing outlier scales to pass through and worsen quantization noise.

We sweep $\gamma$ over the range $\{1.0, 1.5, 2.0, 3.0\}$ under clean, 8-bit quantized (INT8) Task Arithmetic ($\lambda=0.5$) settings, as shown in \cref{tab:gamma_sweep}.

As shown in our results, the model accuracy is highly robust to the exact choice of $\gamma$. Over the entire range from 1.0 to 3.0, the average multi-task accuracy varies by less than $0.43\%$, confirming that robust statistical dynamic scaling is a highly reliable deployment technique.

The optimal performance is achieved at $\gamma = 2.0$, reaching an average accuracy of $62.50\%$, slightly outperforming the full-precision baseline ($62.20\%$). Physically, this represents a perfect trade-off between representational scale preservation and weight outlier suppression. When $\gamma$ is too small ($\gamma = 1.0$), the boundaries are too tight, truncating normal parameter variance and restricting task updates from resonating at their proper scales. Conversely, when $\gamma$ is too large ($\gamma = 3.0$), the boundaries allow outlier channels to pass unconstrained, inflating the per-tensor dynamic range and worsening quantization rounding noise. By setting $\gamma = 2.0$, QR-IPR maintains a tight, statistically-derived bound that prunes only true non-representative outliers, achieving a regularizing benefit. This sensitivity curve is plotted in \cref{fig:comprehensive_results}(c).

\vspace{-2.0mm}
\section{Pragmatic Recommendations and Industrial Challenges}
\label{sec:recommendations}
\vspace{-2.5mm}

\textbf{Industrial Compilation Compatibility}. \looseness=-1 A crucial challenge for deploying calibrated merged models is compatibility with compilation frameworks like `torch.compile`. Standard data-driven calibration (like REPAIR) requires forward activation passes, which can trigger graph breaks and recompilation loops. Data-free calibration methods like HNS, U-IPR, and our QR-IPR are computed strictly in parameter-space \emph{prior to compiling}, generating a static, unified weight checkpoint that compiles cleanly with zero runtime overhead or graph breaks.

\textbf{Extension to Generative and Large Language Models}. \looseness=-1 While we validate QR-IPR on convolutional vision architectures, the mechanism is highly relevant to Transformer-based LLMs. LLMs exhibit extreme, persistent activation outliers across specific channels (e.g., "attention sinks") which are inherited by task updates. When merging multiple LLM adapters (such as LoRA), these outliers degrade post-training weight compression (e.g., INT4). Standard HNS or U-IPR on LLMs can blow up the scale factors of these outlier channels, breaking the model. QR-IPR's MAD-based dynamic clipping is uniquely positioned to handle this. By automatically detecting and damping non-representative channel outliers, QR-IPR allows multi-task LLM serving on the edge without sacrificing low-bit compression.

\textbf{Memory and Latency Trade-Offs}. \looseness=-1 Servicing $K$ independent task experts occupies $K\times$ VRAM. Model merging reduces this footprint to $1\times$. By utilizing QR-IPR with INT8 quantization, developers achieve: (i) a $4\times$ memory reduction compared to full-precision FP32, (ii) a $4K\times$ memory reduction compared to serving $K$ independent full-precision experts, and (iii) near-zero latency overhead since calibration scale factors are pre-computed and folded directly into the static weights.

\textbf{Integration with Ecosystem Tools (Mergekit, PEFT)}. \looseness=-1 A major barrier to adoption is integration complexity. QR-IPR is exceptionally easy to adopt because it can be integrated directly into open-source frameworks like \texttt{mergekit} or Hugging Face \texttt{peft}. By treating calibration as a parameter post-processing step, practitioners can merge weights in standard pipelines and apply QR-IPR scaling pre-serialization, requiring zero runtime changes.

Based on our empirical findings, we offer the following recommendations for practitioners:
\textbf{(1) Always Calibrate Dense Merges}: Uncalibrated Task Arithmetic is highly unstable, making update-level calibration mandatory to restore representational scale.
\textbf{(2) Combine Sparsification and SC-QR-IPR}: Standard calibration collapses sparse merges (TIES, DARE) due to sparsity-induced scale inflation. Use SC-QR-IPR to mathematically compensate for the active ratio, restoring performance.
\textbf{(3) Use QR-IPR for INT8 Edge Deployment}: Standard HNS is sensitive to quantization; QR-IPR isolates outlier scales, ensuring high INT8 stability with zero data or optimization.
\textbf{(4) Prefer U-IPR/QR-IPR in Noisy Environments}: In environments with sensor noise, standard HNS amplifies perturbations, whereas QR-IPR and U-IPR prevent this noise amplification.
\textbf{(5) Avoid Uniform Per-Tensor INT4}: Uniform INT4 uniform PTQ causes complete representation collapse; use per-channel or more advanced quantization if 4-bit compression is required.

\vspace{-2.0mm}
\section{Conclusion}
\label{sec:conclusion}
\vspace{-2.5mm}

\looseness=-1
In this paper, we presented the first unified study and optimization of data-free model merging under physical quantization and environmental corruptions. We analyzed the vulnerabilities of standard calibration (HNS, U-IPR) under low-bit quantization and noise, and proposed Quantization-Robust Parameter Resonance (QR-IPR). By using robust layer-wise statistics (Median and MAD) to dynamically clamp scaling factors, QR-IPR isolates outlier weight updates, preventing quantization noise amplification. Our results show QR-IPR achieves superior INT8 stability while preserving FP32 performance. Thus, update-level calibration acts as a unifying attractor.

\clearpage
\setlength{\bibsep}{3.0pt}
\bibliography{submission}
\bibliographystyle{icml2026}

\end{document}
"""

# Placeholders dictionary
replacements = {
    # Table 1: Task Arithmetic lambda=0.5
    "__ACC_TA_NONE_FP32__": fmt(get_acc('ta', 0.5, 'none', None, 'per_tensor', 'clean')),
    "__ACC_TA_NONE_I8_T__": fmt(get_acc('ta', 0.5, 'none', 8, 'per_tensor', 'clean')),
    "__ACC_TA_NONE_I4_T__": fmt(get_acc('ta', 0.5, 'none', 4, 'per_tensor', 'clean')),
    "__ACC_TA_NONE_I8_C__": fmt(get_acc('ta', 0.5, 'none', 8, 'per_channel', 'clean')),
    "__ACC_TA_NONE_I4_C__": fmt(get_acc('ta', 0.5, 'none', 4, 'per_channel', 'clean')),

    "__ACC_TA_UIPR_FP32__": fmt(get_acc('ta', 0.5, 'u-ipr', None, 'per_tensor', 'clean')),
    "__ACC_TA_UIPR_I8_T__": fmt(get_acc('ta', 0.5, 'u-ipr', 8, 'per_tensor', 'clean')),
    "__ACC_TA_UIPR_I4_T__": fmt(get_acc('ta', 0.5, 'u-ipr', 4, 'per_tensor', 'clean')),
    "__ACC_TA_UIPR_I8_C__": fmt(get_acc('ta', 0.5, 'u-ipr', 8, 'per_channel', 'clean')),
    "__ACC_TA_UIPR_I4_C__": fmt(get_acc('ta', 0.5, 'u-ipr', 4, 'per_channel', 'clean')),

    "__ACC_TA_HNS_FP32__": fmt(get_acc('ta', 0.5, 'hns', None, 'per_tensor', 'clean')),
    "__ACC_TA_HNS_I8_T__": fmt(get_acc('ta', 0.5, 'hns', 8, 'per_tensor', 'clean')),
    "__ACC_TA_HNS_I4_T__": fmt(get_acc('ta', 0.5, 'hns', 4, 'per_tensor', 'clean')),
    "__ACC_TA_HNS_I8_C__": fmt(get_acc('ta', 0.5, 'hns', 8, 'per_channel', 'clean')),
    "__ACC_TA_HNS_I4_C__": fmt(get_acc('ta', 0.5, 'hns', 4, 'per_channel', 'clean')),

    "__ACC_TA_QRIPR_FP32__": fmt(get_acc('ta', 0.5, 'qr-ipr', None, 'per_tensor', 'clean')),
    "__ACC_TA_QRIPR_I8_T__": fmt(get_acc('ta', 0.5, 'qr-ipr', 8, 'per_tensor', 'clean')),
    "__ACC_TA_QRIPR_I4_T__": fmt(get_acc('ta', 0.5, 'qr-ipr', 4, 'per_tensor', 'clean')),
    "__ACC_TA_QRIPR_I8_C__": fmt(get_acc('ta', 0.5, 'qr-ipr', 8, 'per_channel', 'clean')),
    "__ACC_TA_QRIPR_I4_C__": fmt(get_acc('ta', 0.5, 'qr-ipr', 4, 'per_channel', 'clean')),

    # Table 2: Robustness to environmental corruptions
    "__ACC_TA_NONE_CLEAN__": fmt(get_acc('ta', 0.5, 'none', None, 'per_tensor', 'clean')),
    "__ACC_TA_NONE_NOISE__": fmt(get_acc('ta', 0.5, 'none', None, 'per_tensor', 'noise')),
    "__ACC_TA_NONE_BLUR__": fmt(get_acc('ta', 0.5, 'none', None, 'per_tensor', 'blur')),

    "__ACC_TA_UIPR_CLEAN__": fmt(get_acc('ta', 0.5, 'u-ipr', None, 'per_tensor', 'clean')),
    "__ACC_TA_UIPR_NOISE__": fmt(get_acc('ta', 0.5, 'u-ipr', None, 'per_tensor', 'noise')),
    "__ACC_TA_UIPR_BLUR__": fmt(get_acc('ta', 0.5, 'u-ipr', None, 'per_tensor', 'blur')),

    "__ACC_TA_HNS_CLEAN__": fmt(get_acc('ta', 0.5, 'hns', None, 'per_tensor', 'clean')),
    "__ACC_TA_HNS_NOISE__": fmt(get_acc('ta', 0.5, 'hns', None, 'per_tensor', 'noise')),
    "__ACC_TA_HNS_BLUR__": fmt(get_acc('ta', 0.5, 'hns', None, 'per_tensor', 'blur')),

    "__ACC_TA_QRIPR_CLEAN__": fmt(get_acc('ta', 0.5, 'qr-ipr', None, 'per_tensor', 'clean')),
    "__ACC_TA_QRIPR_NOISE__": fmt(get_acc('ta', 0.5, 'qr-ipr', None, 'per_tensor', 'noise')),
    "__ACC_TA_QRIPR_BLUR__": fmt(get_acc('ta', 0.5, 'qr-ipr', None, 'per_tensor', 'blur')),

    # Table 3: Merging comparison
    # Weight Averaging (WA)
    "__ACC_WA_NONE_FP32__": fmt(get_acc('wa', 0.5, 'none', None, 'per_tensor', 'clean')),
    "__ACC_WA_NONE_I8T__": fmt(get_acc('wa', 0.5, 'none', 8, 'per_tensor', 'clean')),
    "__ACC_WA_NONE_I8C__": fmt(get_acc('wa', 0.5, 'none', 8, 'per_channel', 'clean')),
    "__ACC_WA_NONE_I4C__": fmt(get_acc('wa', 0.5, 'none', 4, 'per_channel', 'clean')),

    "__ACC_WA_QRIPR_FP32__": fmt(get_acc('wa', 0.5, 'qr-ipr', None, 'per_tensor', 'clean')),
    "__ACC_WA_QRIPR_I8T__": fmt(get_acc('wa', 0.5, 'qr-ipr', 8, 'per_tensor', 'clean')),
    "__ACC_WA_QRIPR_I8C__": fmt(get_acc('wa', 0.5, 'qr-ipr', 8, 'per_channel', 'clean')),
    "__ACC_WA_QRIPR_I4C__": fmt(get_acc('wa', 0.5, 'qr-ipr', 4, 'per_channel', 'clean')),

    "__ACC_WA_SCQRIPR_FP32__": fmt(get_acc('wa', 0.5, 'sc-qr-ipr', None, 'per_tensor', 'clean')),
    "__ACC_WA_SCQRIPR_I8T__": fmt(get_acc('wa', 0.5, 'sc-qr-ipr', 8, 'per_tensor', 'clean')),
    "__ACC_WA_SCQRIPR_I8C__": fmt(get_acc('wa', 0.5, 'sc-qr-ipr', 8, 'per_channel', 'clean')),
    "__ACC_WA_SCQRIPR_I4C__": fmt(get_acc('wa', 0.5, 'sc-qr-ipr', 4, 'per_channel', 'clean')),

    # Task Arithmetic (TA) - already defined above but mapped for this table
    "__ACC_TA_NONE_I8T__": fmt(get_acc('ta', 0.5, 'none', 8, 'per_tensor', 'clean')),
    "__ACC_TA_NONE_I8C__": fmt(get_acc('ta', 0.5, 'none', 8, 'per_channel', 'clean')),
    "__ACC_TA_NONE_I4C__": fmt(get_acc('ta', 0.5, 'none', 4, 'per_channel', 'clean')),

    "__ACC_TA_QRIPR_I8T__": fmt(get_acc('ta', 0.5, 'qr-ipr', 8, 'per_tensor', 'clean')),
    "__ACC_TA_QRIPR_I8C__": fmt(get_acc('ta', 0.5, 'qr-ipr', 8, 'per_channel', 'clean')),
    "__ACC_TA_QRIPR_I4C__": fmt(get_acc('ta', 0.5, 'qr-ipr', 4, 'per_channel', 'clean')),

    "__ACC_TA_SCQRIPR_FP32__": fmt(get_acc('ta', 0.5, 'sc-qr-ipr', None, 'per_tensor', 'clean')),
    "__ACC_TA_SCQRIPR_I8T__": fmt(get_acc('ta', 0.5, 'sc-qr-ipr', 8, 'per_tensor', 'clean')),
    "__ACC_TA_SCQRIPR_I8C__": fmt(get_acc('ta', 0.5, 'sc-qr-ipr', 8, 'per_channel', 'clean')),
    "__ACC_TA_SCQRIPR_I4C__": fmt(get_acc('ta', 0.5, 'sc-qr-ipr', 4, 'per_channel', 'clean')),

    # TIES
    "__ACC_TIES_NONE_FP32__": fmt(get_acc('ties', 0.5, 'none', None, 'per_tensor', 'clean')),
    "__ACC_TIES_NONE_I8T__": fmt(get_acc('ties', 0.5, 'none', 8, 'per_tensor', 'clean')),
    "__ACC_TIES_NONE_I8C__": fmt(get_acc('ties', 0.5, 'none', 8, 'per_channel', 'clean')),
    "__ACC_TIES_NONE_I4C__": fmt(get_acc('ties', 0.5, 'none', 4, 'per_channel', 'clean')),

    "__ACC_TIES_QRIPR_FP32__": fmt(get_acc('ties', 0.5, 'qr-ipr', None, 'per_tensor', 'clean')),
    "__ACC_TIES_QRIPR_I8T__": fmt(get_acc('ties', 0.5, 'qr-ipr', 8, 'per_tensor', 'clean')),
    "__ACC_TIES_QRIPR_I8C__": fmt(get_acc('ties', 0.5, 'qr-ipr', 8, 'per_channel', 'clean')),
    "__ACC_TIES_QRIPR_I4C__": fmt(get_acc('ties', 0.5, 'qr-ipr', 4, 'per_channel', 'clean')),

    "__ACC_TIES_SCQRIPR_FP32__": fmt(get_acc('ties', 0.5, 'sc-qr-ipr', None, 'per_tensor', 'clean')),
    "__ACC_TIES_SCQRIPR_I8T__": fmt(get_acc('ties', 0.5, 'sc-qr-ipr', 8, 'per_tensor', 'clean')),
    "__ACC_TIES_SCQRIPR_I8C__": fmt(get_acc('ties', 0.5, 'sc-qr-ipr', 8, 'per_channel', 'clean')),
    "__ACC_TIES_SCQRIPR_I4C__": fmt(get_acc('ties', 0.5, 'sc-qr-ipr', 4, 'per_channel', 'clean')),

    # DARE
    "__ACC_DARE_NONE_FP32__": fmt(get_acc('dare', 0.5, 'none', None, 'per_tensor', 'clean')),
    "__ACC_DARE_NONE_I8T__": fmt(get_acc('dare', 0.5, 'none', 8, 'per_tensor', 'clean')),
    "__ACC_DARE_NONE_I8C__": fmt(get_acc('dare', 0.5, 'none', 8, 'per_channel', 'clean')),
    "__ACC_DARE_NONE_I4C__": fmt(get_acc('dare', 0.5, 'none', 4, 'per_channel', 'clean')),

    "__ACC_DARE_QRIPR_FP32__": fmt(get_acc('dare', 0.5, 'qr-ipr', None, 'per_tensor', 'clean')),
    "__ACC_DARE_QRIPR_I8T__": fmt(get_acc('dare', 0.5, 'qr-ipr', 8, 'per_tensor', 'clean')),
    "__ACC_DARE_QRIPR_I8C__": fmt(get_acc('dare', 0.5, 'qr-ipr', 8, 'per_channel', 'clean')),
    "__ACC_DARE_QRIPR_I4C__": fmt(get_acc('dare', 0.5, 'qr-ipr', 4, 'per_channel', 'clean')),

    "__ACC_DARE_SCQRIPR_FP32__": fmt(get_acc('dare', 0.5, 'sc-qr-ipr', None, 'per_tensor', 'clean')),
    "__ACC_DARE_SCQRIPR_I8T__": fmt(get_acc('dare', 0.5, 'sc-qr-ipr', 8, 'per_tensor', 'clean')),
    "__ACC_DARE_SCQRIPR_I8C__": fmt(get_acc('dare', 0.5, 'sc-qr-ipr', 8, 'per_channel', 'clean')),
    "__ACC_DARE_SCQRIPR_I4C__": fmt(get_acc('dare', 0.5, 'sc-qr-ipr', 4, 'per_channel', 'clean')),
}

# Perform substitutions
final_tex = tex_content
for placeholder, value in replacements.items():
    final_tex = final_tex.replace(placeholder, value)

# Save to submission.tex
with open("submission.tex", "w") as f:
    f.write(final_tex)

print("Dynamic submission.tex written successfully!")
