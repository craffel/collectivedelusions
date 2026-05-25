import os
import random
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def get_paper_summary(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    # Return first 8000 characters to get the Abstract, Intro, and key methods
    return content[:8000]

sub5 = get_paper_summary("papers/submission5.txt")
sub6 = get_paper_summary("papers/submission6.txt")
sub9 = get_paper_summary("papers/submission9.txt")

prompt = f"""
You are an expert AI researcher specializing in deep learning, specifically in weight-space model merging and Test-Time Model Merging (TTMM).
We are working on a conference paper submission for a prestigious machine learning conference (ICML/NeurIPS style).
The project centers on "Test-Time Model Merging (TTMM)" for non-stationary, open-world unlabeled test streams.

We have three papers that represent the state of the art and prior work in this domain:

1. **CP-AM (Contrastive Prototypes with Angular Margin)**:
{sub5}

2. **FDF-DPA (Fully Data-Free Dynamic Prototype Adaptation)**:
{sub6}

3. **BK-CoMerge (Bayesian Kronecker-Preconditioned Co-acting TTMM)**:
{sub9}

Your goal is to formulate exactly ten (10) novel, highly significant, and technically sound research ideas that extend or build upon these papers.
Each of the 10 ideas must include:
1. **Title**: A clear, professional title for the proposed method/paper.
2. **Hypothesis**: The core scientific or engineering hypothesis (what are we testing and why do we believe it will work?).
3. **Proposed Methodology**: Concrete, detailed steps on how to implement the method. Explain how we dynamically fuse experts, adapt routing, manage Batch Normalization, and precondition sensitivities on-the-fly.
4. **Expected Results & Impact**: What metrics will improve, why it represents a significant advance, and why it is likely to be accepted.

Ensure the ideas are feasible to implement and evaluate in our setting (which uses a SimpleCNN or ResNet-18 architecture on non-stationary vision streams like MNIST, FashionMNIST, and KMNIST with/without Gaussian noise or other corruptions).

Format the output strictly as markdown. The output should be a list of the 10 ideas.
"""

print("Generating 10 novel research ideas using Gemini...")
model = genai.GenerativeModel("gemini-2.5-pro")
response = model.generate_content(prompt)
ideas_text = response.text

# Parse and select an idea pseudo-randomly
random.seed(20260524) # Seed based on today's date
selected_idx = random.randint(1, 10)

print(f"Selected Idea Index: {selected_idx}")

# Write to progress.md
with open("progress.md", "a", encoding="utf-8") as f:
    f.write("\n# Phase 1: Foundation (Read & Formulate)\n\n")
    f.write("## Synthesized Themes, Contributions, and Limitations from SOTA Papers\n")
    f.write("- **Theme**: Test-Time Model Merging (TTMM) for non-stationary, unlabeled open-world streams.\n")
    f.write("- **Key Contributions**:\n")
    f.write("  1. **CP-AM** introduced Angular/Cosine space representation to overcome L2/Euclidean scale distortion under noise, using CosFace experts and Spherical Cosine Routing.\n")
    f.write("  2. **FDF-DPA** made TTMM data-free by updating prototypes on-the-fly using high-confidence predictions, posterior blending of BN running stats, and Kronecker-Trace guided preconditioning.\n")
    f.write("  3. **BK-CoMerge** unified dynamic Bayesian soft-routing with BN moment-matching buffer fusion, parameterizing merging coefficients as global consensus logits with layer offsets regulated by Kronecker curvature.\n")
    f.write("- **Limitations Identified**:\n")
    f.write("  - Current methods rely heavily on clean/noisy partition transitions but may struggle under gradual domain shifts or sudden, extreme covariate shifts.\n")
    f.write("  - Weight interpolation is prone to representation collapse when merging experts trained on completely distinct target domains (e.g., MNIST vs FashionMNIST) due to activation mismatch in non-linear layers.\n")
    f.write("  - Real-time computing of Kronecker traces or layer sensitivities can still be latency-prohibitive for larger models or fast-moving test streams.\n\n")
    
    f.write("## Ten Proposed Research Ideas\n\n")
    f.write(ideas_text)
    f.write(f"\n\n## Random Selection\n")
    f.write(f"Based on a pseudo-random number generator (seed 20260524), we selected **Idea #{selected_idx}** for execution.\n\n")

print("Ideas successfully logged to progress.md.")
