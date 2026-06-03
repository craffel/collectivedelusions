import os
import random
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

persona_text = """
You are "The Empiricist", an empirically driven researcher who is extremely good at running tons of experiments. Your research philosophy is that true progress in machine learning comes from exhaustive empirical validation and large-scale experimentation. You do not trust an idea until it has been rigorously tested across many datasets, hyperparameters, and seeds.

Key Traits:
* You prioritize comprehensive evaluation over minor theoretical justifications.
* You are a master of scaling up experiments and running massive parallel sweeps.
* You believe the best papers present overwhelming empirical evidence for their claims.
* You always include robust ablation studies to verify every component of a proposed method.

Behavioral Instructions:
When generating ideas, focus on those that can be validated through extensive experimentation.
When designing your methodology, plan for large-scale parallel experiments using available compute.
In your writing, emphasize the breadth and depth of your empirical results.
"""

context_text = """
We have read three recent papers in our workspace:
1. "Demystifying Orthogonal Model Merging: Is Manifold Geometry Doing the Heavy Lifting?"
   - Critique of Orthogonal Model Merging (OrthoMerge). Shows that OrthoMerge's orthogonal component is actually tiny, and Task Arithmetic dominates it. OrthoMerge introduces 88x computational overhead due to SVD. Proposes DMC (Decoupled Magnitude-Corrected) merging.
2. "Pragmatic Multi-Task Model Merging via Task-Conditional Activation Calibration" (TCAC)
   - Proposes Task-Conditional Activation Calibration to fix activation variance collapse and representation shift in deeper layers of merged models, using a tiny calibration set.
3. "Deconstructing Test-Time Model Merging: Is Joint Optimization a Methodological Illusion?"
   - Criticizes joint test-time optimization of merging coefficients and heads (SyMerge). Shows that sequential optimization (coefficients and heads separately) or head-only adaptation works better and is numerically stable.

Our goal is to propose a novel, empirically robust, and significant idea in the domain of deep learning model merging (such as weight merging, representation/activation alignment, calibration, test-time adaptation).
The idea must be highly experimental and allow for a large number of runs, hyperparameter sweeps, multiple datasets (e.g. CIFAR-10, SVHN, FashionMNIST), and multiple seeds to showcase the empirical strength as "The Empiricist".
"""

prompt = f"""
{persona_text}

{context_text}

Please generate exactly 10 novel, high-quality, and detailed research ideas.
For each idea, provide:
1. Title
2. Core Hypothesis
3. Proposed Method
4. Experimental Design (Must be extensive, specifying datasets, hyperparameter sweeps, baseline methods to compare, and ablation studies. This must highlight our "Empiricist" persona!)
5. Expected Results & Potential Impact

Make sure the ideas are technically sound, highly relevant to model merging, and fully testable on ResNet-18 or similar architectures across CIFAR-10, SVHN, and FashionMNIST.

Format your output in clean Markdown.
"""

print("Calling Gemini API to generate ideas...")
model = genai.GenerativeModel('models/gemini-2.5-pro')
response = model.generate_content(prompt)
ideas_content = response.text

# Parse and select one idea based on a pseudo-random number generator
# Let's seed the random generator for reproducibility
random.seed(42)
selected_index = random.randint(1, 10)

selection_prompt = f"""
Here are the 10 ideas generated:

{ideas_content}

We must select Idea #{selected_index} based on our random number generator (seed 42, selection = {selected_index}).
Please write a comprehensive final research progress log. The log must contain:
1. A summary of the 10 brainstormed ideas.
2. The selection process and why Idea #{selected_index} was chosen.
3. The selected idea's title, hypothesis, proposed method, and detailed experimental design.
4. A concrete plan of action for executing this project (Phase 2: Experimentation) and writing the paper (Phase 3).
Ensure the tone of this log strongly reflects the persona of "The Empiricist" (valuing exhaustive empirical testing, sweeps, ablations, multiple seeds).

Format this output as the content for `progress.md`.
"""

print(f"Calling Gemini API to generate progress.md with selected Idea #{selected_index}...")
response_progress = model.generate_content(selection_prompt)
progress_content = response_progress.text

with open("progress.md", "w") as f:
    f.write(progress_content)

print("progress.md written successfully.")
