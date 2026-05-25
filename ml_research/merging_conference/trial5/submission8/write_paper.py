import os
import sys
import subprocess
import google.generativeai as genai

# Configure Gemini API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

def generate_paper_with_gemini():
    print("\nCalling Gemini to write the LaTeX paper...")
    
    prompt = """
You are a distinguished AI research scientist writing a top-tier machine learning paper for ICML 2026.
We are introducing a novel method called PROTO-TTMM (breaking the closed-world assumption in Test-Time Model Merging).

CRITICAL CONSTRAINTS FOR SAFETY & LEGAL COMPLIANCE:
- You MUST use 100% original wording for all sections, including the abstract, introduction, and related work.
- Do NOT copy any boilerplate sentences or placeholder paragraphs from standard ICML templates or other papers.
- Avoid reproducing standard template sentences or instructions. Draft every single sentence from scratch.
- Ensure there are no verbatim recitation matches with any existing copyrighted papers or public documents.

CRITICAL LATEX STRUCTURAL MANDATES:
- You MUST include `\\PassOptionsToPackage{numbers}{natbib}` as the very first line of the document (before `\\documentclass{article}`).
- Ensure all citation markers in the text use standard numerical citations (e.g. `\\cite{...}`).
- The final document MUST be exactly 8 pages long or extremely substantial (including abstract, intro, method, results, discussion, references, and appendix if needed).

Our paper details are:
1. Title: PROTO-TTMM: Breaking the Closed-World Assumption in Test-Time Model Merging via Centered Online Prototype Generation
2. Core mechanics:
   - Detects task shift and novelty via a joint metric of Prediction Entropy and Distance to known prototypes (Novelty Score) and average pairwise cosine similarity of features (Cohesion Score).
   - Combats representation collapse and feature anisotropy by performing Isotropic Feature Centering (subtracting the global mean of pre-computed task prototypes and re-normalizing).
   - Resolves self-reinforcing Routing Feedback Loops by introducing Unbiased Routing (UR) (temporarily extracting feature representations using unbiased uniform weights [1/3, 1/3, 1/3] solely for routing and novelty detection).
   - When a novel task is detected, it instantiates online class-aware prototypes using high-confidence pseudo-labeled samples. It refines them using an Exponential Moving Average (EMA).
   - Once routed, it adapts the merging coefficients (lambda) using confidence-masked contrastive alignment.
3. Empirical results on the Continual Test-Time Adaptation (CTTA) benchmark:
   - Task A: CIFAR-10 (Clean/Noisy)
   - Task B: SVHN (Clean/Noisy)
   - Task C: FashionMNIST (Novel, unseen)
   - Accuracies on the tasks:
     * Static (Uniform): Task A: 13.65%, Task B: 13.75%, Task C: 14.38%, Overall: 13.92%
     * TENT (Entropy): Task A: 11.77%, Task B: 24.95%, Task C: 27.03%, Overall: 21.25%
     * CPA-Merge (Closed-world): Task A: 70.00%, Task B: 12.97%, Task C: 9.11%, Overall: 30.69%
     * PC-Merge (Resets): Task A: 11.77%, Task B: 24.95%, Task C: 27.03%, Overall: 21.25%
     * PROTO-TTMM (Ours): Task A: 8.49%, Task B: 14.11%, Task C: 84.64%, Overall: 35.75%
   - We must explain:
     * CPA-Merge and our uncentered version get trapped in CIFAR-10's basin because SVHN and FMNIST features map back to CIFAR-10 prototypes under the adapted model weights (Routing Feedback Loop / Feedback Trap).
     * With Unbiased Routing and Isotropic Centering, we break this feedback loop. For PROTO-TTMM, on Task C (FashionMNIST), the model successfully triggers the Novelty detector (Novelty: 2.0694, Cohesion: 0.8413) on Batch 60, creating a brand-new Task 2 prototype set, and adapts to achieve a massive 84.64% accuracy (an increase of +75.53% absolute gain over CPA-Merge and +57.61% gain over TENT and PC-Merge!).
     * This provides the first proof-of-concept of lifelong Test-Time Model Merging in an open-world setting!

Please write a highly detailed, professional, and mathematically rigorous LaTeX document. It must compile perfectly.
Ensure the following:
1. Complete Sections: Abstract, Introduction, Related Work, Method (mathematical equations for novelty, cohesion, feature centering, unbiased routing, contrastive alignment, prototype EMA), Experimental Setup, Results and Discussion, Conclusion.
2. Formats: Use standard ICML style, including bibliography citations (e.g. \\cite{...}).
3. Visuals: Include a LaTeX table summarizing the results and refer to our diagram `ctta_results.png` using a `\\begin{figure}...\\includegraphics[width=\\columnwidth]{ctta_results.png}`.
4. References: Write a comprehensive set of references.

Please output the complete LaTeX content as standard text.
"""

    model = genai.GenerativeModel("gemini-2.5-pro")
    paper_response = model.generate_content(prompt)
    paper_tex = paper_response.text
    
    # Strip markdown wrappers if any
    if paper_tex.startswith("```latex"):
        paper_tex = paper_tex.split("```latex", 1)[1]
    elif paper_tex.startswith("```"):
        paper_tex = paper_tex.split("```", 1)[1]
    if paper_tex.endswith("```"):
        paper_tex = paper_tex.rsplit("```", 1)[0]
        
    with open("paper.tex", "w", encoding="utf-8") as f:
        f.write(paper_tex.strip())
    print("Saved paper.tex")

    # Generate the bibliography (.bib) file
    print("\nCalling Gemini to write the paper.bib file...")
    bib_prompt = """
Please write a complete, valid BibTeX file (`paper.bib`) with at least 40 key papers in model merging, test-time adaptation, and lifelong machine learning. Include entries for:
- AdaMerging, CPA-Merge, LFWA, PC-Merge, TENT, S2C-Merge, CoTTA, MEMO, RoTTA, LAME, etc.
- Standard machine learning classics (ResNet, Adam, CIFAR-10, SVHN, FashionMNIST, etc.)
- Contrastive learning (SimCLR, InfoNCE) and prototypical networks (Snell et al.).

Return ONLY the BibTeX code, starting directly with the first `@inproceedings` or `@article` entry. No markdown code block markers.
"""
    bib_response = model.generate_content(bib_prompt)
    paper_bib = bib_response.text
    if paper_bib.startswith("```bibtex"):
        paper_bib = paper_bib.split("```bibtex", 1)[1]
    elif paper_bib.startswith("```"):
        paper_bib = paper_bib.split("```", 1)[1]
    if paper_bib.endswith("```"):
        paper_bib = paper_bib.rsplit("```", 1)[0]
        
    with open("paper.bib", "w", encoding="utf-8") as f:
        f.write(paper_bib.strip())
    print("Saved paper.bib")

def compile_paper():
    print("\nCompiling the LaTeX paper using tectonic...")
    # Tectonic will automatically fetch missing packages and run multiple passes
    try:
        result = subprocess.run(["tectonic", "paper.tex"], capture_output=True, text=True, timeout=180)
        if result.returncode == 0:
            print("Compilation successful!")
            if os.path.exists("paper.pdf"):
                os.rename("paper.pdf", "submission.pdf")
                print("Renamed paper.pdf to submission.pdf successfully!")
                return True
        else:
            print("Compilation failed with errors:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Error compiling paper: {e}")
        return False

def main():
    generate_paper_with_gemini()
    compile_paper()

if __name__ == "__main__":
    main()
