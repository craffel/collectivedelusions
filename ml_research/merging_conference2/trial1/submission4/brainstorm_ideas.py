import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

prompt = """
You are a senior ML researcher acting as the persona "The Methodologist".
Your persona description:
"You are a critical thinker who closely examines current experimental and methodological practices in the field, always willing to rigorously push for improvements. Your research philosophy is that bad methodology leads to false progress, and that the community desperately needs better evaluation protocols, stronger baselines, and more critical analyses of existing trends."

Based on your critical analysis of model merging papers (like SyMerge, OrthoMerge, and SAIM), brainstorm 10 novel research ideas. Each idea must:
1. Be highly aligned with your persona (The Methodologist). It should focus on exposing flaws, testing unverified assumptions, developing fairer/more comprehensive benchmarks, proposing extremely simple but highly tuned/scaled baselines that expose complex methods as overkill, or investigating confounding variables.
2. Be feasible to implement and evaluate empirically using CLIP (ViT-B/32) or standard pre-trained models on classification datasets (e.g., CIFAR-10, CIFAR-100, MNIST, SVHN, EuroSAT) using our cluster's GPU compute (up to 8 H100s).
3. Include:
   - A clear title.
   - The core research question or hypothesis.
   - The methodological flaw/confounder or assumption it targets.
   - The experimental design (including datasets, models, and baselines).
   - Expected results and scientific impact.

Format your output as a clean, structured Markdown list of 10 ideas.
"""

model = genai.GenerativeModel("gemini-3.5-flash")
response = model.generate_content(prompt)
print(response.text)

with open("brainstormed_ideas.md", "w", encoding="utf-8") as f:
    f.write(response.text)

