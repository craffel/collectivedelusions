import google.generativeai as genai
import os

# Configure Gemini API
genai.configure(api_key=os.environ['GEMINI_API_KEY'])

# Read LaTeX paper
with open("submission.tex", "r") as f:
    paper_text = f.read()

# Read reviewing criteria
with open("reviewing_criteria.md", "r") as f:
    criteria_text = f.read()

prompt = f"""
You are an expert reviewer for top-tier machine learning conferences (ICML, NeurIPS, ICLR).
Below is the LaTeX source code of a paper submitted to the conference.
Please provide a highly rigorous, constructive, and detailed peer review of this paper based on the official reviewing criteria provided.

Official Reviewing Criteria:
{criteria_text}

LaTeX Paper Source Code:
{paper_text}

---
Your review must be structured exactly as follows:
1. SUMMARY: A concise summary of the paper's main contributions, approach, and findings.
2. STRENGTHS: Detailed bullet points detailing the strengths of the paper (soundness, presentation, significance, originality).
3. WEAKNESSES: Critical and constructive weaknesses (gaps in methodology, notation inconsistencies, potential overclaims, formatting issues, missing related works, etc.). Be extremely detailed and find at least 3-4 concrete weaknesses that can be resolved.
4. DETAILED QUESTIONS AND CONSTRUCTIVE RECOMMENDATIONS FOR IMPROVEMENT: Concrete, step-by-step suggestions to address the weaknesses.
5. RATINGS:
   - Soundness (excellent, good, fair, poor)
   - Presentation (excellent, good, fair, poor)
   - Significance (excellent, good, fair, poor)
   - Originality (excellent, good, fair, poor)
   - Overall Recommendation (integer score from 1 to 6 with justification)
"""

print("Generating peer review using gemini-2.5-pro...")
model = genai.GenerativeModel("gemini-2.5-pro")
response = model.generate_content(prompt)

with open("self_review.md", "w") as f:
    f.write(response.text)

print("Self-review successfully generated and saved to self_review.md.")
