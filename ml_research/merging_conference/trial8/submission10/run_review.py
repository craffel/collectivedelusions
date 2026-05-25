import google.generativeai as genai
import os
import sys

def main():
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)
        
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    
    # Read submission.tex
    if not os.path.exists("submission.tex"):
        print("Error: submission.tex not found.")
        sys.exit(1)
        
    with open("submission.tex", "r") as f:
        paper_content = f.read()
        
    # Read reviewing_criteria.md
    if not os.path.exists("reviewing_criteria.md"):
        print("Error: reviewing_criteria.md not found.")
        sys.exit(1)
        
    with open("reviewing_criteria.md", "r") as f:
        criteria_content = f.read()
        
    prompt = f"""
You are an expert reviewer for a top-tier machine learning conference (e.g., ICML, NeurIPS, ICLR).
Your task is to provide a highly critical, constructive, and thorough peer review of the following LaTeX paper submission.

We are in the Iterative Refinement phase of the paper development. We want to identify any weaknesses in:
1. Mathematical and theoretical clarity/correctness.
2. Empirical completeness, baseline comparisons, and ablation studies.
3. Quality of writing, structure, and readability.
4. Positioning and literature coverage.

Please evaluate the paper strictly based on the provided reviewing criteria.

CRITERIA:
\"\"\"
{criteria_content}
\"\"\"

PAPER LATEX SOURCE:
\"\"\"
{paper_content}
\"\"\"

Please structure your review as follows:
1. **Summary of the paper**: A concise summary of the paper's goals, methodology, and key results.
2. **Strengths**: At least 3 key strengths of the paper (e.g., novelty, soundness, clear results).
3. **Weaknesses**: At least 3 detailed, critical weaknesses (e.g., missing baselines, leaps in logic, overclaiming, formatting issues, lack of theoretical depth). Be extremely pedantic and rigorous.
4. **Questions and Actionable Suggestions for Improvement**: Specific, concrete questions and recommendations for how the authors can improve the paper's soundness, presentation, significance, and originality.
5. **Overall Recommendation**: Choose one from:
   - 6: Strong Accept
   - 5: Accept
   - 4: Weak Accept
   - 3: Weak Reject
   - 2: Reject
   - 1: Strong Reject
   And provide a clear justification for this score.

Write your review in Markdown.
"""

    print("Calling Gemini API to generate peer review...")
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        review_text = response.text
        
        with open("review.md", "w") as f:
            f.write(review_text)
            
        print("Mock peer review successfully written to review.md!")
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
