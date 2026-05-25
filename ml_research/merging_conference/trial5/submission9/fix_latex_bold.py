import re

with open("submission.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Fix the equal contribution macro typo
text = text.replace(r"\printAffiliationsAndNotice{\icmlequalcontrib}", r"\printAffiliationsAndNotice{\icmlEqualContribution}")

# Replace **something** with \textbf{something}
# Use a regex that matches **...** non-greedily
text_fixed = re.sub(r"\*\*(.*?)\*\*", r"\\textbf{\1}", text)

with open("submission.tex", "w", encoding="utf-8") as f:
    f.write(text_fixed)

print("Successfully fixed LaTeX macro typo and converted Markdown bolding to LaTeX.")
