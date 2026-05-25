with open("paper.tex", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update bibliography reference
text = text.replace("\\bibliography{example_paper}", "\\bibliography{paper}")

# 2. Truncate manual bibliography and add end{document}
target = "\\subsection{A Note on Bibliography Generation}"
if target in text:
    text = text.split(target)[0] + "\n\\end{document}\n"
else:
    # Fallback if target is missing
    target2 = "\\begin{thebibliography}"
    if target2 in text:
        text = text.split(target2)[0] + "\n\\end{document}\n"

with open("paper.tex", "w", encoding="utf-8") as f:
    f.write(text)

print("paper.tex fixed successfully!")
