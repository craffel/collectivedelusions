import re

def clean_file(path):
    with open(path, "r") as f:
        content = f.read()
    
    # Strip leading ```latex or ```bibtex or ```
    content_clean = re.sub(r"^```[a-zA-Z]*\n", "", content)
    # Strip trailing ```
    content_clean = re.sub(r"\n```$", "", content_clean)
    # Also strip general leading/trailing spaces
    content_clean = content_clean.strip()
    
    with open(path, "w") as f:
        f.write(content_clean)
    print(f"Cleaned {path}")

clean_file("paper.tex")
clean_file("paper.bib")
clean_file("progress.md")
