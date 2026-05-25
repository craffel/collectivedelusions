import pypdf

reader = pypdf.PdfReader("submission.pdf")
print(f"Total pages: {len(reader.pages)}")

keywords = [
    "abstract", "introduction", "related work", "method", "evaluation", "setup",
    "results", "ablation", "limitations", "conclusion", "references", "appendix",
    "derivation", "implementation"
]

for idx, page in enumerate(reader.pages):
    text = page.extract_text().lower()
    found = [kw for kw in keywords if kw in text]
    print(f"Page {idx+1}: {found}")
