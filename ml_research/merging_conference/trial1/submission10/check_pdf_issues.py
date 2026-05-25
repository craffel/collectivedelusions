import pypdf

reader = pypdf.PdfReader("submission.pdf")
print("Total Pages:", len(reader.pages))

unresolved_citations = 0
unresolved_references = 0
placeholders = 0

for idx, page in enumerate(reader.pages):
    text = page.extract_text()
    
    # Check for [?] or ?? which usually signify unresolved citations or cross-references
    if "[?]" in text or "[? " in text:
        print(f"WARNING: Potential unresolved citation on page {idx + 1}")
        unresolved_citations += 1
        
    if "??" in text:
        print(f"WARNING: Potential unresolved cross-reference (??) on page {idx + 1}")
        unresolved_references += 1
        
    # Check for raw LaTeX syntax markers that shouldn't appear in final text
    for marker in ["\\ref", "\\cite", "\\textbf", "\\textbf{"]:
        if marker in text:
            print(f"WARNING: Potential raw LaTeX command '{marker}' on page {idx + 1}")
            placeholders += 1

print("\n--- Summary ---")
print("Unresolved Citations:", unresolved_citations)
print("Unresolved References:", unresolved_references)
print("Raw LaTeX Command leakage:", placeholders)
