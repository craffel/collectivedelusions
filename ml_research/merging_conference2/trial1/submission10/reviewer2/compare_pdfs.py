import pypdf

def extract_text(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    text_by_page = []
    for page in reader.pages:
        text_by_page.append(page.extract_text() or "")
    return "\n".join(text_by_page)

txt1 = extract_text("submission.pdf")
txt2 = extract_text("template/example_paper.pdf")

if txt1 == txt2:
    print("The texts of submission.pdf and template/example_paper.pdf are EXACTLY the same.")
else:
    print("There are textual differences!")
    import difflib
    diff = list(difflib.unified_diff(txt1.splitlines(), txt2.splitlines(), lineterm=""))
    print(f"Number of diff lines: {len(diff)}")
    for line in diff[:30]:
        print(line)
