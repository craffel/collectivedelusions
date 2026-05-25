import pypdf

reader = pypdf.PdfReader('submission.pdf')
print(f"Total pages: {len(reader.pages)}")

for idx, page in enumerate(reader.pages):
    text = page.extract_text()
    first_lines = "\n".join(text.splitlines()[:3])
    last_lines = "\n".join(text.splitlines()[-3:])
    print(f"\n--- PAGE {idx+1} ---")
    print(f"--- First lines ---\n{first_lines}")
    print(f"--- Last lines ---\n{last_lines}")
