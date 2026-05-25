import pypdf

reader = pypdf.PdfReader("submission.pdf")
print("Total Pages:", len(reader.pages))
idx = 2  # Page 3 is index 2
print(f"\n--- PAGE {idx + 1} ---")
text = reader.pages[idx].extract_text()
for line in text.split('\n'):
    if "??" in line:
        print("MATCHING LINE:", line)
    else:
        print("LINE:", line)
