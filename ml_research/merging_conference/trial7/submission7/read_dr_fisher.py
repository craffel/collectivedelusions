import pypdf

reader = pypdf.PdfReader("papers/submission10.pdf")

# Search and print pages 3, 4, 5, 9
for p_idx in [2, 3, 4, 8]: # 0-indexed: Page 3, 4, 5, 9
    print(f"\n================ PAGE {p_idx+1} ================")
    text = reader.pages[p_idx].extract_text()
    print(text)
