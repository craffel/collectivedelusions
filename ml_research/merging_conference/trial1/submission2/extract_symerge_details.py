import pypdf

reader = pypdf.PdfReader("papers/0.pdf")
print("=== SyMerge Method Section (Pages 4, 5, 6) ===")
for p_idx in [3, 4, 5]: # pages 4, 5, 6 (0-indexed: 3, 4, 5)
    print(f"--- PAGE {p_idx+1} ---")
    print(reader.pages[p_idx].extract_text()[:4000])
