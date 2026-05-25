import pypdf

reader = pypdf.PdfReader("papers/submission3.pdf")
found = False
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if "cohesion" in text.lower():
        print(f"\n--- Page {i+1} ---")
        lines = text.split("\n")
        for j, line in enumerate(lines):
            if "cohesion" in line.lower() or "c_k" in line.lower() or "ck" in line.lower():
                # print lines around it
                start = max(0, j-3)
                end = min(len(lines), j+4)
                print(f"--- Lines {start+1}-{end} ---")
                for idx in range(start, end):
                    print(f"{idx+1}: {lines[idx]}")
