import pypdf

reader = pypdf.PdfReader('papers/submission10.pdf')
print("=== Page 4 ===")
print(reader.pages[3].extract_text())
print("=== Page 5 ===")
print(reader.pages[4].extract_text())
