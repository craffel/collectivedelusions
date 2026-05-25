import pypdf

reader = pypdf.PdfReader('submission.pdf')
print("=== PAGE 7 FULL CONTENT ===")
print(reader.pages[6].extract_text())
