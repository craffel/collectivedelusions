import pypdf

reader = pypdf.PdfReader('submission.pdf')
print("=== PAGE 8 ===")
print(reader.pages[7].extract_text()[-1000:])  # Print last 1000 chars of page 8
print("\n=== PAGE 9 ===")
print(reader.pages[8].extract_text()[:1000])  # Print first 1000 chars of page 9
