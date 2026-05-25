import pypdf

reader = pypdf.PdfReader("submission.pdf")
print("=== END OF PAGE 8 ===")
print(reader.pages[7].extract_text()[-1000:])
print("\n=== START OF PAGE 9 ===")
print(reader.pages[8].extract_text()[:1000])
