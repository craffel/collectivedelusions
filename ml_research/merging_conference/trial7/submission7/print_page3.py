import pypdf

reader = pypdf.PdfReader("papers/submission3.pdf")
print(reader.pages[2].extract_text()[:4000])
