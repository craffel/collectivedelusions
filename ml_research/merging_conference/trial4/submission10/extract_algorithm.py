from pypdf import PdfReader

reader = PdfReader("papers/submission7.pdf")
# Let's extract page 10 (0-indexed index 9)
text = reader.pages[9].extract_text()
print(text)
