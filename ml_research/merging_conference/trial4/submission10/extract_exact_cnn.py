from pypdf import PdfReader

reader = PdfReader("papers/submission1.pdf")
text = reader.pages[3].extract_text()
lines = text.split("\n")
for i, line in enumerate(lines):
    if "Convolutional Layer 1" in line or "Convolutional Layer 2" in line or "Convolutional Layer 3" in line:
        # Print surrounding lines
        for j in range(max(0, i-2), min(len(lines), i+6)):
            print(f"{j}: {lines[j]}")
        print("-" * 50)
