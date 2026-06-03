import pypdf
import string

def clean_text(text):
    # Keep only printable characters and basic whitespace
    allowed = set(string.printable)
    return "".join(c if c in allowed else " " for c in text)

reader = pypdf.PdfReader('papers/submission10.pdf')
with open('cleaned_pages.txt', 'w') as f:
    f.write("=== Page 4 ===\n")
    f.write(clean_text(reader.pages[3].extract_text()))
    f.write("\n\n=== Page 5 ===\n")
    f.write(clean_text(reader.pages[4].extract_text()))

print("Cleaned text written to cleaned_pages.txt")
