import pypdf

reader = pypdf.PdfReader('submission.pdf')
text = ""
for i, page in enumerate(reader.pages):
    text += f"--- PAGE {i+1} ---\n"
    text += page.extract_text() + "\n"

with open('submission.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print(f"Successfully extracted {len(reader.pages)} pages to submission.txt.")
