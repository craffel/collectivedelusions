import pypdf

reader = pypdf.PdfReader('submission.pdf')
print(f"Total pages: {len(reader.pages)}")

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if 'References' in text:
        print(f"References found on page {i + 1}")
    if 'Append' in text or 'APPENDIX' in text or 'Appendix' in text:
        print(f"Appendix found on page {i + 1}")
