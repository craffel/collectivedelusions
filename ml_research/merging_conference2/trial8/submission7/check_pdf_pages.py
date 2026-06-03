import pypdf

reader = pypdf.PdfReader('submission.pdf')
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    first_line = text.split('\n')[0] if text else "(No text)"
    print(f"Page {i+1}: {first_line[:80]}")
    # Search for section headings
    for line in text.split('\n'):
        if any(h in line.lower() for h in ['abstract', 'introduction', 'related work', 'methodology', 'experiments', 'discussion', 'conclusion', 'references', 'appendix', 'proofs and theoretical analysis']):
            print(f"  -> {line[:80]}")
