import pypdf

def extract_pdf_text(pdf_path, txt_path):
    print(f"Reading {pdf_path}...")
    reader = pypdf.PdfReader(pdf_path)
    print(f"Number of pages: {len(reader.pages)}")
    text_content = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        text_content.append(f"--- PAGE {i+1} ---")
        text_content.append(text)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_content))
    print(f"Text successfully extracted to {txt_path}.")

if __name__ == '__main__':
    extract_pdf_text('submission.pdf', 'submission.txt')
