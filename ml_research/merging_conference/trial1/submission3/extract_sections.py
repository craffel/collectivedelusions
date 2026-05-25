import pypdf
import re

def analyze_pdf(path):
    print(f"==================== {path} ====================")
    reader = pypdf.PdfReader(path)
    text_by_page = []
    headings = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        text_by_page.append(text)
        # Simple regex to find headings (e.g., "1 INTRODUCTION", "2 RELATED WORK", etc.)
        for line in text.split('\n'):
            if re.match(r'^[1-9]\s+[A-Z\s\-]{3,}', line) or re.match(r'^[A-Z][A-Z\s\-]{4,}:', line):
                headings.append((i+1, line.strip()))
                
    print("Headings found:")
    for page_num, heading in headings[:30]:
        print(f"  Page {page_num}: {heading}")
        
    print("\nLet's search for 'dataset', 'optimizer', 'model', or 'learning rate' details:")
    all_text = "\n".join(text_by_page)
    keywords = ["dataset", "CIFAR", "GLUE", "Llama", "epoch", "learning rate", "batch size", "ViT", "RoBERTa"]
    for kw in keywords:
        matches = [m.start() for m in re.finditer(kw, all_text, re.IGNORECASE)]
        print(f"  '{kw}': {len(matches)} occurrences")
        
    # Let's search specifically for the evaluation benchmarks and models in each paper
    print("\n--- Summary of Method & Setup ---")
    # Find abstract
    abstract_start = all_text.lower().find("abstract")
    introduction_start = all_text.lower().find("1 introduction")
    if abstract_start != -1 and introduction_start != -1:
        print("Abstract:")
        print(all_text[abstract_start:introduction_start][:1500])
    elif abstract_start != -1:
        print("Abstract:")
        print(all_text[abstract_start:][:1500])

analyze_pdf("papers/0.pdf")
analyze_pdf("papers/1.pdf")
analyze_pdf("papers/2.pdf")
