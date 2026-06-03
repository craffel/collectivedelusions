import re
from pypdf import PdfReader

papers = ["papers/submission1.pdf", "papers/submission2.pdf", "papers/submission9.pdf"]

for paper in papers:
    print("="*80)
    print(f"Details for {paper}")
    print("="*80)
    reader = PdfReader(paper)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"\n--- Page {i+1} ---\n" + page.extract_text()
    
    # Search for experimental setup keywords
    for keyword in ["ResNet-18", "learning rate", "optimizer", "epoch", "SGD", "Adam", "MNIST", "CIFAR-10", "Fashion-MNIST", "momentum"]:
        matches = [m.start() for m in re.finditer(keyword, text, re.IGNORECASE)]
        if matches:
            print(f"Keyword '{keyword}' found {len(matches)} times. Sample context(s):")
            for idx in matches[:2]:
                start = max(0, idx - 150)
                end = min(len(text), idx + 150)
                context = text[start:end].replace('\n', ' ')
                print(f"  ... {context} ...")
