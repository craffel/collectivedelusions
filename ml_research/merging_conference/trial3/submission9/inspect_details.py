import re

def search_keywords(file_path):
    print(f"\n=====================================")
    print(f"File: {file_path}")
    print(f"=====================================")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # We will search for sections like Introduction, Methodology/Method, Experiments, Results
    # and print some paragraphs to understand the datasets, models, and setups used.
    
    keywords = [
        r"(?:CIFAR-10|SVHN|MNIST|FashionMNIST|KMNIST|split-class|ResNet-18|Vision Transformer|ViT|dataset)",
        r"(?:accuracy|average accuracy|performance|baseline|SAM|OrthoMerge|SPOR|Fisher|SBF)",
        r"(?:Algorithm 1|Table 1|Figure 1|hyperparameter|learning rate|epoch)"
    ]
    
    # Let's find matches and print context
    for kw in keywords:
        print(f"\n--- Searching for: {kw} ---")
        pattern = re.compile(kw, re.IGNORECASE)
        matches = list(pattern.finditer(content))
        print(f"Found {len(matches)} matches.")
        # Print up to 3 distinct matches with surrounding text (approx 200 chars before and after)
        seen_positions = []
        count = 0
        for m in matches:
            pos = m.start()
            # Avoid printing highly overlapping regions
            if any(abs(pos - p) < 300 for p in seen_positions):
                continue
            seen_positions.append(pos)
            start = max(0, pos - 150)
            end = min(len(content), pos + 250)
            print(f"... {content[start:end].strip().replace('\n', ' ')} ...")
            count += 1
            if count >= 3:
                break

if __name__ == "__main__":
    search_keywords("papers_txt/submission4.txt")
    search_keywords("papers_txt/submission5.txt")
    search_keywords("papers_txt/submission8.txt")
