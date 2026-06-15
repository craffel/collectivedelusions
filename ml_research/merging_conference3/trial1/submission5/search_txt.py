def search_file(filepath, keywords):
    print(f"Searching {filepath} for keywords: {keywords}")
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        for kw in keywords:
            if kw.lower() in line.lower():
                print(f"L{i}: {line.strip()}")
                break

search_file("papers/2.txt", ["dataset", "ViT", "ResNet", "Llama", "parameters", "evaluation", "baseline"])
