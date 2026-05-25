from pypdf import PdfReader

reader = PdfReader("papers/submission1.pdf")
print("Searching for datasets in submission1.pdf:")
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if "dataset" in text.lower() or "mnist" in text.lower() or "cifar" in text.lower():
        print(f"--- Page {i+1} mentions datasets/mnist/cifar: ---")
        lines = text.split("\n")
        for line in lines:
            if any(w in line.lower() for w in ["dataset", "mnist", "fashion", "kmnist", "cifar", "svhn"]):
                print(line)
