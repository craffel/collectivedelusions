import pypdf
import re

def search_keywords_in_pdf(pdf_path, keywords):
    print(f"=== Searching keywords in {pdf_path} ===")
    reader = pypdf.PdfReader(pdf_path)
    found_info = {kw: [] for kw in keywords}
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        for kw in keywords:
            # find all sentences containing keyword (case insensitive)
            matches = re.finditer(re.escape(kw), text, re.IGNORECASE)
            for m in matches:
                start = max(0, m.start() - 150)
                end = min(len(text), m.end() + 150)
                context = text[start:end].replace('\n', ' ')
                found_info[kw].append((page_idx + 1, context))
    
    for kw, occurrences in found_info.items():
        print(f"Keyword: '{kw}' - found {len(occurrences)} times.")
        # print first 3 occurrences
        for idx, (page, context) in enumerate(occurrences[:3]):
            print(f"  [Page {page}]: ... {context} ...")
    print("\n")

keywords_to_search = ["dataset", "evaluation", "ViT", "CLIP", "HuggingFace", "model", "github", "task", "parameter", "merging"]

for i in range(3):
    search_keywords_in_pdf(f"papers/{i}.pdf", keywords_to_search)
