import os
import re
import requests

def extract_key(bibtex_str):
    match = re.search(r'@(?:[a-zA-Z]+)\{\s*([^,\s}]+)', bibtex_str)
    if match:
        return match.group(1).strip()
    return None

def main():
    # Load existing bibtex keys to prevent duplicates
    bib_path = "example_paper.bib"
    with open(bib_path, "r", encoding="utf-8") as f:
        existing_content = f.read()
    
    existing_keys = set(re.findall(r'@(?:[a-zA-Z]+)\{\s*([^,\s}]+)', existing_content))
    print(f"Loaded {len(existing_keys)} existing BibTeX keys.")
    
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {"x-api-key": api_key} if api_key else {}
    
    # We will query several topics to get diverse and relevant papers
    queries = [
        "model merging neural networks",
        "test-time adaptation deep learning",
        "sharpness-aware minimization",
        "parameter-efficient fine-tuning adapter",
        "weight averaging deep learning",
        "federated learning model fusion"
    ]
    
    new_bibtex_entries = []
    seen_new_keys = set()
    
    for q in queries:
        print(f"Querying: {q}")
        # Search for papers on Semantic Scholar
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(q)}&fields=title,authors,year,venue,citationStyles&limit=15"
        try:
            r = requests.get(url, headers=headers)
            if r.status_code == 200:
                data = r.json().get("data", [])
                for paper in data:
                    citation_styles = paper.get("citationStyles", {})
                    if citation_styles and "bibtex" in citation_styles:
                        bibtex = citation_styles["bibtex"]
                        key = extract_key(bibtex)
                        if key and key not in existing_keys and key not in seen_new_keys:
                            seen_new_keys.add(key)
                            new_bibtex_entries.append((key, bibtex))
            else:
                print(f"Error status {r.status_code} for query: {q}")
        except Exception as e:
            print(f"Failed query {q}: {e}")
            
    print(f"Found {len(new_bibtex_entries)} unique new BibTeX entries.")
    
    # Let's write the new entries to example_paper.bib
    with open(bib_path, "a", encoding="utf-8") as f:
        f.write("\n\n")
        for key, entry in new_bibtex_entries:
            # Ensure proper separation
            f.write(entry.strip() + "\n\n")
            
    print("Done! Appended new references to example_paper.bib.")

if __name__ == "__main__":
    main()
