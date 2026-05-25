import os
import urllib.request
import urllib.parse
import json

def fetch_semantic_scholar(query, limit=15):
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    encoded_query = urllib.parse.quote_plus(query)
    fields = "title,authors,year,venue,externalIds"
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&fields={fields}&limit={limit}"
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data.get("data", [])
    except Exception as e:
        print(f"Error fetching query '{query}': {e}")
        return []

def clean_author_name(name):
    # Keep only ASCII chars and simple formatting
    return "".join(c for c in name if c.isalnum() or c.isspace() or c == '-')

def to_bibtex(paper):
    title = paper.get("title", "")
    year = paper.get("year", 2024)
    authors = paper.get("authors", [])
    venue = paper.get("venue", "")
    external_ids = paper.get("externalIds", {})
    
    if not title:
        return None
        
    # Generate a cite key
    first_author = "unknown"
    if authors:
        first_author = clean_author_name(authors[0].get("name", "unknown")).split()[-1].lower()
    
    clean_title_words = [w.lower() for w in clean_author_name(title).split() if len(w) > 3]
    title_key = clean_title_words[0] if clean_title_words else "paper"
    cite_key = f"{first_author}{year}{title_key}"
    
    # Remove duplicates or invalid chars in key
    cite_key = "".join(c for c in cite_key if c.isalnum())
    
    author_str = " and ".join(clean_author_name(a.get("name", "")) for a in authors)
    
    arxiv_id = external_ids.get("ArXiv")
    
    if arxiv_id:
        bibtex = f"""@article{{{cite_key},
  title={{{title}}},
  author={{{author_str}}},
  journal={{arXiv preprint arXiv:{arxiv_id}}},
  year={{{year}}}
}}"""
    elif venue:
        bibtex = f"""@inproceedings{{{cite_key},
  title={{{title}}},
  author={{{author_str}}},
  booktitle={{{venue}}},
  year={{{year}}}
}}"""
    else:
        bibtex = f"""@article{{{cite_key},
  title={{{title}}},
  author={{{author_str}}},
  journal={{Semantic Scholar Preprint}},
  year={{{year}}}
}}"""
    return cite_key, bibtex

def main():
    queries = [
        "model merging neural networks",
        "weight averaging neural networks",
        "task arithmetic",
        "sharpness aware minimization",
        "federated learning weight aggregation",
        "permutation symmetries neural networks",
        "rebasin git re-basin",
        "loss landscapes mode connectivity",
        "deep model fusion",
        "model editing language models"
    ]
    
    seen_keys = set()
    # Read existing cite keys from submission.bib if it exists
    if os.path.exists("submission.bib"):
        with open("submission.bib", "r") as f:
            for line in f:
                if line.strip().startswith("@"):
                    try:
                        key = line.split("{")[1].split(",")[0].strip()
                        seen_keys.add(key)
                    except Exception:
                        pass
                        
    print(f"Initially loaded {len(seen_keys)} keys.")
    
    bibtex_entries = []
    for q in queries:
        print(f"Searching for: {q}")
        papers = fetch_semantic_scholar(q, limit=15)
        for p in papers:
            res = to_bibtex(p)
            if res:
                key, bib = res
                if key not in seen_keys:
                    seen_keys.add(key)
                    bibtex_entries.append(bib)
                    
    print(f"Found {len(bibtex_entries)} new papers.")
    
    # Append to submission.bib
    with open("submission.bib", "a") as f:
        f.write("\n\n")
        f.write("\n\n".join(bibtex_entries))
        f.write("\n")
        
    print("Successfully updated submission.bib!")

if __name__ == "__main__":
    main()
