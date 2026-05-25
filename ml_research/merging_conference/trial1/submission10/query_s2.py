import os
import requests
import json

def search_papers(query, limit=20):
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,citationCount,venue,externalIds&limit={limit}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

def format_as_bibtex(paper):
    title = paper.get("title", "")
    year = paper.get("year", "2024")
    authors_list = paper.get("authors", [])
    if not authors_list:
        authors = "Unknown"
        first_author_last = "unknown"
    else:
        authors = " and ".join([a.get("name", "") for a in authors_list])
        first_author_last = authors_list[0].get("name", "").split()[-1].lower() if authors_list[0].get("name") else "unknown"
    
    venue = paper.get("venue", "arXiv preprint")
    if not venue:
        venue = "arXiv preprint"
    
    # Generate unique bibkey
    clean_title_word = "".join([c for c in title.split()[0] if c.isalnum()]).lower() if title.split() else "paper"
    bibkey = f"{first_author_last}{year}{clean_title_word}"
    
    bibtex = f"""@inproceedings{{{bibkey},
  title={{{title}}},
  author={{{authors}}},
  booktitle={{{venue}}},
  year={{{year}}}
}}"""
    return bibkey, bibtex

queries = [
    "model merging",
    "sharpness aware minimization",
    "test-time adaptation"
]

all_bibtex = []
seen_keys = set()

for q in queries:
    print(f"--- Querying: {q} ---")
    papers = search_papers(q, limit=15)
    for p in papers:
        bibkey, bibtex = format_as_bibtex(p)
        if bibkey not in seen_keys:
            seen_keys.add(bibkey)
            all_bibtex.append(bibtex)

print(f"Total compiled bibtex entries: {len(all_bibtex)}")
with open("fetched_papers.bib", "w") as f:
    f.write("\n\n".join(all_bibtex))
print("Saved to fetched_papers.bib")
