import os
import requests
import time

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {"x-api-key": api_key} if api_key else {}

queries = [
    "model merging deep learning",
    "test-time adaptation",
    "weight averaging deep learning",
    "parameter space model merging",
    "test-time model merging",
    "cosface metric learning",
    "batch normalization fusion",
    "fisher information model merging"
]

papers = []
seen_ids = set()

for query in queries:
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,venue,externalIds,citationCount&limit=20"
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            data = r.json()
            for p in data.get("data", []):
                pid = p.get("paperId")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    papers.append(p)
        elif r.status_code == 429:
            print("Rate limited, sleeping...")
            time.sleep(5)
        else:
            print(f"Error {r.status_code} for query: {query}")
    except Exception as e:
        print(f"Exception: {e}")
    time.sleep(1)

print(f"Found {len(papers)} unique papers.")

# Format as BibTeX
bibtex_entries = []
for idx, p in enumerate(papers):
    title = p.get("title", "No Title")
    year = p.get("year") or 2023
    venue = p.get("venue") or "arXiv preprint"
    authors_list = p.get("authors", [])
    if not authors_list:
        authors = "Unknown"
    else:
        authors = " and ".join([a.get("name", "") for a in authors_list if a.get("name")])
    
    # Generate citation key
    first_author = "unknown"
    if authors_list and authors_list[0].get("name"):
        first_author = authors_list[0].get("name").split()[-1].lower()
    # Clean non-alpha from citation key
    first_author = "".join([c for c in first_author if c.isalpha()])
    cite_key = f"{first_author}{year}_{idx}"
    
    # Clean title/venue for LaTeX
    title_clean = title.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    venue_clean = venue.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    
    entry = f"""@article{{{cite_key},
  title={{{title_clean}}},
  author={{{authors}}},
  journal={{{venue_clean}}},
  year={{{year}}}
}}"""
    bibtex_entries.append(entry)

# Write to a temporary file
with open("fetched_references.bib", "w") as f:
    f.write("\n\n".join(bibtex_entries))

print("Done! Written to fetched_references.bib")
