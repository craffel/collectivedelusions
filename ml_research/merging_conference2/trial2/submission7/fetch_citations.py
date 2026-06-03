import os
import requests
import json

api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
headers = {"x-api-key": api_key} if api_key else {}

queries = [
    "model merging deep learning",
    "weight averaging deep learning",
    "parameter editing language models",
    "linear mode connectivity",
    "multi-task learning weight interpolation",
    "lora merging PEFT",
    "representation alignment deep learning"
]

all_papers = {}

for q in queries:
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(q)}&fields=title,authors,year,venue,externalIds,citationCount&limit=20"
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            data = r.json()
            papers = data.get("data", [])
            for p in papers:
                pid = p.get("paperId")
                if pid and pid not in all_papers:
                    all_papers[pid] = p
        else:
            print(f"Error for query '{q}': {r.status_code}")
    except Exception as e:
        print(f"Exception for query '{q}': {e}")

print(f"Total unique papers found: {len(all_papers)}")

# Generate bibtex entries
bib_entries = []
seen_keys = set()

# First read existing bibtex keys to avoid duplicates
existing_bib_path = "template/submission.bib"
if os.path.exists(existing_bib_path):
    with open(existing_bib_path, "r") as f:
        for line in f:
            if line.strip().startswith("@"):
                # Extract key
                try:
                    key = line.split("{")[1].split(",")[0].strip()
                    seen_keys.add(key)
                except:
                    pass

for pid, p in all_papers.items():
    title = p.get("title", "")
    year = p.get("year")
    venue = p.get("venue", "")
    authors_list = p.get("authors", [])
    ext_ids = p.get("externalIds", {})
    
    if not title or not year or not authors_list:
        continue
        
    # Build citation key
    first_author = authors_list[0].get("name", "").split()
    last_name = first_author[-1].lower() if first_author else "anonymous"
    # Clean last name
    last_name = "".join(c for c in last_name if c.isalnum())
    citation_key = f"{last_name}{year}"
    
    # Handle key collisions
    suffix = "a"
    orig_key = citation_key
    while citation_key in seen_keys:
        citation_key = orig_key + suffix
        suffix = chr(ord(suffix) + 1)
        
    seen_keys.add(citation_key)
    
    # Format authors
    authors_formatted = " and ".join([a.get("name", "") for a in authors_list])
    
    # Check if there is an arXiv ID
    arxiv_id = ext_ids.get("ArXiv")
    if arxiv_id:
        entry = f"""@article{{{citation_key},
  title={{{title}}},
  author={{{authors_formatted}}},
  journal={{arXiv preprint arXiv:{arxiv_id}}},
  year={{{year}}}
}}"""
    elif venue:
        entry = f"""@inproceedings{{{citation_key},
  title={{{title}}},
  author={{{authors_formatted}}},
  booktitle={{{venue}}},
  year={{{year}}}
}}"""
    else:
        entry = f"""@article{{{citation_key},
  title={{{title}}},
  author={{{authors_formatted}}},
  journal={{Scientific Report, Semantic Scholar}},
  year={{{year}}}
}}"""
    bib_entries.append(entry)

print(f"Generated {len(bib_entries)} new BibTeX entries.")

# Write to file
with open("new_citations.bib", "w") as f:
    f.write("\n\n".join(bib_entries))
