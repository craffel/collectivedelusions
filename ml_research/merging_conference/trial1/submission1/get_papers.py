import os
import requests
import json
import time

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
headers = {"x-api-key": api_key} if api_key else {}

queries = [
    "model merging",
    "weight averaging",
    "test-time adaptation",
    "sharpness-aware minimization",
    "deep model fusion",
    "multitask learning"
]

all_papers = {}

for q in queries:
    print(f"Searching for: {q}")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(q)}&fields=title,authors,year,venue,citationCount,externalIds,abstract&limit=15"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            for paper in data.get("data", []):
                pid = paper.get("paperId")
                if pid and pid not in all_papers:
                    all_papers[pid] = paper
        else:
            print(f"Error {response.status_code}: {response.text}")
        time.sleep(1) # Be nice to the API
    except Exception as e:
        print(f"Request failed: {e}")

print(f"Found {len(all_papers)} unique papers.")

# Let's generate a BibTeX entry for each paper
bib_entries = []
for pid, paper in all_papers.items():
    title = paper.get("title", "")
    year = paper.get("year")
    venue = paper.get("venue", "")
    authors_list = paper.get("authors", [])
    
    if not title or not year or not authors_list:
        continue
        
    # Format authors
    authors_names = [a.get("name", "") for a in authors_list]
    authors_str = " and ".join(authors_names)
    
    # Create key: first author's last name + year + first word of title (lowercase, alphanumeric)
    first_author = authors_names[0].split()[-1] if authors_names else "anonymous"
    # clean first author name
    first_author = "".join(c for c in first_author if c.isalnum()).lower()
    
    title_words = [w for w in title.split() if w.lower() not in ["a", "an", "the", "on", "of", "in", "to", "for", "with", "and"]]
    first_word = "".join(c for c in title_words[0] if c.isalnum()).lower() if title_words else "paper"
    
    key = f"{first_author}{year}{first_word}"
    
    # Determine publication type and venue format
    venue_str = venue.strip()
    if not venue_str:
        venue_str = "arXiv preprint"
        
    if "arXiv" in venue_str or "preprint" in venue_str.lower():
        bib = f"""@article{{{key},
  title={{{title}}},
  author={{{authors_str}}},
  journal={{arXiv preprint}},
  year={{{year}}}
}}"""
    else:
        bib = f"""@inproceedings{{{key},
  title={{{title}}},
  author={{{authors_str}}},
  booktitle={{{venue_str}}},
  year={{{year}}}
}}"""
    bib_entries.append((key, bib, paper.get("citationCount", 0)))

# Sort by citation count (descending) to get the most prominent ones
bib_entries.sort(key=lambda x: x[2], reverse=True)

print(f"Generated {len(bib_entries)} bib entries.")

# Write to a draft file
with open("fetched_papers.bib", "w") as f:
    for _, bib, _ in bib_entries:
        f.write(bib + "\n\n")

print("Saved to fetched_papers.bib")
