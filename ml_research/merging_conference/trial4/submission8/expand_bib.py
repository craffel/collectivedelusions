import os
import requests
import json
import re

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {}
if api_key:
    headers["x-api-key"] = api_key

queries = [
    "model merging deep learning",
    "test-time adaptation neural network",
    "unsupervised test-time adaptation",
    "natural gradient descent neural network",
    "parameter merging",
    "deep model fusion",
    "multi-task learning weight optimization",
    "elastic weight consolidation",
    "sharpness aware minimization TTA"
]

existing_keys = {
    "devlin2018bert", "vaswani2017attention", "he2016deep", "dosovitskiy2020image",
    "yadav2023ties", "wortsman2022model", "ilharco2022editing", "yang2023adamerging",
    "jung2025symerge", "yu2024dare", "krizhevsky2009learning", "netzer2011reading"
}

bib_entries = []

def clean_name(text):
    if not text:
        return "unknown"
    # Remove non-ascii characters or keep clean
    text = re.sub(r'[^\w\s\-]', '', text)
    return text.strip()

for query in queries:
    print(f"Searching query: {query}")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,year,authors,venue,citationCount&limit=10"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            papers = data.get("data", [])
            for p in papers:
                title = p.get("title")
                year = p.get("year")
                authors = p.get("authors", [])
                venue = p.get("venue")
                
                if not title or not year:
                    continue
                
                # Create BibTeX key
                if authors:
                    first_author = clean_name(authors[0].get("name", "author")).split()[-1].lower()
                else:
                    first_author = "unknown"
                
                words = [w.lower() for w in clean_name(title).split() if len(w) > 3]
                first_word = words[0] if words else "paper"
                bib_key = f"{first_author}{year}{first_word}"
                
                if bib_key in existing_keys:
                    continue
                
                existing_keys.add(bib_key)
                
                # Format authors
                author_names = []
                for a in authors:
                    name = a.get("name")
                    if name:
                        author_names.append(name)
                author_str = " and ".join(author_names) if author_names else "Anonymous"
                
                # Format entry
                entry_type = "article"
                venue_str = venue if venue else ""
                
                # Determine if it's a conference or journal
                if venue_str:
                    if "arXiv" in venue_str or "preprint" in venue_str.lower():
                        entry_type = "article"
                        journal_field = f"journal={{arXiv preprint {venue_str}}}"
                    else:
                        entry_type = "inproceedings"
                        journal_field = f"booktitle={{{venue_str}}}"
                else:
                    entry_type = "article"
                    journal_field = "journal={arXiv preprint}"
                
                bib_entry = f"""@{entry_type}{{{bib_key},
  title={{{title}}},
  author={{{author_str}}},
  {journal_field},
  year={{{year}}}
}}"""
                bib_entries.append((bib_key, bib_entry))
        else:
            print(f"Error {response.status_code} for query '{query}': {response.text}")
    except Exception as e:
        print(f"Request failed for query '{query}': {e}")

print(f"Gathered {len(bib_entries)} unique new BibTeX entries.")

# Write to file
with open("new_refs.bib", "w") as f:
    for _, entry in bib_entries:
        f.write(entry + "\n\n")

print("Done! New refs written to new_refs.bib.")
