import os
import requests
import time

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {}
if api_key:
    headers["x-api-key"] = api_key

queries = [
    "model merging deep learning",
    "weight averaging deep learning",
    "test-time adaptation",
    "linear mode connectivity",
    "task vectors deep learning",
    "Kronecker-factored preconditioning",
    "mixture of experts test-time",
    "out-of-distribution robustness neural networks",
    "parameter-efficient fine-tuning weight space"
]

seen_titles = set()
all_papers = []

print(f"Searching using API Key: {api_key is not None}")

for q in queries:
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(q)}&fields=title,authors,year,venue,journal,externalIds,citationCount&limit=15"
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            data = r.json()
            papers = data.get("data", [])
            print(f"Query '{q}' returned {len(papers)} results.")
            for p in papers:
                title = p.get("title")
                if title and title.lower() not in seen_titles:
                    seen_titles.add(title.lower())
                    all_papers.append(p)
        elif r.status_code == 429:
            print("Rate limited, sleeping...")
            time.sleep(5)
        else:
            print(f"Error {r.status_code} on query {q}: {r.text}")
    except Exception as e:
        print(f"Exception on query {q}: {e}")
    time.sleep(1)

# Format to BibTeX
def sanitize_key(title):
    words = [w.strip(":,.-_()'\"/\\") for w in title.split()]
    words = [w for w in words if w.isalnum() and len(w) > 3]
    if not words:
        return "paper_" + str(int(time.time()))
    key_base = "".join(words[:3]).lower()
    return key_base

bibtex_entries = []
for p in all_papers:
    title = p.get("title")
    authors_list = p.get("authors", [])
    year = p.get("year")
    venue = p.get("venue")
    journal = p.get("journal")
    
    if not title or not authors_list or not year:
        continue
        
    author_names = [a.get("name") for a in authors_list if a.get("name")]
    if not author_names:
        continue
    authors_str = " and ".join(author_names)
    
    key = sanitize_key(title) + str(year)
    
    # Determine type of publication
    entry_type = "article"
    booktitle_or_journal = ""
    
    if venue:
        entry_type = "inproceedings"
        booktitle_or_journal = venue
    elif journal and journal.get("name"):
        entry_type = "article"
        booktitle_or_journal = journal.get("name")
    else:
        entry_type = "article"
        booktitle_or_journal = "arXiv preprint"
        
    # Build BibTeX
    entry = f"@{entry_type}{{{key},\n"
    entry += f"  title={{{title}}},\n"
    entry += f"  author={{{authors_str}}},\n"
    if entry_type == "inproceedings":
        entry += f"  booktitle={{{booktitle_or_journal}}},\n"
    else:
        entry += f"  journal={{{booktitle_or_journal}}},\n"
    entry += f"  year={{{year}}}\n"
    entry += "}"
    bibtex_entries.append((key, entry))

print(f"Generated {len(bibtex_entries)} bibtex entries.")

# Write to a temporary file
with open("fetched_references.bib", "w") as f:
    for key, entry in bibtex_entries:
        f.write(entry + "\n\n")
print("Saved to fetched_references.bib")
