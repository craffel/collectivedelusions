import os
import requests
import json
import time

API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

queries = [
    "Test-Time Model Merging",
    "Test-Time Adaptation",
    "Model Merging deep learning",
    "Weight Interpolation deep learning",
    "Sharpness-Aware Minimization",
    "Hoyer sparsity",
    "Spherical Contrastive Learning",
    "Mixture of Experts test-time",
    "Parameter merging neural networks",
    "Domain adaptation streaming"
]

headers = {}
if API_KEY:
    headers["x-api-key"] = API_KEY

unique_papers = {}

for q in queries:
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(q)}&fields=title,authors,year,venue,externalIds,citationCount&limit=15"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            papers = data.get("data", [])
            print(f"Query '{q}' returned {len(papers)} papers.")
            for p in papers:
                pid = p.get("paperId")
                if pid and pid not in unique_papers:
                    unique_papers[pid] = p
        else:
            print(f"Query '{q}' failed with status {response.status_code}")
        time.sleep(0.5) # rate limit politeness
    except Exception as e:
        print(f"Error querying {q}: {e}")

print(f"Total unique papers collected: {len(unique_papers)}")

# Generate BibTeX entries
bib_entries = []

# Keep track of bibtex keys to avoid duplicates
seen_keys = set()

def make_bib_key(paper):
    authors = paper.get("authors", [])
    if authors:
        last_name = authors[0].get("name", "").split()[-1]
    else:
        last_name = "anon"
    # remove non-alphanumeric
    last_name = "".join(c for c in last_name if c.isalnum()).lower()
    year = paper.get("year")
    if year:
        year_str = str(year)
    else:
        year_str = "empty"
    
    title_words = paper.get("title", "").split()
    first_word = ""
    for w in title_words:
        w_clean = "".join(c for c in w if c.isalnum()).lower()
        if w_clean not in ["the", "a", "an", "on", "of", "and", "in", "for", "with", "to", "by", "from", "at", "as"]:
            first_word = w_clean
            break
    if not first_word and title_words:
        first_word = "".join(c for c in title_words[0] if c.isalnum()).lower()
        
    base_key = f"{last_name}{year_str}{first_word}"
    if not base_key:
        base_key = f"key_{paper.get('paperId')[:8]}"
        
    key = base_key
    counter = 1
    while key in seen_keys:
        key = f"{base_key}_{counter}"
        counter += 1
    seen_keys.add(key)
    return key

for pid, p in unique_papers.items():
    title = p.get("title")
    authors_list = p.get("authors", [])
    year = p.get("year")
    venue = p.get("venue")
    
    if not title:
        continue
        
    # Format authors
    author_names = []
    for a in authors_list:
        name = a.get("name")
        if name:
            author_names.append(name)
    if not author_names:
        author_names = ["Anonymous"]
    authors_str = " and ".join(author_names)
    
    # Format venue / journal
    if not venue:
        venue = "arXiv preprint"
        
    key = make_bib_key(p)
    
    # Check if we have arXiv or DOI
    ext_ids = p.get("externalIds", {})
    arxiv_id = ext_ids.get("ArXiv")
    doi = ext_ids.get("DOI")
    
    bib_type = "article" if "preprint" in venue.lower() or "arxiv" in venue.lower() else "inproceedings"
    
    entry = f"@{bib_type}{{{key},\n"
    entry += f"  author = {{{authors_str}}},\n"
    entry += f"  title = {{{title}}},\n"
    if bib_type == "article":
        entry += f"  journal = {{{venue}}},\n"
    else:
        entry += f"  booktitle = {{{venue}}},\n"
    if year:
        entry += f"  year = {{{year}}},\n"
    if arxiv_id:
        entry += f"  note = {{arXiv:{arxiv_id}}},\n"
    elif doi:
        entry += f"  note = {{DOI:{doi}}},\n"
    entry += "}\n"
    
    bib_entries.append((key, entry))

# Write to file
with open("fetched_papers.bib", "w") as f:
    for _, entry in bib_entries:
        f.write(entry + "\n")

print(f"Successfully wrote {len(bib_entries)} bibtex entries to fetched_papers.bib")

# Save a list of keys with titles so we can cite them easily in our tex file
with open("fetched_papers_keys.json", "w") as f:
    json.dump({key: p.get("title") for key, p in zip(seen_keys, unique_papers.values())}, f, indent=2)
