import os
import requests
import json
import time

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
headers = {}
if api_key:
    headers["x-api-key"] = api_key

queries = [
    "model merging deep learning",
    "test-time adaptation",
    "sharpness-aware minimization",
    "model soups",
    "task arithmetic neural networks",
    "weight averaging neural networks",
    "out of distribution robustness",
    "parameter-efficient fine-tuning merging",
    "federated learning model fusion",
    "unsupervised domain adaptation deep learning"
]

all_papers = {}

for q in queries:
    print(f"Searching for '{q}'...")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q.replace(' ', '+')}&fields=title,authors,year,venue,journal&limit=10"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            for paper in data.get("data", []):
                paper_id = paper.get("paperId")
                if paper_id and paper_id not in all_papers:
                    all_papers[paper_id] = paper
        else:
            print(f"Error {response.status_code} on search '{q}': {response.text}")
        time.sleep(1) # rate limit backoff
    except Exception as e:
        print(f"Exception on '{q}': {e}")

print(f"Found {len(all_papers)} unique papers.")

# Let's clean and format them into BibTeX
bibtex_entries = []
keys_set = set()

# First read existing bib file to avoid duplicate keys
existing_keys = set()
if os.path.exists("example_paper.bib"):
    with open("example_paper.bib", "r") as f:
        content = f.read()
        for line in content.split("\n"):
            if line.strip().startswith("@"):
                try:
                    key = line.split("{")[1].split(",")[0].strip()
                    existing_keys.add(key)
                except:
                    pass

for pid, paper in all_papers.items():
    authors = paper.get("authors", [])
    if not authors:
        continue
    authors_list = [a.get("name") for a in authors if a.get("name")]
    if not authors_list:
        continue
    authors_str = " and ".join(authors_list)
    title = paper.get("title", "").replace("{", "").replace("}", "")
    year = paper.get("year")
    if not year:
        year = 2023
        
    venue = paper.get("venue", "")
    if not venue:
        journal_info = paper.get("journal")
        if journal_info and isinstance(journal_info, dict):
            venue = journal_info.get("name", "")
    if not venue:
        venue = "arXiv preprint"
        
    # generate a clean unique key
    lastname = authors_list[0].split()[-1].lower()
    # keep only alphabet
    lastname = "".join([c for c in lastname if c.isalpha()])
    first_word_title = "".join([c for c in title.split()[0].lower() if c.isalpha()]) if title.split() else "paper"
    key = f"{lastname}{year}{first_word_title}"
    
    # Avoid duplicate keys
    counter = 1
    base_key = key
    while key in existing_keys or key in keys_set:
        key = f"{base_key}_{counter}"
        counter += 1
        
    keys_set.add(key)
    
    entry = f"""@article{{{key},
  title={{{title}}},
  author={{{authors_str}}},
  journal={{{venue}}},
  year={{{year}}}
}}"""
    bibtex_entries.append((key, entry))

print(f"Generated {len(bibtex_entries)} bibtex entries.")

# Write to example_paper.bib
with open("example_paper.bib", "a") as f:
    f.write("\n\n")
    for key, entry in bibtex_entries:
        f.write(entry + "\n\n")

print("Successfully appended to example_paper.bib!")
