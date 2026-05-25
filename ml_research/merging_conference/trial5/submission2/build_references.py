import urllib.request
import json
import os
import re

# Existing keys in example_paper.bib to preserve or avoid duplicating
existing_keys = {
    "wang2021tent", "liang2020shot", "wang2022continual", "zhao2023delta",
    "yadav2023ties", "ilharco2022editing", "yang2024adamerging",
    "anonymous2026s2c", "anonymous2026sata", "anonymous2026lfwa"
}

# Search terms to queries
queries = [
    "test-time adaptation",
    "fully test-time adaptation",
    "continual test-time adaptation",
    "model merging",
    "weight averaging model",
    "model soup",
    "task arithmetic",
    "fisher information merging",
    "parameter merging",
    "dynamic routing mixture of experts",
    "contrastive test-time adaptation",
    "natural gradient descent neural networks",
    "elastic weight consolidation"
]

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

all_papers = []
seen_titles = set()

# Seed seen_titles with existing ones (approximate match via slug)
def slugify(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())

existing_titles = [
    "Tent: Fully Test-Time Adaptation by Entropy Minimization",
    "Do we really need to access the source data? source-hypothesis transfer for unsupervised domain adaptation",
    "Continual test-time domain adaptation",
    "Delta: Degradation-free fully test-time adaptation",
    "Ties-merging: Resolving interference when merging models",
    "Editing models with task arithmetic",
    "AdaMerging: Adaptive Model Merging for Multi-Task Learning",
    "S2c-merge: Teacher-free test-time model merging",
    "Sata-sbf: Teacher-guided test-time model merging",
    "LFWA: Layer-wise Fisher-Weighted Adaptation for Robust Test-Time Model Merging"
]
for title in existing_titles:
    seen_titles.add(slugify(title))

for query in queries:
    print(f"Searching for '{query}'...")
    escaped_query = urllib.parse.quote_plus(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={escaped_query}&fields=title,authors,venue,year,externalIds&limit=15"
    
    req = urllib.request.Request(url, headers={'x-api-key': api_key} if api_key else {})
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode())
            data = res.get("data", [])
            for paper in data:
                title = paper.get("title")
                if not title:
                    continue
                slug = slugify(title)
                if slug in seen_titles:
                    continue
                seen_titles.add(slug)
                all_papers.append(paper)
    except Exception as e:
        print(f"Error searching for '{query}': {e}")

print(f"Found {len(all_papers)} new unique papers.")

# Format as BibTeX
bibtex_entries = []
for paper in all_papers:
    title = paper.get("title")
    authors_list = paper.get("authors", [])
    year = paper.get("year")
    venue = paper.get("venue")
    external_ids = paper.get("externalIds", {})
    
    if not title or not authors_list or not year:
        continue
        
    # Build clean author string
    authors_str = " and ".join([a["name"] for a in authors_list if "name" in a])
    if not authors_str:
        continue
        
    # First author last name for key
    first_author = authors_list[0]["name"]
    # Extract last name
    last_name_match = re.search(r'\s*([^\s]+)$', first_author)
    last_name = last_name_match.group(1) if last_name_match else "author"
    last_name = re.sub(r'[^a-zA-Z]', '', last_name).lower()
    
    # First word of title
    title_words = [w for w in re.sub(r'[^a-zA-Z ]', '', title).split() if len(w) > 3]
    first_word = title_words[0].lower() if title_words else "paper"
    
    bib_key = f"{last_name}{year}{first_word}"
    
    # Avoid duplicate keys
    suffix = 1
    orig_key = bib_key
    while bib_key in existing_keys:
        bib_key = f"{orig_key}_{suffix}"
        suffix += 1
    existing_keys.add(bib_key)
    
    # Determine publication venue
    if venue:
        journal_or_booktitle = f"booktitle={{{venue}}}"
    elif "ArXiv" in external_ids:
        arxiv_id = external_ids["ArXiv"]
        journal_or_booktitle = f"journal={{arXiv preprint arXiv:{arxiv_id}}}"
    else:
        journal_or_booktitle = "journal={arXiv preprint}"
        
    entry = f"""@inproceedings{{{bib_key},
  title={{{title}}},
  author={{{authors_str}}},
  {journal_or_booktitle},
  year={{{year}}}
}}"""
    bibtex_entries.append(entry)

print(f"Generated {len(bibtex_entries)} bibtex entries.")

# Read existing file to make sure we don't overwrite the original 10
with open("example_paper.bib", "r") as f:
    existing_content = f.read()

# Combine them
new_content = existing_content.strip() + "\n\n" + "\n\n".join(bibtex_entries) + "\n"

with open("example_paper.bib", "w") as f:
    f.write(new_content)

print("example_paper.bib updated successfully!")
