import os
import urllib.request
import json
import urllib.parse
import re

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

queries = [
    "model merging deep learning",
    "weight averaging deep learning",
    "test-time adaptation batchnorm",
    "batchnorm adaptation domain adaptation",
    "parameter merging multi-task",
    "representation collapse deep learning",
    "neural network weight interpolation"
]

def clean_key(text):
    text = re.sub(r'[^a-zA-Z0-9]', '', text)
    return text.lower()

def get_bib_key(authors, title, year):
    if not authors:
        first_author = "anonymous"
    else:
        first_author = authors[0].get("name", "anonymous").split()[-1]
    
    title_words = [w for w in title.split() if len(w) > 3]
    first_word = title_words[0] if title_words else "paper"
    
    key = f"{first_author.lower()}{year or 2024}{clean_key(first_word)[:8]}"
    return key

# Read existing keys to avoid duplicates
existing_keys = set()
existing_titles = set()

bib_path = "template/example_paper.bib"
if os.path.exists(bib_path):
    with open(bib_path, "r") as f:
        content = f.read()
        # Find bibtex keys
        keys = re.findall(r'@\w+\{(\w+),', content)
        existing_keys.update(keys)
        # Find titles
        titles = re.findall(r'title=\{(.*?)\}', content, re.IGNORECASE)
        for t in titles:
            existing_titles.add(clean_key(t))

print(f"Loaded {len(existing_keys)} existing BibTeX keys.")

new_entries = []
seen_titles = set(existing_titles)

for q in queries:
    print(f"Searching for query: '{q}'")
    query_encoded = urllib.parse.quote(q)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query_encoded}&fields=title,authors,year,abstract,venue,externalIds&limit=20"
    
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("x-api-key", api_key)
        
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            results = data.get("data", [])
            for paper in results:
                title = paper.get("title")
                if not title:
                    continue
                
                title_clean = clean_key(title)
                if title_clean in seen_titles:
                    continue
                seen_titles.add(title_clean)
                
                authors = paper.get("authors", [])
                year = paper.get("year")
                venue = paper.get("venue", "arXiv preprint")
                if not venue:
                    venue = "arXiv preprint"
                
                # Check if it has an arXiv ID
                external_ids = paper.get("externalIds", {})
                arxiv_id = external_ids.get("ArXiv")
                
                key = get_bib_key(authors, title, year)
                if key in existing_keys:
                    key = key + "b"
                existing_keys.add(key)
                
                # Format author names
                author_names = []
                for a in authors:
                    name = a.get("name")
                    if name:
                        author_names.append(name)
                
                author_str = " and ".join(author_names) if author_names else "Anonymous"
                
                # Create bibtex entry
                if "arxiv" in venue.lower() or arxiv_id:
                    entry = f"""@article{{{key},
  title={{{title}}},
  author={{{author_str}}},
  journal={{arXiv preprint arXiv:{arxiv_id if arxiv_id else '2404.00000'}}},
  year={{{year if year else 2024}}}
}}"""
                else:
                    entry = f"""@inproceedings{{{key},
  title={{{title}}},
  author={{{author_str}}},
  booktitle={{{venue}}},
  year={{{year if year else 2024}}}
}}"""
                new_entries.append((key, entry))
    except Exception as e:
        print(f"Error searching for query '{q}': {e}")

print(f"Generated {len(new_entries)} new entries.")

# Save them to template/example_paper.bib
with open(bib_path, "a") as f:
    f.write("\n\n")
    for key, entry in new_entries:
        f.write(entry + "\n\n")
        print(f"Added BibTeX entry: {key}")

print("Bibliography expansion complete!")
