import os
import requests
import json
import re

def clean_bibtex_key(author, title, year):
    # Extract last name of first author
    if not author:
        author_part = "anon"
    else:
        # e.g., "Wortsman, Mitchell" or "Mitchell Wortsman"
        first_author = author[0].get("name", "anon")
        last_name = first_author.split()[-1]
        author_part = re.sub(r'[^a-zA-Z]', '', last_name).lower()
    
    # Extract first word of title
    words = [w for w in re.sub(r'[^a-zA-Z ]', '', title).split() if len(w) > 3]
    title_part = words[0].lower() if words else "paper"
    
    year_part = str(year)[-2:] if year else "26"
    return f"{author_part}{year_part}{title_part}"

def search_and_format_bibtex(query, existing_keys, limit=20):
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    fields = "title,authors,year,venue,citationStyles"
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields={fields}&limit={limit}"
    
    print(f"Searching for: '{query}'...")
    response = requests.get(url, headers=headers)
    entries = []
    
    if response.status_code == 200:
        data = response.json()
        papers = data.get("data", [])
        for p in papers:
            title = p.get("title")
            authors = p.get("authors", [])
            year = p.get("year")
            venue = p.get("venue")
            citation_styles = p.get("citationStyles", {})
            
            if not title or not authors:
                continue
                
            # If Semantic Scholar already has the BibTeX, use it if it's clean
            bibtex_raw = citation_styles.get("bibtex")
            key = None
            if bibtex_raw:
                # Extract key from raw bibtex
                match = re.search(r'@\w+\{([^,]+),', bibtex_raw)
                if match:
                    key = match.group(1)
            
            if not key:
                key = clean_bibtex_key(authors, title, year)
            
            if key in existing_keys:
                print(f"Skipping duplicate key: {key}")
                continue
                
            existing_keys.add(key)
            
            if bibtex_raw:
                # Clean up/normalize the key in the raw bibtex
                bibtex_clean = re.sub(r'@(\w+)\{[^,]+,', f'@\\1{{{key},', bibtex_raw)
                entries.append((key, bibtex_clean))
            else:
                # Construct manually
                author_names = []
                for a in authors:
                    name = a.get("name")
                    if name:
                        author_names.append(name)
                author_str = " and ".join(author_names)
                
                # Simple guess of journal or booktitle
                booktitle = venue if venue else "arXiv preprint"
                entry = f"""@inproceedings{{{key},
  title={{{title}}},
  author={{{author_str}}},
  booktitle={{{booktitle}}},
  year={{{year if year else 2024}}}
}}"""
                entries.append((key, entry))
    else:
        print(f"Error {response.status_code}: {response.text}")
        
    return entries

def main():
    # Read existing keys from submission.bib
    existing_keys = set()
    if os.path.exists("submission.bib"):
        with open("submission.bib", "r") as f:
            content = f.read()
            for match in re.finditer(r'@\w+\{([^,]+),', content):
                existing_keys.add(match.group(1))
    
    print(f"Loaded {len(existing_keys)} existing keys: {existing_keys}")
    
    queries = [
        "model merging weight averaging",
        "task arithmetic deep learning",
        "TIES-Merging neural networks",
        "sharpness-aware minimization foret",
        "permutation invariance neural network merging",
        "git rebasin weight space",
        "test-time adaptation entropy minimization",
        "model soups deep learning",
        "federated learning weight averaging",
        "parameter efficient fine-tuning merging",
        "Fisher information neural networks weight selection",
        "procrustes alignment neural network representation"
    ]
    
    new_entries = []
    for q in queries:
        entries = search_and_format_bibtex(q, existing_keys, limit=8)
        new_entries.extend(entries)
        if len(existing_keys) >= 65: # Target plenty of keys
            print("Reached target of > 60 total references.")
            break
            
    print(f"Generated {len(new_entries)} new references.")
    
    # Append to submission.bib
    if new_entries:
        with open("submission.bib", "a") as f:
            f.write("\n\n")
            for key, entry in new_entries:
                f.write(entry.strip() + "\n\n")
        print("References successfully written to submission.bib.")

if __name__ == "__main__":
    main()
