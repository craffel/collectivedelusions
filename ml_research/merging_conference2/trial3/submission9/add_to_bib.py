import json
import re
import os

def clean_title(title):
    # Remove braces and non-alphanumeric chars
    return re.sub(r'[^a-zA-Z0-9 ]', '', title).strip()

def get_bib_keys(bib_path):
    keys = set()
    if not os.path.exists(bib_path):
        return keys
    with open(bib_path, 'r') as f:
        content = f.read()
    # match keys like @inproceedings{key, or @article{key,
    found = re.findall(r'@[a-zA-Z]+\s*\{\s*([a-zA-Z0-9_\-]+)\s*,', content)
    return set(found)

def get_bib_titles(bib_path):
    titles = set()
    if not os.path.exists(bib_path):
        return titles
    with open(bib_path, 'r') as f:
        content = f.read()
    # find lines like title = {something} or title = {{something}}
    found = re.findall(r'title\s*=\s*[\{\"]+(.*?)[\}\"]+,', content, re.IGNORECASE)
    for t in found:
        titles.add(clean_title(t).lower())
    return titles

def generate_bib_key(authors, year, title, existing_keys):
    if not authors:
        lastname = "unknown"
    else:
        # Get last name of first author
        first_author = authors[0]
        # split by space and take last word
        parts = first_author.split()
        if parts:
            lastname = parts[-1].lower()
        else:
            lastname = "unknown"
    
    # clean lastname
    lastname = re.sub(r'[^a-z0-9]', '', lastname)
    if not lastname:
        lastname = "unknown"
        
    year_str = str(year) if year else "2024"
    
    # keyword from title
    title_words = [w.lower() for w in title.split() if len(w) > 3]
    kw = title_words[0] if title_words else "paper"
    kw = re.sub(r'[^a-z0-9]', '', kw)
    
    base_key = f"{lastname}{year_str}{kw}"
    key = base_key
    counter = 1
    while key in existing_keys:
        key = f"{base_key}{counter}"
        counter += 1
        
    existing_keys.add(key)
    return key

def format_bibtex(key, paper):
    title = paper["title"]
    authors = " and ".join(paper["authors"]) if paper["authors"] else "Unknown"
    year = paper["year"] if paper["year"] else 2024
    venue = paper["venue"] if paper["venue"] else "arXiv preprint"
    
    # Escape special characters in bibtex
    title_escaped = title.replace("&", "\\&").replace("%", "\\%")
    
    if "arxiv" in venue.lower() or "preprint" in venue.lower() or not venue:
        entry_type = "article"
        field_venue = f"journal   = {{arXiv preprint arXiv:{paper['arxiv']}}}" if paper.get("arxiv") else f"journal   = {{{venue}}}"
    else:
        entry_type = "inproceedings"
        field_venue = f"booktitle = {{{venue}}}"
        
    bib = f"""@{entry_type}{{{key},
  author    = {{{authors}}},
  title     = {{{{{title_escaped}}}}},
  {field_venue},
  year      = {{{year}}}
}}
"""
    return bib

def main():
    bib_path = "example_paper.bib"
    existing_keys = get_bib_keys(bib_path)
    existing_titles = get_bib_titles(bib_path)
    
    print(f"Existing bib keys: {len(existing_keys)}")
    print(f"Existing bib titles: {len(existing_titles)}")
    
    with open("relevant_papers.json") as f:
        new_papers = json.load(f)
        
    # Sort new papers by citation count or year (newest first)
    # Since citationCount might be None, handle it
    new_papers.sort(key=lambda x: (x.get("year") or 0, x.get("citations") or 0), reverse=True)
    
    added_count = 0
    added_entries = []
    added_keys = []
    
    for paper in new_papers:
        title = paper["title"]
        cleaned = clean_title(title).lower()
        if cleaned in existing_titles:
            continue
            
        # check if it's already in the keys by checking if title resembles anything
        # (additional safety)
        key = generate_bib_key(paper["authors"], paper["year"], title, existing_keys)
        bib_entry = format_bibtex(key, paper)
        added_entries.append(bib_entry)
        added_keys.append(key)
        added_count += 1
        
        # We want to add about 35-40 new references to reach a total of ~55 references
        if added_count >= 40:
            break
            
    print(f"Adding {added_count} new papers to {bib_path}")
    
    with open(bib_path, "a") as f:
        f.write("\n\n# New references added during refinement\n\n")
        for entry in added_entries:
            f.write(entry + "\n")
            
    with open("added_keys.json", "w") as f:
        json.dump(added_keys, f, indent=2)
        
    print("Done! Added keys saved to added_keys.json")

if __name__ == "__main__":
    main()
