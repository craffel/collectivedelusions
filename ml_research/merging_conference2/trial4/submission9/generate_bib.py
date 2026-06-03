import json
import re

def clean_key(title):
    # Create a simple citation key from the title
    words = re.findall(r'\b\w+\b', title.lower())
    key_words = [w for w in words if len(w) > 3][:3]
    if not key_words:
        key_words = ["paper"]
    return "".join(key_words)

def main():
    try:
        with open("scholar_papers.json", "r") as f:
            papers = json.load(f)
    except Exception as e:
        print(f"Error loading scholar_papers.json: {e}")
        papers = []
        
    bib_entries = []
    
    # 1. Add the 3 foundational submission papers
    foundational = [
        {
            "key": "submission3",
            "title": "Deconstructing Activation Calibration in Multi-Task Model Merging",
            "authors": [{"name": "Anonymous Authors"}],
            "year": 2026,
            "venue": "Conference Submission 3",
            "journal": "arXiv preprint arXiv:2601.00003"
        },
        {
            "key": "submission7",
            "title": "REDA: Representation Calibration and Decision Boundary Alignment in Model Merging",
            "authors": [{"name": "Anonymous Authors"}],
            "year": 2026,
            "venue": "Conference Submission 7",
            "journal": "arXiv preprint arXiv:2601.00007"
        },
        {
            "key": "submission8",
            "title": "SP-TAAC: Sparsity-Preserving Task-Agnostic Activation Calibration",
            "authors": [{"name": "Anonymous Authors"}],
            "year": 2026,
            "venue": "Conference Submission 8",
            "journal": "arXiv preprint arXiv:2601.00008"
        }
    ]
    
    for f in foundational:
        bib = f"@article{{{f['key']},\n"
        bib += f"  title = {{{f['title']}}},\n"
        bib += f"  author = {{{' and '.join([a['name'] for a in f['authors']])}}},\n"
        bib += f"  journal = {{{f['journal']}}},\n"
        bib += f"  year = {{{f['year']}}}\n"
        bib += "}"
        bib_entries.append(bib)
        
    # 2. Add scholar papers
    seen_keys = set(["submission3", "submission7", "submission8"])
    for p in papers:
        title = p.get("title")
        if not title:
            continue
        key_base = clean_key(title)
        year = p.get("year", 2025)
        
        # Ensure unique key
        key = f"{key_base}{year}"
        counter = 1
        while key in seen_keys:
            key = f"{key_base}{year}_{counter}"
            counter += 1
        seen_keys.add(key)
        
        authors = p.get("authors", [])
        author_str = " and ".join([a.get("name", "Unknown") for a in authors])
        if not author_str:
            author_str = "Anonymous"
            
        venue = p.get("venue", "")
        if not venue:
            venue = "arXiv preprint"
            
        # Escape any unescaped ampersands for LaTeX/BibTeX compatibility
        title = title.replace("&", "\\&")
        author_str = author_str.replace("&", "\\&")
        venue = venue.replace("&", "\\&")
            
        bib = f"@article{{{key},\n"
        bib += f"  title = {{{title}}},\n"
        bib += f"  author = {{{author_str}}},\n"
        bib += f"  journal = {{{venue}}},\n"
        bib += f"  year = {{{year}}}\n"
        bib += "}"
        bib_entries.append(bib)
        
    with open("submission.bib", "w") as f:
        f.write("\n\n".join(bib_entries))
        
    print(f"Generated submission.bib with {len(bib_entries)} entries.")

if __name__ == "__main__":
    main()
