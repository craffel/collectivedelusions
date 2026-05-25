import os
import requests
import json
import re

def search_semantic_scholar(query, limit=15):
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {"x-api-key": api_key} if api_key else {}
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "paperId,title,authors,year,venue,citationCount,externalIds"
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get("data", [])
        else:
            print(f"Error {response.status_code} for query '{query}': {response.text}")
            return []
    except Exception as e:
        print(f"Exception for query '{query}': {e}")
        return []

def clean_name(name):
    # Remove non-ascii or special chars for bibkey
    name = re.sub(r'[^a-zA-Z0-9]', '', name)
    return name

def make_bib_entry(paper):
    title = paper.get("title", "No Title")
    authors = paper.get("authors", [])
    year = paper.get("year", 2024)
    venue = paper.get("venue", "")
    
    if not authors:
        author_str = "Anonymous"
        first_author = "anon"
    else:
        author_names = [a.get("name", "") for a in authors if a.get("name")]
        if not author_names:
            author_str = "Anonymous"
            first_author = "anon"
        else:
            author_str = " and ".join(author_names)
            first_author = clean_name(author_names[0].split()[-1]) if author_names[0].split() else "author"
    
    first_word_title = clean_name(title.split()[0]) if title.split() else "paper"
    bibkey = f"{first_author.lower()}{year}{first_word_title.lower()}"
    
    # Clean venue
    if not venue:
        # Check if arXiv in externalIds
        ext_ids = paper.get("externalIds", {})
        if ext_ids and "ArXiv" in ext_ids:
            venue = f"arXiv preprint arXiv:{ext_ids['ArXiv']}"
        else:
            venue = "CoRR"
            
    entry = f"@article{{{bibkey},\n"
    entry += f"  title={{{title}}},\n"
    entry += f"  author={{{author_str}}},\n"
    entry += f"  journal={{{venue}}},\n"
    entry += f"  year={{{year}}},\n"
    entry += f"  publisher={{IEEE}}\n"
    entry += f"}}\n\n"
    return bibkey, entry

def main():
    queries = [
        "model merging machine learning",
        "task arithmetic model merging",
        "model soups deep learning",
        "test-time adaptation deep learning",
        "sharpness-aware minimization deep learning",
        "flatness generalization deep learning",
        "unsupervised test-time adaptation",
        "federated learning model merging",
        "linear mode connectivity",
        "weight ensembling neural networks",
        "out of distribution robustness test-time",
        "knowledge distillation soft cross entropy"
    ]
    
    all_papers = {}
    bib_entries = []
    keys_seen = set()
    
    for q in queries:
        print(f"Searching for '{q}'...")
        results = search_semantic_scholar(q, limit=8)
        for p in results:
            pid = p.get("paperId")
            if pid and pid not in all_papers:
                all_papers[pid] = p
                
    print(f"Found {len(all_papers)} unique papers. Generating bib entries...")
    
    for pid, p in all_papers.items():
        key, entry = make_bib_entry(p)
        if key not in keys_seen:
            keys_seen.add(key)
            bib_entries.append(entry)
            
    # Write to submission.bib
    with open("submission.bib", "w") as f:
        # Add a few classic references manually if needed
        f.write("% Auto-generated bibliography for SATT-Merge\n\n")
        for entry in bib_entries:
            f.write(entry)
            
    print(f"Successfully wrote {len(bib_entries)} references to submission.bib")

if __name__ == "__main__":
    main()
