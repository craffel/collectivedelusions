import os
import requests
import json

def search_papers(query, limit=20):
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {"x-api-key": api_key} if api_key else {}
    
    url = f"https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "fields": "title,authors,year,citationStyles,openAccessPdf",
        "limit": limit
    }
    
    r = requests.get(url, params=params, headers=headers)
    if r.status_code != 200:
        print(f"Error: {r.status_code}")
        print(r.text)
        return []
    
    return r.json().get("data", [])

if __name__ == "__main__":
    queries = [
        "model merging weight averaging",
        "task arithmetic ties-merging",
        "batchnorm calibration transfer learning",
        "federated learning batchnorm normalization"
    ]
    
    all_bibtexs = []
    seen_titles = set()
    
    for q in queries:
        print(f"--- Querying: {q} ---")
        papers = search_papers(q, limit=15)
        for p in papers:
            title = p.get("title")
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            bibtex = p.get("citationStyles", {}).get("bibtex")
            if bibtex:
                all_bibtexs.append((title, bibtex))
                print(f"Found: {title}")
                
    print(f"\nTotal bibtexs found: {len(all_bibtexs)}")
    with open("fetched_bibtexs.txt", "w") as f:
        for title, bibtex in all_bibtexs:
            f.write(f"% {title}\n{bibtex}\n\n")
