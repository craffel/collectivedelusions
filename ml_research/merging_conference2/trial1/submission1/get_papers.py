import os
import requests
import json
import re

def search_papers(query, limit=15):
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "fields": "paperId,title,authors,year,venue,citationStyles",
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json().get("data", [])
        else:
            print(f"Error {response.status_code} for query '{query}': {response.text}")
            return []
    except Exception as e:
        print(f"Exception for query '{query}': {e}")
        return []

def main():
    queries = [
        "model merging",
        "weight averaging deep learning",
        "task arithmetic",
        "ties merging",
        "model soup",
        "linear mode connectivity",
        "federated learning weight averaging",
        "loss landscape model merging",
        "parameter grafting",
        "test-time model adaptation merging",
        "SVD model compression merging",
        "procrustes model merging"
    ]
    
    seen_titles = set()
    bibtex_entries = []
    
    # Also add some manual entries or ensure we get at least 60 entries
    for q in queries:
        print(f"Searching for: {q}")
        papers = search_papers(q, limit=15)
        for p in papers:
            title = p.get("title", "").strip().lower()
            if not title or title in seen_titles:
                continue
            
            citation_styles = p.get("citationStyles")
            if citation_styles and "bibtex" in citation_styles:
                bibtex = citation_styles["bibtex"]
                # Clean up some common issues in SS bibtex if any
                seen_titles.add(title)
                bibtex_entries.append(bibtex)
                print(f"  Added: {p.get('title')}")
                
        print(f"Current unique bibtex count: {len(bibtex_entries)}")
        
    print(f"Total unique bibtex count collected: {len(bibtex_entries)}")
    
    with open("submission.bib", "w", encoding="utf-8") as f:
        for entry in bibtex_entries:
            f.write(entry + "\n\n")
            
    print("Saved to submission.bib")

if __name__ == "__main__":
    main()
