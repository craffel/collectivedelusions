import os
import urllib.request
import json
import urllib.parse

def search_papers(query, limit=20):
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
        
    query_encoded = urllib.parse.quote(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query_encoded}&fields=title,authors,year,citationCount,venue&limit={limit}"
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data.get("data", [])
    except Exception as e:
        print(f"Error searching for query '{query}': {e}")
        return []

def main():
    queries = [
        "model merging",
        "activation calibration deep learning",
        "ties-merging",
        "weight averaging neural networks",
        "representation calibration"
    ]
    
    all_papers = {}
    for q in queries:
        print(f"\nSearching for: {q}")
        papers = search_papers(q, limit=10)
        for p in papers:
            pid = p.get("paperId")
            if pid and pid not in all_papers:
                all_papers[pid] = p
                
    print(f"\nFound {len(all_papers)} unique papers.")
    for i, (pid, p) in enumerate(all_papers.items()):
        authors_list = ", ".join([a["name"] for a in p.get("authors", [])[:3]])
        if len(p.get("authors", [])) > 3:
            authors_list += " et al."
        print(f"[{i+1}] {p.get('title')} ({p.get('year')}) - {authors_list} - Citations: {p.get('citationCount')}")
        
    # Save the search results to a file
    with open("scholar_papers.json", "w") as f:
        json.dump(list(all_papers.values()), f, indent=4)
    print("\nSaved search results to scholar_papers.json")

if __name__ == "__main__":
    main()
