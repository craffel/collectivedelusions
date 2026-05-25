import os
import requests
import json

def search_semantic_scholar(query, limit=10):
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
        
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,abstract,citationCount,venue,externalIds&limit={limit}"
    
    print(f"Searching for: '{query}'")
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        results = response.json()
        return results.get("data", [])
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

if __name__ == "__main__":
    queries = [
        "test-time adaptation model merging",
        "elastic weight consolidation test-time adaptation",
        "sharpness-aware test-time adaptation",
        "model merging multitask learning"
    ]
    
    all_papers = {}
    for q in queries:
        papers = search_semantic_scholar(q, limit=5)
        for p in papers:
            all_papers[p['paperId']] = p
            
    print(f"\nFound {len(all_papers)} unique papers.\n")
    for pid, p in list(all_papers.items())[:15]:
        print(f"Title: {p.get('title')}")
        print(f"Authors: {', '.join([a['name'] for a in p.get('authors', [])])}")
        print(f"Year: {p.get('year')}")
        print(f"Venue: {p.get('venue')}")
        print(f"Citations: {p.get('citationCount')}")
        print(f"DOI/ArXiv: {p.get('externalIds', {})}")
        print("-" * 40)
        
    # Save results to a json file
    with open("retrieved_papers.json", "w") as f:
        json.dump(list(all_papers.values()), f, indent=2)
