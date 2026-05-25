import os
import requests
import json

def search_semantic_scholar(query, limit=10):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=paperId,title,authors,year,abstract,openAccessPdf&limit={limit}"
    headers = {}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
        
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("data", [])
        else:
            print(f"Error {response.status_code}: {response.text}")
            return []
    except Exception as e:
        print("Request failed:", e)
        return []

def main():
    queries = [
        "model merging machine learning",
        "test-time adaptation",
        "Fisher information model merging",
        "federated model merging",
        "multi-task learning weight interpolation"
    ]
    
    all_papers = {}
    for q in queries:
        print(f"Searching for: '{q}'...")
        papers = search_semantic_scholar(q, limit=5)
        for p in papers:
            all_papers[p["paperId"]] = {
                "title": p.get("title"),
                "year": p.get("year"),
                "authors": [a.get("name") for a in p.get("authors", [])],
                "abstract": p.get("abstract")
            }
            
    with open("retrieved_papers.json", "w") as f:
        json.dump(all_papers, f, indent=2)
        
    print(f"\nSuccessfully retrieved {len(all_papers)} unique papers and saved to retrieved_papers.json")

if __name__ == "__main__":
    main()
