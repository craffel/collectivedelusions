import os
import requests
import json

def search_papers(query, limit=10):
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
        
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,abstract,openAccessPdf&openAccessPdf&limit={limit}"
    print(f"Querying: {url}")
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data.get("data", [])
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

if __name__ == "__main__":
    queries = [
        "task arithmetic",
        "ties-merging",
        "adamerging",
        "model soups",
        "test-time adaptation"
    ]
    all_results = {}
    for q in queries:
        results = search_papers(q, limit=5)
        all_results[q] = results
        
    with open("search_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("Done! Saved to search_results.json")
