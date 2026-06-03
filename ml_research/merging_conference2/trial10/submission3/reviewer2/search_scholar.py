import os
import requests
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {"x-api-key": api_key} if api_key else {}

def search_papers(query, limit=10):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,abstract,citationCount,openAccessPdf&limit={limit}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

queries = [
    "isotropic parameter resonance model merging",
    "holographic norm scaling model merging",
    "representation collapse model merging",
    "variance decay model merging"
]

results = {}
for q in queries:
    print(f"Searching for: '{q}'...")
    papers = search_papers(q, limit=5)
    results[q] = []
    for p in papers:
        results[q].append({
            "paperId": p.get("paperId"),
            "title": p.get("title"),
            "year": p.get("year"),
            "citationCount": p.get("citationCount"),
            "openAccessPdf": p.get("openAccessPdf"),
            "abstract": p.get("abstract")
        })

with open("search_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Search complete. Results written to search_results.json")
