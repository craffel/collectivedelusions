import os
import requests
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
headers = {"x-api-key": api_key} if api_key else {}

def search_semantic_scholar(query, limit=20):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,citationCount,venue,externalIds&limit={limit}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

queries = [
    "model merging deep learning",
    "weight averaging deep learning",
    "model soups",
    "batchnorm calibration domain adaptation",
    "test time adaptation batchnorm"
]

all_papers = {}
for q in queries:
    print(f"Searching for: {q}")
    results = search_semantic_scholar(q, limit=15)
    for p in results:
        paper_id = p.get("paperId")
        if paper_id and paper_id not in all_papers:
            all_papers[paper_id] = p

print(f"Found {len(all_papers)} unique papers.")
with open("semantic_scholar_results.json", "w") as f:
    json.dump(list(all_papers.values()), f, indent=2)
