import os
import requests
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {"x-api-key": api_key} if api_key else {}

queries = [
    "model merging deep learning",
    "weight averaging deep learning",
    "activation calibration model merging",
    "multitask model merging",
    "parameter fusion neural networks"
]

results = []
for q in queries:
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(q)}&fields=title,authors,year,venue,externalIds,citationCount&limit=15"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        for paper in data.get("data", []):
            results.append({
                "title": paper.get("title"),
                "authors": [a["name"] for a in paper.get("authors", [])] if paper.get("authors") else [],
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "arxiv": paper.get("externalIds", {}).get("ArXiv"),
                "doi": paper.get("externalIds", {}).get("DOI"),
                "citations": paper.get("citationCount")
            })
    else:
        print(f"Error for query '{q}': {response.status_code}")

# De-duplicate results
unique_results = {}
for r in results:
    if r["title"].lower() not in unique_results:
        unique_results[r["title"].lower()] = r

print(f"Found {len(unique_results)} unique papers.")
with open("found_papers.json", "w") as f:
    json.dump(list(unique_results.values()), f, indent=2)
