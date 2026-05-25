import os
import requests
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
headers = {}
if api_key:
    headers["x-api-key"] = api_key

def search_papers(query, limit=15):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,venue,externalIds&limit={limit}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

queries = [
    "model merging deep learning",
    "test-time adaptation",
    "git re-basin model merging",
    "model soups",
    "ties-merging yadav",
    "dare model merging yu"
]

all_papers = []
seen_titles = set()

for q in queries:
    print(f"Searching for: {q}")
    papers = search_papers(q, limit=10)
    for p in papers:
        title = p.get("title", "")
        if title.lower() not in seen_titles:
            seen_titles.add(title.lower())
            all_papers.append(p)

print(f"Found {len(all_papers)} unique papers.")

# Write results to a json file
with open("found_papers.json", "w") as f:
    json.dump(all_papers, f, indent=2)

print("Saved to found_papers.json")
