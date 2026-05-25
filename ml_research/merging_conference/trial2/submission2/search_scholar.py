import os
import requests
import json
import time

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {}
if api_key:
    headers["x-api-key"] = api_key

def search_papers(query, limit=5):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "fields": "title,authors,year,abstract,openAccessPdf,citationCount",
        "limit": limit,
        "openAccessPdf": ""
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 429:
            print("Rate limited. Sleeping for 5 seconds...")
            time.sleep(5)
            response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json().get("data", [])
        else:
            print(f"Error searching {query}: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Exception searching {query}: {e}")
        return []

queries = [
    "sharpness aware model merging",
    "SAM model merging",
    "test time model merging",
    "LoRA model merging",
    "orthogonal model merging"
]

results = {}
for q in queries:
    print(f"Searching for '{q}'...")
    papers = search_papers(q, limit=3)
    results[q] = papers
    time.sleep(1) # politely sleep between requests

with open("literature_search_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Done! Saved to literature_search_results.json.")
