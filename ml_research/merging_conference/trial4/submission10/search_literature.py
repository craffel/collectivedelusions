import os
import requests
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {"x-api-key": api_key} if api_key else {}

url = "https://api.semanticscholar.org/graph/v1/paper/search"
params = {
    "query": "test-time model merging",
    "fields": "title,authors,year,abstract,openAccessPdf",
    "limit": 10
}

print("Searching Semantic Scholar for 'test-time model merging'...")
response = requests.get(url, params=params, headers=headers)
if response.status_code == 200:
    results = response.json().get("data", [])
    print(f"Found {len(results)} results:\n")
    for i, paper in enumerate(results):
        print(f"[{i+1}] Title: {paper.get('title')}")
        print(f"    Year: {paper.get('year')}")
        abstract = paper.get('abstract')
        if abstract:
            print(f"    Abstract: {abstract[:400]}...")
        else:
            print("    Abstract: N/A")
        print(f"    PDF URL: {paper.get('openAccessPdf', {}).get('url') if paper.get('openAccessPdf') else 'N/A'}")
        print("-" * 40)
else:
    print(f"Search failed with status code {response.status_code}: {response.text}")
