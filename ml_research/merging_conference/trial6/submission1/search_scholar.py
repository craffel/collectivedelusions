import requests
import os
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {}
if api_key:
    headers["x-api-key"] = api_key

query = "model merging test-time"
url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,openAccessPdf,abstract&limit=10"

print(f"Searching for: '{query}'")
response = requests.get(url, headers=headers)
if response.status_code == 200:
    results = response.json()
    print(f"Found {results.get('total', 0)} papers.")
    for i, paper in enumerate(results.get("data", [])):
        print(f"\n[{i+1}] {paper.get('title')} ({paper.get('year')})")
        print(f"ID: {paper.get('paperId')}")
        print(f"Abstract: {paper.get('abstract')[:300]}...")
        print(f"PDF URL: {paper.get('openAccessPdf')}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
