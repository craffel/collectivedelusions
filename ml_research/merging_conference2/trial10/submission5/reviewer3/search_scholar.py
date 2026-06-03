import os
import requests
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {"x-api-key": api_key} if api_key else {}

queries = [
    "model merging Wasserstein",
    "model merging optimal transport",
    "Wasserstein-calibrated parameter resonance"
]

for query in queries:
    print(f"\n=== Query: {query} ===")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,abstract,openAccessPdf&limit=3"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        for i, paper in enumerate(data.get("data", [])):
            print(f"\nPaper {i+1}: {paper.get('title')} ({paper.get('year')})")
            print(f"ID: {paper.get('paperId')}")
            print(f"Abstract: {paper.get('abstract')[:300]}...")
            pdf = paper.get("openAccessPdf")
            if pdf:
                print(f"PDF URL: {pdf.get('url')}")
            else:
                print("No open access PDF")
    else:
        print(f"Error {response.status_code}: {response.text}")
