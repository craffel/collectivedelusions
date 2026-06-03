import requests
import os
import json

api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
headers = {}
if api_key:
    headers["x-api-key"] = api_key

queries = [
    "model merging activation calibration",
    "multi task model merging representation collapse",
    "neural network weight alignment model merging"
]

results = {}
for q in queries:
    print(f"Searching for: '{q}'...")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(q)}&fields=title,authors,year,abstract,openAccessPdf&openAccessPdf&limit=5"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        results[q] = data.get("data", [])
    else:
        print(f"Error {response.status_code}: {response.text}")

print("\n--- Search Results ---")
for q, papers in results.items():
    print(f"\nQuery: {q}")
    for i, paper in enumerate(papers):
        pdf_url = paper.get("openAccessPdf", {}).get("url") if paper.get("openAccessPdf") else None
        print(f"{i+1}. {paper['title']} ({paper.get('year')})")
        print(f"   Paper ID: {paper['paperId']}")
        print(f"   PDF URL: {pdf_url}")
        if paper.get("abstract"):
            print(f"   Abstract: {paper['abstract'][:300]}...")
