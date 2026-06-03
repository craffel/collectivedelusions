import os
import requests

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {"x-api-key": api_key} if api_key else {}

queries = [
    "REPAIR model merging activation",
    "activation calibration model merging",
    "ties-merging",
    "dare-merging"
]

for query in queries:
    print(f"\n=== Query: {query} ===")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query.replace(' ', '+')}&fields=title,authors,year,citationCount,abstract&limit=5"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        for paper in data.get("data", []):
            print(f"Title: {paper.get('title')}")
            print(f"Year: {paper.get('year')}, Citations: {paper.get('citationCount')}")
            authors = ", ".join([a['name'] for a in paper.get('authors', [])])
            print(f"Authors: {authors}")
            abstract = paper.get('abstract') or ""
            print(f"Abstract: {abstract[:200]}...")
            print("-" * 40)
    else:
        print(f"Error {response.status_code}: {response.text}")
