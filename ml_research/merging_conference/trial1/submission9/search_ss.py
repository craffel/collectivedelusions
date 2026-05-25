import os
import requests
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {}
if api_key:
    headers["x-api-key"] = api_key

def search_papers(query, limit=5):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,abstract,openAccessPdf,citationCount&openAccessPdf&limit={limit}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

queries = [
    "model merging task vectors",
    "test-time adaptive model merging",
    "orthogonal model merging",
    "isotropic model merging"
]

results = {}
for q in queries:
    print(f"=== Searching for: '{q}' ===")
    papers = search_papers(q, limit=3)
    results[q] = []
    for paper in papers:
        print(f"Title: {paper.get('title')}")
        print(f"Year: {paper.get('year')}")
        print(f"Citations: {paper.get('citationCount')}")
        pdf = paper.get('openAccessPdf')
        pdf_url = pdf.get('url') if pdf else "None"
        print(f"PDF: {pdf_url}")
        abstract = paper.get('abstract') or ""
        print(f"Abstract: {abstract[:300]}...")
        print("-" * 40)
        results[q].append({
            "paperId": paper.get("paperId"),
            "title": paper.get("title"),
            "year": paper.get("year"),
            "abstract": abstract,
            "pdf_url": pdf_url
        })

with open("literature_results.json", "w") as f:
    json.dump(results, f, indent=2)
