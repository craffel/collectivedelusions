import os
import requests
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {"x-api-key": api_key} if api_key else {}

def search_semantic_scholar(query, limit=5):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,abstract,openAccessPdf&openAccessPdf&limit={limit}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

queries = [
    "model merging task arithmetic",
    "orthogonal model merging",
    "sharpness aware model merging",
    "test time model merging adaptation"
]

for q in queries:
    print(f"=== Query: {q} ===")
    results = search_semantic_scholar(q, limit=3)
    if results and 'data' in results:
        for idx, paper in enumerate(results['data']):
            print(f"[{idx+1}] {paper.get('title')} ({paper.get('year')})")
            print(f"    Authors: {', '.join([a['name'] for a in paper.get('authors', [])])}")
            print(f"    PaperID: {paper.get('paperId')}")
            pdf_info = paper.get('openAccessPdf')
            print(f"    PDF URL: {pdf_info.get('url') if pdf_info else 'None'}")
            print(f"    Abstract: {(paper.get('abstract') or '')[:300]}...")
            print()
