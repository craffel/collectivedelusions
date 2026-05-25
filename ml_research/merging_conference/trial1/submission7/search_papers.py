import os
import requests
import json

def search_semantic_scholar(query, limit=10):
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {"x-api-key": api_key} if api_key else {}
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "paperId,title,authors,year,abstract,openAccessPdf,citationCount"
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

if __name__ == "__main__":
    queries = [
        "model merging test-time adaptation",
        "model merging sharpness",
        "orthogonal model merging",
        "model merging singular value decomposition",
        "continual learning model merging"
    ]
    for q in queries:
        print(f"=== Query: {q} ===")
        results = search_semantic_scholar(q, limit=3)
        if results and "data" in results:
            for paper in results["data"]:
                print(f"Title: {paper.get('title')}")
                print(f"Year: {paper.get('year')}")
                print(f"Paper ID: {paper.get('paperId')}")
                print(f"OA PDF: {paper.get('openAccessPdf')}")
                print(f"Citations: {paper.get('citationCount')}")
                print("-" * 40)
        print("\n")
