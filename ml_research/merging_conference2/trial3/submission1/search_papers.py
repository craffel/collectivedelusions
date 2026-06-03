import os
import requests
import json

def search_scholar(query, limit=10):
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    params = {
        "query": query,
        "fields": "title,authors,year,citationCount,abstract,openAccessPdf",
        "limit": limit,
        "openAccessPdf": ""
    }
    
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

if __name__ == "__main__":
    queries = [
        "model merging representation alignment",
        "model merging activation calibration",
        "variance collapse model merging",
        "deep neural network model merging head adaptation"
    ]
    for q in queries:
        print(f"=== Query: {q} ===")
        results = search_scholar(q, limit=5)
        if results and "data" in results:
            for paper in results["data"]:
                print(f"Title: {paper.get('title')}")
                print(f"Year: {paper.get('year')}")
                print(f"Citations: {paper.get('citationCount')}")
                print(f"PDF URL: {paper.get('openAccessPdf', {}).get('url') if paper.get('openAccessPdf') else 'None'}")
                abstract = paper.get('abstract')
                if abstract:
                    print(f"Abstract: {abstract[:200]}...")
                print("-" * 40)
        print("\n" + "="*80 + "\n")
