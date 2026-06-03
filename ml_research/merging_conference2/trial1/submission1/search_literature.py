import os
import requests
import json

def search_papers(query, limit=10):
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    url = f"https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "fields": "paperId,title,authors,year,openAccessPdf,abstract",
        "openAccessPdf": "",
        "limit": limit
    }
    
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

if __name__ == "__main__":
    queries = [
        "model merging SVD",
        "isotropic model merging",
        "orthogonal model merging",
        "model merging task arithmetic",
        "test-time model merging"
    ]
    
    for q in queries:
        print(f"=== Query: {q} ===")
        papers = search_papers(q, limit=3)
        for i, paper in enumerate(papers):
            print(f"{i+1}. {paper.get('title')} ({paper.get('year')})")
            print(f"   ID: {paper.get('paperId')}")
            pdf_info = paper.get('openAccessPdf')
            pdf_url = pdf_info.get('url') if pdf_info else "None"
            print(f"   PDF: {pdf_url}")
            abstract = paper.get('abstract')
            if abstract:
                # print first 200 chars of abstract
                print(f"   Abstract: {abstract[:200]}...")
            print()
