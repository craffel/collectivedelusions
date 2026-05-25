import os
import requests
import json

def search_papers(query, limit=10):
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,abstract,openAccessPdf&limit={limit}"
    print(f"Searching: {url}")
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data.get("data", [])
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

if __name__ == "__main__":
    queries = [
        "model merging test-time adaptation",
        "sharpness-aware minimization model merging",
        "parameter-efficient fine-tuning model merging"
    ]
    for q in queries:
        print(f"\n=== Query: {q} ===")
        results = search_papers(q, limit=5)
        for r in results:
            print(f"- Title: {r.get('title')}")
            print(f"  Year: {r.get('year')}")
            print(f"  Abstract: {r.get('abstract')[:300] if r.get('abstract') else 'N/A'}...")
            pdf = r.get("openAccessPdf")
            print(f"  PDF: {pdf.get('url') if pdf else 'N/A'}")
