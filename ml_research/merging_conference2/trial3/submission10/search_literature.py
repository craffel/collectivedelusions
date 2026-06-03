import requests
import os
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {"x-api-key": api_key} if api_key else {}

def search_papers(query, limit=10):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,abstract,openAccessPdf&openAccessPdf&limit={limit}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    queries = [
        "model merging activation calibration",
        "model merging REPAIR",
        "task arithmetic representation calibration"
    ]
    for q in queries:
        print(f"=== Query: {q} ===")
        results = search_papers(q, limit=5)
        if results and "data" in results:
            for paper in results["data"]:
                print(f"Title: {paper.get('title')}")
                print(f"Year: {paper.get('year')}")
                print(f"Authors: {', '.join([a['name'] for a in paper.get('authors', [])])}")
                pdf_info = paper.get('openAccessPdf')
                pdf_url = pdf_info.get('url') if pdf_info else None
                print(f"PDF URL: {pdf_url}")
                print("-" * 40)
