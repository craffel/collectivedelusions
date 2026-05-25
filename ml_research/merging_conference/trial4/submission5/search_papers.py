import os
import requests
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {"x-api-key": api_key} if api_key else {}

def search_papers(query, limit=10):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=paperId,title,authors,year,abstract,openAccessPdf&openAccessPdf&limit={limit}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

queries = [
    "test-time model merging",
    "test-time adaptation model merging",
    "SATA-TTA",
    "SyMerge"
]

for q in queries:
    print("="*60)
    print(f"Query: {q}")
    print("="*60)
    results = search_papers(q, limit=5)
    if results and "data" in results:
        for idx, paper in enumerate(results["data"]):
            print(f"{idx+1}. {paper.get('title')} ({paper.get('year')})")
            print(f"ID: {paper.get('paperId')}")
            pdf = paper.get("openAccessPdf")
            print(f"PDF URL: {pdf.get('url') if pdf else 'None'}")
            abstract = paper.get("abstract")
            if abstract:
                print(f"Abstract: {abstract[:400]}...")
            else:
                print("Abstract: None")
            print("-" * 40)
    else:
        print("No results or error.")
