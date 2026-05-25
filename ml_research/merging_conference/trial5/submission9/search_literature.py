import requests
import json
import os

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {}
if api_key:
    headers["x-api-key"] = api_key

query = "test-time model merging"
url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,abstract,openAccessPdf,citationCount&limit=10"

print(f"Querying Semantic Scholar for: '{query}'...")
try:
    response = requests.get(url, headers=headers)
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Total results: {data.get('total', 0)}")
        for i, paper in enumerate(data.get("data", [])):
            print("\n" + "="*50)
            print(f"{i+1}. {paper.get('title')} ({paper.get('year')})")
            print(f"Citations: {paper.get('citationCount', 0)}")
            print(f"ID: {paper.get('paperId')}")
            pdf_info = paper.get("openAccessPdf")
            if pdf_info:
                print(f"PDF URL: {pdf_info.get('url')}")
            else:
                print("PDF: None")
            abstract = paper.get("abstract")
            if abstract:
                print(f"Abstract: {abstract[:800]}...")
            else:
                print("Abstract: None")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
