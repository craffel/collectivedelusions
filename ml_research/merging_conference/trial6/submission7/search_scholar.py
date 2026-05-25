import os
import requests
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
headers = {}
if api_key:
    headers["x-api-key"] = api_key

query = "test-time model merging"
url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,abstract,citationCount,openAccessPdf&limit=10"

print(f"Searching for query: {query}")
response = requests.get(url, headers=headers)
if response.status_code == 200:
    results = response.json()
    print(f"Found {len(results.get('data', []))} results:")
    for idx, paper in enumerate(results.get('data', [])):
        print(f"\n[{idx+1}] Title: {paper.get('title')}")
        print(f"Year: {paper.get('year')} | Citations: {paper.get('citationCount')}")
        print(f"Paper ID: {paper.get('paperId')}")
        pdf_info = paper.get('openAccessPdf')
        if pdf_info:
            print(f"PDF URL: {pdf_info.get('url')}")
        else:
            print("PDF: None")
        abstract = paper.get('abstract', '')
        if abstract:
            print(f"Abstract: {abstract[:300]}...")
else:
    print(f"Error {response.status_code}: {response.text}")
