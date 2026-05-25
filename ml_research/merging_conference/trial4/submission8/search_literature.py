import os
import requests
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {}
if api_key:
    headers["x-api-key"] = api_key

queries = [
    "test-time adaptation model merging",
    "AdaMerging",
    "SyMerge",
    "test-time model merging"
]

for query in queries:
    print(f"\n=== Query: {query} ===")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,year,abstract,openAccessPdf,citationCount&openAccessPdf&limit=5"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            papers = data.get("data", [])
            for i, p in enumerate(papers):
                print(f"{i+1}. Title: {p.get('title')}")
                print(f"   Year: {p.get('year')}")
                print(f"   Citations: {p.get('citationCount')}")
                pdf = p.get("openAccessPdf")
                print(f"   PDF: {pdf.get('url') if pdf else 'None'}")
                abstract = p.get('abstract')
                if abstract:
                    print(f"   Abstract: {abstract[:200]}...")
                print("-" * 40)
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
