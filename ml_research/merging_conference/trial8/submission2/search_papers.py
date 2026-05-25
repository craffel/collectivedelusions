import requests
import os
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {}
if api_key:
    headers["x-api-key"] = api_key

query = "Test-Time Model Merging"
url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,openAccessPdf,abstract&openAccessPdf&limit=5"

try:
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        print(f"Found {data.get('total', 0)} papers.")
        for paper in data.get("data", []):
            print(f"Title: {paper.get('title')}")
            print(f"Year: {paper.get('year')}")
            print(f"URL: {paper.get('openAccessPdf', {}).get('url') if paper.get('openAccessPdf') else 'None'}")
            print(f"Abstract: {paper.get('abstract', '')[:300]}...")
            print("-"*50)
    else:
        print(f"Error: {response.status_code}, {response.text}")
except Exception as e:
    print(f"Exception: {e}")
