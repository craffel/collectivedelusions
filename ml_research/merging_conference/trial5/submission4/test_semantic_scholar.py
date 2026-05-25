import os
import requests

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {}
if api_key:
    headers["x-api-key"] = api_key
    print("Using API Key")
else:
    print("No API Key found, using free tier")

query = "S2C-Merge"
url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,openAccessPdf,abstract&openAccessPdf&limit=10"

response = requests.get(url, headers=headers)
if response.status_code == 200:
    results = response.json()
    print(f"Found {results.get('total', 0)} papers.")
    for paper in results.get("data", []):
        print(f"Title: {paper.get('title')}")
        print(f"Year: {paper.get('year')}")
        pdf_info = paper.get("openAccessPdf")
        if pdf_info:
            print(f"PDF URL: {pdf_info.get('url')}")
        else:
            print("No open access PDF")
        print("-" * 40)
else:
    print(f"Error {response.status_code}: {response.text}")
