import os
import requests

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {"x-api-key": api_key} if api_key else {}

def search_papers(query, limit=10):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields=title,authors,year,openAccessPdf,abstract&openAccessPdf&limit={limit}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

results = search_papers("test-time model merging", limit=10)
if results and "data" in results:
    for paper in results["data"]:
        print(f"Title: {paper.get('title')}")
        print(f"Year: {paper.get('year')}")
        pdf = paper.get('openAccessPdf')
        print(f"PDF URL: {pdf.get('url') if pdf else 'None'}")
        print("-" * 50)
else:
    print("No results found.")
