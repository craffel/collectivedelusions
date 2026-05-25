import os
import requests
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {"x-api-key": api_key} if api_key else {}

def search_papers(query, limit=5):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,abstract,citationCount,openAccessPdf&limit={limit}"
    print(f"Searching for: {query}")
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            papers = data.get("data", [])
            print(f"Found {len(papers)} papers:")
            for idx, p in enumerate(papers):
                print(f"[{idx+1}] {p.get('title')} ({p.get('year')})")
                print(f"    Citations: {p.get('citationCount')}")
                pdf_info = p.get('openAccessPdf')
                print(f"    PDF URL: {pdf_info.get('url') if pdf_info else 'None'}")
                print(f"    Abstract: {p.get('abstract')[:300] if p.get('abstract') else 'No abstract'}")
                print()
        else:
            print(f"Failed to search: {response.status_code} - {response.text}")
    except Exception as e:
        print("Error during search:", e)

search_papers("model merging sharpness aware", limit=3)
search_papers("SAM model merging", limit=3)
search_papers("linear mode connectivity SAM", limit=3)
