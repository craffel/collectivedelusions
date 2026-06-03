import urllib.request
import json
import os

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

def search_papers(query, limit=5):
    print(f"=== Searching for: '{query}' ===")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&fields=title,authors,year,abstract,openAccessPdf&openAccessPdf&limit={limit}"
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            results = data.get("data", [])
            for p in results:
                print(f"Title: {p.get('title')}")
                print(f"Year: {p.get('year')}")
                pdf_info = p.get("openAccessPdf")
                pdf_url = pdf_info.get("url") if pdf_info else "None"
                print(f"PDF URL: {pdf_url}")
                abstract = p.get("abstract", "")
                if abstract:
                    print(f"Abstract: {abstract[:300]}...")
                print("-"*20)
    except Exception as e:
        print("Error:", e)

search_papers("model merging deep learning", 4)
search_papers("task arithmetic model merging", 4)
search_papers("ties-merging deep learning", 4)
search_papers("REPAIR model merging neural networks", 4)
