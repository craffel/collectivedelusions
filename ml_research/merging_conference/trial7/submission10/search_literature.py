import urllib.request
import json
import os

def search_semantic_scholar(query, limit=10):
    print(f"Searching Semantic Scholar for: {query}")
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&fields=title,authors,year,abstract,citationCount,openAccessPdf&openAccessPdf&limit={limit}"
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            results = data.get("data", [])
            print(f"Found {len(results)} papers.")
            for i, paper in enumerate(results):
                print(f"\n[{i+1}] {paper.get('title')} ({paper.get('year')})")
                authors = ", ".join([a.get("name") for a in paper.get("authors", [])])
                print(f"Authors: {authors}")
                print(f"Citations: {paper.get('citationCount')}")
                pdf_info = paper.get("openAccessPdf")
                if pdf_info:
                    print(f"PDF URL: {pdf_info.get('url')}")
                abstract = paper.get("abstract")
                if abstract:
                    print(f"Abstract: {abstract[:300]}...")
    except Exception as e:
        print(f"Error during search: {e}")

if __name__ == "__main__":
    search_semantic_scholar("Enneng Yang model merging", limit=10)
    print("\n" + "="*80 + "\n")
    search_semantic_scholar("test-time model merging", limit=10)
