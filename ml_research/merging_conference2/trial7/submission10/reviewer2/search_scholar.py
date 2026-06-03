import os
import urllib.request
import urllib.parse
import json

def search_paper(query):
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    print(f"API Key present: {bool(api_key)}")
    
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
        
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&fields=title,authors,year,abstract,openAccessPdf&limit=5"
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            results = data.get("data", [])
            print(f"Found {len(results)} results:")
            for i, paper in enumerate(results):
                print(f"\n[{i+1}] Title: {paper.get('title')}")
                print(f"Year: {paper.get('year')}")
                authors_list = paper.get('authors')
                if authors_list:
                    authors = [a.get('name') for a in authors_list if a and a.get('name')]
                    print(f"Authors: {', '.join(authors)}")
                else:
                    print("Authors: None")
                print(f"Paper ID: {paper.get('paperId')}")
                print(f"Abstract: {paper.get('abstract', '')[:300]}...")
                pdf_info = paper.get('openAccessPdf')
                if pdf_info:
                    print(f"PDF URL: {pdf_info.get('url')}")
    except Exception as e:
        print(f"Error querying API: {e}")

if __name__ == "__main__":
    search_paper("REPAIR: Renormalizing representations after model merging")
