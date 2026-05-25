import os
import urllib.request
import urllib.parse
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
query = "test-time model adaptation model merging"
url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote(query)}&fields=title,authors,year,abstract,openAccessPdf&openAccessPdf&limit=10"

req = urllib.request.Request(url)
if api_key:
    req.add_header("x-api-key", api_key)

print(f"Searching Semantic Scholar for query: {query}")
try:
    with urllib.request.urlopen(req) as response:
        if response.status == 200:
            data = json.loads(response.read().decode())
            results = data.get("data", [])
            print(f"Found {len(results)} results:\n")
            for i, paper in enumerate(results):
                title = paper.get("title", "No Title")
                year = paper.get("year", "No Year")
                
                authors_list = paper.get("authors") or []
                authors = ", ".join([a.get("name", "") for a in authors_list if a and isinstance(a, dict)])
                
                abstract = paper.get("abstract") or "No abstract available"
                pdf_info = paper.get("openAccessPdf")
                pdf_url = pdf_info.get("url") if (pdf_info and isinstance(pdf_info, dict)) else "None"
                
                print(f"[{i+1}] {title} ({year})")
                print(f"Authors: {authors}")
                print(f"PDF URL: {pdf_url}")
                print(f"Abstract: {abstract[:400]}...")
                print("-" * 80)
        else:
            print(f"Error: {response.status}")
except Exception as e:
    import traceback
    print(f"Request failed: {e}")
    traceback.print_exc()
