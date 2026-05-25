import os
import urllib.request
import urllib.parse
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
query = "test-time model merging"
encoded_query = urllib.parse.quote(query)
url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&fields=title,authors,year,abstract,openAccessPdf&openAccessPdf&limit=5"

req = urllib.request.Request(url)
if api_key:
    req.add_header("x-api-key", api_key)

try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode())
        print(f"Found {len(res.get('data', []))} papers:")
        for idx, paper in enumerate(res.get("data", [])):
            print(f"\n[{idx+1}] {paper.get('title')} ({paper.get('year')})")
            print(f"ID: {paper.get('paperId')}")
            pdf_info = paper.get("openAccessPdf")
            print(f"PDF URL: {pdf_info.get('url') if pdf_info else 'None'}")
            print(f"Abstract: {paper.get('abstract')[:300]}...")
except Exception as e:
    print(f"Error querying Semantic Scholar: {e}")
