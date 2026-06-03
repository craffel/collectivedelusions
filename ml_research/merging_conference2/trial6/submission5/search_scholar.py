import os
import urllib.request
import json
import urllib.parse

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
query = "model merging activation calibration"
query_encoded = urllib.parse.quote(query)
url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query_encoded}&fields=title,authors,year,abstract,openAccessPdf&openAccessPdf&limit=10"

req = urllib.request.Request(url)
if api_key:
    req.add_header("x-api-key", api_key)

try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        results = data.get("data", [])
        print(f"Found {len(results)} results:")
        for paper in results:
            print(f"Title: {paper.get('title')}")
            print(f"Year: {paper.get('year')}")
            print(f"URL: {paper.get('openAccessPdf', {}).get('url') if paper.get('openAccessPdf') else 'None'}")
            print(f"Abstract: {paper.get('abstract')[:300] if paper.get('abstract') else 'None'}...")
            print("-" * 40)
except Exception as e:
    print(f"Error: {e}")
