import urllib.request
import urllib.parse
import json
import os

query = "model merging activation calibration"
api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

# URL encode the query
params = {
    "query": query,
    "fields": "title,authors,year,openAccessPdf,abstract",
    "limit": 5
}
url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(params)

req = urllib.request.Request(url)
if api_key:
    req.add_header("x-api-key", api_key)

try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print(f"Found {len(data.get('data', []))} papers:")
        for paper in data.get("data", []):
            print("-" * 50)
            print(f"Title: {paper.get('title')}")
            print(f"Year: {paper.get('year')}")
            pdf_info = paper.get("openAccessPdf")
            if pdf_info:
                print(f"PDF URL: {pdf_info.get('url')}")
            else:
                print("PDF: None")
            print(f"Abstract: {paper.get('abstract', '')[:200]}...")
except Exception as e:
    print(f"Error querying Semantic Scholar: {e}")
