import os
import urllib.request
import urllib.parse
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

queries = [
    "model merging neural networks",
    "sharpness-aware minimization weight space",
    "test-time adaptation stream"
]

for query in queries:
    print(f"\n===== QUERY: {query} =====")
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&fields=title,authors,year,abstract,openAccessPdf&openAccessPdf&limit=3"

    req = urllib.request.Request(url)
    if api_key:
        req.add_header("x-api-key", api_key)

    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode())
            data = res.get('data', [])
            print(f"Found {len(data)} papers:")
            for idx, paper in enumerate(data):
                title = paper.get('title')
                year = paper.get('year')
                paper_id = paper.get('paperId')
                pdf_info = paper.get("openAccessPdf")
                pdf_url = pdf_info.get('url') if pdf_info else 'None'
                abstract = paper.get('abstract')
                abstract_snippet = abstract[:200] if abstract else "No abstract"
                print(f"[{idx+1}] {title} ({year}) | ID: {paper_id}")
                print(f"    PDF URL: {pdf_url}")
                print(f"    Abstract: {abstract_snippet}...")
    except Exception as e:
        print(f"Error querying for '{query}': {e}")
