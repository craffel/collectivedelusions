import urllib.request
import json
import os

def search_semantic_scholar(query, limit=5):
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    query_encoded = urllib.parse.quote_plus(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query_encoded}&fields=title,authors,year,abstract,openAccessPdf&openAccessPdf&limit={limit}"
    
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("x-api-key", api_key)
        
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data.get("data", [])
    except Exception as e:
        print(f"Error querying API: {e}")
        return []

if __name__ == "__main__":
    queries = [
        "Test-Time Model Merging",
        "Test-Time Adaptation model merging",
        "Sharpness-Aware Minimization model merging"
    ]
    for q in queries:
        print(f"\n--- Query: {q} ---")
        results = search_semantic_scholar(q, limit=3)
        for i, paper in enumerate(results):
            print(f"{i+1}. {paper.get('title')} ({paper.get('year')})")
            print(f"   ID: {paper.get('paperId')}")
            pdf_info = paper.get('openAccessPdf')
            pdf_url = pdf_info.get('url') if pdf_info else "None"
            print(f"   PDF URL: {pdf_url}")
            abstract = paper.get('abstract', '')
            if abstract:
                print(f"   Abstract: {abstract[:200]}...")
