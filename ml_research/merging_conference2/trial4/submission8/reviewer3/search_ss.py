import urllib.request
import json
import os

api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

def search_paper(query, limit=5):
    query_encoded = urllib.parse.quote(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query_encoded}&fields=title,authors,year,abstract,citationCount,openAccessPdf&limit={limit}"
    
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("x-api-key", api_key)
        
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data
    except Exception as e:
        print("Error during API request:", e)
        return None

if __name__ == "__main__":
    queries = [
        "REDA model merging",
        "SP-TAAC",
        "Sparsity-Preserving Task-Agnostic Calibration",
        "classifier head alignment model merging"
    ]
    for q in queries:
        print(f"========================================\nSearching for: {q}")
        res = search_paper(q, 3)
        if res and "data" in res:
            for p in res["data"]:
                print(f"- Title: {p.get('title')}")
                print(f"  Year: {p.get('year')}")
                print(f"  Citations: {p.get('citationCount')}")
                print(f"  PDF URL: {p.get('openAccessPdf', {}).get('url') if p.get('openAccessPdf') else 'None'}")
                print()
        else:
            print("No results or error.")

