import os
import requests
import json
import sys

def search_semantic_scholar(query, limit=10):
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "paperId,title,authors,year,abstract,openAccessPdf"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "model merging"
    results = search_semantic_scholar(query)
    if results and "data" in results:
        print(json.dumps(results["data"], indent=2))
    else:
        print("No results found.")
