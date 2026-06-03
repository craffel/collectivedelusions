import os
import requests
import json

def search_semantic_scholar(query, limit=5):
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
        
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "fields": "title,authors,year,abstract,citationCount,venue",
        "limit": limit
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        results = response.json().get("data", [])
        return results
    except Exception as e:
        print(f"Error querying Semantic Scholar: {e}")
        return []

if __name__ == "__main__":
    queries = [
        "model merging representation collapse",
        "post-training quantization model merging",
        "data-free parameter calibration neural network"
    ]
    
    for q in queries:
        print(f"\nSearching for: '{q}'")
        print("-" * 50)
        results = search_semantic_scholar(q, limit=3)
        for i, paper in enumerate(results):
            print(f"{i+1}. {paper.get('title')} ({paper.get('year')})")
            authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])])
            print(f"   Authors: {authors}")
            print(f"   Venue: {paper.get('venue')}")
            print(f"   Citations: {paper.get('citationCount')}")
            abstract = paper.get('abstract', '')
            if abstract:
                print(f"   Abstract: {abstract[:200]}...")
            print()
