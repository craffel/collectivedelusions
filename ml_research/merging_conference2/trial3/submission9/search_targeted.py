import os
import requests
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {"x-api-key": api_key} if api_key else {}

queries = [
    "TIES-Merging",
    "DARE model merging",
    "Git Re-Basin",
    "Model soups",
    "AdaMerging",
    "REPAIR model merging",
    "Task Arithmetic model merging",
    "ZipIt model merging",
    "activation calibration model merging",
    "model merging survey",
    "stochastic weight averaging",
    "sharpness-aware minimization weight averaging",
    "federated weight averaging",
    "layer-wise model merging",
    "representation alignment model merging"
]

results = []
for q in queries:
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(q)}&fields=title,authors,year,venue,externalIds,citationCount&limit=20"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        for paper in data.get("data", []):
            results.append({
                "title": paper.get("title"),
                "authors": [a["name"] for a in paper.get("authors", [])] if paper.get("authors") else [],
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "arxiv": paper.get("externalIds", {}).get("ArXiv"),
                "doi": paper.get("externalIds", {}).get("DOI"),
                "citations": paper.get("citationCount")
            })
    else:
        print(f"Error for query '{q}': {response.status_code}")

# Merge with existing found_papers.json
if os.path.exists("found_papers.json"):
    with open("found_papers.json") as f:
        existing = json.load(f)
    results.extend(existing)

# De-duplicate results
unique_results = {}
for r in results:
    if r["title"].lower() not in unique_results:
        unique_results[r["title"].lower()] = r

print(f"Combined and found {len(unique_results)} unique papers.")

# Filter by relevance keywords
keywords = [
    "merg", "averag", "fusion", "soup", "arithmetic", "align", 
    "calibrat", "conflict", "deconflict", "ties", "dare", "re-basin", 
    "rebasin", "flat", "minima", "landscape", "multitask", "multi-task"
]

relevant_papers = []
for title, paper in unique_results.items():
    # check if any keyword is in title (case insensitive)
    if any(kw in title for kw in keywords):
        # filter out some noisy non-ML papers
        exclude_kw = ["crash", "freeway", "lake", "highway", "traffic", "vehicle", "intrusion", "iot", "alzheimer"]
        if not any(ek in title for ek in exclude_kw):
            relevant_papers.append(paper)

print(f"Filtered to {len(relevant_papers)} relevant papers.")
with open("relevant_papers.json", "w") as f:
    json.dump(relevant_papers, f, indent=2)
