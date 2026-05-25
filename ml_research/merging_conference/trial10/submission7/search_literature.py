import requests
import os
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {"x-api-key": api_key} if api_key else {}

queries = [
    "test-time model merging",
    "test-time adaptation model merging",
    "noise robust model merging",
    "data-free model merging"
]

results_dict = {}

for q in queries:
    print(f"Searching for: '{q}'")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(q)}&fields=title,authors,year,abstract,openAccessPdf,citationCount&limit=8"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            papers = data.get("data", [])
            print(f"Found {len(papers)} papers.")
            results_dict[q] = papers
            for paper in papers:
                title = paper.get("title")
                year = paper.get("year")
                citations = paper.get("citationCount", 0)
                pdf_info = paper.get("openAccessPdf")
                has_pdf = "Yes" if pdf_info and pdf_info.get("url") else "No"
                print(f"- {title} ({year}) [Citations: {citations}, PDF: {has_pdf}]")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")
    print()

with open("literature_search_results.json", "w") as f:
    json.dump(results_dict, f, indent=2)
