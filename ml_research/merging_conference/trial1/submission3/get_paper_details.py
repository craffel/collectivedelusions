import os
import requests

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {"x-api-key": api_key} if api_key else {}

title = "Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning"
url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(title)}&fields=title,authors,year,abstract,citationCount,openAccessPdf&limit=1"
response = requests.get(url, headers=headers)
if response.status_code == 200:
    data = response.json()
    papers = data.get("data", [])
    if papers:
        p = papers[0]
        print("Title:", p.get("title"))
        print("Authors:", [a.get("name") for a in p.get("authors", [])])
        print("Year:", p.get("year"))
        print("Abstract:")
        print(p.get("abstract"))
    else:
        print("Paper not found on Semantic Scholar.")
else:
    print("Request failed:", response.status_code)
