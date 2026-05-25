import os
import requests

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {"x-api-key": api_key} if api_key else {}
query = "test-time adaptation model merging"
url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query.replace(' ', '+')}&fields=title,authors,year,abstract,openAccessPdf&limit=10"
response = requests.get(url, headers=headers)
if response.status_code == 200:
    data = response.json()
    for i, p in enumerate(data.get("data", [])):
        print(f"[{i+1}] {p.get('title')} ({p.get('year')})")
        print(f"ID: {p.get('paperId')}")
        pdf = p.get("openAccessPdf")
        print(f"PDF: {pdf.get('url') if pdf else None}")
        print(f"Abstract: {p.get('abstract')[:200] if p.get('abstract') else 'None'}...")
        print("-" * 40)
else:
    print(response.status_code, response.text)
