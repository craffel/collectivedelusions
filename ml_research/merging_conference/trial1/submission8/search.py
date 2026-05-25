import os
import requests
import json

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {}
if api_key:
    headers["x-api-key"] = api_key

query = "model merging test-time adaptation"
url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,authors,year,abstract,openAccessPdf&openAccessPdf&limit=5"

try:
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2))
    else:
        print(f"Error {response.status_code}: {response.text}")
except Exception as e:
    print(f"Exception: {e}")
