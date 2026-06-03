import os
import requests
import time
import re

API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

queries = [
    "model merging deep learning",
    "weight averaging deep learning",
    "task arithmetic model merging",
    "ties-merging deep learning",
    "representation similarity CKA neural network",
    "multi-task learning deep learning",
    "mixture of experts gating routing",
    "model merging activation calibration",
    "deep representation learning CKA",
    "gradient surgery multi-task learning"
]

def fetch_bibtex_for_query(query, limit=10):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&fields=title,citationStyles&limit={limit}"
    headers = {}
    if API_KEY:
        headers["x-api-key"] = API_KEY
    
    print(f"Searching for: {query} ...")
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 429:
            print("Rate limited. Waiting 3 seconds...")
            time.sleep(3)
            response = requests.get(url, headers=headers, timeout=15)
            
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return []
            
        data = response.json()
        bibtexs = []
        for paper in data.get("data", []):
            citation_styles = paper.get("citationStyles", {})
            if citation_styles and "bibtex" in citation_styles:
                bibtexs.append(citation_styles["bibtex"])
        return bibtexs
    except Exception as e:
        print(f"Exception during search: {e}")
        return []

def clean_bibtex(bib_text):
    # Some bibtex strings returned have weird characters or headers like @['JournalArticle', 'Conference']{...
    # Let's clean the opening tag @['JournalArticle']{key, -> @article{key,
    # or just keep it simple. Usually they start with @[something]{key,
    # Let's replace @\[.*?\]\{ with @article{
    cleaned = re.sub(r"^@\[.*?\]\{", "@article{", bib_text, flags=re.MULTILINE)
    # Also sometimes we have a bracket of types. Let's fix those
    cleaned = re.sub(r"^@\['[^']*'\]\{", "@article{", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^@\['[^']*',\s*'[^']*'\]\{", "@article{", cleaned, flags=re.MULTILINE)
    return cleaned

def main():
    collected_bibtexs = []
    seen_keys = set()
    
    # Read existing bib keys to avoid duplicates
    existing_content = ""
    if os.path.exists("example_paper.bib"):
        with open("example_paper.bib", "r") as f:
            existing_content = f.read()
            for key in re.findall(r"@\w+\{([^,]+),", existing_content):
                seen_keys.add(key)
                
    print(f"Initially found {len(seen_keys)} keys in example_paper.bib.")
    
    for q in queries:
        bibs = fetch_bibtex_for_query(q, limit=8)
        for b in bibs:
            b_clean = clean_bibtex(b)
            # Find key in the cleaned bib
            match = re.search(r"@\w+\{([^,]+),", b_clean)
            if match:
                key = match.group(1)
                if key not in seen_keys:
                    seen_keys.add(key)
                    collected_bibtexs.append(b_clean)
        time.sleep(1) # Polite delay
        
    print(f"Collected {len(collected_bibtexs)} new bibliography entries.")
    
    # Append to existing bib
    if collected_bibtexs:
        with open("example_paper.bib", "a") as f:
            f.write("\n\n" + "\n\n".join(collected_bibtexs))
        print("Updated example_paper.bib successfully!")
    else:
        print("No new bibliography entries collected.")

if __name__ == "__main__":
    main()
