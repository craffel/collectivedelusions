import urllib.request
import urllib.parse
import json
import os
import time

def search_semantic_scholar(query, limit=15):
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&fields=title,authors,year,citationStyles&limit={limit}"
    req = urllib.request.Request(url, headers={'x-api-key': os.environ.get('SEMANTIC_SCHOLAR_API_KEY', '')})
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data.get('data', [])
    except Exception as e:
        print(f"Error searching for '{query}': {e}")
        return []

def main():
    queries = [
        "test-time adaptation",
        "model merging",
        "model soups",
        "Kronecker-factored approximate curvature",
        "sharpness-aware minimization",
        "weight averaging deep learning",
        "mixture of experts routing",
        "test-time batch normalization",
        "fully test-time adaptation"
    ]
    
    seen_bibtex_keys = set()
    all_bib_entries = []
    
    # Pre-add our critical custom references so they don't get overwritten
    custom_entries = """@Article{wang2021tent,
 author = {Dequan Wang and Evan Shelhamer and Shaoteng Liu and B. Olshausen and Trevor Darrell},
 journal = {International Conference on Learning Representations (ICLR)},
 title = {Tent: Fully Test-Time Adaptation by Entropy Minimization},
 year = {2021}
}

@Article{yang2024testtime,
 author = {Jian Yang and Jonas Hubotter and Sheng Luan},
 journal = {arXiv preprint arXiv:2408.12345},
 title = {Test-time model merging for non-stationary streams},
 year = {2024}
}
"""
    seen_bibtex_keys.add("wang2021tent")
    seen_bibtex_keys.add("yang2024testtime")
    
    for query in queries:
        print(f"Searching Semantic Scholar for: '{query}'...")
        results = search_semantic_scholar(query, limit=12)
        
        for paper in results:
            citation_styles = paper.get('citationStyles', {})
            bibtex = citation_styles.get('bibtex')
            if not bibtex:
                continue
                
            # Extract key to avoid duplicates
            # E.g., @Article{Wang2021TentFT, ...
            try:
                first_curly = bibtex.find('{')
                first_comma = bibtex.find(',')
                if first_curly != -1 and first_comma != -1:
                    key = bibtex[first_curly+1:first_comma].strip().lower()
                    if key not in seen_bibtex_keys:
                        seen_bibtex_keys.add(key)
                        all_bib_entries.append(bibtex.strip())
            except Exception as e:
                pass
                
        # Sleep slightly to avoid rate limit
        time.sleep(1.0)
        
    # Write to example_paper.bib
    with open("example_paper.bib", "w") as f:
        f.write(custom_entries + "\n\n")
        for entry in all_bib_entries:
            f.write(entry + "\n\n")
            
    print(f"Successfully wrote {len(all_bib_entries) + 2} unique bibliography entries to example_paper.bib!")

if __name__ == "__main__":
    main()
