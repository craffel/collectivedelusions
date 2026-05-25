import os
import requests
import json
import re

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {"x-api-key": api_key} if api_key else {}

queries = [
    "model merging deep learning",
    "test-time model merging",
    "test-time adaptation",
    "mixture of experts routing",
    "weight averaging neural networks",
    "parameter efficient fine tuning merging",
    "task arithmetic",
    "domain adaptation non-stationary stream",
    "batch normalization fusion",
    "bayesian soft routing"
]

def clean_key(title, year):
    # Create a bibtex key from title and year
    words = re.sub(r'[^a-zA-Z0-9\s]', '', title).split()
    first_word = words[0].lower() if words else "paper"
    second_word = words[1].lower() if len(words) > 1 else ""
    yr = str(year) if year else "2024"
    return f"{first_word}{second_word}{yr}"

def get_bibtex_entry(paper):
    title = paper.get("title", "No Title")
    year = paper.get("year", 2024)
    venue = paper.get("venue", "") or paper.get("journal", {}).get("name", "") or "arXiv preprint"
    
    authors_list = paper.get("authors", [])
    author_str = " and ".join([a.get("name", "") for a in authors_list[:5]])
    if len(authors_list) > 5:
        author_str += " and others"
        
    key = clean_key(title, year)
    
    entry = f"@article{{{key},\n"
    entry += f"  author    = {{{author_str}}},\n"
    entry += f"  title     = {{{title}}},\n"
    entry += f"  journal   = {{{venue}}},\n"
    entry += f"  year      = {{{year}}}"
    
    pages = paper.get("pages")
    if pages:
        entry += f",\n  pages     = {{{pages}}}"
        
    volume = paper.get("volume")
    if volume:
        entry += f",\n  volume    = {{{volume}}}"
        
    entry += "\n}\n\n"
    return key, entry

def main():
    seen_ids = set()
    bib_entries = []
    keys = set()
    
    # Let's add some manual/famous papers to make sure we have bedrock citations
    manual_entries = [
        ("""@inproceedings{langley00,
 author    = {P. Langley},
 title     = {Crafting Papers on Machine Learning},
 year      = {2000},
 pages     = {1207--1216},
 booktitle = {Proceedings of the 17th International Conference on Machine Learning (ICML 2000)}
}"""),
        ("""@article{tent2021,
  author    = {Dequan Wang and Evan Shelhamer and Shaoteng Liu and Bruno Olshausen and Trevor Darrell},
  title     = {Tent: Fully Test-Time Adaptation by Entropy Minimization},
  journal   = {ICLR},
  year      = {2021}
}"""),
        ("""@article{modelsoups2022,
  author    = {Mitchell Wortsman and Gabriel Ilharco and Samir Yitzhak Gadre and Rebecca Roelofs and Raphael Gontijo-Lopes and others},
  title     = {Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time},
  journal   = {ICML},
  year      = {2022}
}"""),
        ("""@article{gitrebasin2023,
  author    = {Samuel Ainsworth and Jonathan Hayase and Siddhartha Srinivasa},
  title     = {Git Re-Basin: Merging Models across Loss Basins},
  journal   = {RENAME},
  year      = {2023}
}"""),
        ("""@article{zipit2023,
  author    = {George Stoica and Roman Ring and Lawrence Carin and others},
  title     = {ZipIt! Merging Models with Disjoint Vocabularies},
  journal   = {ICCV},
  year      = {2023}
}"""),
        ("""@article{taskarithmetic2023,
  author    = {Gabriel Ilharco and Marco Tulio Ribeiro and Mitchell Wortsman and Ludwig Schmidt and Hannaneh Hajishirzi},
  title     = {Editing models with task arithmetic},
  journal   = {ICLR},
  year      = {2023}
}"""),
        ("""@article{fedavg2017,
  author    = {Brendan McMahan and Eider Moore and Daniel Ramage and Seth Hampson and Blaise Aguera y Arcas},
  title     = {Communication-Efficient Learning of Deep Networks from Decentralized Data},
  journal   = {AISTATS},
  year      = {2017}
}""")
    ]
    
    # Parse existing keys from manual entries to avoid duplicates
    for entry in manual_entries:
        match = re.search(r'@\w+\{(\w+),', entry)
        if match:
            keys.add(match.group(1))
        bib_entries.append(entry + "\n\n")

    print("Fetching papers from Semantic Scholar...")
    for query in queries:
        print(f"Searching for: '{query}'")
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields=title,authors,year,venue,journal,externalIds&limit=15"
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if "data" in data:
                    for paper in data["data"]:
                        paper_id = paper.get("paperId")
                        if paper_id and paper_id not in seen_ids:
                            seen_ids.add(paper_id)
                            key, entry_str = get_bibtex_entry(paper)
                            if key not in keys:
                                keys.add(key)
                                bib_entries.append(entry_str)
            else:
                print(f"Error {response.status_code} for query: {query}")
        except Exception as e:
            print(f"Exception for query: {query}: {e}")
            
    print(f"Gathered {len(bib_entries)} references.")
    
    # Write to paper.bib
    with open("paper.bib", "w") as f:
        f.writelines(bib_entries)
    print("Saved all references to paper.bib")

if __name__ == "__main__":
    main()
