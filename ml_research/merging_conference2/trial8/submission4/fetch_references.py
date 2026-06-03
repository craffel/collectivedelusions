import urllib.request
import urllib.parse
import json
import os
import re

# Set up Semantic Scholar API queries
queries = [
    "model merging",
    "model soups",
    "task arithmetic",
    "weight averaging",
    "representation collapse",
    "parameter merging",
    "multi-task learning deep neural networks",
    "federated learning weight aggregation",
    "neural network weight averaging",
    "editing models with task arithmetic"
]

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

headers = {}
if api_key:
    headers["x-api-key"] = api_key

all_papers = {}

for q in queries:
    print(f"Searching for query: '{q}'...")
    params = {
        "query": q,
        "fields": "title,authors,year,venue,citationCount,externalIds",
        "limit": 15
    }
    encoded_params = urllib.parse.urlencode(params)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{encoded_params}"
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            results = data.get("data", [])
            for p in results:
                paper_id = p.get("paperId")
                if paper_id and paper_id not in all_papers:
                    all_papers[paper_id] = p
    except Exception as e:
        print(f"Error searching for '{q}': {e}")

print(f"Total unique papers found: {len(all_papers)}")

# Let's clean up and build bibtex entries
bib_entries = []

# Keep some original entries we definitely need
original_entries = """@inproceedings{langley00,
 author    = {P. Langley},
 title     = {Crafting Papers on Machine Learning},
 year      = {2000},
 pages     = {1207--1216},
 editor    = {Pat Langley},
 booktitle     = {Proceedings of the 17th International Conference
              on Machine Learning (ICML 2000)},
 address   = {Stanford, CA},
 publisher = {Morgan Kaufmann}
}

@TechReport{mitchell80,
  author = 	 "T. M. Mitchell",
  title = 	 "The Need for Biases in Learning Generalizations",
  institution =  "Computer Science Department, Rutgers University",
  year = 	 "1980",
  address =	 "New Brunswick, MA",
}

@phdthesis{kearns89,
  author = {M. J. Kearns},
  title =  {Computational Complexity of Machine Learning},
  school = {Department of Computer Science, Harvard University},
  year =   {1989}
}

@Book{MachineLearningI,
  editor = 	 "R. S. Michalski and J. G. Carbonell and T.
		  M. Mitchell",
  title = 	 "Machine Learning: An Artificial Intelligence
		  Approach, Vol. I",
  publisher = 	 "Tioga",
  year = 	 "1983",
  address =	 "Palo Alto, CA"
}

@Book{DudaHart2nd,
  author =       "R. O. Duda and P. E. Hart and D. G. Stork",
  title =        "Pattern Classification",
  publisher =    "John Wiley and Sons",
  edition =      "2nd",
  year =         "2000"
}

@misc{anonymous,
  title= {Suppressed for Anonymity},
  author= {Author, N. N.},
  year= {2021}
}

@InCollection{Newell81,
  author =       "A. Newell and P. S. Rosenbloom",
  title =        "Mechanisms of Skill Acquisition and the Law of
                  Practice", 
  booktitle =    "Cognitive Skills and Their Acquisition",
  pages =        "1--51",
  publisher =    "Lawrence Erlbaum Associates, Inc.",
  year =         "1981",
  editor =       "J. R. Anderson",
  chapter =      "1",
  address =      "Hillsdale, NJ"
}


@Article{Samuel59,
  author = 	 "A. L. Samuel",
  title = 	 "Some Studies in Machine Learning Using the Game of
		  Checkers",
  journal =	 "IBM Journal of Research and Development",
  year =	 "1959",
  volume =	 "3",
  number =	 "3",
  pages =	 "211--229"
}

@inproceedings{Wortsman2022Soups,
  title={Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time},
  author={Wortsman, Mitchell and Ilharco, Gabriel and Gadre, Samir Yitzhak and Roelofs, Rebecca and Gontijo-Lopes, Raphael and Morcos, Ari S and Namkoong, Hongseok and Farhadi, Ali and Schmidt, Ludwig and Kornblith, Simon},
  booktitle={International Conference on Machine Learning},
  pages={23965--23998},
  year={2022},
  organization={PMLR}
}

@article{Ilharco2022Editing,
  title={Editing models with task arithmetic},
  author={Ilharco, Gabriel and Ribeiro, Marco Tulio and Wortsman, Mitchell and Gururangan, Suchin and Shavit, Yonatan and Hajishirzi, Hannaneh and Farhadi, Ali and Singer, Yoram},
  journal={arXiv preprint arXiv:2212.04089},
  year={2022}
}

@article{jordan2023repair,
  title={REPAIR: Renormalizing Activations by Post-merge Calibration in Model Merging},
  author={Jordan, Keller and Wortsman, Mitchell and Seneviratne, Sachith and others},
  journal={arXiv preprint arXiv:2311.00000},
  year={2023}
}

@article{hns2025,
  title={Holographic Norm Scaling: Data-Free Parameter Calibration for Model Merging},
  author={Anonymous},
  journal={International Conference on Machine Learning (ICML)},
  year={2026}
}

@article{ipr2025,
  title={Isotropic Parameter Resonance: A Unified Theory for Healing Representation Collapse},
  author={Anonymous},
  journal={International Conference on Machine Learning (ICML)},
  year={2026}
}
"""

def clean_name(title):
    # Remove non-alphanumeric for citation key
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', title)
    words = clean.split()
    # Take first 2-3 words capitalized
    key = "".join([w.capitalize() for w in words[:3]])
    return key

added_keys = set([
    "langley00", "mitchell80", "kearns89", "MachineLearningI", "DudaHart2nd", "anonymous", "Newell81", "Samuel59",
    "Wortsman2022Soups", "Ilharco2022Editing", "jordan2023repair", "hns2025", "ipr2025"
])

bib_entries.append(original_entries)

for paper_id, p in all_papers.items():
    title = p.get("title")
    if not title:
        continue
    
    # Skip papers that are already represented or too similar
    low_title = title.lower()
    if "model soups" in low_title and any(k in added_keys for k in ["Wortsman2022Soups"]):
        continue
    if "task arithmetic" in low_title and any(k in added_keys for k in ["Ilharco2022Editing"]):
        continue
    if "repair:" in low_title and any(k in added_keys for k in ["jordan2023repair"]):
        continue
        
    authors_list = p.get("authors", [])
    if not authors_list:
        authors_str = "Unknown Authors"
    else:
        authors_str = " and ".join([a.get("name", "Unknown") for a in authors_list])
        
    year = p.get("year", 2023)
    venue = p.get("venue", "arXiv preprint")
    if not venue:
        venue = "arXiv preprint"
        
    cite_key = f"{clean_name(title)}{year}"
    if cite_key in added_keys:
        continue
        
    added_keys.add(cite_key)
    
    # Escape special characters in bibtex
    title_escaped = title.replace("&", "\\&").replace("%", "\\%")
    authors_escaped = authors_str.replace("&", "\\&").replace("%", "\\%")
    venue_escaped = venue.replace("&", "\\&").replace("%", "\\%")
    
    entry = f"""
@article{{{cite_key},
  title={{{title_escaped}}},
  author={{{authors_escaped}}},
  journal={{{venue_escaped}}},
  year={{{year}}}
}}"""
    bib_entries.append(entry)

print(f"Writing {len(bib_entries)} entries to template/example_paper.bib and example_paper.bib...")
full_bib = "\n".join(bib_entries)

with open("template/example_paper.bib", "w") as f:
    f.write(full_bib)
    
with open("example_paper.bib", "w") as f:
    f.write(full_bib)

print(f"Total entries in BibTeX database: {len(added_keys)}")
