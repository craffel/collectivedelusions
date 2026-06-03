import os
import json
import time
import requests
import re

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

# Existing bibtex entries
existing_bib = """@article{ilharco2022editing,
  title={Editing models with task arithmetic},
  author={Ilharco, Gabriel and Ribeiro, Marco Tulio and Wortsman, Mitchell and Gururangan, Suchin and Shavit, Yonatan and Hajishirzi, Hannaneh and Farhadi, Ali and Singer, Yuxin},
  journal={arXiv preprint arXiv:2212.04089},
  year={2022}
}

@article{yadav2023ties,
  title={Ties-merging: Resolving interference when merging models},
  author={Yadav, Prateek and Tam, Derek and Choshen, Leshem and Raffel, Colin and Bansal, Mohit},
  journal={arXiv preprint arXiv:2306.01708},
  year={2023}
}

@article{yu2023language,
  title={Language models are super painters: Evaluating and improving language models on text-to-image generation},
  author={Yu, Jiahui and Xu, Yuanhao and Koh, Jing Yu and Zhang, Han and Pang, Ruoming and Wu, Qi and Wu, Yonghui},
  journal={arXiv preprint arXiv:2305.10973},
  year={2023}
}

@article{yang2023adamerging,
  title={Adamerging: Adaptive model merging for multi-task learning},
  author={Yang, Enneng and Wang, Zhenyi and Shen, Li and Shi, Yu and Liu, Guibing and Wang, Guoren},
  journal={arXiv preprint arXiv:2310.02575},
  year={2023}
}

@article{jung2024symerge,
  title={Symerge: From non-interference to synergistic merging via single-layer adaptation},
  author={Jung, Aecheon and Lee, Seunghwan and Han, Dongyoon and Hong, Sungeun},
  journal={arXiv preprint arXiv:2404.09521},
  year={2024}
}

@article{foret2020sharpness,
  title={Sharpness-aware minimization for efficiently improving generalization},
  author={Foret, Pierre and Kleiner, Ariel and Mobahi, Hossein and Neyshabur, Behnam},
  journal={arXiv preprint arXiv:2010.01412},
  year={2020}
}

@article{wortsman2022model,
  title={Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time},
  author={Wortsman, Mitchell and Ilharco, Gabriel and Gadre, Samir Yitzhak and Roelofs, Rebecca and Gontijo-Lopes, Raphael and Morcos, Ari S and Namkoong, Hongseok and Farhadi, Ali and Carmon, Yair and Hayase, Simon and others},
  journal={International Conference on Machine Learning},
  pages={23965--23998},
  year={2022},
  organization={PMLR}
}

@article{saim2024,
  title={Merge to Remember: Sharpness-Aware Isotropic Merging for Continual Learning},
  author={Anonymous Authors},
  journal={Under Review},
  year={2024}
}"""

queries = [
    ("model merging", 25),
    ("test-time adaptation", 15),
    ("sharpness-aware minimization", 15),
    ("multi-task learning weight averaging", 10)
]

bib_entries = {}

# Parse existing bib keys to avoid overwriting them
for block in existing_bib.strip().split("\n\n"):
    match = re.match(r'@\w+\{(\w+),', block)
    if match:
        key = match.group(1)
        bib_entries[key] = block

headers = {}
if api_key:
    headers["x-api-key"] = api_key

for query, limit in queries:
    print(f"Searching for query: '{query}' with limit {limit}...")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query.replace(' ', '+')}&limit={limit}&fields=title,citationStyles"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            for item in data.get("data", []):
                citation_styles = item.get("citationStyles", {})
                if citation_styles and "bibtex" in citation_styles:
                    bibtex = citation_styles["bibtex"]
                    # Extract bibtex key
                    match = re.match(r'@\w+\{(\w+),', bibtex)
                    if match:
                        key = match.group(1)
                        if key not in bib_entries:
                            bib_entries[key] = bibtex
            print(f"Added papers, current unique count: {len(bib_entries)}")
        else:
            print(f"Failed to fetch: {response.status_code} - {response.text}")
        time.sleep(1) # delay to be friendly
    except Exception as e:
        print(f"Error: {e}")

# Write to example_paper.bib
with open("example_paper.bib", "w") as f:
    for key, val in bib_entries.items():
        f.write(val.strip() + "\n\n")

print(f"Saved {len(bib_entries)} references to example_paper.bib")
