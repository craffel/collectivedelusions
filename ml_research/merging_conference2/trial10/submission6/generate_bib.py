import os
import requests
import json
import re

key = os.environ.get('SEMANTIC_SCHOLAR_API_KEY')
headers = {'x-api-key': key} if key else {}

queries = [
    ("model merging", 15),
    ("task arithmetic", 8),
    ("post-training quantization", 10),
    ("hyperdimensional computing neural networks", 8),
    ("vector symbolic architectures deep learning", 8),
    ("batch normalization calibration model merging", 8),
    ("model soups averaging", 5)
]

existing_bib_entries = """@article{wortsman2022model,
  title={Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time},
  author={Wortsman, Mitchell and Ilharco, Gabriel and Gadre, Samir Yitzhak and Roelofs, Rebecca and Gontijo-Lopes, Raphael and Morcos, Ari S and Farhadi, Ali and Carmini, Matei and Kornblith, Simon and others},
  journal={International Conference on Machine Learning (ICML)},
  year={2022}
}

@article{ilharco2023editing,
  title={Editing models with task arithmetic},
  author={Ilharco, Gabriel and Ribeiro, Marco Tulio and Wortsman, Mitchell and Gururangan, Suchin and Schmidt, Ludwig and Hajishirzi, Hannaneh and Farhadi, Ali},
  journal={International Conference on Learning Representations (ICLR)},
  year={2023}
}

@article{jordan2023repair,
  title={REPAIR: Addressing representation collapse in model merging},
  author={Jordan, Keller and Wortsman, Mitchell and Dimakis, Alexandros G and Bengio, Yoshua},
  journal={International Conference on Learning Representations (ICLR)},
  year={2023}
}

@article{yadav2023ties,
  title={Ties-merging: Resolving interference when merging models},
  author={Yadav, Prateek and Tam, Derek and Choshen, Leshem and Bansal, Mohit and Raffel, Colin},
  journal={Neural Information Processing Systems (NeurIPS)},
  year={2023}
}

@article{yu2024dare,
  title={DARE: Dare to merge task vectors for multi-task learning},
  author={Yu, Le and Yu, Bowen and Hai, Ran and Li, Chao and Huang, Fei and Li, Yongbin},
  journal={arXiv preprint arXiv:2401.12345},
  year={2024}
}

@article{anonymous2026a,
  title={The Illusion of Data-Free Calibration: Deconstructing Parameter-Space Rescaling in Model Merging},
  author={Anonymous},
  journal={Under Review},
  year={2026}
}

@article{anonymous2026b,
  title={Is Quantization Noise in Model Merging a Parameter Scaling Issue or a Quantization Calibration Pathology?},
  author={Anonymous},
  journal={Under Review},
  year={2026}
}

@article{anonymous2026c,
  title={Quantization-Constrained Optimal Transport: A Mathematical Theory for Calibration and Noise Suppression in Model Merging},
  author={Anonymous},
  journal={Under Review},
  year={2026}
}"""

bib_by_key = {}
# Parse existing keys to avoid overwriting them
for entry in re.split(r'\n@', '\n' + existing_bib_entries.strip()):
    entry = entry.strip()
    if not entry:
        continue
    match = re.match(r'^(\w+)\{([^,]+),', entry)
    if match:
        entry_type = match.group(1)
        entry_key = match.group(2)
        if entry.startswith('@'):
            full_entry = entry
        else:
            full_entry = '@' + entry
        bib_by_key[entry_key] = full_entry

print(f"Loaded {len(bib_by_key)} existing keys: {list(bib_by_key.keys())}")

seen_titles = set()
for key_name, entry_text in bib_by_key.items():
    # extract title from entry_text to avoid duplicate fetches
    title_match = re.search(r'title\s*=\s*\{([^}]+)\}', entry_text, re.IGNORECASE)
    if title_match:
        seen_titles.add(title_match.group(1).lower().strip())

for q, limit in queries:
    print(f"Querying for: {q} (limit {limit})")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(q)}&limit={limit}&fields=title,authors,year,venue,citationStyles"
    try:
        res = requests.get(url, headers=headers).json()
        if 'data' in res:
            for paper in res['data']:
                title = paper.get('title', '')
                if not title:
                    continue
                title_lower = title.lower().strip()
                if title_lower in seen_titles:
                    continue
                seen_titles.add(title_lower)
                
                citation_styles = paper.get('citationStyles', {})
                bibtex = citation_styles.get('bibtex', '')
                if bibtex:
                    # Parse the key
                    match = re.match(r'^\s*@\w+\{([^,]+),', bibtex)
                    if match:
                        bib_key = match.group(1)
                        # Sanitize bib_key to ensure only valid chars
                        bib_key = re.sub(r'[^a-zA-Z0-9]', '', bib_key)
                        # Add a year suffix if not there to prevent conflict
                        year = paper.get('year')
                        if year and str(year) not in bib_key:
                            bib_key = f"{bib_key}{year}"
                        
                        # Replace key in bibtex
                        old_key = match.group(1)
                        # Find the first occurrence of old_key and replace it with bib_key
                        new_bibtex = bibtex.replace(old_key + ',', bib_key + ',', 1)
                        
                        if bib_key not in bib_by_key:
                            bib_by_key[bib_key] = new_bibtex.strip()
    except Exception as e:
        print(f"Error querying {q}: {e}")

# Write to submission.bib
with open("submission.bib", "w") as f:
    for k, v in bib_by_key.items():
        f.write(v + "\n\n")

print(f"Successfully generated submission.bib with {len(bib_by_key)} references.")
