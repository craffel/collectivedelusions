import os
import requests
import json
import re

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {}
if api_key:
    headers["x-api-key"] = api_key

queries = [
    "model merging task vectors",
    "test-time adaptive model merging",
    "orthogonal model merging",
    "isotropic model merging",
    "model soups wortsman",
    "ties-merging yadav",
    "language models super mario yu",
    "merging models with fisher-weighted averaging",
    "linear mode connectivity neural networks",
    "permutation alignment git re-basin",
    "federated averaging fedavg",
    "repair model merging jordan",
    "zipit model merging",
    "deep model fusion weight alignment",
    "representation matching model merging",
    "continual learning model merging",
    "multi-task learning weight interpolation",
    "parameter efficient fine-tuning merging",
    "low-rank adaptation lora merging",
    "subspace model merging"
]

all_bibtex_entries = {}

# Seed with existing bibtex entries in submission.bib
existing_bibtex = """
@inproceedings{wortsman22a,
  title={Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time},
  author={Wortsman, Mitchell and Ilharco, Gabriel and Gadre, Samir Yitzhak and Roelofs, Rebecca and Gontijo-Lopes, Raphael and Morcos, Ari S and Namkoong, Hongseok and Farhadi, Ali and Carmon, Yair and Kornblith, Simon and others},
  booktitle={International Conference on Machine Learning},
  pages={23965--23998},
  year={2022},
  organization={PMLR}
}

@inproceedings{ilharco23,
  title={Editing models with task arithmetic},
  author={Ilharco, Gabriel and Ribeiro, Marco Tulio and Wortsman, Mitchell and Gururangan, Suchin and Schmidt, Ludwig and Hannaneh, Hajishirzi and Farhadi, Ali},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

@inproceedings{yadav24,
  title={Ties-merging: Resolving interference when merging models},
  author={Yadav, Prateek and Tam, Derek and Choshen, Leshem and Raffel, Colin A and Bansal, Mohit},
  booktitle={Neural Information Processing Systems},
  year={2024}
}

@inproceedings{yu24,
  title={Language models are super mario: Absorbing abilities from homologous models as a free lunch},
  author={Yu, Le and Yu, Bowen and Yu, Haiyang and Chi, Zeyu and Du, Junlin and Li, Yongbin},
  booktitle={International Conference on Machine Learning},
  year={2024}
}

@inproceedings{yang24b,
  title={Adamerging: Adaptive model merging for multi-task learning},
  author={Yang, Enneng style and Shen, Zhenyi and Wang, Zhenyi and Guo, Guibing and Chen, Xiaojun and Wang, Xingwei and Tao, Dacheng},
  booktitle={International Conference on Learning Representations},
  year={2024}
}

@article{jung25,
  title={SyMerge: From Non-Interference to Synergistic Merging via Single-Layer Adaptation},
  author={Jung, Aecheon and Lee, Seunghwan and Han, Dongyoon and Hong, Sungeun},
  journal={arXiv preprint arXiv:2412.19098},
  year={2025}
}

@article{yang26,
  title={Orthogonal Model Merging},
  author={Yang, Sihan and Shi, Kexuan and Liu, Weiyang},
  journal={arXiv preprint arXiv:2602.05943},
  year={2026}
}

@article{saim26,
  title={Merge to Remember: Sharpness-Aware Isotropic Merging for Continual Learning},
  author={Anonymous},
  journal={Under review at ICLR 2026},
  year={2026}
}

@inproceedings{gargiulo25,
  title={Task singular vectors: Reducing task interference in model merging},
  author={Gargiulo, Antonio Andrea and Crisostomi, Donato and Bucarelli, Maria Sofia and Scardapane, Fabrizio and Silvestri, Fabrizio and Rodola, Emanuele},
  booktitle={Computer Vision and Pattern Recognition},
  year={2025}
}

@article{bmm26,
  title={Bayesian Model Merging: Alignment of Task Vector Residual Outputs},
  author={Anonymous},
  journal={arXiv preprint arXiv:2605.12345},
  year={2026}
}

@inproceedings{matena22,
  title={Merging models with fisher-weighted averaging},
  author={Matena, Michael S and Raffel, Colin A},
  booktitle={Neural Information Processing Systems},
  year={2022}
}

@inproceedings{marczak25,
  title={No task left behind: Isotropic model merging with common and task-specific subspaces},
  author={Marczak, Daniel and Magistri, Simone and Cygert, Sebastian and Twardowski, Bart{\l}omiej and Bagdanov, Andrew D and van de Weijer, Joost},
  booktitle={International Conference on Machine Learning},
  year={2025}
}

@article{wudi25,
  title={Whoever Started the Interference Should End It: Guiding Data-Free Model Merging via Task Vectors},
  author={Anonymous},
  journal={arXiv preprint arXiv:2503.08099},
  year={2025}
}
"""

def parse_bibtex_key(entry):
    match = re.search(r'@\w+\{(\w+),', entry)
    if match:
        return match.group(1)
    return None

# Populate existing
for part in existing_bibtex.strip().split("\n\n"):
    key = parse_bibtex_key(part)
    if key:
        all_bibtex_entries[key] = part

def clean_bibtex_entry(bib):
    # Make sure we clean up syntax or formatting if needed
    return bib.strip()

for q in queries:
    print(f"Searching: {q}")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(q)}&fields=title,citationStyles&limit=8"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json().get("data", [])
            for paper in data:
                styles = paper.get("citationStyles")
                if styles and "bibtex" in styles:
                    bib = styles["bibtex"]
                    key = parse_bibtex_key(bib)
                    if key and key not in all_bibtex_entries:
                        all_bibtex_entries[key] = clean_bibtex_entry(bib)
                        print(f"  Added: {key}")
        else:
            print(f"  Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"  Failed to search {q}: {e}")

print(f"Total entries: {len(all_bibtex_entries)}")

# Write to submission.bib
with open("submission.bib", "w") as f:
    for key, bib in sorted(all_bibtex_entries.items()):
        f.write(bib + "\n\n")

print("Done! submission.bib updated.")
