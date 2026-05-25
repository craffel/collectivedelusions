import requests
import json
import os
import re

api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
headers = {}
if api_key:
    headers["x-api-key"] = api_key

# List of search queries to cover all relevant areas
queries = [
    "model merging deep learning",
    "model soups wortsman",
    "task arithmetic merging",
    "ties-merging resolving interference",
    "dare model merging language models",
    "git re-basin loss basins",
    "test-time adaptation entropy",
    "continual test-time domain adaptation",
    "gradient surgery multi-task",
    "multi-task learning gradient conflict",
    "fisher information deep learning",
    "natural gradient descent neural networks",
    "overcoming catastrophic forgetting ewc",
    "online test-time adaptation robust",
    "parameter-efficient fine-tuning merging",
    "collaborative learning model fusion",
    "mixture of experts test-time"
]

existing_bibs = [
    "ilharco23", "adam_kingma", "tent_wang", "cotta_wang", "adamerging",
    "pcgrad", "dare_merging", "cpa_merge", "lfwa_merge", "pc_merge",
    "resnet", "mnist_dataset", "fmnist_dataset", "kmnist_dataset"
]

all_entries = {}

# Seed with existing manually constructed ones so we don't lose them
# (Parsed from submission.bib content we read earlier)
all_entries["ilharco23"] = """@inproceedings{ilharco23,
  title     = {Editing Models with Task Arithmetic},
  author    = {Ilharco, Gabriel and Wortsman, Mitchell and Samir, Aniruddha and Schmidt, Ludwig},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023}
}"""

all_entries["adam_kingma"] = """@inproceedings{adam_kingma,
  title     = {Adam: A Method for Stochastic Optimization},
  author    = {Kingma, Diederik P. and Ba, Jimmy},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2015}
}"""

all_entries["tent_wang"] = """@inproceedings{tent_wang,
  title     = {Tent: Fully Test-Time Adaptation by Entropy Minimization},
  author    = {Wang, Dequan and Shelhamer, Evan and Liu, Shaoteng and Olshausen, Bruno and Darrell, Trevor},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021}
}"""

all_entries["cotta_wang"] = """@inproceedings{cotta_wang,
  title     = {Continual Test-Time Domain Adaptation},
  author    = {Wang, Qin and Fink, Olga and Van Gool, Luc and Dai, Dengxin},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}"""

all_entries["adamerging"] = """@inproceedings{adamerging,
  title     = {AdaMerging: Adaptive Model Merging for Multi-Task Learning},
  author    = {Yang, Lichang and Zhang, Hongling and Wang, Kun and Chen, Lei and Zhao, Bin and Wang, Donglin},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}"""

all_entries["pcgrad"] = """@inproceedings{pcgrad,
  title     = {Gradient Surgery for Multi-Task Learning},
  author    = {Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey and Hausman, Karol},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2020}
}"""

all_entries["dare_merging"] = """@article{dare_merging,
  title     = {Language Models are Super-Scalers of Task Vectors},
  author    = {Yu, Le and Yu, Bowen and Yu, Haiyang and Li, Jiacheng and Huang, Fei and Li, Yongbin},
  journal   = {arXiv preprint arXiv:2309.03079},
  year      = {2023}
}"""

all_entries["cpa_merge"] = """@article{cpa_merge,
  title     = {Contrastive Prototype Alignment with Dynamic Task Routing for Teacher-Free Test-Time Model Merging},
  author    = {Anonymous},
  journal   = {Under Review for ICML},
  year      = {2026}
}"""

all_entries["lfwa_merge"] = """@article{lfwa_merge,
  title     = {LFW A: Layer-wise Fisher-Weighted Adaptation for Robust Test-Time Model Merging},
  author    = {Anonymous},
  journal   = {Under Review for ICML},
  year      = {2026}
}"""

all_entries["pc_merge"] = """@article{pc_merge,
  title     = {PC-Merge: Projecting Conflicting Gradients with Dynamic Optimizer Resets for Robust Test-Time Model Merging},
  author    = {Anonymous},
  journal   = {Under Review for ICML},
  year      = {2026}
}"""

all_entries["resnet"] = """@article{resnet,
  title     = {Deep Residual Learning for Image Recognition},
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  journal   = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}"""

all_entries["mnist_dataset"] = """@article{mnist_dataset,
  title     = {Gradient-based learning applied to document recognition},
  author    = {LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  journal   = {Proceedings of the IEEE},
  year      = {1998}
}"""

all_entries["fmnist_dataset"] = """@article{fmnist_dataset,
  title     = {Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
  author    = {Xiao, Han and Rasul, Kashif and Vollgraf, Roland},
  journal   = {arXiv preprint arXiv:1708.07747},
  year      = {2017}
}"""

all_entries["kmnist_dataset"] = """@article{kmnist_dataset,
  title     = {Deep Learning for Classical Japanese Literature},
  author    = {Clanuwat, Tarin and Bober-Irizar, Mikel and Kitamoto, Asanobu and Lamb, Alex and Yamamoto, Kazuaki and Croft, David},
  journal   = {arXiv preprint arXiv:1812.01718},
  year      = {2018}
}"""


def make_citation_key(title, authors, year):
    # Get first author last name
    if not authors:
        name = "unknown"
    else:
        first_author = authors[0].get("name", "unknown")
        name = first_author.split()[-1].lower()
    # Remove non-alpha
    name = re.sub(r'[^a-z0-9]', '', name)
    
    # Get first word of title
    title_words = title.split()
    first_word = "paper"
    for word in title_words:
        word_clean = re.sub(r'[^a-zA-Z0-9]', '', word).lower()
        if len(word_clean) > 3 and word_clean not in ["with", "from", "that", "this", "your", "deep", "neural", "learning", "model", "models", "multi", "task"]:
            first_word = word_clean
            break
            
    year_str = str(year) if year else "2024"
    key = f"{name}_{first_word}_{year_str}"
    return key


def clean_latex(text):
    if not text:
        return ""
    # Escape some common latex special characters but keep braces
    # Remove non-ascii
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text


print("Querying Semantic Scholar for diverse queries...")

for q in queries:
    print(f"Searching: {q}")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(q)}&fields=title,authors,year,venue,citationCount&limit=5"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            papers = data.get("data", [])
            for p in papers:
                title = p.get("title")
                authors = p.get("authors", [])
                year = p.get("year")
                venue = p.get("venue")
                
                if not title or not authors:
                    continue
                    
                key = make_citation_key(title, authors, year)
                if key in all_entries:
                    continue
                    
                author_names = " and ".join([a.get("name") for a in authors if a.get("name")])
                author_names = clean_latex(author_names)
                title_clean = clean_latex(title)
                venue_clean = clean_latex(venue) if venue else "arXiv preprint"
                
                if not venue_clean or venue_clean == "arXiv preprint" or "arxiv" in venue_clean.lower():
                    entry = f"""@article{{{key},
  title     = {{{title_clean}}},
  author    = {{{author_names}}},
  journal   = {{arXiv preprint}},
  year      = {{{year}}}
}}"""
                else:
                    entry = f"""@inproceedings{{{key},
  title     = {{{title_clean}}},
  author    = {{{author_names}}},
  booktitle = {{{venue_clean}}},
  year      = {{{year}}}
}}"""
                all_entries[key] = entry
        else:
            print(f"  Error: {response.status_code}")
    except Exception as e:
        print(f"  Failed: {e}")

print(f"Total compiled unique references: {len(all_entries)}")

# Write to submission.bib
with open("submission.bib", "w") as f:
    for key, entry in all_entries.items():
        f.write(entry + "\n\n")

print("Successfully wrote expanded bibliography to submission.bib")
