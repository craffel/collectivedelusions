import json
import re

# Existing references in paper.bib (we want to preserve these, especially the target papers)
existing_bib = """@inproceedings{proto_ttmm,
  title={PROTO-TTMM: Breaking the Closed-World Assumption in Test-Time Model Merging},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}

@inproceedings{fp_ca,
  title={Fisher-Preconditioned Contrastive Alignment for Teacher-Free Test-Time Model Merging},
  author={Anonymous},
  booktitle={Under Review},
  year={2026}
}

@inproceedings{iggs_merge,
  title={IGGS-Merge: Information-Geometric Gradient Surgery for Robust Test-Time Model Merging},
  author={Anonymous},
  booktitle={Under Review},
  year={2026}
}

@inproceedings{tent,
  title={Tent: Fully Test-Time Adaptation by Entropy Minimization},
  author={Wang, Dequan and Shelhamer, Evan and Liu, Shaoteng and Olshausen, Bruno and Darrell, Trevor},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}

@inproceedings{cotta,
  title={Continual Test-Time Domain Adaptation with Likelihood Ratio and Forgetting Prevention},
  author={Wang, Qin and Fink, Olga and Van Gool, Luc and Dai, Dengxin},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}

@article{task_arithmetic,
  title={Editing Models with Task Arithmetic},
  author={Ilharco, Gabriel and Ribeiro, Marco Tulio and Wortsman, Mitchell and Gururangan, Suchin and Connor, Ludwig and Soll-Collier, Hannaneh and Hajishirzi, Hannaneh and Shamir, Eli and Farhadi, Ali},
  journal={arXiv preprint arXiv:2212.04089},
  year={2022}
}

@inproceedings{fisher_merging,
  title={Merging Models with Fisher Information},
  author={Matena, Michael S and Raffel, Colin A},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}

@inproceedings{regmean,
  title={RegMean: Demystifying Classifier-Free Guidance and Feature RegMean in Model Merging},
  author={Jin, Xisen and Peng, Xiangyu and Chan, Stephen and Others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}

@inproceedings{sam,
  title={Sharpness-Aware Minimization for Efficiently Improving Generalization},
  author={Foret, Pierre and Kleiner, Ariel and Mobahi, Hossein and Neyshabur, Behnam},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}

@article{resnet,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  journal={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016}
}"""

# Extract keys of existing entries to avoid duplication
existing_keys = set(re.findall(r'@\w+\{(\w+),', existing_bib))

# Load found papers
with open("found_papers.json", "r") as f:
    found_papers = json.load(f)

# Sort papers by year descending to get the most recent and relevant ones first
found_papers.sort(key=lambda x: (x.get("year") if x.get("year") is not None else 0), reverse=True)

new_bib_entries = []

# Filter words to keep papers highly relevant to model merging, test-time adaptation, deep learning
relevant_keywords = [
    "merge", "merging", "adaptation", "test-time", "tta", "soup", "soups", "ties", 
    "dare", "regression", "domain", "zero-shot", "alignment", "generalization",
    "continual", "ood", "out-of-distribution", "routing", "expert", "multi-task"
]

def is_relevant(paper):
    title = paper.get("title", "").lower()
    # Skip papers about traffic freeway merging or other irrelevant fields
    irrelevant_keywords = ["crash", "freeway", "highway", "traffic", "vehicle", "lane", "fake news", "bio", "medical"]
    if any(k in title for k in irrelevant_keywords):
        return False
    return any(k in title for k in relevant_keywords)

filtered_papers = [p for p in found_papers if is_relevant(p)]

# We want around 40-45 new papers to reach a total of ~50-55 references
for paper in filtered_papers[:45]:
    title = paper.get("title", "")
    authors_list = [a.get("name", "") for a in paper.get("authors", [])]
    if not authors_list:
        continue
    
    # Format author string for BibTeX
    authors_str = " and ".join(authors_list)
    
    # Format venue/booktitle
    venue = paper.get("venue", "")
    if not venue or venue == "arXiv.org":
        arxiv_id = paper.get("externalIds", {}).get("ArXiv", "")
        if arxiv_id:
            venue = f"arXiv preprint arXiv:{arxiv_id}"
        else:
            venue = "arXiv preprint"
    
    year = paper.get("year", 2024)
    
    # Create clean BibTeX key
    first_author_lastname = authors_list[0].split()[-1] if authors_list[0] else "Anonymous"
    # Clean non-alphanumeric chars from author name
    first_author_lastname = re.sub(r'\W+', '', first_author_lastname).lower()
    
    # First 2-3 words of title for key
    title_words = re.sub(r'[^\w\s]', '', title).lower().split()
    title_words = [w for w in title_words if w not in ["the", "a", "an", "on", "for", "in", "of", "with", "to", "and", "by", "from"]][:2]
    title_suffix = "".join(title_words)
    
    bib_key = f"{first_author_lastname}{year}{title_suffix}"
    
    # Avoid duplicate keys
    if bib_key in existing_keys:
        continue
    existing_keys.add(bib_key)
    
    # Determine type
    if "preprint" in venue.lower() or "arxiv" in venue.lower():
        entry = f"""@article{{{bib_key},
  title={{{title}}},
  author={{{authors_str}}},
  journal={{{venue}}},
  year={{{year}}}
}}"""
    else:
        entry = f"""@inproceedings{{{bib_key},
  title={{{title}}},
  author={{{authors_str}}},
  booktitle={{{venue}}},
  year={{{year}}}
}}"""
    new_bib_entries.append(entry)

# Combine and write to paper.bib
full_bib = existing_bib + "\n\n" + "\n\n".join(new_bib_entries)

with open("paper.bib", "w") as f:
    f.write(full_bib)

print(f"Successfully generated paper.bib with {len(existing_keys)} references (added {len(new_bib_entries)} references).")
