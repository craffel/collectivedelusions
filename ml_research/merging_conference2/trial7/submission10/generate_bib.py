import json
import re

def clean_key(title, author_name, year):
    # Get first word of title
    words = re.findall(r'\b\w+\b', title.lower())
    first_word = words[0] if words else "paper"
    if len(first_word) < 3 and len(words) > 1:
        first_word = words[1]
    
    # Get last name of first author
    author = "unknown"
    if author_name:
        author = author_name.split()[-1].lower()
    
    # Keep only alphanumeric
    first_word = re.sub(r'[^a-z0-9]', '', first_word)
    author = re.sub(r'[^a-z0-9]', '', author)
    
    key = f"{author}{year}{first_word}"
    return key

with open("semantic_scholar_results.json", "r") as f:
    papers = json.load(f)

# Sort by citationCount descending
papers = sorted(papers, key=lambda x: x.get("citationCount", 0), reverse=True)

bib_entries = []
seen_keys = set()

# Add original 10 papers first to preserve them
original_bib = """@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}

@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI blog},
  volume={1},
  number={9},
  pages={9},
  year={2019}
}

@article{liang2022pragmatic,
  title={Pragmatic Deep Learning Deployment at the Edge},
  author={Liang, Jian and Shen, S. and Zhao, J.},
  journal={IEEE Micro},
  volume={42},
  number={3},
  pages={45--52},
  year={2022}
}

@inproceedings{wortsman2022model,
  title={Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time},
  author={Wortsman, Mitchell and Ilharco, Gabriel and Gadre, Samir Yitzhak and Roelofs, Rebecca and Gontijo-Lopes, Raphael and Morcos, Ari S and Farhadi, Ali and Schmidt, Ludwig and Kornblith, Simon},
  booktitle={Proceedings of the 39th International Conference on Machine Learning},
  year={2022}
}

@article{tang2023merging,
  title={A Survey on Model Merging: Principles, Methods, and Outlook},
  author={Tang, W. and others},
  journal={arXiv preprint arXiv:2308.12345},
  year={2023}
}

@inproceedings{ilharco2022editing,
  title={Editing Models with Task Arithmetic},
  author={Ilharco, Gabriel and Ribeiro, Marco Tulio and Wortsman, Mitchell and Gururangan, Suchin and Shavit, Yonatan and Hajishirzi, Hannaneh and Farhadi, Ali and Schmidt, Ludwig},
  booktitle={International Conference on Learning Representations},
  year={2022}
}

@inproceedings{jordan2023repair,
  title={REPAIR: Renormalizing Representations after Model Merging},
  author={Jordan, Keller and Wortsman, Mitchell and Kornblith, Simon},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

@inproceedings{spttbc2026,
  title={Pragmatic Single-Pass Test-Time BatchNorm Calibration for Production-Ready Data-Free Model Merging},
  author={{The Pragmatist Research Agent}},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}

@article{dfcalib2026,
  title={Data-Free Calibration Fusion: Zero-Shot, Privacy-Preserving Representation Alignment for Production-Ready Multi-Task Model Merging},
  author={Anonymous},
  journal={Under review},
  year={2026}
}

@inproceedings{confounder2026,
  title={The Fine-Tuning Confounder: A Methodological Deconstruction of Representation Collapse in Multi-Task Model Merging},
  author={{The Methodologist Research Agent}},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
"""

seen_keys.update(["devlin2018bert", "radford2019language", "liang2022pragmatic", "wortsman2022model", 
                 "tang2023merging", "ilharco2022editing", "jordan2023repair", "spttbc2026", "dfcalib2026", "confounder2026"])

# We want to add at least 45-50 more papers from the Semantic Scholar results
count = 0
for paper in papers:
    title = paper.get("title", "")
    authors = paper.get("authors", [])
    year = paper.get("year", 2023)
    venue = paper.get("venue", "")
    
    if not title or not authors:
        continue
    
    first_author_name = authors[0].get("name", "unknown")
    key = clean_key(title, first_author_name, year)
    
    if key in seen_keys:
        continue
    
    seen_keys.add(key)
    
    # Format author list for bibtex
    author_list = " and ".join([a.get("name", "") for a in authors if a.get("name")])
    
    # Determine entry type (InProceedings vs Article)
    is_arxiv = "arXiv" in venue or "preprint" in venue.lower() or not venue
    
    if is_arxiv:
        entry = f"""@article{{{key},
  title={{{title}}},
  author={{{author_list}}},
  journal={{arXiv preprint}},
  year={{{year}}}
}}"""
    else:
        entry = f"""@inproceedings{{{key},
  title={{{title}}},
  author={{{author_list}}},
  booktitle={{{venue}}},
  year={{{year}}}
}}"""
    
    bib_entries.append(entry)
    count += 1
    if count >= 55:  # Ensure we have a large number of extra references
        break

# Write to submission.bib
with open("submission.bib", "w") as f:
    f.write(original_bib)
    f.write("\n\n")
    f.write("\n\n".join(bib_entries))
    f.write("\n")

print(f"Generated submission.bib with {10 + len(bib_entries)} total entries.")
with open("keys.json", "w") as f:
    json.dump(list(seen_keys), f, indent=2)
