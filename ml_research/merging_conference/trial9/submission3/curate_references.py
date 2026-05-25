import re

# Read fetched_references
with open("fetched_references.bib", "r") as f:
    content = f.read()

# Parse entries
entries = re.split(r'@article\{', content)[1:]
parsed_entries = []

for entry in entries:
    lines = entry.strip().split('\n')
    if not lines:
        continue
    key = lines[0].split(',')[0].strip()
    
    # Extract fields
    title_match = re.search(r'title=\{(.*?)\}', entry, re.DOTALL)
    author_match = re.search(r'author=\{(.*?)\}', entry, re.DOTALL)
    journal_match = re.search(r'journal=\{(.*?)\}', entry, re.DOTALL)
    year_match = re.search(r'year=\{(.*?)\}', entry, re.DOTALL)
    
    title = title_match.group(1).strip() if title_match else ""
    author = author_match.group(1).strip() if author_match else ""
    journal = journal_match.group(1).strip() if journal_match else ""
    year = year_match.group(1).strip() if year_match else "2023"
    
    parsed_entries.append({
        'key': key,
        'title': title,
        'author': author,
        'journal': journal,
        'year': year,
        'full': "@article{" + entry.strip()
    })

# Group into categories
categories = {
    'merging': [],
    'tta': [],
    'metric': [],
    'bn_fisher': []
}

for p in parsed_entries:
    t_lower = p['title'].lower()
    j_lower = p['journal'].lower()
    
    # Check metric learning
    if any(k in t_lower for k in ["cosface", "arcface", "angular", "cosine", "spherical", "metric", "contrastive", "margin"]):
        categories['metric'].append(p)
    # Check merging
    elif any(k in t_lower for k in ["merge", "merging", "weight averaging", "interpolation", "task vector", "parameter space", "ties"]):
        categories['merging'].append(p)
    # Check tta
    elif any(k in t_lower for k in ["test-time", "test time", "adaptation", "sfda", "tent", "tta"]):
        categories['tta'].append(p)
    # Check BN, Fisher, Bayesian
    elif any(k in t_lower for k in ["fisher", "batch normalization", "statistics", "bn", "covariate", "precondition", "kronecker", "regularization", "coherence", "bayesian"]):
        categories['bn_fisher'].append(p)
    else:
        # Default to merging or tta as fallback
        if "adaptation" in t_lower or "domain" in t_lower:
            categories['tta'].append(p)
        else:
            categories['merging'].append(p)

print("Categorization results:")
for cat, lst in categories.items():
    print(f"  {cat}: {len(lst)} papers")

# Let's select the top papers from each category to form a list of 60 papers
selected_papers = []
max_per_cat = 15

for cat in ['merging', 'tta', 'metric', 'bn_fisher']:
    selected_papers.extend(categories[cat][:max_per_cat])

# Write selected to example_paper.bib
original_bib = """@inproceedings{langley00,
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
"""

with open("example_paper.bib", "w") as f:
    f.write(original_bib)
    f.write("\n\n")
    for p in selected_papers:
        f.write(p['full'])
        f.write("\n\n")

print(f"Curated {len(selected_papers)} papers and wrote them to example_paper.bib.")

# Print categories and keys
for cat in ['merging', 'tta', 'metric', 'bn_fisher']:
    print(f"\n--- {cat.upper()} ---")
    keys = [p['key'] for p in categories[cat][:max_per_cat]]
    print(", ".join(keys))
