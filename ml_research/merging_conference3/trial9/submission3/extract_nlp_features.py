import os
import urllib.request
import zipfile
import importlib.metadata
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print("Starting NLP feature extraction script...")

# 1. Patch Hugging Face metadata check
orig = importlib.metadata.version
importlib.metadata.version = lambda p: '0.35.0' if p == 'huggingface-hub' else orig(p)

import transformers
import transformers.tokenization_utils_base
transformers.tokenization_utils_base.list_repo_templates = lambda *args, **kwargs: []

from transformers import AutoTokenizer, AutoModel

# 2. Download and unzip datasets if not already downloaded
urls = {
    "SST-2": "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
    "CoLA": "https://dl.fbaipublicfiles.com/glue/data/CoLA.zip",
    "RTE": "https://dl.fbaipublicfiles.com/glue/data/RTE.zip",
    "QNLI": "https://dl.fbaipublicfiles.com/glue/data/QNLI.zip"
}

os.makedirs("data/nlp", exist_ok=True)
for task, url in urls.items():
    zip_path = f"data/nlp/{task}.zip"
    if not os.path.exists(zip_path):
        print(f"Downloading {task}...")
        urllib.request.urlretrieve(url, zip_path)
    
    task_dir = f"data/nlp/{task}"
    if not os.path.exists(task_dir):
        print(f"Extracting {task}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/nlp")

print("All datasets downloaded and extracted successfully.")

# 3. Load DistilBERT model & tokenizer
print("Loading DistilBERT model and tokenizer...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# 4. Extract raw sentences from each task (500 samples per task)
task_sentences = {}
print("Parsing raw TSV files...")

# SST-2
with open("data/nlp/SST-2/train.tsv", "r", encoding="utf-8") as f:
    lines = f.readlines()
    sentences = []
    # Skip header
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            sentences.append(parts[0])
        if len(sentences) == 500:
            break
    task_sentences["SST-2"] = sentences

# CoLA
with open("data/nlp/CoLA/train.tsv", "r", encoding="utf-8") as f:
    lines = f.readlines()
    sentences = []
    # No header
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            sentences.append(parts[3])
        if len(sentences) == 500:
            break
    task_sentences["CoLA"] = sentences

# RTE
with open("data/nlp/RTE/train.tsv", "r", encoding="utf-8") as f:
    lines = f.readlines()
    sentences = []
    # Skip header
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            sentences.append(parts[1] + " " + parts[2])
        if len(sentences) == 500:
            break
    task_sentences["RTE"] = sentences

# QNLI
with open("data/nlp/QNLI/train.tsv", "r", encoding="utf-8") as f:
    lines = f.readlines()
    sentences = []
    # Skip header
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            sentences.append(parts[1] + " " + parts[2])
        if len(sentences) == 500:
            break
    task_sentences["QNLI"] = sentences

print({task: len(sents) for task, sents in task_sentences.items()})

# 5. Extract DistilBERT CLS embeddings in batches
extracted_embeddings = {}
print("Extracting sentence embeddings via DistilBERT...")
batch_size = 64

with torch.no_grad():
    for task_idx, (task_name, sentences) in enumerate(task_sentences.items()):
        print(f"Processing task {task_name}...")
        task_embs = []
        for i in range(0, len(sentences), batch_size):
            batch_sents = sentences[i : i + batch_size]
            encoded = tokenizer(batch_sents, padding=True, truncation=True, max_length=128, return_tensors="pt")
            outputs = model(**encoded)
            # CLS token is at index 0 of the sequence dimension
            cls_embs = outputs.last_hidden_state[:, 0, :].numpy()
            task_embs.append(cls_embs)
        
        extracted_embeddings[task_name] = np.concatenate(task_embs, axis=0)

# 6. Apply unsupervised K-Means clustering to create 10 classes per task
print("Clustering embeddings into 10 semantic classes per task using KMeans...")
task_classes = {}
for task_name, embs in extracted_embeddings.items():
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embs)
    task_classes[task_name] = labels
    print(f"Task {task_name}: Created 10 clusters of sizes: {np.bincount(labels)}")

# 7. Merge all tasks together, apply PCA to project to 192 dimensions
print("Merging task representations and applying PCA...")
all_embs = []
all_tasks = []
all_classes = []

task_list = ["SST-2", "CoLA", "RTE", "QNLI"]
for task_idx, task_name in enumerate(task_list):
    all_embs.append(extracted_embeddings[task_name])
    all_tasks.append(np.full(500, task_idx))
    all_classes.append(task_classes[task_name])

all_embs = np.concatenate(all_embs, axis=0)  # (2000, 768)
all_tasks = np.concatenate(all_tasks, axis=0)  # (2000,)
all_classes = np.concatenate(all_classes, axis=0)  # (2000,)

pca = PCA(n_components=192, random_state=42)
projected_embs = pca.fit_transform(all_embs)  # (2000, 192)

# 8. Normalize projected embeddings to have mean norm of 1.0 (to match coordinate sandbox configuration Rh = 1.0)
projected_embs = torch.tensor(projected_embs, dtype=torch.float32)
norms = projected_embs.norm(dim=-1, keepdim=True)
normalized_embs = projected_embs / (norms + 1e-10) # Unit norm each
mean_norm = normalized_embs.norm(dim=-1).mean().item()
print(f"Normalized embeddings to unit norm. Mean norm: {mean_norm}")

# Save the dataset
save_dict = {
    "features": normalized_embs,
    "tasks": torch.tensor(all_tasks, dtype=torch.long),
    "classes": torch.tensor(all_classes, dtype=torch.long)
}
torch.save(save_dict, "data/nlp_real_world_features.pt")
print("Successfully saved NLP real-world dataset to data/nlp_real_world_features.pt!")
