import sys
import os
sys.path.insert(0, os.path.abspath("./custom_libs"))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import time

model_name = "bert-base-uncased"
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

texts = ["this movie was absolutely fantastic and wonderful"] * 8
labels = [1] * 8
encodings = tokenizer(texts, truncation=True, padding=True, max_length=32, return_tensors="pt")
ds = TensorDataset(encodings["input_ids"], encodings["attention_mask"], torch.tensor(labels))
dl = DataLoader(ds, batch_size=8, shuffle=True)

optimizer = AdamW(model.parameters(), lr=1e-3)
model.train()

start = time.time()
for batch in dl:
    optimizer.zero_grad()
    input_ids, attn_mask, labels = batch
    outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
end = time.time()
print(f"Time for 1 step of bert-base-uncased: {end - start:.4f} seconds")
