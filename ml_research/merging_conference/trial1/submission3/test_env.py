import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

print("Environment check:")
try:
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)

try:
    dataset = load_dataset("glue", "sst2", split="train[:100]")
    print(f"Dataset loaded successfully! Length: {len(dataset)}")
except Exception as e:
    print("Error loading dataset:", e)
