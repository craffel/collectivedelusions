from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
model = AutoModel.from_pretrained('prajjwal1/bert-tiny')

inputs = tokenizer("Hello world!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
print("Hidden states shape:", outputs.last_hidden_state.shape)
print("Pooler output shape:", outputs.pooler_output.shape)
