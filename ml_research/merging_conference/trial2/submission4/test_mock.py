import sys
sys.modules['flash_attn'] = None
sys.modules['flash_attn.bert_padding'] = None
sys.modules['flash_attn_2_cuda'] = None

try:
    from transformers import ViTForImageClassification
    print("Successfully imported ViTForImageClassification!")
except Exception as e:
    print("Import failed:", e)
