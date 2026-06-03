from datasets import load_dataset
try:
    print("Loading nielsr/eurosat...")
    eurosat = load_dataset("nielsr/eurosat")
    print("Loaded. Split keys:", list(eurosat.keys()))
    print("Size:", len(eurosat['train']))
except Exception as e:
    print("Failed to load nielsr/eurosat:", e)
