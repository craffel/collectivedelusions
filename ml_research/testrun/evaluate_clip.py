import torch
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from tqdm import tqdm

def get_class_names(task):
    if task == "mnist":
        return [f"a photo of the number {i}" for i in range(10)]
    elif task == "gtsrb":
        # Simplified GTSRB classes
        return [f"a road sign of type {i}" for i in range(43)]
    elif task == "eurosat":
        return ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
    return ["object"]

def evaluate_clip_task(model, processor, task, device, split="test", num_samples=100):
    class_names = get_class_names(task)
    
    # Load dataset
    try:
        if task == "mnist":
            ds = load_dataset("mnist", split=split)
            img_key = "image"
            label_key = "label"
        elif task == "eurosat":
            # Try phelber/eurosat
            try:
                ds = load_dataset("phelber/eurosat", "rgb", split="test")
            except:
                ds = load_dataset("nateraw/eurosat", split="test")
            img_key = "image"
            label_key = "label"
        elif task == "gtsrb":
            # meow-meow/gtsrb or tanganke/gtsrb
            try:
                ds = load_dataset("meow-meow/gtsrb", split="test")
            except:
                ds = load_dataset("tanganke/gtsrb", split="test")
            img_key = "image"
            label_key = "label"
        else:
            return 0.0
    except Exception as e:
        print(f"Error loading dataset for {task}: {e}")
        return 0.0

    # Tokenize class names
    inputs_text = processor(text=class_names, return_tensors="pt", padding=True).to(device)
    
    correct = 0
    total = 0
    
    indices = list(range(min(num_samples, len(ds))))
    for i in tqdm(indices, desc=f"Evaluating {task}"):
        image = ds[i][img_key]
        label = ds[i][label_key]
        
        inputs_img = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=inputs_text['input_ids'], 
                            pixel_values=inputs_img['pixel_values'],
                            attention_mask=inputs_text['attention_mask'])
            
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            pred = probs.argmax().item()
            
            if pred == label:
                correct += 1
            total += 1
            
    accuracy = correct / total
    return accuracy
