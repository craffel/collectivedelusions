import os
import sys
import time
import torch

torch.set_num_threads(8)

sys.path.insert(0, os.path.abspath('AdaMerging/src'))

import types
from modeling import ClassificationHead, ImageClassifier
sys.modules['src'] = types.ModuleType('src')
sys.modules['src.models'] = types.ModuleType('src.models')
sys.modules['src.models.modeling'] = types.ModuleType('src.models.modeling')
sys.modules['src.models.modeling'].ClassificationHead = ClassificationHead

import open_clip
from local_datasets.registry import get_dataset

if __name__ == "__main__":
    device = "cpu"
    print("Loading model...")
    clip_model, _, val_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    
    print("Loading dataset...")
    ds = get_dataset('MNIST', val_preprocess, location='data')
    test_full = ds.test_dataset
    print(f"MNIST Test size: {len(test_full)}")
    
    dataloader = torch.utils.data.DataLoader(test_full, batch_size=256, shuffle=False, num_workers=2)
    
    # We will evaluate just the first 1000 images to estimate speed
    start_time = time.time()
    count = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if isinstance(data, dict):
                images = data['images']
            else:
                images, _ = data
            features = clip_model.encode_image(images)
            count += images.size(0)
            if count >= 1000:
                break
    elapsed = time.time() - start_time
    print(f"Evaluated {count} images in {elapsed:.2f} seconds.")
    print(f"Estimated time for full 10,000 images: {elapsed * 10:.2f} seconds.")
