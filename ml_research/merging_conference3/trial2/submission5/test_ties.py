import os
import sys
import gc
import torch

torch.set_num_threads(8)

sys.path.insert(0, os.path.abspath('AdaMerging/src'))

import types
from modeling import ClassificationHead, ImageClassifier
sys.modules['src'] = types.ModuleType('src')
sys.modules['src.models'] = types.ModuleType('src.models')
sys.modules['src.models.modeling'] = types.ModuleType('src.models.modeling')
sys.modules['src.models.modeling'].ClassificationHead = ClassificationHead

from ties_merging_utils import state_dict_to_vector, vector_to_state_dict, ties_merging

if __name__ == "__main__":
    print("Loading pre-trained state dict...")
    pretrained_sd = torch.load('checkpoints/ViT-B-32/zeroshot.pt', map_location='cpu')
    
    datasets_list = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    
    # Extract task vectors on the fly and delete finetuned state dicts immediately
    task_vectors_sd = []
    for dataset in datasets_list:
        print(f"Loading expert state dict for {dataset}...")
        ft_sd = torch.load(f'checkpoints/ViT-B-32/{dataset}/finetuned.pt', map_location='cpu')
        tv = {}
        for k in pretrained_sd.keys():
            tv[k] = ft_sd[k] - pretrained_sd[k]
        task_vectors_sd.append(tv)
        del ft_sd
        gc.collect()
        
    print("Flattening checkpoints...")
    remove_keys = []
    
    # We can delete pretrained_sd for a moment, but wait, we need its keys and structure.
    # We can keep a copy of keys/shapes or just keep it.
    flat_tvs = []
    for tv in task_vectors_sd:
        flat_tv = state_dict_to_vector(tv, remove_keys)
        flat_tvs.append(flat_tv)
        
    # Delete task_vectors_sd to free memory
    del task_vectors_sd
    gc.collect()
    
    flat_tvs_tensor = torch.vstack(flat_tvs)
    del flat_tvs
    gc.collect()
    
    print(f"Flat task vectors shape: {flat_tvs_tensor.shape}")
    
    print("Running ties_merging...")
    merged_tv = ties_merging(flat_tvs_tensor, reset_thresh=20, merge_func="dis-sum")
    print("TIES-Merging finished successfully!")
    
    # Let's clean up TIES variables
    del flat_tvs_tensor
    del merged_tv
    gc.collect()
    print("Cleanup successful.")
