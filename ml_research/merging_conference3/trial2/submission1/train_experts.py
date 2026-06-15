import os
import sys
import torch
import functools
# Patch torch.load globally to handle unpickling of custom classes
torch.load = functools.partial(torch.load, weights_only=False)

import torch.nn as nn
from tqdm import tqdm

# Add AdaMerging/src to path
sys.path.insert(0, os.path.abspath('AdaMerging/src'))

from modeling import ImageEncoder, ImageClassifier
from heads import get_classification_head
from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize

class Args:
    def __init__(self):
        self.model = 'ViT-B-32'
        self.save = os.path.abspath('checkpoints/ViT-B-32')
        self.data_location = os.path.abspath('data')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.openclip_cachedir = os.path.abspath('openclip_cache')
        self.batch_size = 128
        self.cache_dir = None

def main():
    args = Args()
    os.makedirs(args.save, exist_ok=True)
    os.makedirs(args.openclip_cachedir, exist_ok=True)
    os.makedirs(args.data_location, exist_ok=True)

    print("Loading base pre-trained image encoder...")
    base_encoder = ImageEncoder(args, keep_lang=False)
    zeroshot_path = os.path.join(args.save, 'zeroshot.pt')
    torch.save(base_encoder, zeroshot_path)
    print(f"Saved base pre-trained image encoder to {zeroshot_path}")

    datasets_to_train = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    
    for dataset_name in datasets_to_train:
        print(f"\n=========================================")
        print(f"Training Expert for {dataset_name}")
        print(f"=========================================")
        
        # Reload encoder to start from pre-trained
        image_encoder = ImageEncoder(args, keep_lang=False).to(args.device)
        
        # Load classification head
        print("Loading/building classification head...")
        classification_head = get_classification_head(args, dataset_name).to(args.device)
        
        # Build classifier
        classifier = ImageClassifier(image_encoder, classification_head)
        classifier.freeze_head()
        classifier.to(args.device)
        
        # Load dataset
        dataset = get_dataset(
            dataset_name,
            classifier.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size
        )
        
        dataloader = get_dataloader(dataset, is_train=True, args=args)
        
        # Train visual encoder
        optimizer = torch.optim.AdamW(classifier.image_encoder.parameters(), lr=2e-5, weight_decay=0.1)
        criterion = nn.CrossEntropyLoss()
        
        classifier.train()
        
        # Limit to max 5 steps to keep it fast but effective
        max_steps = 5
        step = 0
        epoch = 0
        
        while step < max_steps:
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                if step >= max_steps:
                    break
                batch = maybe_dictionarize(batch)
                x = batch['images'].to(args.device)
                y = batch['labels'].to(args.device)
                
                optimizer.zero_grad()
                outputs = classifier(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
                
                step += 1
                
            acc = 100. * correct / total
            print(f"Epoch {epoch+1} Complete. Steps: {step}, Loss: {epoch_loss/len(dataloader):.4f}, Acc: {acc:.2f}%")
            epoch += 1
            
        # Save fine-tuned encoder
        dataset_dir = os.path.join(args.save, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        finetuned_path = os.path.join(dataset_dir, 'finetuned.pt')
        torch.save(classifier.image_encoder, finetuned_path)
        print(f"Saved fine-tuned image encoder to {finetuned_path}")
        
        # Evaluate final model
        classifier.eval()
        test_dataloader = get_dataloader(dataset, is_train=False, args=args)
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                if i > 1:
                    break
                batch = maybe_dictionarize(batch)
                x = batch['images'].to(args.device)
                y = batch['labels'].to(args.device)
                outputs = classifier(x)
                _, predicted = outputs.max(1)
                test_total += y.size(0)
                test_correct += predicted.eq(y).sum().item()
        
        test_acc = 100. * test_correct / test_total
        print(f"Final Test Accuracy for {dataset_name}: {test_acc:.2f}%")

if __name__ == '__main__':
    main()
