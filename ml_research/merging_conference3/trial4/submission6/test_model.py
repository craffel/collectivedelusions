import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AdaMerging', 'src'))

from modeling import ImageEncoder
from args import parse_arguments

def main():
    args = parse_arguments()
    args.model = 'ViT-B-32'
    args.device = 'cpu'
    
    # Try instantiating ImageEncoder
    print("Instantiating ImageEncoder...")
    image_encoder = ImageEncoder(args, keep_lang=False)
    print("ImageEncoder instantiated successfully!")
    
    # Print a few keys of image_encoder state dict
    sd_keys = list(image_encoder.state_dict().keys())
    print("ImageEncoder state dict keys (first 10):", sd_keys[:10])
    
    # Load our zeroshot.pt state dict
    zeroshot_path = os.path.join('checkpoints', 'ViT-B-32', 'zeroshot.pt')
    if os.path.exists(zeroshot_path):
        print(f"Loading {zeroshot_path}...")
        checkpoint_sd = torch.load(zeroshot_path, map_location='cpu')
        checkpoint_keys = list(checkpoint_sd.keys())
        print("Checkpoint state dict keys (first 10):", checkpoint_keys[:10])
        
        # Check if they match or if checkpoint_sd is indeed a state_dict of ImageEncoder
        print(f"Number of keys in ImageEncoder: {len(sd_keys)}")
        print(f"Number of keys in Checkpoint: {len(checkpoint_keys)}")
        
        # Test loading
        try:
            image_encoder.load_state_dict(checkpoint_sd, strict=True)
            print("Successfully loaded checkpoint state_dict with strict=True!")
        except Exception as e:
            print("Failed loading with strict=True, trying strict=False...")
            try:
                image_encoder.load_state_dict(checkpoint_sd, strict=False)
                print("Successfully loaded checkpoint state_dict with strict=False!")
            except Exception as e2:
                print(f"Error loading checkpoint: {e2}")

if __name__ == '__main__':
    main()
