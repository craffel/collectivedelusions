import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

def generate_shape_image(shape_type, size=224, seed=None):
    if seed is not None:
        np.random.seed(seed)
    bg_color = np.random.randint(20, 100, size=3)
    img = Image.new("RGB", (size, size), tuple(bg_color))
    draw = ImageDraw.Draw(img)
    
    # Add background pixel noise
    pixels = np.array(img)
    noise = np.random.normal(0, 15, pixels.shape).astype(np.int16)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    draw = ImageDraw.Draw(img)
    
    shape_color = tuple(np.random.randint(150, 255, size=3))
    center_x = np.random.randint(int(size*0.3), int(size*0.7))
    center_y = np.random.randint(int(size*0.3), int(size*0.7))
    r = np.random.randint(int(size*0.15), int(size*0.3))
    
    if shape_type == "circle":
        draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], fill=shape_color)
    elif shape_type == "square":
        draw.rectangle([center_x - r, center_y - r, center_x + r, center_y + r], fill=shape_color)
    elif shape_type == "triangle":
        p1 = (center_x, center_y - r)
        p2 = (center_x - r, center_y + r)
        p3 = (center_x + r, center_y + r)
        draw.polygon([p1, p2, p3], fill=shape_color)
    elif shape_type == "cross":
        w = np.random.randint(8, 20)
        draw.rectangle([center_x - w, center_y - r, center_x + w, center_y + r], fill=shape_color)
        draw.rectangle([center_x - r, center_y - w, center_x + r, center_y + w], fill=shape_color)
        
    return img

class FeatureExtractor:
    def __init__(self):
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.eval()
        self.activations = {}
        self.hooks = []
        self.hooks.append(self.model.layer1.register_forward_hook(self._get_hook("stage1")))
        self.hooks.append(self.model.layer2.register_forward_hook(self._get_hook("stage2")))
        self.hooks.append(self.model.layer3.register_forward_hook(self._get_hook("stage3")))
        self.hooks.append(self.model.layer4.register_forward_hook(self._get_hook("stage4")))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _get_hook(self, name):
        def hook(model, input, output):
            self.activations[name] = torch.mean(output, dim=[2, 3]).detach()
        return hook
        
    def extract(self, img):
        tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            self.model(tensor)
        feats = {}
        for name in ["stage1", "stage2", "stage3", "stage4"]:
            feat = self.activations[name].squeeze(0).numpy()
            norm = np.linalg.norm(feat)
            feats[name] = feat / norm if norm > 0 else feat
        return feats
        
    def close(self):
        for h in self.hooks:
            h.remove()

def main():
    extractor = FeatureExtractor()
    shape_types = ["circle", "square", "triangle", "cross"]
    K = len(shape_types)
    
    # Generate data
    num_cal = 32
    num_eval = 25
    
    cal_feats = {name: [] for name in ["stage1", "stage2", "stage3", "stage4"]}
    cal_y = []
    
    for k, shape in enumerate(shape_types):
        for i in range(num_cal):
            img = generate_shape_image(shape, seed=42 + i + k*100)
            feats = extractor.extract(img)
            for name in ["stage1", "stage2", "stage3", "stage4"]:
                cal_feats[name].append(feats[name])
            cal_y.append(k)
            
    # Compute stage-specific centroids
    centroids = {}
    for name in ["stage1", "stage2", "stage3", "stage4"]:
        centroids[name] = []
        for k in range(K):
            k_feats = [cal_feats[name][i] for i in range(len(cal_y)) if cal_y[i] == k]
            centroid = np.mean(k_feats, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            centroids[name].append(centroid)
        centroids[name] = np.array(centroids[name])
        
    # Evaluate on fresh test data
    eval_feats = {name: [] for name in ["stage1", "stage2", "stage3", "stage4"]}
    eval_y = []
    for k, shape in enumerate(shape_types):
        for i in range(num_eval):
            img = generate_shape_image(shape, seed=2026 + i + k*200)
            feats = extractor.extract(img)
            for name in ["stage1", "stage2", "stage3", "stage4"]:
                eval_feats[name].append(feats[name])
            eval_y.append(k)
            
    # Measure nearest-centroid accuracy at each stage
    print("Nearest-Centroid Accuracy at each ResNet-18 stage:")
    for name in ["stage1", "stage2", "stage3", "stage4"]:
        correct = 0
        for i in range(len(eval_y)):
            h = eval_feats[name][i]
            sims = np.dot(centroids[name], h)
            pred = np.argmax(sims)
            if pred == eval_y[i]:
                correct += 1
        print(f"  {name}: {correct / len(eval_y) * 100:.2f}%")
        
    extractor.close()

if __name__ == "__main__":
    main()
