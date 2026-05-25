import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
for name, param in model.named_parameters():
    if 'visual' in name:
        if 'proj' in name or 'ln_post' in name or 'class_embedding' in name:
            print(name, param.shape)
