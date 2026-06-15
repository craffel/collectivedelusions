import sys
sys.path.insert(0, './local_packages')
import torch
import torchvision.datasets as dset
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
from torch.func import functional_call

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
model_mnist = CLIPVisionModel.from_pretrained('tanganke/clip-vit-base-patch32_mnist').to(device)
model_svhn = CLIPVisionModel.from_pretrained('tanganke/clip-vit-base-patch32_svhn').to(device)

base_params = {n: p for n, p in clip_model.vision_model.named_parameters()}
mnist_params = {n: p for n, p in model_mnist.vision_model.named_parameters()}
svhn_params = {n: p for n, p in model_svhn.vision_model.named_parameters()}

target_param_names = [n for n in base_params.keys() if "encoder.layers." in n]

task_vectors = {
    0: {n: mnist_params[n] - base_params[n] for n in target_param_names},
    1: {n: svhn_params[n] - base_params[n] for n in target_param_names}
}

# Precompute text embeddings
def precompute_text_embeds(classes):
    inputs = processor(text=classes, padding=True, return_tensors='pt').to(device)
    with torch.no_grad():
        text_outputs = clip_model.text_model(**inputs)
        text_embeds = clip_model.text_projection(text_outputs.pooler_output)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return text_embeds

text_embeds_mnist = precompute_text_embeds([f"a photo of the number {i}" for i in range(10)])
text_embeds_svhn = precompute_text_embeds([f"a photo of the number {i}" for i in range(10)])

# Load datasets
mnist_dataset = dset.MNIST(root='./data', train=False, download=False)
svhn_dataset = dset.SVHN(root='./data', split='test', download=False)

torch.manual_seed(42)
mnist_adapt_indices = torch.randperm(len(mnist_dataset))[:50].tolist()
svhn_adapt_indices = torch.randperm(len(svhn_dataset))[:50].tolist()

mnist_test_indices = torch.randperm(len(mnist_dataset))[50:150].tolist()
svhn_test_indices = torch.randperm(len(svhn_dataset))[50:150].tolist()

mnist_adapt_imgs = [mnist_dataset[i][0] for i in mnist_adapt_indices]
svhn_adapt_imgs = [svhn_dataset[i][0] for i in svhn_adapt_indices]

mnist_test_imgs = [mnist_dataset[i][0] for i in mnist_test_indices]
mnist_test_labels = torch.tensor([mnist_dataset[i][1] for i in mnist_test_indices]).to(device)

svhn_test_imgs = [svhn_dataset[i][0] for i in svhn_test_indices]
svhn_test_labels = torch.tensor([svhn_dataset[i][1] for i in svhn_test_indices]).to(device)

inputs_mnist_adapt = processor(images=mnist_adapt_imgs, return_tensors='pt').pixel_values.to(device)
inputs_svhn_adapt = processor(images=svhn_adapt_imgs, return_tensors='pt').pixel_values.to(device)

inputs_mnist_test = processor(images=mnist_test_imgs, return_tensors='pt').pixel_values.to(device)
inputs_svhn_test = processor(images=svhn_test_imgs, return_tensors='pt').pixel_values.to(device)

L, K, degree = 12, 2, 2

def get_monomial_design_matrix():
    l_indices = torch.linspace(0.0, 1.0, L).to(device)
    return torch.stack([l_indices ** j for j in range(degree + 1)], dim=1)

def get_chebyshev_design_matrix():
    l_indices = torch.linspace(0, L - 1, L).to(device)
    x = 2.0 * l_indices / (L - 1) - 1.0
    C = [torch.ones_like(x)]
    if degree >= 1:
        C.append(x)
    for j in range(2, degree + 1):
        C.append(2.0 * x * C[-1] - C[-2])
    return torch.stack(C, dim=1)

V = get_monomial_design_matrix()
C = get_chebyshev_design_matrix()

def evaluate_accuracy(merged_vision_params, test_pixel_values, text_embeds, labels):
    with torch.no_grad():
        vision_outputs = functional_call(clip_model.vision_model, merged_vision_params, args=(), kwargs={'pixel_values': test_pixel_values})
        image_embeds = clip_model.visual_projection(vision_outputs.pooler_output)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        logits = image_embeds @ text_embeds.t()
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
    return acc * 100.0

def entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()

def run_experiment(method_name, lr):
    if method_name == 'AdaMerging':
        lambdas = torch.full((K, L), 0.5, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([lambdas], lr=lr)
    elif method_name == 'PolyMerge':
        gammas = [torch.full((K,), 0.5 if j==0 else 0.0, device=device, requires_grad=True) for j in range(degree + 1)]
        optimizer = torch.optim.Adam(gammas, lr=lr)
    elif method_name == 'ChebyMerge':
        alphas = [torch.full((K,), 0.5 if j==0 else 0.0, device=device, requires_grad=True) for j in range(degree + 1)]
        optimizer = torch.optim.Adam(alphas, lr=lr)
    elif method_name == 'ChebyMerge + CSD':
        alphas = [torch.full((K,), 0.5 if j==0 else 0.0, device=device, requires_grad=True) for j in range(degree + 1)]
        param_groups = [{'params': [alphas[j]], 'lr': lr * (0.2 ** j)} for j in range(degree + 1)]
        optimizer = torch.optim.Adam(param_groups)

    for step in range(20):
        if method_name == 'AdaMerging':
            lambdas_step = lambdas
        elif method_name == 'PolyMerge':
            lambdas_step = sum(gammas[j].unsqueeze(1) * V[:, j].unsqueeze(0) for j in range(degree + 1))
        elif method_name in ['ChebyMerge', 'ChebyMerge + CSD']:
            lambdas_step = sum(alphas[j].unsqueeze(1) * C[:, j].unsqueeze(0) for j in range(degree + 1))

        merged_params = {}
        for name, param in base_params.items():
            if name in target_param_names:
                parts = name.split(".")
                l_idx = int(parts[parts.index("layers") + 1])
                merged_params[name] = param + lambdas_step[0, l_idx] * task_vectors[0][name] + lambdas_step[1, l_idx] * task_vectors[1][name]
            else:
                merged_params[name] = param

        outputs_mnist = functional_call(clip_model.vision_model, merged_params, args=(), kwargs={'pixel_values': inputs_mnist_adapt})
        outputs_svhn = functional_call(clip_model.vision_model, merged_params, args=(), kwargs={'pixel_values': inputs_svhn_adapt})

        embeds_mnist = clip_model.visual_projection(outputs_mnist.pooler_output)
        embeds_mnist = embeds_mnist / embeds_mnist.norm(dim=-1, keepdim=True)
        embeds_svhn = clip_model.visual_projection(outputs_svhn.pooler_output)
        embeds_svhn = embeds_svhn / embeds_svhn.norm(dim=-1, keepdim=True)

        logits_mnist = embeds_mnist @ text_embeds_mnist.t()
        logits_svhn = embeds_svhn @ text_embeds_svhn.t()

        loss = entropy(logits_mnist) + entropy(logits_svhn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final Evaluation
    with torch.no_grad():
        if method_name == 'AdaMerging':
            lambdas_step = lambdas
        elif method_name == 'PolyMerge':
            lambdas_step = sum(gammas[j].unsqueeze(1) * V[:, j].unsqueeze(0) for j in range(degree + 1))
        elif method_name in ['ChebyMerge', 'ChebyMerge + CSD']:
            lambdas_step = sum(alphas[j].unsqueeze(1) * C[:, j].unsqueeze(0) for j in range(degree + 1))

        final_params = {}
        for name, param in base_params.items():
            if name in target_param_names:
                parts = name.split(".")
                l_idx = int(parts[parts.index("layers") + 1])
                final_params[name] = param + lambdas_step[0, l_idx] * task_vectors[0][name] + lambdas_step[1, l_idx] * task_vectors[1][name]
            else:
                final_params[name] = param

    acc_mnist = evaluate_accuracy(final_params, inputs_mnist_test, text_embeds_mnist, mnist_test_labels)
    acc_svhn = evaluate_accuracy(final_params, inputs_svhn_test, text_embeds_svhn, svhn_test_labels)
    return (acc_mnist + acc_svhn) / 2.0

lrs = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
methods = ['AdaMerging', 'PolyMerge', 'ChebyMerge', 'ChebyMerge + CSD']

print("Running learning rate sweep...")
for m in methods:
    print(f"\nMethod: {m}")
    for lr in lrs:
        avg_acc = run_experiment(m, lr)
        print(f"  LR: {lr:<6} | Final Avg Acc: {avg_acc:.2f}%")
