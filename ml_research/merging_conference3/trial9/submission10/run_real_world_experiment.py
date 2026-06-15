import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Insert our local package directory to resolve huggingface-hub dependency
sys.path.insert(0, './local_packages')
from transformers import AutoModel, AutoTokenizer

device = torch.device("cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Define the Multi-Lora Wrapper for Linear layers
class MultiLoraLinear(nn.Module):
    def __init__(self, original_linear, K=3, r=8, lora_alpha=16):
        super().__init__()
        self.original_linear = original_linear # frozen
        self.K = K
        self.r = r
        self.scaling = lora_alpha / r
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.zeros(r, in_features)) for _ in range(K)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(out_features, r)) for _ in range(K)
        ])
        for k in range(K):
            nn.init.kaiming_uniform_(self.lora_A[k], a=np.sqrt(5))
            nn.init.zeros_(self.lora_B[k])
            
    def forward(self, x, alpha=None):
        base_out = self.original_linear(x)
        if alpha is None:
            return base_out
            
        # x shape: (B, SeqLen, D) or (B, D)
        # alpha shape: (B, K)
        # We run each lora adapter and scale by alpha[:, k]
        blended_lora = torch.zeros_like(base_out)
        for k in range(self.K):
            la = torch.matmul(x, self.lora_A[k].t()) # (B, SeqLen, r)
            lb = torch.matmul(la, self.lora_B[k].t()) * self.scaling # (B, SeqLen, out)
            
            # Extract routing weights for expert k and unsqueeze to match dimensions
            ak = alpha[:, k] # (B,)
            if len(x.shape) == 3: # (B, SeqLen, D)
                ak = ak.unsqueeze(-1).unsqueeze(-1)
            else: # (B, D)
                ak = ak.unsqueeze(-1)
            blended_lora = blended_lora + lb * ak
            
        return base_out + blended_lora

# Define 3 distinct synthetic NLP tasks
def generate_synthetic_data():
    # Task 0: Sentiment Analysis (Positive / Negative)
    task0_data = {
        "train_pos": [
            "I absolutely loved this movie!", "It was a fantastic and beautiful story.",
            "The performance of the actors was brilliant and amazing.", "Highly recommend this masterpiece, truly wonderful.",
            "An incredible journey of emotions, exceptionally well done.", "I was completely blown away by this outstanding film.",
            "A magnificent work of art that everyone should experience.", "Simply superb, the directing and script were flawless.",
            "What a heartwarming and delightful experience from start to finish.", "A brilliant cinematic achievement that is deeply moving."
        ] * 10,
        "train_neg": [
            "This movie was absolutely terrible.", "I hated every minute of it, complete waste of time.",
            "The acting was awful and the plot was boring.", "A total disaster of a film, do not watch.",
            "Extremely disappointing, dull characters and bad pacing.", "A pointless and frustrating waste of talent and energy.",
            "I was bored to death, easily one of the worst movies ever.", "The script is laughable and the execution is painful.",
            "A tasteless and uninspired mess that is hard to sit through.", "Do not buy a ticket for this garbage of a production."
        ] * 10,
        "cal_pos": [
            "A very enjoyable film with great charm.", "Sincere, engaging, and beautifully filmed.",
            "A solid performance from the lead cast.", "Refreshing and wonderfully written script."
        ] * 4, # 16 samples total
        "cal_neg": [
            "Very poorly done, lacks any depth or interest.", "A complete mess that fails to deliver.",
            "The story makes no sense and the acting is stiff.", "A boring and uninspired waste of time."
        ] * 4,
        "test_pos": [
            "An absolute delight, highly entertaining!", "Immensely satisfying with a brilliant climax.",
            "A beautiful and poignant narrative.", "Captivating and exceptionally well-crafted.",
            "Wonderful experience, I would highly recommend it."
        ] * 8, # 40 samples total
        "test_neg": [
            "A dreadful and painful film to watch.", "Completely unwatchable, a complete failure.",
            "Stupid plot, awful directing, and terrible acting.", "A cheap and lazy attempt at a thriller.",
            "Such a massive waste of time and money."
        ] * 8
    }

    # Task 1: Topic Classification (Sports / Science)
    task1_data = {
        "train_pos": [ # Sports
            "The team won the championship match after a thrilling overtime.", "He scored a beautiful goal in the second half of the game.",
            "The basketball player made a stunning buzzer-beater shot.", "The tennis match was intense, with both athletes playing perfectly.",
            "She broke the world record in the 100-meter dash today.", "The football coach was extremely proud of the players' performance.",
            "The stadium was packed with excited fans cheering for their team.", "He hit a home run in the bottom of the ninth inning.",
            "An incredible athletic display of skill and speed on the field.", "They secured the gold medal in the relay tournament."
        ] * 10,
        "train_neg": [ # Science
            "The scientists discovered a new chemical element in the lab.", "Quantum physics explains the complex behavior of subatomic particles.",
            "A new study on DNA genetic sequencing was published today.", "The space telescope captured stunning images of distant galaxies.",
            "The chemical reaction produced a stable compound under pressure.", "Researchers are investigating the effects of temperature on enzymes.",
            "The theory of relativity revolutionized our understanding of space.", "Geologists found ancient fossils embedded in the rock layer.",
            "The mathematical algorithm successfully modeled the biological system.", "A breakthrough in nuclear fusion could provide clean energy."
        ] * 10,
        "cal_pos": [ # Sports
            "The tournament ended with a spectacular final match.", "He trained hard and won the marathon easily.",
            "The goalkeeper made several incredible saves during play.", "A great victory for the visiting team last night."
        ] * 4,
        "cal_neg": [ # Science
            "The laboratory analysis confirmed the chemical structure.", "Astronomers detected an unusual radio signal from space.",
            "The experiment validated the initial hypothesis perfectly.", "They published their research findings in a scientific journal."
        ] * 4,
        "test_pos": [ # Sports
            "An amazing athletic achievement by the young runner.", "The championship cup was awarded to the winning squad.",
            "He scored three touchdowns during the playoff game.", "The Olympic team trained diligently for four years.",
            "A brilliant strategy by the coach led to victory."
        ] * 8,
        "test_neg": [ # Science
            "The study provides deep insights into molecular biology.", "The physicists observed the particle collision in the accelerator.",
            "A new theory describes the expansion of the universe.", "The laboratory developed a highly sensitive temperature sensor.",
            "The research paper covers the genetics of plant adaptation."
        ] * 8
    }

    # Task 2: Sentence Type Classification (Question / Statement)
    task2_data = {
        "train_pos": [ # Question
            "Where is the nearest library located in this town?", "Who is the author of this famous classical novel?",
            "Why does the temperature drop significantly at night?", "When will the next solar eclipse be visible here?",
            "How do cells convert nutrients into usable energy?", "Is there any water on the surface of Mars?",
            "What is the exact chemical composition of water?", "Which country has the largest population in the world?",
            "Are there any alternative explanations for this phenomenon?", "Can we predict volcanic eruptions using seismic data?"
        ] * 10,
        "train_neg": [ # Statement
            "The library is closed on Sundays and national holidays.", "This book was written by a famous 19th-century author.",
            "The earth rotates on its axis once every twenty-four hours.", "The sun is a medium-sized star at the center of our system.",
            "Cells use mitochondria to produce adenosine triphosphate.", "Mars has a thin atmosphere composed mostly of carbon dioxide.",
            "Water consists of two hydrogen atoms and one oxygen atom.", "China and India are the two most populous nations.",
            "There are several factors contributing to global climate patterns.", "Seismic sensors measure the vibrations of the earth's crust."
        ] * 10,
        "cal_pos": [ # Question
            "Where can I find more research papers on this?", "Who designed the architecture of this computer system?",
            "Why are leaves green during the spring season?", "When was the first artificial satellite launched?"
        ] * 4,
        "cal_neg": [ # Statement
            "Leaves contain chlorophyll which absorbs blue and red light.", "The computer architecture utilizes a multi-core processor.",
            "The satellite orbit is decaying slowly over several decades.", "The research project was funded by a national grant."
        ] * 4,
        "test_pos": [ # Question
            "What are the primary symptoms of the disease?", "Which algorithm is most efficient for sorting array elements?",
            "Why do birds migrate south during the winter?", "How does the immune system recognize foreign pathgens?",
            "Is it possible to travel faster than the speed of light?"
        ] * 8,
        "test_neg": [ # Statement
            "Sorting algorithms organize data in a specified order.", "Migrating birds navigate using the earth's magnetic field.",
            "The immune system produces specific antibodies against antigens.", "Nothing can travel faster than light in a vacuum.",
            "The patient exhibited mild symptoms of a common cold."
        ] * 8
    }

    return [task0_data, task1_data, task2_data]

# Helper to tokenize datasets
def tokenize_sentences(sentences_pos, sentences_neg, tokenizer, max_length=32):
    sentences = sentences_pos + sentences_neg
    labels = [1] * len(sentences_pos) + [0] * len(sentences_neg)
    
    encoded = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return encoded["input_ids"], encoded["attention_mask"], torch.tensor(labels, dtype=torch.long)

class BERTMultiTaskWrapper(nn.Module):
    def __init__(self, base_model, K=3):
        super().__init__()
        self.base_model = base_model
        self.K = K
        
        # We replace the key linear projections in the last layer of the encoder with MultiLoraLinear
        # BertModel has self.base_model.encoder.layer
        self.target_layer = self.base_model.encoder.layer[-1]
        
        # Replace attention query, key, value, and output dense
        self.target_layer.attention.self.query = MultiLoraLinear(self.target_layer.attention.self.query, K=K)
        self.target_layer.attention.self.key = MultiLoraLinear(self.target_layer.attention.self.key, K=K)
        self.target_layer.attention.self.value = MultiLoraLinear(self.target_layer.attention.self.value, K=K)
        self.target_layer.attention.output.dense = MultiLoraLinear(self.target_layer.attention.output.dense, K=K)
        
        # Replace MLP intermediate and output dense
        self.target_layer.intermediate.dense = MultiLoraLinear(self.target_layer.intermediate.dense, K=K)
        self.target_layer.output.dense = MultiLoraLinear(self.target_layer.output.dense, K=K)
        
        # 3 separate binary classification heads (one for each task)
        self.heads = nn.ModuleList([
            nn.Linear(base_model.config.hidden_size, 2) for _ in range(K)
        ])
        
    def forward(self, input_ids, attention_mask, alpha=None):
        # We pass alpha to the modified linear layers
        # To do this, we override how layers are called or we set a global alpha state in the modified layers
        # A clean way is to temporarily set a dynamic alpha attribute on the wrapped layers
        for name, module in self.named_modules():
            if isinstance(module, MultiLoraLinear):
                module.alpha_state = alpha
                
        # Now we modify MultiLoraLinear's forward to use this state if it exists
        # Let's override the forward call dynamically or let MultiLoraLinear access it
        # We can just run the standard base_model forward
        # Let's intercept the inputs and run custom forward if needed, or simply let the internal modified layers read the state
        # Let's inspect MultiLoraLinear.forward: it checks if 'alpha' is passed.
        # Since the standard BERT forward won't pass alpha to the modules, we can set an `alpha` attribute on the module
        # and modify MultiLoraLinear to check self.alpha_state!
        # Let's do that!
        for name, module in self.named_modules():
            if isinstance(module, MultiLoraLinear):
                module.current_alpha = alpha
                
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output is outputs.pooler_output (shape: B, 128)
        pooled = outputs.pooler_output
        return pooled

    def get_logits(self, pooled, task_id):
        # Pass through the task-specific classification head
        return self.heads[task_id](pooled)

# Override MultiLoraLinear forward to check for active state
def custom_multilora_forward(self, x):
    alpha = getattr(self, "current_alpha", None)
    base_out = self.original_linear(x)
    if alpha is None:
        return base_out
    
    # x shape: (B, SeqLen, D) or (B, D)
    # alpha shape: (B, K)
    blended_lora = torch.zeros_like(base_out)
    for k in range(self.K):
        la = torch.matmul(x, self.lora_A[k].t())
        lb = torch.matmul(la, self.lora_B[k].t()) * self.scaling
        
        ak = alpha[:, k]
        if len(x.shape) == 3: # (B, SeqLen, D)
            ak = ak.unsqueeze(-1).unsqueeze(-1)
        else: # (B, D)
            ak = ak.unsqueeze(-1)
        blended_lora = blended_lora + lb * ak
        
    return base_out + blended_lora

MultiLoraLinear.forward = custom_multilora_forward

def extract_early_activations(model, input_ids, attention_mask):
    """
    Extract CLS hidden state of Layer 0 (adapter-free) as early representation z_b \in \mathbb{R}^{128}.
    """
    with torch.no_grad():
        # Get embeddings
        embeddings = model.base_model.embeddings(input_ids=input_ids)
        # Run Layer 0
        layer_outputs = model.base_model.encoder.layer[0](embeddings, attention_mask=None)
        # Layer 0 output shape: (B, SeqLen, 128)
        # Extract CLS token representation (index 0)
        cls_rep = layer_outputs[0][:, 0, :] # (B, 128)
    return cls_rep

# Functions for optimizing ensembling temperatures
def optimize_erm(cal_energies_norm, P, epochs=100, lr=0.05, tau0=0.2):
    N, K = cal_energies_norm.size()
    w = torch.full((K,), np.log(tau0), requires_grad=True, device=device)
    optimizer = torch.optim.Adam([w], lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        tau = torch.clamp(torch.exp(w), min=0.01, max=10.0)
        alpha = torch.softmax(cal_energies_norm / tau, dim=-1)
        loss = torch.mean(1.0 - torch.sum(alpha * P, dim=-1))
        loss.backward()
        optimizer.step()
    return torch.clamp(torch.exp(w), min=0.01, max=10.0).detach()

def optimize_pac_zca(cal_energies_norm, P, epochs=100, lr=0.05, sigma0_sq=5.0, delta=0.05, tau0=0.2):
    N, K = cal_energies_norm.size()
    w = torch.full((K,), np.log(tau0), requires_grad=True, device=device)
    w0 = torch.full((K,), np.log(tau0), device=device)
    optimizer = torch.optim.Adam([w], lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        tau = torch.clamp(torch.exp(w), min=0.01, max=10.0)
        alpha = torch.softmax(cal_energies_norm / tau, dim=-1)
        cal_loss = torch.mean(1.0 - torch.sum(alpha * P, dim=-1))
        kl = torch.sum((w - w0)**2) / (2.0 * sigma0_sq)
        bound = cal_loss + torch.sqrt((kl + np.log(2.0 * np.sqrt(N) / delta)) / (2.0 * N))
        bound.backward()
        optimizer.step()
    return torch.clamp(torch.exp(w), min=0.01, max=10.0).detach()

def optimize_dirichlet_pac(cal_energies_norm, P, epochs=100, lr=0.05, tau0=0.2, delta=0.05):
    N, K = cal_energies_norm.size()
    w = torch.full((K,), np.log(tau0), requires_grad=True, device=device)
    optimizer = torch.optim.Adam([w], lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        tau = torch.clamp(torch.exp(w), min=0.01, max=10.0)
        exponent = torch.clamp(cal_energies_norm / tau, min=-50.0, max=50.0)
        a_b = torch.exp(exponent)
        exponent0 = torch.clamp(cal_energies_norm / tau0, min=-50.0, max=50.0)
        a0_b = torch.exp(exponent0)
        alpha = a_b / torch.sum(a_b, dim=-1, keepdim=True)
        cal_loss = torch.mean(1.0 - torch.sum(alpha * P, dim=-1))
        
        sum_a = torch.sum(a_b, dim=-1)
        sum_a0 = torch.sum(a0_b, dim=-1)
        kl_samples = (torch.lgamma(sum_a) - torch.lgamma(sum_a0)
                      - torch.sum(torch.lgamma(a_b), dim=-1)
                      + torch.sum(torch.lgamma(a0_b), dim=-1)
                      + torch.sum((a_b - a0_b) * (torch.digamma(a_b) - torch.digamma(sum_a).unsqueeze(-1)), dim=-1))
        mean_kl = torch.mean(kl_samples)
        bound = cal_loss + torch.sqrt((mean_kl + np.log(2.0 * np.sqrt(N) / delta)) / (2.0 * N))
        bound.backward()
        optimizer.step()
    return torch.clamp(torch.exp(w), min=0.01, max=10.0).detach()

def optimize_dirichlet_pem(cal_energies_norm, P_all, lambda_div=1.0, epochs=100, lr=0.05, tau0=0.2, delta=0.05):
    N, K = cal_energies_norm.size()
    w = torch.full((K,), np.log(tau0), requires_grad=True, device=device)
    optimizer = torch.optim.Adam([w], lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        tau = torch.clamp(torch.exp(w), min=0.01, max=10.0)
        exponent = torch.clamp(cal_energies_norm / tau, min=-50.0, max=50.0)
        a_b = torch.exp(exponent)
        exponent0 = torch.clamp(cal_energies_norm / tau0, min=-50.0, max=50.0)
        a0_b = torch.exp(exponent0)
        alpha = a_b / torch.sum(a_b, dim=-1, keepdim=True)
        
        ensembled_probs = torch.sum(alpha.unsqueeze(-1) * P_all, dim=1) # (N, C)
        ensembled_probs_clamped = torch.clamp(ensembled_probs, min=1e-15)
        entropy = -torch.sum(ensembled_probs_clamped * torch.log(ensembled_probs_clamped), dim=-1) # (N,)
        individual_entropy = torch.mean(entropy) / np.log(2.0) # binary, so C=2 classes per head
        
        mean_alpha = torch.mean(alpha, dim=0)
        mean_alpha_clamped = torch.clamp(mean_alpha, min=1e-15)
        diversity_entropy = -torch.sum(mean_alpha_clamped * torch.log(mean_alpha_clamped)) / np.log(K)
        
        cal_loss = individual_entropy - lambda_div * diversity_entropy
        
        sum_a = torch.sum(a_b, dim=-1)
        sum_a0 = torch.sum(a0_b, dim=-1)
        kl_samples = (torch.lgamma(sum_a) - torch.lgamma(sum_a0)
                      - torch.sum(torch.lgamma(a_b), dim=-1)
                      + torch.sum(torch.lgamma(a0_b), dim=-1)
                      + torch.sum((a_b - a0_b) * (torch.digamma(a_b) - torch.digamma(sum_a).unsqueeze(-1)), dim=-1))
        mean_kl = torch.mean(kl_samples)
        bound = cal_loss + torch.sqrt((mean_kl + np.log(2.0 * np.sqrt(N) / delta)) / (2.0 * N))
        bound.backward()
        optimizer.step()
    return torch.clamp(torch.exp(w), min=0.01, max=10.0).detach()

def run_real_world_experiment(seed=42, model_name="prajjwal1/bert-tiny"):
    print(f"\n--- Running Real-World Transformer Experiment ({model_name}, Seed: {seed}) ---")
    set_seed(seed)
    K = 3
    
    # 1. Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    
    # Freeze all original BERT parameters
    for param in base_model.parameters():
        param.requires_grad = False
        
    model = BERTMultiTaskWrapper(base_model, K=K).to(device)
    
    # 2. Generate raw text datasets for 3 tasks
    raw_data = generate_synthetic_data()
    
    # 3. Tokenize and wrap into DataLoaders
    train_loaders = []
    cal_inputs = []
    test_inputs = []
    
    for k in range(K):
        # Tokenize train pos/neg
        tr_ids, tr_masks, tr_labels = tokenize_sentences(raw_data[k]["train_pos"], raw_data[k]["train_neg"], tokenizer)
        train_ds = TensorDataset(tr_ids, tr_masks, tr_labels)
        train_loaders.append(DataLoader(train_ds, batch_size=16, shuffle=True))
        
        # Tokenize cal pos/neg
        cal_ids, cal_masks, cal_labels = tokenize_sentences(raw_data[k]["cal_pos"], raw_data[k]["cal_neg"], tokenizer)
        cal_inputs.append((cal_ids, cal_masks, cal_labels))
        
        # Tokenize test pos/neg
        t_ids, t_masks, t_labels = tokenize_sentences(raw_data[k]["test_pos"], raw_data[k]["test_neg"], tokenizer)
        test_inputs.append((t_ids, t_masks, t_labels))
        
    # 4. Train each task adapter sequentially (only lora parameters and specific classification head)
    for k in range(K):
        print(f"Training Adapter for Task {k}...")
        # Unfreeze task k adapter parameters and head
        for param in model.parameters():
            param.requires_grad = False
            
        for param in model.target_layer.attention.self.query.lora_A[k].parameters() if hasattr(model.target_layer.attention.self.query.lora_A[k], 'parameters') else [model.target_layer.attention.self.query.lora_A[k]]:
            param.requires_grad = True
        for param in model.target_layer.attention.self.query.lora_B[k].parameters() if hasattr(model.target_layer.attention.self.query.lora_B[k], 'parameters') else [model.target_layer.attention.self.query.lora_B[k]]:
            param.requires_grad = True
        for param in model.target_layer.attention.self.key.lora_A[k].parameters() if hasattr(model.target_layer.attention.self.key.lora_A[k], 'parameters') else [model.target_layer.attention.self.key.lora_A[k]]:
            param.requires_grad = True
        for param in model.target_layer.attention.self.key.lora_B[k].parameters() if hasattr(model.target_layer.attention.self.key.lora_B[k], 'parameters') else [model.target_layer.attention.self.key.lora_B[k]]:
            param.requires_grad = True
        for param in model.target_layer.attention.self.value.lora_A[k].parameters() if hasattr(model.target_layer.attention.self.value.lora_A[k], 'parameters') else [model.target_layer.attention.self.value.lora_A[k]]:
            param.requires_grad = True
        for param in model.target_layer.attention.self.value.lora_B[k].parameters() if hasattr(model.target_layer.attention.self.value.lora_B[k], 'parameters') else [model.target_layer.attention.self.value.lora_B[k]]:
            param.requires_grad = True
        for param in model.target_layer.attention.output.dense.lora_A[k].parameters() if hasattr(model.target_layer.attention.output.dense.lora_A[k], 'parameters') else [model.target_layer.attention.output.dense.lora_A[k]]:
            param.requires_grad = True
        for param in model.target_layer.attention.output.dense.lora_B[k].parameters() if hasattr(model.target_layer.attention.output.dense.lora_B[k], 'parameters') else [model.target_layer.attention.output.dense.lora_B[k]]:
            param.requires_grad = True
            
        for param in model.target_layer.intermediate.dense.lora_A[k].parameters() if hasattr(model.target_layer.intermediate.dense.lora_A[k], 'parameters') else [model.target_layer.intermediate.dense.lora_A[k]]:
            param.requires_grad = True
        for param in model.target_layer.intermediate.dense.lora_B[k].parameters() if hasattr(model.target_layer.intermediate.dense.lora_B[k], 'parameters') else [model.target_layer.intermediate.dense.lora_B[k]]:
            param.requires_grad = True
        for param in model.target_layer.output.dense.lora_A[k].parameters() if hasattr(model.target_layer.output.dense.lora_A[k], 'parameters') else [model.target_layer.output.dense.lora_A[k]]:
            param.requires_grad = True
        for param in model.target_layer.output.dense.lora_B[k].parameters() if hasattr(model.target_layer.output.dense.lora_B[k], 'parameters') else [model.target_layer.output.dense.lora_B[k]]:
            param.requires_grad = True
            
        for param in model.heads[k].parameters():
            param.requires_grad = True
            
        # Optimize with Adam
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # We set alpha to one-hot vector during training of expert k
        train_alpha = torch.zeros(16, K, device=device)
        train_alpha[:, k] = 1.0
        
        model.train()
        for epoch in range(10): # 10 epochs is plenty to overfit these simple datasets
            total_loss = 0.0
            for batch_ids, batch_masks, batch_labels in train_loaders[k]:
                optimizer.zero_grad()
                # Run with task-specific routing weight
                # Adjust training alpha to size of the actual batch
                batch_alpha = train_alpha[:batch_ids.size(0)]
                pooled = model(batch_ids, batch_masks, alpha=batch_alpha)
                logits = model.get_logits(pooled, k)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1}/10, Loss: {total_loss:.4f}", flush=True)
            
    # Freeze everything post-training
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    # 5. Extract intermediate activations and run the Sample-Splitting protocol for Calibration
    # Each calibration set has 16 samples.
    # Split into Subset 1 (Prior Calibration Set: 8 samples/task) and Subset 2 (Optimization Calibration Set: 8 samples/task)
    prior_act = []
    prior_lbl = []
    
    opt_act = []
    opt_lbl = []
    opt_tids = []
    opt_ids_list = []
    opt_masks_list = []
    
    for k in range(K):
        ids, masks, lbls = cal_inputs[k]
        
        # Subset 1 (Prior Calibration Set, indices 0-7)
        ids_prior, masks_prior = ids[:8], masks[:8]
        act_prior = extract_early_activations(model, ids_prior, masks_prior)
        prior_act.append(act_prior)
        prior_lbl.append(lbls[:8])
        
        # Subset 2 (Optimization Calibration Set, indices 8-15)
        ids_opt, masks_opt = ids[8:16], masks[8:16]
        act_opt = extract_early_activations(model, ids_opt, masks_opt)
        opt_act.append(act_opt)
        opt_lbl.append(lbls[8:16])
        opt_tids.append(torch.full((8,), k, dtype=torch.long))
        opt_ids_list.append(ids_opt)
        opt_masks_list.append(masks_opt)
        
    prior_act = torch.cat(prior_act, dim=0) # (24, 128)
    prior_lbl = torch.cat(prior_lbl, dim=0)
    
    opt_act = torch.cat(opt_act, dim=0) # (24, 128)
    opt_lbl = torch.cat(opt_lbl, dim=0)
    opt_tids = torch.cat(opt_tids, dim=0)
    opt_ids = torch.cat(opt_ids_list, dim=0)
    opt_masks = torch.cat(opt_masks_list, dim=0)
    
    # 6. Extract task-specific subspaces from Subset 1 using SVD
    # We choose d = 4 subspace dimension (out of D = 128)
    d = 4
    V_d = []
    for k in range(K):
        # We look at prior activations from task k (indices k*8 to (k+1)*8)
        Z_k = prior_act[k*8:(k+1)*8] # (8, 128)
        U, S, Vt = torch.linalg.svd(Z_k, full_matrices=False)
        # SVD coordinate matrix is Vt.T[:, :d]
        V_d.append(Vt.T[:, :d])
        
    # 7. Compute SEP coordinates on Subset 2 and test sets
    opt_energies = torch.zeros(opt_act.size(0), K, device=device)
    for k in range(K):
        opt_energies[:, k] = torch.norm(torch.matmul(opt_act, V_d[k]), dim=-1)
    # Perform energy normalization
    opt_energies_norm = opt_energies / torch.sum(opt_energies, dim=-1, keepdim=True)
    
    # Let's aggregate our test sets: total 120 samples
    test_ids_list = []
    test_masks_list = []
    test_lbl_list = []
    test_tids_list = []
    for k in range(K):
        t_ids, t_masks, t_labels = test_inputs[k]
        test_ids_list.append(t_ids)
        test_masks_list.append(t_masks)
        test_lbl_list.append(t_labels)
        test_tids_list.append(torch.full((t_ids.size(0),), k, dtype=torch.long))
        
    test_ids = torch.cat(test_ids_list, dim=0)
    test_masks = torch.cat(test_masks_list, dim=0)
    test_lbls = torch.cat(test_lbl_list, dim=0)
    test_tids = torch.cat(test_tids_list, dim=0)
    
    # Extract early activations for test set
    test_act = extract_early_activations(model, test_ids, test_masks) # (120, 128)
    test_energies = torch.zeros(test_act.size(0), K, device=device)
    for k in range(K):
        test_energies[:, k] = torch.norm(torch.matmul(test_act, V_d[k]), dim=-1)
    test_energies_norm = test_energies / torch.sum(test_energies, dim=-1, keepdim=True)
    
    # 8. Precompute prediction probabilities of experts on Subset 2
    # P has shape (N_opt, K)
    N_opt = opt_act.size(0) # 24
    P = torch.zeros(N_opt, K, device=device)
    P_all = torch.zeros(N_opt, K, 2, device=device) # binary heads, so C = 2
    
    with torch.no_grad():
        for k in range(K):
            # Evaluate using expert k's one-hot weights
            expert_alpha = torch.zeros(N_opt, K, device=device)
            expert_alpha[:, k] = 1.0
            
            # Run model with this routing weight
            pooled = model(opt_ids, opt_masks, alpha=expert_alpha)
            
            # For each task expert, we obtain predictions on the corresponding task head
            for b in range(N_opt):
                task_id = opt_tids[b].item()
                logits = model.get_logits(pooled[b:b+1], task_id)
                probs = torch.softmax(logits, dim=-1).squeeze(0) # (2,)
                
                correct_label = opt_lbl[b].item()
                P[b, k] = probs[correct_label].item()
                P_all[b, k, :] = probs
                
    # 9. Optimize Router Temperatures
    tau0 = 0.05
    tau_erm = optimize_erm(opt_energies_norm, P, tau0=tau0)
    tau_pac_zca = optimize_pac_zca(opt_energies_norm, P, tau0=tau0)
    tau_dirichlet = optimize_dirichlet_pac(opt_energies_norm, P, tau0=tau0)
    tau_pem_div = optimize_dirichlet_pem(opt_energies_norm, P_all, lambda_div=1.0, tau0=tau0)
    
    print(f"Optimized Temperatures:")
    print(f"  Temp-Only ERM: {tau_erm.tolist()}")
    print(f"  PAC-ZCA:       {tau_pac_zca.tolist()}")
    print(f"  Dirichlet-PAC: {tau_dirichlet.tolist()}")
    print(f"  PEM-Div:       {tau_pem_div.tolist()}")
    
    # 10. Evaluation on Test Set
    # We evaluate ensembled networks under several routing strategies:
    # A. Expert Ceiling (Oracle): dispatch each test query to its correct expert
    # B. Uniform Merging: static routing alpha_k = 1/K = 0.33
    # C. SABLE (SEP-Block) Norm: static routing with tau = 0.05
    # D. Temp-Only ERM: dynamic softmax ensembling with optimized ERM temperatures
    # E. PAC-ZCA: dynamic softmax ensembling with optimized ZCA temperatures
    # F. Dirichlet-PAC (Ours): dynamic Dirichlet ensembling with optimized temperatures
    # G. PEM-Div (Ours): dynamic Dirichlet ensembling with PEM-Div optimized temperatures
    
    eval_strategies = {
        "Expert Ceiling": None,
        "Uniform Merging": torch.full((test_ids.size(0), K), 1.0 / K),
        "SABLE (SEP-Block) Norm": torch.softmax(test_energies_norm / 0.05, dim=-1),
        "Temp-Only ERM": torch.softmax(test_energies_norm / tau_erm, dim=-1),
        "PAC-ZCA": torch.softmax(test_energies_norm / tau_pac_zca, dim=-1),
        "Dirichlet-PAC (Ours)": None, # computed via Dirichlet concentration mapping
        "PEM-Div (Ours)": None # computed via Dirichlet concentration mapping
    }
    
    # Handle Dirichlet-PAC and PEM-Div ensembling weight mappings
    # Dirichlet-PAC
    exp_dirichlet = torch.clamp(test_energies_norm / tau_dirichlet, min=-50.0, max=50.0)
    a_dirichlet = torch.exp(exp_dirichlet)
    alpha_dirichlet = a_dirichlet / torch.sum(a_dirichlet, dim=-1, keepdim=True)
    eval_strategies["Dirichlet-PAC (Ours)"] = alpha_dirichlet
    
    # PEM-Div
    exp_pem = torch.clamp(test_energies_norm / tau_pem_div, min=-50.0, max=50.0)
    a_pem = torch.exp(exp_pem)
    alpha_pem = a_pem / torch.sum(a_pem, dim=-1, keepdim=True)
    eval_strategies["PEM-Div (Ours)"] = alpha_pem
    
    # Run evaluation
    results = {}
    with torch.no_grad():
        # Expert Ceiling
        correct_ceiling = 0
        total_test = test_ids.size(0)
        for b in range(total_test):
            task_id = test_tids[b].item()
            expert_alpha = torch.zeros(1, K, device=device)
            expert_alpha[0, task_id] = 1.0
            
            pooled = model(test_ids[b:b+1], test_masks[b:b+1], alpha=expert_alpha)
            logits = model.get_logits(pooled, task_id)
            pred = torch.argmax(logits, dim=-1).item()
            if pred == test_lbls[b].item():
                correct_ceiling += 1
        results["Expert Ceiling"] = correct_ceiling / total_test
        
        # Other strategies
        for name, alpha in eval_strategies.items():
            if alpha is None: # Handled dynamically
                continue
            correct = 0
            # Run parallel or batch inference
            pooled = model(test_ids, test_masks, alpha=alpha)
            for b in range(total_test):
                task_id = test_tids[b].item()
                logits = model.get_logits(pooled[b:b+1], task_id)
                pred = torch.argmax(logits, dim=-1).item()
                if pred == test_lbls[b].item():
                    correct += 1
            results[name] = correct / total_test
            
    print("\nReal-World Test Accuracies:")
    for name, acc in results.items():
        print(f"  {name:25s}: {acc*100:6.2f}%")
        
    return results

if __name__ == "__main__":
    import json
    seeds = [42, 43, 44, 45, 46]
    all_runs = []
    for s in seeds:
        all_runs.append(run_real_world_experiment(s))
        
    # Aggregate results
    keys = all_runs[0].keys()
    aggregated = {}
    for k in keys:
        vals = [run[k] for run in all_runs]
        mean = np.mean(vals) * 100.0
        std = np.std(vals) * 100.0
        aggregated[k] = {"mean": mean, "std": std}
        
    print("\nAggregated Real-World Results (Mean ± SD % over 5 Seeds):")
    for k in keys:
        print(f"  {k:25s}: {aggregated[k]['mean']:6.2f}% ± {aggregated[k]['std']:4.2f}%")
        
    # Save results to a json file
    with open("real_world_results.json", "w") as f:
        json.dump(aggregated, f, indent=2)
