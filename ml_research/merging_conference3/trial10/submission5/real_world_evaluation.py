import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import time

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_20newsgroups_data():
    # Load newsgroups
    categories = [
        # Computers (Comp)
        'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
        # Recreation (Rec)
        'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
        # Science (Sci)
        'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
        # Talk (Talk)
        'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc', 'alt.atheism', 'soc.religion.christian'
    ]
    
    train_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    test_data = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))
    
    # Map raw categories to 4 meta-domains (K=4)
    cat_to_domain = {}
    for cat in train_data.target_names:
        if cat.startswith('comp'):
            cat_to_domain[cat] = 0 # Computers
        elif cat.startswith('rec'):
            cat_to_domain[cat] = 1 # Recreation
        elif cat.startswith('sci'):
            cat_to_domain[cat] = 2 # Science
        else:
            cat_to_domain[cat] = 3 # Talk/Religion
            
    train_targets = np.array([cat_to_domain[train_data.target_names[t]] for t in train_data.target])
    test_targets = np.array([cat_to_domain[test_data.target_names[t]] for t in test_data.target])
    
    return train_data.data, train_targets, test_data.data, test_targets

class ExpertMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        h = self.relu(self.fc1(x))
        out = self.fc2(h)
        return h, out

def train_experts(X_train, y_train, K=4, input_dim=1024):
    print("Training 4 specialized experts on 20newsgroups...")
    experts = []
    
    for k in range(K):
        # Create a biased training set for expert k
        # 80% of samples from domain k, 20% from others
        idx_k = np.where(y_train == k)[0]
        idx_others = np.where(y_train != k)[0]
        
        # Balance sizes
        n_k = len(idx_k)
        n_others = n_k // 4
        
        selected_others = np.random.choice(idx_others, n_others, replace=False)
        biased_indices = np.concatenate([idx_k, selected_others])
        np.random.shuffle(biased_indices)
        
        X_biased = torch.FloatTensor(X_train[biased_indices])
        y_biased = torch.LongTensor(y_train[biased_indices])
        
        model = ExpertMLP(input_dim=input_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        model.train()
        # Train for 5 epochs
        for epoch in range(5):
            optimizer.zero_grad()
            _, outputs = model(X_biased)
            loss = criterion(outputs, y_biased)
            loss.backward()
            optimizer.step()
            
        model.eval()
        experts.append(model)
        
        # Evaluate expert k on domain k vs other domains
        with torch.no_grad():
            _, train_outputs = model(torch.FloatTensor(X_train))
            preds = torch.argmax(train_outputs, dim=1).numpy()
            acc_k = np.mean(preds[y_train == k] == k)
            acc_others = np.mean(preds[y_train != k] == y_train[y_train != k])
            print(f"  Expert {k} trained. Accuracy on own domain: {acc_k:.2%}, Accuracy on other domains: {acc_others:.2%}")
            
    return experts

def run_real_world_simulation(seed=42):
    set_seed(seed)
    
    # 1. Load Data
    train_texts, train_targets, test_texts, test_targets = get_20newsgroups_data()
    
    # Filter out empty documents
    train_indices = [i for i, text in enumerate(train_texts) if len(text.strip()) > 10]
    test_indices = [i for i, text in enumerate(test_texts) if len(text.strip()) > 10]
    
    train_texts = [train_texts[i] for i in train_indices]
    train_targets = train_targets[train_indices]
    test_texts = [test_texts[i] for i in test_indices]
    test_targets = test_targets[test_indices]
    
    # 2. Extract TF-IDF features
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=1024, stop_words='english')
    X_train_sparse = vectorizer.fit_transform(train_texts)
    X_test_sparse = vectorizer.transform(test_texts)
    
    X_train = X_train_sparse.toarray()
    X_test = X_test_sparse.toarray()
    
    # 3. Train Experts
    K = 4
    experts = train_experts(X_train, train_targets, K=K, input_dim=1024)
    
    # 4. Calibration Phase: Extract layer-wise centroids of expert hidden representations
    print("Calibrating expert centroids...")
    # For simplicity, our multi-layer ensembling model has L=5 layers:
    # L=3 is input features (TF-IDF vector)
    # L=4 is the hidden representation from Expert MLP (128-dim)
    # L=5 is the final predictions from Expert MLP (4-dim)
    # We will route at layer 4 and layer 5.
    
    # Centroids at layer 3 (input TF-IDF)
    centroids_l3 = []
    for k in range(K):
        idx = np.where(train_targets == k)[0][:64] # Use 64 samples
        centroids_l3.append(np.mean(X_train[idx], axis=0))
    centroids_l3 = np.array(centroids_l3)
    
    # Centroids at layer 4 (hidden representation 128-dim)
    centroids_l4 = []
    for k in range(K):
        idx = np.where(train_targets == k)[0][:64]
        with torch.no_grad():
            h_list = []
            for expert_idx, expert in enumerate(experts):
                h, _ = expert(torch.FloatTensor(X_train[idx]))
                h_list.append(h.numpy())
            # Concatenate or average across experts to represent the layer-wise hidden state centroids
            avg_h = np.mean(np.array(h_list), axis=0) # shape: 64 x 128
            centroids_l4.append(np.mean(avg_h, axis=0))
    centroids_l4 = np.array(centroids_l4)
    
    # 5. Construct Block-Structured Stream from Test Set
    print("Constructing block-structured test stream...")
    # We want a stream of block_len = 50 samples of the same domain, transitioning suddenly.
    block_len = 50
    stream_texts = []
    stream_targets = []
    stream_features = []
    
    # Group test indices by target domain
    test_idx_by_domain = {k: np.where(test_targets == k)[0] for k in range(K)}
    
    num_blocks = 16 # 16 blocks * 50 samples = 800 samples total
    for b in range(num_blocks):
        domain = b % K
        # Select 50 samples without replacement from this domain
        available_idx = test_idx_by_domain[domain]
        selected_idx = np.random.choice(available_idx, block_len, replace=True)
        for idx in selected_idx:
            stream_targets.append(domain)
            stream_features.append(X_test[idx])
            
    stream_targets = np.array(stream_targets)
    stream_features = np.array(stream_features)
    num_samples = len(stream_targets)
    print(f"Stream constructed. Total samples: {num_samples}. Meta-blocks of length: {block_len}")
    
    # 6. Evaluate Ensembling Methods
    methods = ["Uniform", "SABLE_0.005", "ChemMerge_0.05", "Momentum_Merge_Advanced", "UGR"]
    results = {m: {"acc": [], "jitter": []} for m in methods}
    
    for method in methods:
        accuracies = []
        # We track stateful values layer-by-layer
        # L=3 is input, L=4 is hidden, L=5 is output
        # For UGR, state is on sphere
        if method == "UGR":
            prev_s = np.ones(K) / np.sqrt(K)
            
        all_layer_alphas = [] # track alphas to compute routing jitter
        
        for i in range(num_samples):
            feat = stream_features[i]
            target_domain = stream_targets[i]
            
            # Step 1: Initialize temporal coupling
            if method == "UGR":
                s_state = np.copy(prev_s)
                alpha_l3 = s_state ** 2
            else:
                sample_C = np.ones(K) * 0.25
                alpha_l3 = np.ones(K) * 0.25
                
            sample_alphas = [alpha_l3]
            
            # Layer 4 ensembling (hidden representations)
            # Similarity extraction at layer 3 using centroids_l3
            norm_feat = np.linalg.norm(feat) + 1e-8
            norm_c3 = np.linalg.norm(centroids_l3, axis=1) + 1e-8
            sim3 = np.dot(centroids_l3, feat) / (norm_feat * norm_c3)
            
            # Target weight vector w
            if method == "SABLE_0.005":
                temp = 0.005
            elif method == "ChemMerge_0.05":
                temp = 0.050
            elif method == "Momentum_Merge_Advanced":
                temp = 0.005
            elif method == "UGR":
                temp = 0.010
            else: # Uniform
                temp = 1.0
                
            shifted_sim3 = (sim3 - np.max(sim3)) / temp
            w3 = np.exp(shifted_sim3)
            w3 = w3 / np.sum(w3)
            
            # Update weights at layer 4
            if method == "Uniform":
                alpha_l4 = np.ones(K) * 0.25
            elif method == "SABLE_0.005":
                alpha_l4 = w3
            elif method == "ChemMerge_0.05":
                dt = 1.5
                k_decay = 0.3
                sample_C = np.clip(sample_C + dt * (w3 * (1.0 - sample_C) - k_decay * sample_C), 0.0, 1.0)
                alpha_l4 = sample_C / (np.sum(sample_C) + 1e-8)
            elif method == "Momentum_Merge_Advanced":
                beta = 0.60
                alpha_l4 = (1.0 - beta) * w3 + beta * alpha_l3
            elif method == "UGR":
                eta = 0.80
                w_sphere = w3 / (np.linalg.norm(w3) + 1e-6)
                c_val = np.dot(s_state, w_sphere)
                if np.abs(c_val) < 1.0 - 1e-6:
                    v_orth = w_sphere - c_val * s_state
                    u_vec = v_orth / (np.linalg.norm(v_orth) + 1e-8)
                    theta = eta * np.arccos(np.clip(c_val, -1.0, 1.0))
                    s_state = np.cos(theta) * s_state + np.sin(theta) * u_vec
                alpha_l4 = s_state ** 2
                
            sample_alphas.append(alpha_l4)
            
            # Layer 5 ensembling (outputs)
            # Evaluate each expert to get hidden representation
            with torch.no_grad():
                h_vals = []
                out_vals = []
                for expert in experts:
                    h, out = expert(torch.FloatTensor(feat))
                    h_vals.append(h.numpy())
                    out_vals.append(out.numpy())
                    
                h_vals = np.array(h_vals) # shape: K x 128
                out_vals = np.array(out_vals) # shape: K x 4
                
            # Blended hidden representation using alpha_l4
            blended_h = np.dot(alpha_l4, h_vals) # 128-dim
            
            # Similarity extraction at layer 4 using centroids_l4 and blended_h
            norm_bh = np.linalg.norm(blended_h) + 1e-8
            norm_c4 = np.linalg.norm(centroids_l4, axis=1) + 1e-8
            sim4 = np.dot(centroids_l4, blended_h) / (norm_bh * norm_c4)
            
            shifted_sim4 = (sim4 - np.max(sim4)) / temp
            w4 = np.exp(shifted_sim4)
            w4 = w4 / np.sum(w4)
            
            # Update weights at layer 5
            if method == "Uniform":
                alpha_l5 = np.ones(K) * 0.25
            elif method == "SABLE_0.005":
                alpha_l5 = w4
            elif method == "ChemMerge_0.05":
                sample_C = np.clip(sample_C + dt * (w4 * (1.0 - sample_C) - k_decay * sample_C), 0.0, 1.0)
                alpha_l5 = sample_C / (np.sum(sample_C) + 1e-8)
            elif method == "Momentum_Merge_Advanced":
                alpha_l5 = (1.0 - beta) * w4 + beta * alpha_l4
            elif method == "UGR":
                w_sphere = w4 / (np.linalg.norm(w4) + 1e-6)
                c_val = np.dot(s_state, w_sphere)
                if np.abs(c_val) < 1.0 - 1e-6:
                    v_orth = w_sphere - c_val * s_state
                    u_vec = v_orth / (np.linalg.norm(v_orth) + 1e-8)
                    theta = eta * np.arccos(np.clip(c_val, -1.0, 1.0))
                    s_state = np.cos(theta) * s_state + np.sin(theta) * u_vec
                alpha_l5 = s_state ** 2
                
            sample_alphas.append(alpha_l5)
            all_layer_alphas.append(sample_alphas)
            
            # Blended prediction using alpha_l5
            blended_out = np.dot(alpha_l5, out_vals) # 4-dim
            pred_domain = np.argmax(blended_out)
            accuracies.append(pred_domain == target_domain)
            
            # Store final state for temporal coupling
            if method == "UGR":
                prev_s = np.copy(s_state)
                
        # Compute metrics
        mean_acc = np.mean(accuracies)
        
        # Jitter: mean-squared coordinate difference between consecutive layers (layer 3 to 4, layer 4 to 5)
        jitters = []
        for i in range(num_samples):
            alphas = all_layer_alphas[i] # list of length 3: [alpha_l3, alpha_l4, alpha_l5]
            j34 = np.mean((alphas[1] - alphas[0]) ** 2)
            j45 = np.mean((alphas[2] - alphas[1]) ** 2)
            jitters.append((j34 + j45) / 2.0)
            
        mean_jitter = np.mean(jitters)
        
        results[method]["acc"] = mean_acc
        results[method]["jitter"] = mean_jitter
        print(f"Method {method:25s} | Accuracy: {mean_acc:.2%} | Jitter: {mean_jitter:.6f}")
        
if __name__ == "__main__":
    run_real_world_simulation()
