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
    categories = [
        'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
        'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
        'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
        'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc', 'alt.atheism', 'soc.religion.christian'
    ]
    train_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    test_data = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))
    
    cat_to_domain = {}
    for cat in train_data.target_names:
        if cat.startswith('comp'):
            cat_to_domain[cat] = 0
        elif cat.startswith('rec'):
            cat_to_domain[cat] = 1
        elif cat.startswith('sci'):
            cat_to_domain[cat] = 2
        else:
            cat_to_domain[cat] = 3
            
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
    experts = []
    for k in range(K):
        idx_k = np.where(y_train == k)[0]
        idx_others = np.where(y_train != k)[0]
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
        for epoch in range(5):
            optimizer.zero_grad()
            _, outputs = model(X_biased)
            loss = criterion(outputs, y_biased)
            loss.backward()
            optimizer.step()
        model.eval()
        experts.append(model)
    return experts

def run_benchmark():
    print("Initializing benchmark...")
    train_texts, train_targets, test_texts, test_targets = get_20newsgroups_data()
    train_indices = [i for i, text in enumerate(train_texts) if len(text.strip()) > 10]
    test_indices = [i for i, text in enumerate(test_texts) if len(text.strip()) > 10]
    train_texts = [train_texts[i] for i in train_indices]
    train_targets = train_targets[train_indices]
    test_texts = [test_texts[i] for i in test_indices]
    test_targets = test_targets[test_indices]
    
    vectorizer = TfidfVectorizer(max_features=1024, stop_words='english')
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()
    
    K = 4
    set_seed(42)
    experts = train_experts(X_train, train_targets, K=K, input_dim=1024)
    
    # Calibrate centroids
    centroids_l3 = []
    for k in range(K):
        idx = np.where(train_targets == k)[0][:64]
        centroids_l3.append(np.mean(X_train[idx], axis=0))
    centroids_l3 = np.array(centroids_l3)
    
    centroids_l4 = []
    for k in range(K):
        idx = np.where(train_targets == k)[0][:64]
        with torch.no_grad():
            h_list = []
            for expert in experts:
                h, _ = expert(torch.FloatTensor(X_train[idx]))
                h_list.append(h.numpy())
            avg_h = np.mean(np.array(h_list), axis=0)
            centroids_l4.append(np.mean(avg_h, axis=0))
    centroids_l4 = np.array(centroids_l4)
    
    # Test stream (800 samples)
    block_len = 50
    stream_targets = []
    stream_features = []
    test_idx_by_domain = {k: np.where(test_targets == k)[0] for k in range(K)}
    
    num_blocks = 16
    for b in range(num_blocks):
        domain = b % K
        available_idx = test_idx_by_domain[domain]
        selected_idx = np.random.choice(available_idx, block_len, replace=True)
        for idx in selected_idx:
            stream_targets.append(domain)
            stream_features.append(X_test[idx])
            
    stream_targets = np.array(stream_targets)
    stream_features = np.array(stream_features)
    num_samples = len(stream_targets)
    
    methods = [
        "Uniform",
        "SABLE",
        "ChemMerge",
        "ChemMerge (Coupled)",
        "Momentum-Merge",
        "Momentum-Merge (Coupled)",
        "UGR",
        "UGR_Hybrid_Reset",
        "UGR_Softmax_Free"
    ]
    
    configs = {
        "Uniform": {},
        "SABLE": {"temp": 0.05},
        "ChemMerge": {"temp": 0.05, "dt": 1.5, "k_decay": 0.3},
        "ChemMerge (Coupled)": {"temp": 0.05, "dt": 1.5, "k_decay": 0.3},
        "Momentum-Merge": {"temp": 0.05, "beta": 0.80},
        "Momentum-Merge (Coupled)": {"temp": 0.05, "beta": 0.80},
        "UGR": {"temp": 0.05, "eta": 0.10},
        "UGR_Hybrid_Reset": {"temp": 0.05, "eta": 0.10},
        "UGR_Softmax_Free": {"eta": 0.10}
    }
    
    num_runs = 10
    timings = {m: [] for m in methods}
    
    print(f"Running benchmarks over {num_runs} repeated passes of the {num_samples}-sample stream...")
    
    # Warmup
    for method in methods:
        _ = run_stream(method, stream_features, stream_targets, experts, centroids_l3, centroids_l4, configs[method], K)
        
    # Timed runs
    for run in range(num_runs):
        print(f"Pass {run+1}/{num_runs}...")
        for method in methods:
            t0 = time.perf_counter()
            _ = run_stream(method, stream_features, stream_targets, experts, centroids_l3, centroids_l4, configs[method], K)
            t1 = time.perf_counter()
            timings[method].append(t1 - t0)
            
    print("\n" + "="*80)
    print("REAL-WORLD SERVER ROUTING LATENCY & THROUGHPUT BENCHMARK")
    print("="*80)
    print("| Method | Total Stream Time (s) | Latency per Query (ms) | Throughput (Queries/s) |")
    print("| :--- | :---: | :---: | :---: |")
    for method in methods:
        times = np.array(timings[method])
        avg_time = np.mean(times)
        std_time = np.std(times)
        latency_ms = (avg_time / num_samples) * 1000.0
        latency_std = (std_time / num_samples) * 1000.0
        throughput = num_samples / avg_time
        throughput_std = (std_time / (avg_time ** 2)) * num_samples
        print(f"| **{method}** | {avg_time:.4f} ± {std_time:.4f}s | {latency_ms:.3f} ± {latency_std:.3f} ms | {throughput:.1f} ± {throughput_std:.1f} QPS |")
    print("="*80)

def run_stream(method, stream_features, stream_targets, experts, centroids_l3, centroids_l4, cfg, K):
    num_samples = len(stream_targets)
    accuracies = []
    
    if method == "UGR":
        prev_s = np.ones(K) / np.sqrt(K)
    elif method == "UGR_Hybrid_Reset":
        prev_s_hr = np.ones(K) / np.sqrt(K)
    elif method == "UGR_Softmax_Free":
        prev_s_sf = np.ones(K) / np.sqrt(K)
    elif "Coupled" in method:
        if "ChemMerge" in method:
            prev_C = np.ones(K) * 0.25
        elif "Momentum-Merge" in method:
            prev_alpha_out = np.ones(K) * 0.25
            
    for i in range(num_samples):
        feat = stream_features[i]
        target_domain = stream_targets[i]
        
        # Layer 3 init and state load
        if method == "UGR":
            s_state = np.copy(prev_s)
            alpha_l3 = s_state ** 2
        elif method == "UGR_Hybrid_Reset":
            s_state = np.copy(prev_s_hr)
            alpha_l3 = s_state ** 2
        elif method == "UGR_Softmax_Free":
            s_state = np.copy(prev_s_sf)
            alpha_l3 = s_state ** 2
        elif method == "Momentum-Merge (Coupled)":
            alpha_l3 = np.copy(prev_alpha_out)
        elif method == "ChemMerge (Coupled)":
            sample_C = np.copy(prev_C)
            alpha_l3 = sample_C / (np.sum(sample_C) + 1e-8)
        else:
            alpha_l3 = np.ones(K) * 0.25
            if "ChemMerge" in method:
                sample_C = np.ones(K) * 0.25
                
        # Layer 4 ensembling
        norm_feat = np.linalg.norm(feat) + 1e-8
        norm_c3 = np.linalg.norm(centroids_l3, axis=1) + 1e-8
        sim3 = np.dot(centroids_l3, feat) / (norm_feat * norm_c3)
        
        if method != "Uniform":
            if method == "UGR_Softmax_Free":
                relu_sim3 = np.maximum(sim3, 0.0)
                sum_relu3 = np.sum(relu_sim3)
                if sum_relu3 > 1e-8:
                    w3 = relu_sim3 / sum_relu3
                else:
                    w3 = np.ones(K) / K
            else:
                temp = cfg["temp"]
                shifted_sim3 = (sim3 - np.max(sim3)) / temp
                w3 = np.exp(shifted_sim3)
                w3 = w3 / np.sum(w3)
            
        if method == "Uniform":
            alpha_l4 = np.ones(K) * 0.25
        elif method == "SABLE":
            alpha_l4 = w3
        elif "ChemMerge" in method:
            dt = cfg["dt"]
            k_decay = cfg["k_decay"]
            sample_C = np.clip(sample_C + dt * (w3 * (1.0 - sample_C) - k_decay * sample_C), 0.0, 1.0)
            alpha_l4 = sample_C / (np.sum(sample_C) + 1e-8)
        elif "Momentum-Merge" in method:
            beta = cfg["beta"]
            alpha_l4 = (1.0 - beta) * w3 + beta * alpha_l3
        elif method == "UGR" or method == "UGR_Hybrid_Reset" or method == "UGR_Softmax_Free":
            eta = cfg["eta"]
            w_sphere = w3 / (np.linalg.norm(w3) + 1e-6)
            
            if method == "UGR_Hybrid_Reset":
                c_val3 = np.dot(s_state, w_sphere)
                if c_val3 < 0.50:
                    s_state = np.ones(K) / np.sqrt(K)
                    alpha_l3 = s_state ** 2
                    
            c_val = np.dot(s_state, w_sphere)
            if np.abs(c_val) < 1.0 - 1e-6:
                v_orth = w_sphere - c_val * s_state
                u_vec = v_orth / (np.linalg.norm(v_orth) + 1e-8)
                theta = eta * np.arccos(np.clip(c_val, -1.0, 1.0))
                s_state = np.cos(theta) * s_state + np.sin(theta) * u_vec
            alpha_l4 = s_state ** 2
            
        # Layer 5 ensembling
        with torch.no_grad():
            h_vals = []
            out_vals = []
            for expert in experts:
                h, out = expert(torch.FloatTensor(feat))
                h_vals.append(h.numpy())
                out_vals.append(out.numpy())
            h_vals = np.array(h_vals)
            out_vals = np.array(out_vals)
            
        blended_h = np.dot(alpha_l4, h_vals)
        norm_bh = np.linalg.norm(blended_h) + 1e-8
        norm_c4 = np.linalg.norm(centroids_l4, axis=1) + 1e-8
        sim4 = np.dot(centroids_l4, blended_h) / (norm_bh * norm_c4)
        
        if method != "Uniform":
            if method == "UGR_Softmax_Free":
                relu_sim4 = np.maximum(sim4, 0.0)
                sum_relu4 = np.sum(relu_sim4)
                if sum_relu4 > 1e-8:
                    w4 = relu_sim4 / sum_relu4
                else:
                    w4 = np.ones(K) / K
            else:
                shifted_sim4 = (sim4 - np.max(sim4)) / temp
                w4 = np.exp(shifted_sim4)
                w4 = w4 / np.sum(w4)
            
        if method == "Uniform":
            alpha_l5 = np.ones(K) * 0.25
        elif method == "SABLE":
            alpha_l5 = w4
        elif "ChemMerge" in method:
            sample_C = np.clip(sample_C + dt * (w4 * (1.0 - sample_C) - k_decay * sample_C), 0.0, 1.0)
            alpha_l5 = sample_C / (np.sum(sample_C) + 1e-8)
        elif "Momentum-Merge" in method:
            alpha_l5 = (1.0 - beta) * w4 + beta * alpha_l4
        elif method == "UGR" or method == "UGR_Hybrid_Reset" or method == "UGR_Softmax_Free":
            w_sphere = w4 / (np.linalg.norm(w4) + 1e-6)
            c_val = np.dot(s_state, w_sphere)
            if np.abs(c_val) < 1.0 - 1e-6:
                v_orth = w_sphere - c_val * s_state
                u_vec = v_orth / (np.linalg.norm(v_orth) + 1e-8)
                theta = eta * np.arccos(np.clip(c_val, -1.0, 1.0))
                s_state = np.cos(theta) * s_state + np.sin(theta) * u_vec
            alpha_l5 = s_state ** 2
            
        blended_out = np.dot(alpha_l5, out_vals)
        pred_domain = np.argmax(blended_out)
        accuracies.append(pred_domain == target_domain)
        
        # State save for next query
        if method == "UGR":
            prev_s = np.copy(s_state)
        elif method == "UGR_Hybrid_Reset":
            prev_s_hr = np.copy(s_state)
        elif method == "UGR_Softmax_Free":
            prev_s_sf = np.copy(s_state)
        elif method == "Momentum-Merge (Coupled)":
            prev_alpha_out = np.copy(alpha_l5)
        elif method == "ChemMerge (Coupled)":
            prev_C = np.copy(sample_C)
            
    return np.mean(accuracies)

if __name__ == "__main__":
    run_benchmark()
