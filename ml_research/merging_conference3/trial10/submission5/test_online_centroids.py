import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

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

def run_online_centroid_test():
    print("Loading 20newsgroups dataset...")
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
    seed = 42
    set_seed(seed)
    
    print("Training expert models...")
    experts = train_experts(X_train, train_targets, K=K, input_dim=1024)
    
    # 1. Standard calibrated centroids
    print("Computing standard calibrated centroids...")
    calibrated_l3 = []
    for k in range(K):
        idx = np.where(train_targets == k)[0][:64]
        calibrated_l3.append(np.mean(X_train[idx], axis=0))
    calibrated_l3 = np.array(calibrated_l3)
    
    calibrated_l4 = []
    for k in range(K):
        idx = np.where(train_targets == k)[0][:64]
        with torch.no_grad():
            h_list = []
            for expert in experts:
                h, _ = expert(torch.FloatTensor(X_train[idx]))
                h_list.append(h.numpy())
            avg_h = np.mean(np.array(h_list), axis=0)
            calibrated_l4.append(np.mean(avg_h, axis=0))
    calibrated_l4 = np.array(calibrated_l4)
    
    # Test stream setup
    block_len = 50
    stream_targets = []
    stream_features = []
    test_idx_by_domain = {k: np.where(test_targets == k)[0] for k in range(K)}
    
    # Let's make a longer stream of 40 blocks (2000 samples) to allow centroids to adapt/converge
    num_blocks = 40
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
    
    # Config for UGR
    cfg = {"temp": 0.05, "eta": 0.10}
    gamma_0 = 0.05 # decay rate for online learning
    
    # Run 1: UGR with Calibrated Centroids (Static)
    print("\nEvaluating UGR with STATIC CALIBRATED centroids...")
    set_seed(seed)
    prev_s = np.ones(K) / np.sqrt(K)
    static_correct = 0
    static_jitters = []
    prev_alpha_l4 = None
    
    for i in range(num_samples):
        feat = stream_features[i]
        target_domain = stream_targets[i]
        
        s_state = np.copy(prev_s)
        alpha_l3 = s_state ** 2
        
        norm_feat = np.linalg.norm(feat) + 1e-8
        norm_c3 = np.linalg.norm(calibrated_l3, axis=1) + 1e-8
        sim3 = np.dot(calibrated_l3, feat) / (norm_feat * norm_c3)
        
        temp = cfg["temp"]
        shifted_sim3 = (sim3 - np.max(sim3)) / temp
        w3 = np.exp(shifted_sim3)
        w3 = w3 / np.sum(w3)
        
        eta = cfg["eta"]
        w_sphere = w3 / (np.linalg.norm(w3) + 1e-6)
        
        c_val = np.dot(s_state, w_sphere)
        if np.abs(c_val) < 1.0 - 1e-6:
            v_orth = w_sphere - c_val * s_state
            u_vec = v_orth / (np.linalg.norm(v_orth) + 1e-8)
            theta = eta * np.arccos(np.clip(c_val, -1.0, 1.0))
            s_state = np.cos(theta) * s_state + np.sin(theta) * u_vec
        alpha_l4 = s_state ** 2
        prev_s = s_state
        
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
        norm_c4 = np.linalg.norm(calibrated_l4, axis=1) + 1e-8
        sim4 = np.dot(calibrated_l4, blended_h) / (norm_bh * norm_c4)
        
        shifted_sim4 = (sim4 - np.max(sim4)) / temp
        w4 = np.exp(shifted_sim4)
        w4 = w4 / np.sum(w4)
        w_sphere4 = w4 / (np.linalg.norm(w4) + 1e-6)
        
        c_val4 = np.dot(s_state, w_sphere4)
        if np.abs(c_val4) < 1.0 - 1e-6:
            v_orth4 = w_sphere4 - c_val4 * s_state
            u_vec4 = v_orth4 / (np.linalg.norm(v_orth4) + 1e-8)
            theta4 = eta * np.arccos(np.clip(c_val4, -1.0, 1.0))
            s_state = np.cos(theta4) * s_state + np.sin(theta4) * u_vec4
        alpha_out = s_state ** 2
        prev_s = s_state
        
        # Classification
        pred_out = np.dot(alpha_out, out_vals)
        pred_domain = np.argmax(pred_out)
        if pred_domain == target_domain:
            static_correct += 1
            
        if prev_alpha_l4 is not None:
            static_jitters.append(np.sum((alpha_l4 - prev_alpha_l4) ** 2))
        prev_alpha_l4 = alpha_l4
        
    static_acc = static_correct / num_samples
    static_jitter = np.mean(static_jitters)
    print(f"Static Calibrated UGR: Accuracy = {static_acc*100:.2f}%, Jitter = {static_jitter:.6f}")
    
    # Run 2: UGR with RANDOM Centroids + ONLINE adaptation
    print("\nEvaluating UGR starting with RANDOM centroids + ONLINE adaptation...")
    set_seed(seed)
    prev_s = np.ones(K) / np.sqrt(K)
    online_correct = 0
    online_last_400_correct = 0
    online_jitters = []
    prev_alpha_l4 = None
    
    # Initialize random centroids
    online_c3 = np.random.normal(0, 1, size=calibrated_l3.shape)
    # Project to make them unit length or keep positive if text
    online_c3 = np.abs(online_c3) # keep non-negative for tf-idf
    online_c4 = np.random.normal(0, 1, size=calibrated_l4.shape)
    
    for i in range(num_samples):
        feat = stream_features[i]
        target_domain = stream_targets[i]
        
        s_state = np.copy(prev_s)
        alpha_l3 = s_state ** 2
        
        norm_feat = np.linalg.norm(feat) + 1e-8
        norm_c3 = np.linalg.norm(online_c3, axis=1) + 1e-8
        sim3 = np.dot(online_c3, feat) / (norm_feat * norm_c3)
        
        temp = cfg["temp"]
        shifted_sim3 = (sim3 - np.max(sim3)) / temp
        w3 = np.exp(shifted_sim3)
        w3 = w3 / np.sum(w3)
        
        eta = cfg["eta"]
        w_sphere = w3 / (np.linalg.norm(w3) + 1e-6)
        
        c_val = np.dot(s_state, w_sphere)
        if np.abs(c_val) < 1.0 - 1e-6:
            v_orth = w_sphere - c_val * s_state
            u_vec = v_orth / (np.linalg.norm(v_orth) + 1e-8)
            theta = eta * np.arccos(np.clip(c_val, -1.0, 1.0))
            s_state = np.cos(theta) * s_state + np.sin(theta) * u_vec
        alpha_l4 = s_state ** 2
        prev_s = s_state
        
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
        norm_c4 = np.linalg.norm(online_c4, axis=1) + 1e-8
        sim4 = np.dot(online_c4, blended_h) / (norm_bh * norm_c4)
        
        shifted_sim4 = (sim4 - np.max(sim4)) / temp
        w4 = np.exp(shifted_sim4)
        w4 = w4 / np.sum(w4)
        w_sphere4 = w4 / (np.linalg.norm(w4) + 1e-6)
        
        c_val4 = np.dot(s_state, w_sphere4)
        if np.abs(c_val4) < 1.0 - 1e-6:
            v_orth4 = w_sphere4 - c_val4 * s_state
            u_vec4 = v_orth4 / (np.linalg.norm(v_orth4) + 1e-8)
            theta4 = eta * np.arccos(np.clip(c_val4, -1.0, 1.0))
            s_state = np.cos(theta4) * s_state + np.sin(theta4) * u_vec4
        alpha_out = s_state ** 2
        prev_s = s_state
        
    # Classification
        pred_out = np.dot(alpha_out, out_vals)
        pred_domain = np.argmax(pred_out)
        if pred_domain == target_domain:
            online_correct += 1
            if i >= num_samples - 400:
                online_last_400_correct += 1
            
        if prev_alpha_l4 is not None:
            online_jitters.append(np.sum((alpha_l4 - prev_alpha_l4) ** 2))
        prev_alpha_l4 = alpha_l4
        
        # --- ONLINE CENTROID ADAPTATION ---
        # Update online_c3 (using routing coefficients alpha_l4) and online_c4 (using alpha_out)
        for k in range(K):
            gamma_k_l3 = gamma_0 * alpha_l4[k]
            online_c3[k] = (1.0 - gamma_k_l3) * online_c3[k] + gamma_k_l3 * feat
            
            gamma_k_l4 = gamma_0 * alpha_out[k]
            online_c4[k] = (1.0 - gamma_k_l4) * online_c4[k] + gamma_k_l4 * blended_h
            
    online_acc = online_correct / num_samples
    online_jitter = np.mean(online_jitters)
    print(f"Online Adapted UGR (from RANDOM): Accuracy = {online_acc*100:.2f}%, Jitter = {online_jitter:.6f}")
    print(f"Online Adapted UGR (from RANDOM) Final 400 samples: Accuracy = {online_last_400_correct/400*100:.2f}%")
    
    # Compute Cosine Similarity between adapted centroids and true calibrated centroids
    print("\nAnalyzing Centroid Alignment after online adaptation...")
    for k in range(K):
        # Layer 3 similarity
        cos_sim3 = np.dot(online_c3[k], calibrated_l3[k]) / (np.linalg.norm(online_c3[k]) * np.linalg.norm(calibrated_l3[k]) + 1e-8)
        # Layer 4 similarity
        cos_sim4 = np.dot(online_c4[k], calibrated_l4[k]) / (np.linalg.norm(online_c4[k]) * np.linalg.norm(calibrated_l4[k]) + 1e-8)
        print(f"Expert {k}: Layer 3 Centroid Cosine Similarity = {cos_sim3:.4f}, Layer 4 Centroid Cosine Similarity = {cos_sim4:.4f}")

if __name__ == "__main__":
    run_online_centroid_test()
