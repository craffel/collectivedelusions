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

def run_evaluation(seed=42):
    set_seed(seed)
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
    
    # Test stream
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
    
    # Sweep over UGR hyperparameters
    print("\nSweeping UGR hyperparameters:")
    for temp in [0.005, 0.01, 0.05, 0.1]:
        for eta in [0.1, 0.3, 0.5, 0.7, 0.9]:
            # Run simulation
            accuracies = []
            jitters = []
            prev_s = np.ones(K) / np.sqrt(K)
            
            all_layer_alphas = []
            for i in range(num_samples):
                feat = stream_features[i]
                target_domain = stream_targets[i]
                
                s_state = np.copy(prev_s)
                alpha_l3 = s_state ** 2
                sample_alphas = [alpha_l3]
                
                # L4 routing
                norm_feat = np.linalg.norm(feat) + 1e-8
                norm_c3 = np.linalg.norm(centroids_l3, axis=1) + 1e-8
                sim3 = np.dot(centroids_l3, feat) / (norm_feat * norm_c3)
                shifted_sim3 = (sim3 - np.max(sim3)) / temp
                w3 = np.exp(shifted_sim3)
                w3 = w3 / np.sum(w3)
                
                w_sphere = w3 / (np.linalg.norm(w3) + 1e-6)
                c_val = np.dot(s_state, w_sphere)
                if np.abs(c_val) < 1.0 - 1e-6:
                    v_orth = w_sphere - c_val * s_state
                    u_vec = v_orth / (np.linalg.norm(v_orth) + 1e-8)
                    theta = eta * np.arccos(np.clip(c_val, -1.0, 1.0))
                    s_state = np.cos(theta) * s_state + np.sin(theta) * u_vec
                alpha_l4 = s_state ** 2
                sample_alphas.append(alpha_l4)
                
                # L5 routing
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
                shifted_sim4 = (sim4 - np.max(sim4)) / temp
                w4 = np.exp(shifted_sim4)
                w4 = w4 / np.sum(w4)
                
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
                blended_out = np.dot(alpha_l5, out_vals)
                pred_domain = np.argmax(blended_out)
                accuracies.append(pred_domain == target_domain)
                
                prev_s = np.copy(s_state)
                
            mean_acc = np.mean(accuracies)
            # Compute Jitter
            for i in range(num_samples):
                alphas = all_layer_alphas[i]
                j34 = np.mean((alphas[1] - alphas[0]) ** 2)
                j45 = np.mean((alphas[2] - alphas[1]) ** 2)
                jitters.append((j34 + j45) / 2.0)
            mean_jitter = np.mean(jitters)
            print(f"UGR | temp={temp:.3f} | eta={eta:.2f} | Acc: {mean_acc:.2%} | Jitter: {mean_jitter:.6f}")

if __name__ == "__main__":
    run_evaluation()
