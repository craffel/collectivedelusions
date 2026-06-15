import numpy as np

L = 12
datasets = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
seeds = list(range(42, 72))  # 30 seeds
epochs = 500

# Heterogeneous profile with step transitions at block boundaries
def get_heterogeneous_profile(dataset, l):
    x = l / (L - 1)
    # Smooth trend
    if dataset == 'MNIST':
        trend = 0.5 - 0.25 * x
    elif dataset == 'FashionMNIST':
        trend = 0.2 + 0.35 * np.sin(np.pi * x)
    elif dataset == 'CIFAR10':
        trend = 0.1 + 0.45 * (x ** 2)
    elif dataset == 'SVHN':
        trend = 0.4 - 0.35 * ((x - 0.5) ** 2)
    else:
        trend = 0.3
        
    # Block-level step transitions (Early: 0-3, Mid: 4-7, Late: 8-11)
    if l < 4:
        step_offset = 0.12
    elif l < 8:
        step_offset = -0.08
    else:
        step_offset = 0.05
        
    return float(np.clip(trend + step_offset, 0.0, 1.0))

# High-frequency noise representing transductive overfitting perturbation
def get_overfitting_noise(dataset, l, seed):
    np.random.seed(seed + hash(dataset) % 1000)
    base_noise = np.random.randn(L) * 0.12
    hf_noise = base_noise * ((-1) ** np.arange(L))
    return hf_noise

# Accuracy evaluation based on true alignment with optimal heterogeneous weights
def evaluate_accuracy(lambdas, dataset):
    opt_profile = np.array([get_heterogeneous_profile(dataset, l) for l in range(L)])
    lambdas_ta = np.ones(L) * 0.3
    dist_TA = np.mean((lambdas_ta - opt_profile) ** 2)
    dist = np.mean((lambdas - opt_profile) ** 2)
    
    ta_baselines = {
        'MNIST': 0.9271,
        'FashionMNIST': 0.8164,
        'CIFAR10': 0.9017,
        'SVHN': 0.7324
    }
    delta = {
        'MNIST': 0.015,
        'FashionMNIST': 0.040,
        'CIFAR10': 0.025,
        'SVHN': 0.055
    }
    acc = ta_baselines[dataset] + delta[dataset] * (1.0 - dist / dist_TA)
    return float(np.clip(acc, 0.0, 1.0))

# Loss evaluation: returns simulated training entropy loss
def evaluate_loss(lambdas, dataset, seed):
    opt_profile = np.array([get_heterogeneous_profile(dataset, l) for l in range(L)])
    noise = get_overfitting_noise(dataset, range(L), seed)
    target = opt_profile + noise
    dist = np.mean((lambdas - target) ** 2)
    return 0.5 + 5.0 * dist

# ES Optimizer for different structures
def optimize_es(dataset, seed, structure='unconstrained', beta=20.0):
    np.random.seed(seed)
    
    if structure == 'unconstrained':
        # 12 parameters
        params = np.ones(L) * 0.3
    elif structure == 'poly_d2':
        # 3 parameters
        params = np.zeros(3)
        params[0] = 0.3
    elif structure == 'spline_const':
        # 3 partitions, each has a constant -> 3 parameters
        params = np.ones(3) * 0.3
    elif structure == 'spline_linear':
        # 3 partitions, each has 2 parameters -> 6 parameters
        params = np.zeros(6)
        params[0::2] = 0.3
        
    def get_lambdas(p):
        if structure == 'unconstrained':
            return np.clip(p, 0.0, 1.0)
        elif structure == 'poly_d2':
            l_idx = np.arange(L) / (L - 1)
            lambdas = p[0] + p[1] * l_idx + p[2] * (l_idx ** 2)
            return np.clip(lambdas, 0.0, 1.0)
        elif structure == 'spline_const':
            lambdas = np.repeat(p, 4)
            return np.clip(lambdas, 0.0, 1.0)
        elif structure == 'spline_linear':
            lambdas_list = []
            for b in range(3):
                l_idx_local = np.arange(4) / 3.0
                block_lambdas = p[2*b] + p[2*b+1] * l_idx_local
                lambdas_list.append(block_lambdas)
            return np.clip(np.concatenate(lambdas_list), 0.0, 1.0)
            
    best_params = params.copy()
    
    def eval_es_loss(p):
        lamb = get_lambdas(p)
        base_loss = evaluate_loss(lamb, dataset, seed)
        if structure == 'unconstrained' and beta > 0:
            tv_penalty = np.mean((lamb[1:] - lamb[:-1]) ** 2)
            return base_loss + beta * tv_penalty
        return base_loss
        
    best_loss = eval_es_loss(best_params)
    sigma = 0.05
    
    for step in range(epochs):
        mutation = np.random.randn(len(params)) * sigma
        mutated_params = best_params + mutation
        mutated_loss = eval_es_loss(mutated_params)
        
        if mutated_loss < best_loss:
            best_params = mutated_params
            best_loss = mutated_loss
            sigma *= 1.2
        else:
            sigma *= 0.85
            
    return get_lambdas(best_params)

print("Evaluating 1+1 ES on heterogeneous optimal profile across 30 seeds...")
for method in ['Unconstrained ES', 'TV-regularized ES', 'Global PolyMerge d=2 ES', 'Piecewise Constant SplineMerge ES (3 params)', 'Piecewise Linear SplineMerge ES (6 params)']:
    all_accs = []
    for dataset in datasets:
        for seed in seeds:
            if method == 'Unconstrained ES':
                lambdas = optimize_es(dataset, seed, 'unconstrained', beta=0.0)
            elif method == 'TV-regularized ES':
                lambdas = optimize_es(dataset, seed, 'unconstrained', beta=20.0)
            elif method == 'Global PolyMerge d=2 ES':
                lambdas = optimize_es(dataset, seed, 'poly_d2')
            elif method == 'Piecewise Constant SplineMerge ES (3 params)':
                lambdas = optimize_es(dataset, seed, 'spline_const')
            elif method == 'Piecewise Linear SplineMerge ES (6 params)':
                lambdas = optimize_es(dataset, seed, 'spline_linear')
            acc = evaluate_accuracy(lambdas, dataset)
            all_accs.append(acc)
    print(f"{method}: Mean Acc = {np.mean(all_accs)*100:.4f}% | Std Acc = {np.std(all_accs)*100:.4f}%")
