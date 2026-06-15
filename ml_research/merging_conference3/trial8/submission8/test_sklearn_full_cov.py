import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

X = np.random.randn(10, 4)
gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(X)

print("covariances_ shape:", gmm.covariances_.shape)
print("precisions_cholesky_ shape:", gmm.precisions_cholesky_.shape)

# Let's see if _compute_precision_cholesky is available and how it works
try:
    prec = _compute_precision_cholesky(gmm.covariances_, 'full')
    print("Computed precisions_cholesky_ successfully!")
except Exception as e:
    print("Error computing:", e)
