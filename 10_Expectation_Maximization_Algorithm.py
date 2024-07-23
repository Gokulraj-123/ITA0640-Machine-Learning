import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse
np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Fit a Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)
y_gmm = gmm.predict(X)

# Extract parameters
means = gmm.means_
covariances = gmm.covariances_
weights = gmm.weights_

# Print parameters
print("Means:")
print(means)
print("\nCovariances:")
print(covariances)
print("\nWeights:")
print(weights)

# Plot the results
plt.figure(figsize=(10, 6))

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, s=40, cmap='viridis', label='Data')

# Plot the Gaussian Mixture Models
ax = plt.gca()
for i in range(gmm.n_components):
    mean = means[i]
    cov = covariances[i]
    
    # Draw the ellipse representing the covariance
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180.0 * angle / np.pi
    ell = Ellipse(mean, v[0], v[1], angle=angle, color='red', alpha=0.5)
    ax.add_patch(ell)

plt.title('Gaussian Mixture Model')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
