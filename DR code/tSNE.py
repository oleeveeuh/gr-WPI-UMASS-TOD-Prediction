import pandas as pd
import numpy as np
from sklearn.preprocessing import  MinMaxScaler
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist
import itertools
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



path = (r"/Users/olivialiau/Downloads/REUDATA/BA11_60_MM_train.csv")
df = pd.read_csv(path)

scaler = MinMaxScaler()
df['age_normalized'] = scaler.fit_transform(df[['Age']])
df.drop(columns=['Age'], inplace=True)
# print(df)

pca = PCA(n_components=50)
pca_features = pca.fit_transform(df)

perplexities = [10, 15, 20, 25, 30, 35]
learning_rates = [200, 300, 400, 500, 600, 800, 1000]
n_iters = [1000, 2000, 3000]
n_components = [2, 3, 4, 5]


def calculate_residual_variance(original_data, embedded_data):
    high_dim_distances = pdist(original_data, metric='euclidean')
    low_dim_distances = pdist(embedded_data, metric='euclidean')
    residuals = high_dim_distances - low_dim_distances
    residual_variance = np.sum(residuals**2) / np.sum(high_dim_distances**2)
    return residual_variance

best_params = None
best_residual_variance = float('inf')

for n_components, perplexity, learning_rate, n_iter in itertools.product(n_components, perplexities, learning_rates, n_iters):
    tsne = TSNE(init = 'pca', n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, method = 'exact', random_state=42, n_jobs = -1)
    tsne_results = tsne.fit_transform(pca_features)
    residual_variance = calculate_residual_variance(df, tsne_results)
    
    if residual_variance < best_residual_variance:
        best_residual_variance = residual_variance
        best_params = {
            'perplexity': perplexity,
            'learning_rate': learning_rate,
            'n_iter': n_iter,
            'n_components' : n_components
        }

print(f'Best Parameters: {best_params}')
print(f'Best Residual Variance: {best_residual_variance:.4f}')


tsne = TSNE(**best_params, random_state=42, init = 'pca', method = 'exact', n_jobs = -1)
best_tsne_results = tsne.fit_transform(pca_features)
plt.scatter(best_tsne_results[:, 0], best_tsne_results[:, 1])
plt.title('Best t-SNE Embedding')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
