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
df_filtered = df.drop(columns=['Age', 'TOD_pos'])
# print(df_filtered)

pca = PCA(n_components=50)
pca_features = pca.fit_transform(df_filtered)

def calculate_residual_variance(original_data, embedded_data):
    high_dim_distances = pdist(original_data, metric='euclidean')
    low_dim_distances = pdist(embedded_data, metric='euclidean')
    residuals = high_dim_distances - low_dim_distances
    residual_variance = np.sum(residuals**2) / np.sum(high_dim_distances**2)
    return residual_variance


# Define the ranges for parameters
perplexities = [10, 15, 20, 25, 30, 35, 40, 50]
learning_rates = [200, 300, 400, 500, 600, 700, 800, 1000]
n_iters = [1000, 2000, 3000]

tsne_result = []
best_parameters = []

for run in range(5):
    best_residual_variance = float('inf')
    best_params = {}
    
    for perplexity in perplexities:
        for learning_rate in learning_rates:
            for iter in n_iters:
                tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=iter, random_state=None)
                tsne_result = tsne.fit_transform(pca_features)
                
                residual_variance = calculate_residual_variance(df_filtered, tsne_result)
                
                if residual_variance < best_residual_variance:
                    best_residual_variance = residual_variance
                    best_parameters = {
                        'perplexity': perplexity,
                        'learning_rate': learning_rate,
                        'n_iters' : iter
                    }
                    tsne_result = tsne.fit_transform(pca_features)

    
print("Best parameters from each run:", best_parameters)
print(f"Residual Variance: {best_residual_variance:.4f}")

plt.figure(figsize=(10, 8))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
plt.title('Averaged t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()



# tsne_results = []

# for i in range(5):
#     tsne = TSNE(**best_parameters, random_state=None, init = 'pca', method = 'exact', n_jobs = -1)
#     tsne_result = tsne.fit_transform(pca_features)
#     tsne_results.append(tsne_result)

# tsne_results_array = np.array(tsne_results)
# average_tsne_result = np.mean(tsne_results_array, axis=0)

# final_residual_variance = calculate_residual_variance(df_filtered, tsne_result)
# print(f"Residual Variance: {final_residual_variance:.4f}")

# plt.figure(figsize=(10, 8))
# plt.scatter(average_tsne_result[:, 0], average_tsne_result[:, 1])
# plt.title('Averaged t-SNE Visualization')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
# plt.show()




# tsne = TSNE(**best_params, random_state=42, init = 'pca', method = 'exact', n_jobs = -1)
# best_tsne_results = tsne.fit_transform(pca_features)
# plt.scatter(best_tsne_results[:, 0], best_tsne_results[:, 1])
# plt.title('Best t-SNE Embedding')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.show()

# finaldf = pd.DataFrame(best_tsne_results)
# finaldf['TOD'] = df['TOD_pos']
# finaldf.to_csv('output_file.csv', index=False)

