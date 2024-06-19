import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import csv

path = (r"/Users/olivialiau/Downloads/REUDATA/BA11_60_MM_train.csv")

df = pd.read_csv(path)

df.drop(columns=['Unnamed: 0', 'X'], inplace=True)
df['Sex'] = df['Sex'].replace('M', '1')
df['Sex'] = df['Sex'].replace('F', '0')
#print(df)


# perplexity = np.arange(5, 55, 5)
# y = []
# x = []

# for i in perplexity:
#     x.append(i)
#     model = TSNE(method = 'exact', n_iter=1000, n_components=2, init="pca", perplexity=i, learning_rate = 200, random_state = 42)
#     reduced = model.fit_transform(df)
#     y.append(model.kl_divergence_)
    
# plt.plot(x, y, marker='o', linestyle='-', color='b', label='Line 1')
# print(min(y))
# plt.show()




tsne = TSNE(init = 'pca', n_components=2, perplexity=50, learning_rate=300, n_iter=1000, random_state=42, n_jobs = -1)
#tsne = TSNE(init = 'pca', n_components=2, perplexity=50, learning_rate=200, n_iter=1000, random_state=42, method = 'exact', n_jobs = -1)
tsne_results = tsne.fit_transform(df)
print(tsne.kl_divergence_)

tsne_df = pd.DataFrame(tsne_results, columns=['Dimension 1', 'Dimension 2'])
plt.figure(figsize=(10, 8))
plt.scatter(tsne_df['Dimension 1'], tsne_df['Dimension 2'])
plt.title('t-SNE Results')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()


