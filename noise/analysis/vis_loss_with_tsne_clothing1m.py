import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.signal import savgol_filter

losses = np.load('/home/lixiaoyu/project/ecg/challenge2020/test/MLNT/losses_resnet50_baseline_nopretrain.npy')
losses = losses[0:1000000, 1:11]

X_tsne = TSNE(n_components=2, random_state=33).fit_transform(losses)
plt.figure(figsize=(10, 10))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=4, label="t-SNE-clothing1m")
plt.legend()
# plt.savefig('../figs/t-SNE-uniform-corrupProb0.6-s4-w5-o3.png')
plt.show()



print('done')