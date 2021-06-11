import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.signal import savgol_filter

a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.tile(a, (4900, 1))
c = b.T
true_label = np.reshape(c, (-1,))

# losses = np.load('./imbFactor0.01_loss_filtered_w5_order3.npy')
# losses = np.load('./loss_filtered_w11_order3.npy')
# losses = np.load('./losses_corruptionProb0.6_corruptionTypeunif.npy')
# losses = np.load('/home/lixiaoyu/project/ecg/challenge2020/test/meta-weight-net/results/losses_corruptionProb0.6_corruptionTypeunif_filtered_w25_order3.npy')
losses = np.load('/home/lixiaoyu/project/ecg/challenge2020/test/meta-weight-net/results/losses_corruptionProb0.6_corruptionTypeunif_filtered_w5_order3.npy')
targets = np.load('/home/lixiaoyu/project/ecg/challenge2020/test/meta-weight-net/results/targets_corruptionProb0.6_corruptionTypeunif.npy')

label_of_noise = np.zeros(len(targets))
tmp = targets - true_label
false_indexes = np.where(tmp != 0)
label_of_noise[false_indexes] = 1

# losses = savgol_filter(losses, 11, 3, axis=1)

X_tsne = TSNE(n_components=2, random_state=33).fit_transform(losses)
plt.figure(figsize=(10, 10))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label_of_noise, s=4, label="t-SNE-uniform-corrupProb0.6-w5-o3")
plt.legend()
plt.savefig('t-SNE-uniform-corrupProb0.6-s4-w5-o3.png')
# plt.show()



print('done')