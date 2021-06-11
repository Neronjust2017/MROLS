import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.signal import savgol_filter

a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.tile(a, (4900, 1))
c = b.T
true_label = np.reshape(c, (-1,))

losses = ['../results/cifar10/loss_sequence/losses_imbFactor0.005_average3_filtered_w5_order3_normed.npy',
          '../results/cifar10/loss_sequence/losses_imbFactor0.2_average3_filtered_w5_order3_normed.npy',
          '../results/cifar10/loss_sequence/losses_imbFactor1.0_average3_filtered_w5_order3_normed.npy']

targets = ['../results/cifar10/loss_sequence/targets_imbFactor0.005_seed1.npy',
           '../results/cifar10/loss_sequence/targets_imbFactor0.2_seed1.npy',
           '../results/cifar10/loss_sequence/targets_imbFactor1.0_seed1.npy']

titles = ['T-SNE imbFactor0.005',
          'T-SNE imbFactor0.2',
          'T-SNE imbFactor1.0']

for l, t, title in zip(losses, targets, titles):
    loss = np.load(l)
    target = np.load(t)

    # label_of_noise = np.zeros(len(target))
    # tmp = target - true_label
    # false_indexes = np.where(tmp != 0)
    # label_of_noise[false_indexes] = 1

    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(loss)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=target, s=4, label=title)
    plt.legend()
    plt.savefig(title + '.png')

    print('done')