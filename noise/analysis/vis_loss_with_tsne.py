import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.signal import savgol_filter

# cifar-10
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.tile(a, (4900, 1))
c = b.T
true_label = np.reshape(c, (-1,))

# cifar-100
# a = np.array([i for i in range(100)])
# b = np.tile(a, (490, 1))
# c = b.T
# true_label = np.reshape(c, (-1,))

losses = ['../results/cifar10/loss_sequence/losses_corruptionProb0.2_corruptionTypeflip2_filtered_w5_order3_normed.npy',
          '../results/cifar10/loss_sequence/losses_corruptionProb0.4_corruptionTypeflip2_filtered_w5_order3_normed.npy',
          '../results/cifar10/loss_sequence/losses_corruptionProb0.4_corruptionTypeunif_filtered_w5_order3_normed.npy',
          '../results/cifar10/loss_sequence/losses_corruptionProb0.6_corruptionTypeunif_filtered_w5_order3_normed.npy']

targets = ['../results/cifar10/train_labels/targets_corruptionProb0.2_corruptionTypeflip2.npy',
           '../results/cifar10/train_labels/targets_corruptionProb0.4_corruptionTypeflip2.npy',
           '../results/cifar10/train_labels/targets_corruptionProb0.4_corruptionTypeunif.npy',
           '../results/cifar10/train_labels/targets_corruptionProb0.6_corruptionTypeunif.npy']

titles = ['20% Asymmetric Noise',
          '40% Asymmetric Noise',
          '40% Symmetric Noise',
          '60% Symmetric Noise',
          ]

for l, t, title in zip(losses, targets, titles):
    loss = np.load(l)
    target = np.load(t)

    print(2)
    label_of_noise = np.zeros(len(target))
    tmp = target - true_label
    false_indexes = np.where(tmp != 0)
    label_of_noise[false_indexes] = 1


    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(loss)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label_of_noise, s=4, label=title)
    plt.legend(markerscale=2)
    plt.savefig(title + '.png')

    print('done')