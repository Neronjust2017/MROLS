import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# # losses = np.load('./imbFactor0.01_loss_filtered_w5_order3.npy')
# # losses = np.load('./loss_filtered_w11_order3.npy')
# # losses = np.load('./losses_corruptionProb0.6_corruptionTypeunif.npy')
# losses = np.load('/data/ecg/challenge2020/saved/loss_43101_100.npy')
# # losses = np.load('/home/lixiaoyu/project/ecg/challenge2020/test/meta-weight-net/results/losses_corruptionProb0.6_corruptionTypeflip2.npy')
# # targets = np.load('/home/lixiaoyu/project/ecg/challenge2020/test/meta-weight-net/results/targets_corruptionProb0.6_corruptionTypeflip2_trialFlip2.npy')
#
#
# losses = losses[42901:, 1:42]
# # losses = savgol_filter(losses, 11, 3, axis=1)
# is_label_noise = False
# for i in range(200):
#     plt.plot(range(len(losses[0])), losses[i], alpha=1)
#     # is_label_noise = targets[i] == 0
#     plt.show()
#     # print(is_label_noise)
#     # plt.savefig('./figs/losses/{}_loss.jpg'.format(str(i)))
# print('done')
#


losses = np.load('../results/losses_corruptionProb0.6_corruptionTypeunif_filtered_w5_order3.npy')
np.save('../results/losses_corruptionProb0.6_corruptionTypeunif_filtered_w5_order3_normed.npy', losses/np.amax(losses))