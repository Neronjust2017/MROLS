import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def norm_data(a):
    a = np.nan_to_num(a)
    a = a - np.mean(a)
    a = a / np.std(a)
    return a
losses1 = np.load('../results/entropy_corruptionProb0.6_corruptionTypeunif_seed1.npy')
losses2 = np.load('../results/entropy_corruptionProb0.6_corruptionTypeunif_seed2.npy')
losses3 = np.load('../results/entropy_corruptionProb0.6_corruptionTypeunif_seed3.npy')
losses4 = np.load('../results/entropy_corruptionProb0.6_corruptionTypeunif_seed4.npy')
losses5 = np.load('../results/entropy_corruptionProb0.6_corruptionTypeunif_seed5.npy')

losses = (losses1 + losses2 + losses3 + losses4 + losses5) / 5
losses = norm_data(losses)

np.save('../results/entropy_corruptionProb0.6_corruptionTypeunif_average5.npy', losses)

losses2save = []
w_length = 5
poly_order = 3
for loss in losses:
    loss = savgol_filter(loss, w_length, poly_order)
    losses2save.append(loss)
losses2save = np.array(losses2save)
np.save('../results/entropy_corruptionProb0.6_corruptionTypeunif_average5_filtered_w{}_order{}.npy'.format(w_length, poly_order),
        losses2save / np.amax(losses2save))


losses2save = []
w_length = 7
poly_order = 3
for loss in losses:
    loss = savgol_filter(loss, w_length, poly_order)
    losses2save.append(loss)
losses2save = np.array(losses2save)
np.save('../results/losses_corruptionProb0.6_corruptionTypeunif_average5_filtered_w{}_order{}_normed.npy'.format(w_length, poly_order),
        losses2save / np.amax(losses2save))