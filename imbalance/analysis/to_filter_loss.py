import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

losses1 = np.load('../results/cifar10/loss_sequence/losses_imbFactor1.0_seed1.npy')
losses2 = np.load('../results/cifar10/loss_sequence/losses_imbFactor1.0_seed2.npy')
losses3 = np.load('../results/cifar10/loss_sequence/losses_imbFactor1.0_seed3.npy')

losses = (losses1 + losses2 + losses3) / 3

np.save('../results/cifar10/loss_sequence/losses_imbFactor1.0_average3.npy', losses)

np.save('../results/cifar10/loss_sequence/losses_imbFactor1.0_average3_normed.npy', losses/np.amax(losses))
losses2save = []
w_length = 5
poly_order = 3
for loss in losses:
    loss = savgol_filter(loss, w_length, poly_order)
    losses2save.append(loss)
losses2save = np.array(losses2save)
np.save('../results/cifar10/loss_sequence/losses_imbFactor1.0_average3_filtered_w{}_order{}_normed.npy'.format(w_length, poly_order),
        losses2save / np.amax(losses2save))
