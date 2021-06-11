import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

is_to_stack_losses = False
is_to_average_losses = True
if is_to_average_losses == True:
    # losses = np.load('./losses_imbFactor0.01.npy')
    # losses = savgol_filter(losses, 11, 3, axis=1)
    # losses = np.load('/home/lixiaoyu/project/ecg/challenge2020/test/Meta-weight-net_class-imbalance/losses_imbFactor0.02.npy')

    losses1 = np.load('./results/cifar10/loss_sequence/losses_imbFactor0.02_seed1.npy')
    losses2 = np.load('./results/cifar10/loss_sequence/losses_imbFactor0.02_seed2.npy')
    losses3 = np.load('./results/cifar10/loss_sequence/losses_imbFactor0.02_seed3.npy')

    losses = (losses1 + losses2 + losses3 ) / 3
    losses = losses - np.mean(losses)
    losses = losses / np.std(losses)

    np.save('./results/cifar10/loss_sequence/losses_imbFactor0.02_average3.npy', losses)

    losses2save = []
    w_length = 5
    poly_order = 3
    for loss in losses:
        loss = savgol_filter(loss, w_length, poly_order)
        losses2save.append(loss)
    losses2save = np.array(losses2save)
    np.save('./results/cifar10/loss_sequence/losses_imbFactor0.02_average3_filtered_w{}_order{}.npy'.format(w_length, poly_order),
            losses2save / np.amax(losses2save))


    losses2save = []
    w_length = 7
    poly_order = 3
    for loss in losses:
        loss = savgol_filter(loss, w_length, poly_order)
        losses2save.append(loss)
    losses2save = np.array(losses2save)
    np.save('./results/cifar10/loss_sequence/losses_imbFactor0.02_average3_filtered_w{}_order{}.npy'.format(w_length, poly_order),
            losses2save / np.amax(losses2save))


if is_to_stack_losses == True:
    def norm_data(a):
        a = a - np.mean(a)
        a = a / np.std(a)
        a = np.expand_dims(a, 1)
        return a


    losses1 = np.load('./losses_imbFactor0.02_network333_seed1.npy')
    losses1 = norm_data(losses1)
    losses2 = np.load('./losses_imbFactor0.02_network444_seed1.npy')
    losses2 = norm_data(losses2)
    losses3 = np.load('./losses_imbFactor0.02_seed1.npy')
    losses3 = norm_data(losses3)
    losses4 = np.load('./losses_imbFactor0.02_network666_seed1.npy')
    losses4 = norm_data(losses4)
    losses5 = np.load('./losses_imbFactor0.02_network777_seed1.npy')
    losses5 = norm_data(losses5)

    # losses = (losses1 + losses2 + losses3 + losses4 + losses5) / 5
    losses = np.concatenate([losses1, losses2, losses3, losses4, losses5], axis=1)

    np.save('losses_imbFactor0.02_stack5_normed.npy', losses)