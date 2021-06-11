import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if meta loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, model_path='model_checkpoint.pth', vnet_path='vnet_checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time meta loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each meta loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.meta_loss_min = np.Inf
        self.delta = delta
        self.model_path = model_path
        self.vnet_path = vnet_path
        self.trace_func = trace_func
    def __call__(self, meta_loss, model, vnet):

        score = -meta_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(meta_loss, model,vnet)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(meta_loss, model, vnet)
            self.counter = 0

    def save_checkpoint(self, meta_loss, model, vnet):
        '''Saves model when metaidation loss decrease.'''
        if self.verbose:
            self.trace_func(f'meta loss decreased ({self.meta_loss_min:.6f} --> {meta_loss:.6f}).  Saving model and vnet...')
        torch.save(model.state_dict(), self.model_path)
        torch.save(vnet.state_dict(), self.vnet_path)
        self.meta_loss_min = meta_loss

class EarlyStopping2:
    """Early stops the training if meta loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, model_path='model_checkpoint.pth', vnet_path_1='vnet_checkpoint.pth', vnet_path_2='vnet_checkpoint_2.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time meta loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each meta loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.meta_loss_min = np.Inf
        self.delta = delta
        self.model_path = model_path
        self.vnet_path_1 = vnet_path_1
        self.vnet_path_2 = vnet_path_2
        self.trace_func = trace_func
    def __call__(self, meta_loss, model, vnet_1, vnet_2):

        score = -meta_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(meta_loss, model, vnet_1, vnet_2)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(meta_loss, model, vnet_1, vnet_2)
            self.counter = 0

    def save_checkpoint(self, meta_loss, model, vnet_1, vnet_2):
        '''Saves model when metaidation loss decrease.'''
        if self.verbose:
            self.trace_func(f'meta loss decreased ({self.meta_loss_min:.6f} --> {meta_loss:.6f}).  Saving model and vnet...')
        torch.save(model.state_dict(), self.model_path)
        torch.save(vnet_1.state_dict(), self.vnet_path_1)
        torch.save(vnet_2.state_dict(), self.vnet_path_2)
        self.meta_loss_min = meta_loss