import numpy as np
import torch
from tqdm import tqdm
import os, pdb

def Callback():
    #TODO: Callback func. https://dzlab.github.io/dl/2019/03/16/pytorch-training-loop/
    def __init__(self): pass
    def on_train_begin(self): pass
    def on_train_end(self): pass
    def on_epoch_begin(self): pass
    def on_epoch_end(self): pass
    def on_batch_begin(self): pass
    def on_batch_end(self): pass
    def on_loss_begin(self): pass
    def on_loss_end(self): pass
    def on_step_begin(self): pass
    def on_step_end(self): pass

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    # source https://blog.csdn.net/qq_37430422/article/details/103638681
    def __init__(self, args, patience=50, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 50
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.val_on_psnr = args.val_on_psnr
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

        self.basedir = args.basedir
        self.model_name = args.model_name

        self.out_folder = os.path.join(self.basedir, self.model_name)
        self.ckpt_save_path = os.path.join(self.out_folder, 'checkpoint.pt')
        if not os.path.isdir(self.out_folder):
            os.mkdir(self.out_folder)

    def __call__(self, val_loss, model, epoch=-1, save_multiple=False, save_all=False, val_psnr=None):
        
        # find maximum psnr
        if self.val_on_psnr:
            score = val_psnr
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_psnr, model, epoch=epoch, save_multiple=save_multiple)
            elif score < self.best_score + self.delta:
                self.counter += 1

                if self.counter >= self.patience:
                    self.early_stop = True
                
                if save_all: # save all ckpt
                    self.save_checkpoint(val_psnr, model, epoch=epoch, save_multiple=True, update_best=False)
            else: # save best ckpt only
                self.best_score = score
                self.save_checkpoint(val_psnr, model, epoch=epoch, save_multiple=save_multiple)
                self.counter = 0

        # find minimum loss
        else:
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, epoch=epoch, save_multiple=save_multiple)
            elif score < self.best_score + self.delta:
                self.counter += 1

                if self.counter >= self.patience:
                    self.early_stop = True
                
                if save_all: # save all ckpt
                    self.save_checkpoint(val_loss, model, epoch=epoch, save_multiple=True, update_best=False)
            else: # save best ckpt only
                self.best_score = score
                self.save_checkpoint(val_loss, model, epoch=epoch, save_multiple=save_multiple)
                self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch=-1, save_multiple=False, update_best=True):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            tqdm.write(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        ckpt_save_path = self.ckpt_save_path
        if save_multiple:
            ckpt_save_path = ckpt_save_path[:-3]+f'-{epoch:04d}-{val_loss:.4f}.pt'

        torch.save(model.state_dict(), ckpt_save_path)
        if update_best:
            self.val_loss_min = val_loss
    
    def isBestModel(self):
        ''' Check if current model the best one.
        get early stop counter, if counter==0: it means current model has the best validation loss
        '''
        return self.counter==0