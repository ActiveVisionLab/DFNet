import utils.set_sys_path
import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from models.ray_utils import *
from models.nerfw import * # NeRF-w and NeRF-hist
from models.options import config_parser
from models.rendering import *
from dataset_loaders.load_7Scenes import load_7Scenes_dataloader_NeRF
from dataset_loaders.load_Cambridge import load_Cambridge_dataloader_NeRF

# losses
from models.losses import loss_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(0)
import random
random.seed(0)

parser = config_parser()
args = parser.parse_args()

def train_on_epoch_nerfw(args, train_dl, H, W, focal, N_rand, optimizer, loss_func, global_step, render_kwargs_train):
    for batch_idx, (target, pose, img_idx) in enumerate(train_dl):
        target = target[0].permute(1,2,0).to(device)
        pose = pose.reshape(3,4).to(device) # reshape to 3x4 rot matrix
        img_idx = img_idx.to(device)

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if N_rand is not None:
            rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='ij'), -1)  # (H, W, 2)
            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        # #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, retraw=True, img_idx=img_idx, **render_kwargs_train)
        optimizer.zero_grad()

        # compute loss
        results = {}
        results['rgb_fine'] = rgb
        results['rgb_coarse'] = extras['rgb0']
        results['beta'] = extras['beta']
        results['transient_sigmas'] = extras['transient_sigmas']

        loss_d = loss_func(results, target_s)
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            img_loss = img2mse(rgb, target_s)
            psnr = mse2psnr(img_loss)
        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        torch.set_default_tensor_type('torch.FloatTensor')
    return loss, psnr

def train_nerf(args, train_dl, val_dl, hwf, i_split, near, far, render_poses=None, render_img=None):

    i_train, i_val, i_test = i_split
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    if args.reduce_embedding==2:
        render_kwargs_train['i_epoch'] = -1
        render_kwargs_test['i_epoch'] = -1

    if args.render_test:
        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)
        if args.reduce_embedding==2:
            render_kwargs_test['i_epoch'] = global_step
        render_test(args, train_dl, val_dl, hwf, start, render_kwargs_test)
        return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    # use_batching = not args.no_batching

    N_epoch = args.epochs + 1 # epoch
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)


    # loss function
    loss_func = loss_dict['nerfw'](coef=1)

    for i in trange(start, N_epoch):
        time0 = time.time()
        if args.reduce_embedding==2:
            render_kwargs_train['i_epoch'] = i
        loss, psnr = train_on_epoch_nerfw(args, train_dl, H, W, focal, N_rand, optimizer, loss_func, global_step, render_kwargs_train)
        dt = time.time()-time0
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0 and i!=0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if args.N_importance > 0: # have fine sample network
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'embedding_a_state_dict': render_kwargs_train['embedding_a'].state_dict(),
                    'embedding_t_state_dict': render_kwargs_train['embedding_t'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)

        if i%args.i_testset==0 and i > 0: # run thru all validation set

            # clean GPU memory before testing, try to avoid OOM
            torch.cuda.empty_cache()

            if args.reduce_embedding==2:
                render_kwargs_test['i_epoch'] = i
            trainsavedir = os.path.join(basedir, expname, 'trainset_{:06d}'.format(i))
            os.makedirs(trainsavedir, exist_ok=True)
            images_train = []
            poses_train = []
            index_train = []
            j_skip = 10 # save holdout view render result Trainset/j_skip
            # randomly choose some holdout views from training set
            for batch_idx, (img, pose, img_idx) in enumerate(train_dl):
                if batch_idx % j_skip != 0:
                    continue
                img_val = img.permute(0,2,3,1) # (1,H,W,3)
                pose_val = torch.zeros(1,4,4)
                pose_val[0,:3,:4] = pose.reshape(3,4)[:3,:4] # (1,3,4))
                pose_val[0,3,3] = 1.
                images_train.append(img_val)
                poses_train.append(pose_val)
                index_train.append(img_idx)
            images_train = torch.cat(images_train, dim=0).numpy()
            poses_train = torch.cat(poses_train, dim=0).to(device)
            index_train = torch.cat(index_train, dim=0).to(device)
            print('train poses shape', poses_train.shape)

            with torch.no_grad():
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
                render_path(args, poses_train, hwf, args.chunk, render_kwargs_test, gt_imgs=images_train, savedir=trainsavedir, img_ids=index_train)
                torch.set_default_tensor_type('torch.FloatTensor')
            print('Saved train set')
            del images_train
            del poses_train

            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            images_val = []
            poses_val = []
            index_val = []
            # views from validation set
            for img, pose, img_idx in val_dl:
                img_val = img.permute(0,2,3,1) # (1,H,W,3)
                pose_val = torch.zeros(1,4,4)
                pose_val[0,:3,:4] = pose.reshape(3,4)[:3,:4] # (1,3,4))
                pose_val[0,3,3] = 1.
                images_val.append(img_val)
                poses_val.append(pose_val)
                index_val.append(img_idx)

            images_val = torch.cat(images_val, dim=0).numpy()
            poses_val = torch.cat(poses_val, dim=0).to(device)
            index_val = torch.cat(index_val, dim=0).to(device)
            print('test poses shape', poses_val.shape)

            with torch.no_grad():
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
                render_path(args, poses_val, hwf, args.chunk, render_kwargs_test, gt_imgs=images_val, savedir=testsavedir, img_ids=index_val)
                torch.set_default_tensor_type('torch.FloatTensor')
            print('Saved test set')

            # clean GPU memory after testing
            torch.cuda.empty_cache()
            del images_val
            del poses_val
    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1

def train():

    print(parser.format_values())

    # Load data
    if args.dataset_type == '7Scenes':

        train_dl, val_dl, hwf, i_split, bds, render_poses, render_img = load_7Scenes_dataloader_NeRF(args)
        near = bds[0]
        far = bds[1]

        print('NEAR FAR', near, far)
        train_nerf(args, train_dl, val_dl, hwf, i_split, near, far, render_poses, render_img)
        return

    elif args.dataset_type == 'Cambridge':

        train_dl, val_dl, hwf, i_split, bds, render_poses, render_img = load_Cambridge_dataloader_NeRF(args)
        near = bds[0]
        far = bds[1]

        print('NEAR FAR', near, far)
        train_nerf(args, train_dl, val_dl, hwf, i_split, near, far, render_poses, render_img)
        return

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

if __name__=='__main__':

    train()