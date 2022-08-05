import math
import os
import time
import pprint
import random
import pdb
import cv2
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import numpy as np
from torchvision import models
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dm.pose_model import preprocess_data, get_error_in_q
# from dm.prepare_data import prepare_data
from models.nerfw import create_nerf, to8b, img2mse, mse2psnr
from models.ray_utils import get_rays
from models.rendering import render
from utils.utils import freeze_bn_layer_train
''' FeatureNet models '''
# from feature.model import PoseNetV2 as FeatureNet
# from feature.model import EfficientNetB3 as FeatureNet
from feature.dfnet import DFNet as FeatureNet
from feature.dfnet import DFNet_s as FeatureNet3
# from feature.efficientnet import EfficientNetB3 as FeatureNet
# from feature.efficientnet import EfficientNetB0 as FeatureNet

''' PoseNet Models '''
# from feature.efficientnet import EfficientNetB0 as PoseNet
from feature.dfnet import DFNet as PoseNet
from feature.dfnet import DFNet_s as PoseNet3
from torchvision.utils import save_image

def disable_model_grad(model):
    ''' set whole model to requires_grad=False, this is for nerf model '''
    print("disable_model_grad...")
    for module in model.modules():
        # print("this is a layer:", module)
        if hasattr(module, 'weight'):
            module.weight.requires_grad_(False)
        if hasattr(module, 'bias'):
            module.bias.requires_grad_(False)
    return model

def inference_pose_regression(args, data, device, model):
    """
    Inference the Pose Regression Network
    Inputs:
        args: parsed argument
        data: Input image in shape (batchsize, channels, H, W)
        device: gpu device
        model: PoseNet model
    Outputs:
        pose: Predicted Pose in shape (batchsize, 3, 4)
    """
    inputs = data.to(device)
    if args.preprocess_ImgNet:
        inputs = preprocess_data(inputs, device)
    predict_pose = model(inputs)
    pose = predict_pose.reshape(args.batch_size, 3, 4)

    if args.svd_reg:
        # pdb.set_trace()
        R_torch = pose[:,:3,:3].clone() # debug
        u,s,v=torch.svd(R_torch)
        Rs = torch.matmul(u, v.transpose(-2,-1))
        pose[:,:3,:3] = Rs
    return pose

def rgb_loss(rgb, target, extras):
    ''' Compute RGB MSE Loss, original from NeRF Paper '''
    # Compute MSE loss between predicted and true RGB.
    img_loss = img2mse(rgb, target)
    loss = img_loss

    # Add MSE loss for coarse-grained model
    if 'rgb0' in extras:
        img_loss0 = img2mse(extras['rgb0'], target)
        loss += img_loss0
    return loss

def PoseLoss(args, pose_, pose, device):
    loss_func = nn.MSELoss()
    predict_pose = pose_.reshape(args.batch_size, 12).to(device) # maynot need reshape
    pose_loss = loss_func(predict_pose, pose)
    return pose_loss

def load_exisiting_model(args, isFeatureNet=False):
    ''' Load a pretrained DFNet model '''
    if isFeatureNet==False: # load the Pose Estimator F
        if args.DFNet_s:
            model = PoseNet3()
            model.load_state_dict(torch.load(args.pretrain_model_path))
        else:
            model = PoseNet()
            model.load_state_dict(torch.load(args.pretrain_model_path))
        return model
    else:
        if args.DFNet_s: # load the Feature Extractor G
            model = FeatureNet3()
            model.load_state_dict(torch.load(args.pretrain_featurenet_path))
        else:
            model=FeatureNet()
            model.load_state_dict(torch.load(args.pretrain_featurenet_path))
        return model

def prepare_batch_render(args, pose, batch_size, target_, H, W, focal, half_res=True, rand=True):
    ''' Break batch of images into rays '''
    target_ = target_.permute(0, 2, 3, 1).numpy()#.squeeze(0) # convert to numpy image
    if half_res:
        N_rand = batch_size * (H//2) * (W//2)
        target_half = np.stack([cv2.resize(target_[i], (H//2, W//2), interpolation=cv2.INTER_AREA) for i in range(batch_size)], 0)
        target_half = torch.Tensor(target_half)
        
        rays = torch.stack([torch.stack(get_rays(H//2, W//2, focal/2, pose[i]), 0) for i in range(batch_size)], 0) # [N, ro+rd, H, W, 3] (130, 2, 100, 100, 3)
        rays_rgb = torch.cat((rays, target_half[:, None, ...]), 1)

    else:
        # N_rand = batch_size * H * W
        N_rand = args.N_rand
        target_ = torch.Tensor(target_)
        rays = torch.stack([torch.stack(get_rays(H, W, focal, pose[i]), 0) for i in range(batch_size)], 0) # [N, ro+rd, H, W, 3] (130, 2, 200, 200, 3)
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = torch.cat([rays, target_[:, None, ...]], 1)

    # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.permute(0, 2, 3, 1, 4)
    
    # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = torch.reshape(rays_rgb, (-1, 3, 3))

    if 1:
        #print('shuffle rays')
        rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]

    # Random over all images
    batch = rays_rgb[:N_rand].permute(1, 0 , 2) # [B, 2+1, 3*?] # (4096, 3, 3)
    batch_rays, target_s = batch[:2], batch[2] # [2, 4096, 3], [4096, 3]

    return batch_rays, target_s

def fix_coord_supp(args, pose, world_setup_dict, device=None):
    # this function needs to be fixed because it is taking args.pose_scale
    '''supplementary fix_coord() for direct matching
    Inputs:
        args: parsed argument
        pose: pose [N, 3, 4]
        device: cpu or gpu
    Outputs:
        pose: converted Pose in shape [N, 3, 4]
    '''
    sc=world_setup_dict['pose_scale'] # manual tuned factor, align with colmap scale
    if device is None:
        move_all_cam_vec = torch.Tensor(world_setup_dict['move_all_cam_vec'])
    else:
        move_all_cam_vec = torch.Tensor(world_setup_dict['move_all_cam_vec']).to(device)
    sc2 = world_setup_dict['pose_scale2']
    pose[:,:3,3] *= sc
    # move center of camera pose
    pose[:, :3, 3] += move_all_cam_vec
    pose[:,:3,3] *= sc2
    return pose

def eval_on_batch(args, data, model, pose, img_idx, hwf, half_res, device, **render_kwargs_test):
    ''' Perform 1 step of eval'''
    with torch.no_grad():
        H, W, focal = hwf
        target_ = deepcopy(data)
        pose_ = inference_pose_regression(args, data, device, model)
        device_cpu = torch.device('cpu')
        pose_ = pose_.to(device_cpu) # put predict pose back to cpu
        pose_nerf = pose_.clone()

        if args.NeRFH:
            # rescale the predicted pose to nerf scales
            pose_nerf = fix_coord_supp(args, pose_nerf)

        batch_rays, target = prepare_batch_render(args, pose_nerf, args.batch_size, target_, H, W, focal, half_res)
        batch_rays = batch_rays.to(device)
        target = target.to(device)
        pose = pose.to(device)
        img_idx = img_idx.to(device)

        # every new tensor from onward is in GPU
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if half_res:
            rgb, disp, acc, extras = render(H//2, W//2, focal/2, chunk=args.chunk, rays=batch_rays, img_idx=img_idx, **render_kwargs_test)
        else:
            rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, img_idx=img_idx, **render_kwargs_test)

        loss = PoseLoss(args, pose_, pose, device)
        psnr = mse2psnr(img2mse(rgb, target))

        # end of every new tensor from onward is in GPU
        torch.set_default_tensor_type('torch.FloatTensor')

        iter_loss = loss.to(device_cpu).detach().numpy()
        iter_loss = np.array([iter_loss])

        iter_psnr = psnr.to(device_cpu).detach().numpy()
    return iter_loss, iter_psnr

def eval_on_epoch(args, data_loaders, model, hwf, half_res, device, **render_kwargs_test):
    ''' Perform 1 epoch of training with batch '''
    model.eval()
    batch_size = 1
    
    train_dl, val_dl, test_dl = data_loaders

    total_loss = []
    total_psnr = []
    
    ####  Core optimization loop  #####
    for data, pose, img_idx in val_dl:
        # training one step with batch_size = args.batch_size
        loss, psnr = eval_on_batch(args, data, model, pose, img_idx, hwf, half_res, device, **render_kwargs_test)
        total_loss.append(loss.item())
        total_psnr.append(psnr.item())
    total_loss_mean = np.mean(total_loss)
    total_psnr_mean = np.mean(total_psnr)
    return total_loss_mean, total_psnr_mean

def train_on_batch(args, data, model, pose, img_idx, hwf, optimizer, half_res, device, **render_kwargs_test):
    ''' Perform 1 step of training'''
    H, W, focal = hwf
    target_ = deepcopy(data)
    pose_ = inference_pose_regression(args, data, device, model)
    device_cpu = torch.device('cpu')
    pose_ = pose_.to(device_cpu) # put predict pose back to cpu
    pose_nerf = pose_.clone()
    if args.NeRFH:
        # rescale the predicted pose to nerf scales
        pose_nerf = fix_coord_supp(args, pose_nerf)
    
    batch_rays, target = prepare_batch_render(args, pose_nerf, args.batch_size, target_, H, W, focal, half_res)
    batch_rays = batch_rays.to(device)
    target = target.to(device)
    pose = pose.to(device)
    img_idx = img_idx.to(device)

    # # every new tensor from onward is in GPU
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if half_res:
        rgb, disp, acc, extras = render(H//2, W//2, focal/2, chunk=args.chunk, rays=batch_rays, img_idx=img_idx, **render_kwargs_test)
    else:
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, img_idx=img_idx, **render_kwargs_test)

    ### Loss Design Here ###
    # Compute RGB MSE Loss
    photo_loss = rgb_loss(rgb, target, extras)

    # Compute Combine Loss if needed
    if args.combine_loss:
        pose_loss = PoseLoss(args, pose_, pose, device)
        loss = args.combine_loss_w[0] * pose_loss + args.combine_loss_w[1] * photo_loss
        # print("img_idx: {}, pose_loss: {}, rgb_loss {}, total_loss {}".format(img_idx, args.combine_loss_w[0] * pose_loss, args.combine_loss_w[1] * photo_loss, loss))

    ### Loss Design End
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    psnr = mse2psnr(img2mse(rgb, target))

    # end of every new tensor from onward is in GPU
    torch.set_default_tensor_type('torch.FloatTensor')

    iter_loss = loss.to(device_cpu).detach().numpy()
    iter_loss = np.array([iter_loss])

    iter_psnr = psnr.to(device_cpu).detach().numpy()
    return iter_loss, iter_psnr


def train_on_epoch(args, data_loaders, model, hwf, optimizer, half_res, device, **render_kwargs_test):
    ''' Perform 1 epoch of training with batch '''
    model.train()
    model = freeze_bn_layer_train(model)

    # Prepare dataloaders for PoseNet, each batch contains (image, pose)
    train_dl, val_dl, test_dl = data_loaders
    total_loss = []
    total_psnr = []
    
    ####  Core optimization loop  #####
    for data, pose, img_idx in train_dl:
        # print("img_idx: {}, pose: {}".format(img_idx, pose) )
        # training one step with batch_size = args.batch_size
        loss, psnr = train_on_batch(args, data, model, pose, img_idx, hwf, optimizer, half_res, device, **render_kwargs_test)
        total_loss.append(loss.item())
        total_psnr.append(psnr.item())
    total_loss_mean = np.mean(total_loss)
    total_psnr_mean = np.mean(total_psnr)
    return total_loss_mean, total_psnr_mean

def save_val_result_7Scenes(args, epoch, val_dl, model, hwf, half_res, device, num_samples=1, **render_kwargs_test):
    ''' Perform inference on a random val image and save the result '''
    half_res=False # save half res image to reduce computational speed
    model.eval()
    i = 0
    for batch in val_dl:
        if args.NeRFH:
            data, pose, img_idx = batch
        else:
            data, pose = batch
        if i >= num_samples:
            return
        H, W, focal = hwf
        target_ = deepcopy(data)
        img_idx = img_idx.to(device)

        pose_ = inference_pose_regression(args, data, device, model)
        device_cpu = torch.device('cpu')
        pose_ = pose_.to(device_cpu) # put predict_pose back to cpu

        pose_nerf = pose_.clone()
        if args.NeRFH:
            # rescale the predicted pose to nerf scales
            pose_nerf = fix_coord_supp(args, pose_nerf)
        
        # every new tensor from onward is in GPU
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        with torch.no_grad():
            if half_res:
                rgb, disp, acc, extras = render(H//2, W//2, focal/2, chunk=args.chunk, c2w=pose_nerf[0].to(device), img_idx=img_idx, **render_kwargs_test)
            else:
                rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf[0].to(device), img_idx=img_idx, **render_kwargs_test)

        ### Set Save Dir ###
        if num_samples <=1:
            out_folder = os.path.join(args.basedir, args.model_name, 'val_imgs')
        else:
            out_folder = os.path.join(args.basedir, args.model_name, 'val_imgs_batches')
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)

        if half_res:
            target_img = F.interpolate(target_, scale_factor=0.5, mode='area').permute(0,2,3,1).reshape(H//2,W//2,3)
            rgb_img = rgb.reshape(H//2,W//2,3)
        else:
            target_img = target_.permute(0,2,3,1).reshape(H,W,3)
            rgb_img = rgb.reshape(H,W,3)
        target_img_to_save = to8b(target_img.to(device_cpu).detach().numpy())
        rgb_img_to_save = to8b(rgb_img.to(device_cpu).detach().numpy())
        # save NeRF Rendered RGB Img
        import imageio
        if num_samples <= 1:
            imageio.imwrite(os.path.join(out_folder, '{0:04d}_gt.png'.format(epoch)), target_img_to_save)
            imageio.imwrite(os.path.join(out_folder, '{0:04d}.png'.format(epoch)), rgb_img_to_save)
        else:
            imageio.imwrite(os.path.join(out_folder, '{0:04d}_gt.png'.format(i)), target_img_to_save)
            imageio.imwrite(os.path.join(out_folder, '{0:04d}.png'.format(i)), rgb_img_to_save)
        
        # end of every new tensor from onward is in GPU
        torch.set_default_tensor_type('torch.FloatTensor')
        i = i+1


def train_nerf_tracking(args, model, optimizer, i_split, hwf, near, far, device, early_stopping, images=None, poses_train=None, train_dl=None, val_dl=None, test_dl=None):
    ''' finetune pretrained PoseNet using NeRF '''
    half_res = False # direct-pn paper settings
    # half_res = True # This half_res is to further downsample the output image of nerf to 100x100
    # Prepare dataloaders for PoseNet, each batch contains (image, pose)
    if args.dataset_type != '7Scenes' and args.dataset_type != 'Cambridge': # blender dataset
        train_dl, val_dl, test_dl = prepare_data(args, images, poses_train, i_split)
        _, render_kwargs_test, _, grad_vars, _ = create_nerf(args) # llff nerf
    else:
        # load NeRF model
        _, render_kwargs_test, start, grad_vars, _ = create_nerf(args)
        global_step = start
        if args.reduce_embedding==2:
            render_kwargs_test['i_epoch'] = global_step
    
    # # only freeze 150MB? effectiveness of this towards training result is yet unknown
    render_kwargs_test['embedding_a'] = disable_model_grad(render_kwargs_test['embedding_a'])
    render_kwargs_test['embedding_t'] = disable_model_grad(render_kwargs_test['embedding_t'])
    render_kwargs_test['network_fn'] = disable_model_grad(render_kwargs_test['network_fn'])
    render_kwargs_test['network_fine'] = disable_model_grad(render_kwargs_test['network_fine'])

    data_loaders = [train_dl, val_dl, test_dl]
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    # render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    i_train, i_val, i_test = i_split

    N_epoch = 2001
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    time0 = time.time()

    model_log = tqdm(total=0, position = 1, bar_format='{desc}')
    for epoch in tqdm(range(N_epoch), desc='epochs'):
        #train 1 epoch with batch_size = 1
        loss, psnr = train_on_epoch(args, data_loaders, model, hwf, optimizer, half_res, device, **render_kwargs_test)
        
        val_loss, val_psnr = eval_on_epoch(args, data_loaders, model, hwf, half_res, device, **render_kwargs_test)

        tqdm.write('At epoch {0:4d} : train loss: {1:.4f}, train psnr: {2:.4f}, val loss: {3:.4f}, val psnr: {4:.4f}'.format(epoch, loss, psnr, val_loss, val_psnr))

        # check wether to early stop
        early_stopping(val_loss, model, epoch=epoch, save_multiple=(not args.no_save_multiple), save_all=args.save_all_ckpt)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        model_log.set_description_str(f'Best val loss: {early_stopping.val_loss_min:.4f}')

        if (epoch % 1 == 0) and (args.pose_only != 1):
            ### run one single image save the result
            if args.dataset_type == '7Scenes' or args.dataset_type == 'llff' or args.dataset_type == 'Cambridge':
                save_val_result_7Scenes(args, epoch, val_dl, model, hwf, half_res, device, **render_kwargs_test)

        if epoch % args.i_eval == 0:
            # calculate position and angular error
            get_error_in_q(args, val_dl, model, len(val_dl.dataset), device, batch_size=1)
