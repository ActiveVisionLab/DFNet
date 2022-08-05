import utils.set_sys_path
import os, sys
import numpy as np
import imageio
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
from torchsummary import summary
# from torchinfo import summary
import matplotlib.pyplot as plt

from dm.pose_model import *
from dm.direct_pose_model import *
from dm.callbacks import EarlyStopping
# from dm.prepare_data import prepare_data, load_dataset
from dm.options import config_parser
from models.rendering import render_path
from models.nerfw import to8b
from dataset_loaders.load_7Scenes import load_7Scenes_dataloader
from dataset_loaders.load_Cambridge import load_Cambridge_dataloader
from utils.utils import freeze_bn_layer
from feature.direct_feature_matching import train_feature_matching
# import torch.onnx

parser = config_parser()
args = parser.parse_args()
device = torch.device('cuda:0') # this is really controlled in train.sh

def render_test(args, train_dl, val_dl, hwf, start, model, device, render_kwargs_test):
    model.eval()

    # ### Eval Training set result
    if args.render_video_train:
        images_train = []
        poses_train = []
        # views from train set
        for img, pose in train_dl:
            predict_pose = inference_pose_regression(args, img, device, model)
            device_cpu = torch.device('cpu')
            predict_pose = predict_pose.to(device_cpu) # put predict pose back to cpu

            img_val = img.permute(0,2,3,1) # (1,240,320,3)
            pose_val = torch.zeros(1,4,4)
            pose_val[0,:3,:4] = predict_pose.reshape(3,4)[:3,:4] # (1,3,4))
            pose_val[0,3,3] = 1.
            images_train.append(img_val)
            poses_train.append(pose_val)

        images_train = torch.cat(images_train, dim=0).numpy()
        poses_train = torch.cat(poses_train, dim=0)
        print('train poses shape', poses_train.shape)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        with torch.no_grad():
            rgbs, disps = render_path(poses_train.to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images_train, savedir=None)
        torch.set_default_tensor_type('torch.FloatTensor')
        print('Saving trainset as video', rgbs.shape, disps.shape)
        moviebase = os.path.join(args.basedir, args.model_name, '{}_trainset_{:06d}_'.format(args.model_name, start))
        imageio.mimwrite(moviebase + 'train_rgb.mp4', to8b(rgbs), fps=15, quality=8)
        imageio.mimwrite(moviebase + 'train_disp.mp4', to8b(disps / np.max(disps)), fps=15, quality=8)

    ### Eval Validation set result
    if args.render_video_test:
        images_val = []
        poses_val = []
        # views from val set
        for img, pose in val_dl: 
            predict_pose = inference_pose_regression(args, img, device, model)
            device_cpu = torch.device('cpu')
            predict_pose = predict_pose.to(device_cpu) # put predict pose back to cpu

            img_val = img.permute(0,2,3,1) # (1,240,360,3)
            pose_val = torch.zeros(1,4,4)
            pose_val[0,:3,:4] = predict_pose.reshape(3,4)[:3,:4] # (1,3,4))
            pose_val[0,3,3] = 1.
            images_val.append(img_val)
            poses_val.append(pose_val)

        images_val = torch.cat(images_val, dim=0).numpy()
        poses_val = torch.cat(poses_val, dim=0)
        print('test poses shape', poses_val.shape)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        with torch.no_grad():
            rgbs, disps = render_path(poses_val.to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images_val, savedir=None)
        torch.set_default_tensor_type('torch.FloatTensor')
        print('Saving testset as video', rgbs.shape, disps.shape)
        moviebase = os.path.join(args.basedir, args.model_name, '{}_test_{:06d}_'.format(args.model_name, start))
        imageio.mimwrite(moviebase + 'test_rgb.mp4', to8b(rgbs), fps=15, quality=8)
        imageio.mimwrite(moviebase + 'test_disp.mp4', to8b(disps / np.max(disps)), fps=15, quality=8)
    return

def train():
    print(parser.format_values())
    # Load data
    if args.dataset_type == '7Scenes':
        train_dl, val_dl, test_dl, hwf, i_split, near, far = load_7Scenes_dataloader(args)
    elif args.dataset_type == 'Cambridge':
        train_dl, val_dl, test_dl, hwf, i_split, near, far = load_Cambridge_dataloader(args)
    else:
        print("please choose dataset_type: 7Scenes or Cambridge, exiting...")
        sys.exit()

    ### pose regression module, here requires a pretrained DFNet for Pose Estimator F
    assert(args.pretrain_model_path != '') # make sure to add the pretrained model with --pretrain_model_path
    # load pretrained DFNet model
    model = load_exisiting_model(args)

    if args.freezeBN:
        model = freeze_bn_layer(model)
    model.to(device)

    ### feature extraction module, here requires a pretrained DFNet for Feature Extractor G using --pretrain_featurenet_path
    if args.pretrain_featurenet_path == '':
        print('Use the same DFNet for Feature Extraction and Pose Regression')
        feat_model = load_exisiting_model(args)
    else: 
        # you can optionally load different pretrained DFNet for feature extractor and pose estimator
        feat_model = load_exisiting_model(args, isFeatureNet=True)

    feat_model.eval()
    feat_model.to(device)

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) #weight_decay=weight_decay, **kwargs

    # set callbacks parameters
    early_stopping = EarlyStopping(args, patience=args.patience[0], verbose=False)

    # start training
    if args.dataset_type == '7Scenes':
        train_feature_matching(args, model, feat_model, optimizer, i_split, hwf, near, far, device, early_stopping, train_dl=train_dl, val_dl=val_dl, test_dl=test_dl)
    elif args.dataset_type == 'Cambridge':
        train_feature_matching(args, model, feat_model, optimizer, i_split, hwf, near, far, device, early_stopping, train_dl=train_dl, val_dl=val_dl, test_dl=test_dl)

def eval():
    print(parser.format_values())
    # Load data
    if args.dataset_type == '7Scenes':
        train_dl, val_dl, test_dl, hwf, i_split, near, far = load_7Scenes_dataloader(args)
    elif args.dataset_type == 'Cambridge':
        train_dl, val_dl, test_dl, hwf, i_split, near, far = load_Cambridge_dataloader(args)
    else:
        print("please choose dataset_type: 7Scenes or Cambridge, exiting...")
        sys.exit()
    
    # load pretrained DFNet_dm model
    model = load_exisiting_model(args)
    if args.freezeBN:
        model = freeze_bn_layer(model)
    model.to(device)

    print(len(test_dl.dataset))

    get_error_in_q(args, test_dl, model, len(test_dl.dataset), device, batch_size=1)

if __name__ == '__main__':
    if args.eval:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        eval()
    else:
        train()
