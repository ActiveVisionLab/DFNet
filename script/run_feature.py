import sys
sys.path.append('../')
import torch
from torch import nn, optim
from torchvision.utils import save_image
import os, pdb
from torchsummary import summary
from dataset_loaders.load_7Scenes import load_7Scenes_dataloader
from dataset_loaders.load_Cambridge import load_Cambridge_dataloader
import os.path as osp
import numpy as np
from utils.utils import plot_features, save_image_saliancy, save_image_saliancy_single
from utils.utils import freeze_bn_layer, freeze_bn_layer_train
from models.nerfw import create_nerf
from tqdm import tqdm
from dm.callbacks import EarlyStopping
from feature.dfnet import DFNet, DFNet_s
# from feature.efficientnet import EfficientNetB3 as DFNet
# from feature.efficientnet import EfficientNetB0 as DFNet
from feature.misc import *
from feature.options import config_parser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(0)
torch.manual_seed(0)
import random
random.seed(0)

def tmp_plot(target_in, rgb_in, features_target, features_rgb):
    ''' 
    print 1 pair of salient feature map
    '''
    print("for debug only...")
    pdb.set_trace()
    ### plot featues with pixel-wise addition
    save_image(target_in[1], './tmp/target_in.png')
    save_image(rgb_in[1], './tmp/rgb_in.png')
    save_image_saliancy(features_target[1], './tmp/target', True)
    save_image_saliancy(features_rgb[1], './tmp/rgb', True)
    ### plot featues seperately
    save_image(target_in[1], './tmp/target_in.png')
    save_image(rgb_in[1], './tmp/rgb_in.png')
    plot_features(features_target[:,1:2,...], './tmp/target', False)
    plot_features(features_rgb[:,1:2,...], './tmp/rgb', False)
    sys.exit()

def tmp_plot2(target_in, rgb_in, features_target, features_rgb, i=0):
    '''
    print 1 pair of batch of salient feature map
    :param: target_in [B, 3, H, W]
    :param: rgb_in [B, 3, H, W]
    :param: features_target [B, C, H, W]
    :param: features_rgb [B, C, H, W]
    :param: frame index i of batch
    '''
    print("for debug only...")
    save_image(target_in[i], './tmp/target_in.png')
    save_image(rgb_in[i], './tmp/rgb_in.png')
    pdb.set_trace()
    features_t = features_target[i].clone()[:, None, :, :]
    features_r = features_rgb[i].clone()[:, None, :, :]
    save_image_saliancy(features_t, './tmp/target', True)
    save_image_saliancy(features_r, './tmp/rgb', True)

def tmp_plot3(target_in, rgb_in, features_target, features_rgb, i=0):
    '''
    print 1 pair of 1 sample of salient feature map
    :param: target_in [B, 3, H, W]
    :param: rgb_in [B, 3, H, W]
    :param: features_target [B, C, H, W]
    :param: features_rgb [B, C, H, W]
    :param: frame index i of batch
    '''
    print("for debug only...")
    save_image(target_in[i], './tmp/target_in.png')
    save_image(rgb_in[i], './tmp/rgb_in.png')
    features_t = features_target[i].clone()[:, None, :, :]
    features_r = features_rgb[i].clone()[:, None, :, :]
    save_image_saliancy(features_t[0], './tmp/target', True)
    save_image_saliancy(features_r[0], './tmp/rgb', True)

def lognuniform(low=-2, high=0, size=1, base=10):
    ''' sample from log uniform distribution between 0.01~1 '''
    return np.power(base, np.random.uniform(low, high, size))

def getrelpose(pose1, pose2):
    ''' get relative pose from abs pose pose1 to abs pose pose2 
    R^{v}_{gt} = R_v * R_gt.T
    :param: pose1 [B, 3, 4]
    :param: pose2 [B, 3, 4]
    return rel_pose [B, 3, 4]
    '''
    assert(pose1.shape == pose2.shape)
    rel_pose = pose1 - pose2 # compute translation term difference
    rel_pose[:,:3,:3] = pose2[:,:3,:3] @ torch.transpose(pose1[:,:3,:3], 1, 2) # compute rotation term difference
    return rel_pose

parser = config_parser()
args = parser.parse_args()

def train_on_batch(args, targets, rgbs, poses, feat_model, dset_size, FeatureLoss, optimizer, hwf):
    ''' core training loop for featurenet'''
    feat_model.train()
    H, W, focal = hwf
    H, W = int(H), int(W)
    if args.freezeBN:
        feat_model = freeze_bn_layer_train(feat_model)

    train_loss_epoch = []
    select_inds = np.random.choice(dset_size, size=[dset_size], replace=False)  # (N_rand,)

    batch_size=args.featurenet_batch_size # manual setting, use smaller batch size like featurenet_batch_size = 4 if OOM
    if dset_size % batch_size == 0:
        N_iters = dset_size//batch_size
    else:
        N_iters = dset_size//batch_size + 1
    i_batch = 0

    for i in range(0, N_iters):
        if i_batch + batch_size > dset_size:
            i_batch = 0
            break
        i_inds = select_inds[i_batch:i_batch+batch_size]
        i_batch = i_batch + batch_size

        # convert input shape to [B, 3, H, W]
        target_in = targets[i_inds].clone().permute(0,3,1,2).to(device)
        rgb_in = rgbs[i_inds].clone().permute(0,3,1,2).to(device)
        pose = poses[i_inds].clone().reshape(batch_size, 12).to(device)
        pose = torch.cat([pose, pose]) # double gt pose tensor

        features, predict_pose = feat_model(torch.cat([target_in, rgb_in]), True, upsampleH=H, upsampleW=W) # features: (1, [2, B, C, H, W])

        # get features_target and features_rgb
        if args.DFNet:
            features_target = features[0] # [3, B, C, H, W]
            features_rgb = features[1]
        else:
            features_target = features[0][0]
            features_rgb = features[0][1]

        # svd, seems not very benificial here, therefore removed

        if args.poselossonly:
            loss_pose = PoseLoss(args, predict_pose, pose, device) # target
            loss = loss_pose
        elif args.featurelossonly: # Not good. To be removed later
            loss_f = FeatureLoss(features_rgb, features_target)
            loss = loss_f
        else:
            loss_pose = PoseLoss(args, predict_pose, pose, device) # target
            if args.tripletloss:
                loss_f = triplet_loss_hard_negative_mining_plus(features_rgb, features_target, margin=args.triplet_margin)
            else:
                loss_f = FeatureLoss(features_rgb, features_target)
            loss = loss_pose + loss_f

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_epoch.append(loss.item())
    train_loss = np.mean(train_loss_epoch)
    return train_loss

def train_on_batch_with_random_view_synthesis(args, targets, rgbs, poses, virtue_view, poses_perturb, feat_model, dset_size, FeatureLoss, optimizer, hwf, img_idxs, render_kwargs_test):
    ''' we implement random view synthesis for generating more views to help training posenet '''
    feat_model.train()

    H, W, focal = hwf
    H, W = int(H), int(W)

    if args.freezeBN:
        feat_model = freeze_bn_layer_train(feat_model)

    train_loss_epoch = []

    # random generate batch_size of idx
    select_inds = np.random.choice(dset_size, size=[dset_size], replace=False)  # (N_rand,)

    batch_size=args.featurenet_batch_size # manual setting, use smaller batch size like featurenet_batch_size = 4 if OOM
    if dset_size % batch_size == 0:
        N_iters = dset_size//batch_size
    else:
        N_iters = dset_size//batch_size + 1
    
    i_batch = 0
    for i in range(0, N_iters):
        if i_batch + batch_size > dset_size:
            i_batch = 0
            break
        i_inds = select_inds[i_batch:i_batch+batch_size]
        i_batch = i_batch + batch_size

        # convert input shape to [B, 3, H, W]
        target_in = targets[i_inds].clone().permute(0,3,1,2).to(device)
        rgb_in = rgbs[i_inds].clone().permute(0,3,1,2).to(device)
        pose = poses[i_inds].clone().reshape(batch_size, 12).to(device)
        rgb_perturb = virtue_view[i_inds].clone().permute(0,3,1,2).to(device)
        pose_perturb = poses_perturb[i_inds].clone().reshape(batch_size, 12).to(device)

        # inference feature model for GT and nerf image
        pose = torch.cat([pose, pose]) # double gt pose tensor
        features, predict_pose = feat_model(torch.cat([target_in, rgb_in]), return_feature=True, upsampleH=H, upsampleW=W) # features: (1, [2, B, C, H, W])

        # get features_target and features_rgb
        if args.DFNet:
            features_target = features[0] # [3, B, C, H, W]
            features_rgb = features[1]

        loss_pose = PoseLoss(args, predict_pose, pose, device) # target

        if args.tripletloss:
            loss_f = triplet_loss_hard_negative_mining_plus(features_rgb, features_target, margin=args.triplet_margin)
        else:
            loss_f = FeatureLoss(features_rgb, features_target) # feature Maybe change to s2d-ce loss

        # inference model for RVS image
        _, virtue_pose = feat_model(rgb_perturb.to(device), False)

        # add relative pose loss here. TODO: This FeatureLoss is nn.MSE. Should be fixed later
        loss_pose_perturb = PoseLoss(args, virtue_pose, pose_perturb, device)
        loss = args.combine_loss_w[0]*loss_pose + args.combine_loss_w[1]*loss_f + args.combine_loss_w[2]*loss_pose_perturb

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_epoch.append(loss.item())
    train_loss = np.mean(train_loss_epoch)
    return train_loss

def train_feature(args, train_dl, val_dl, test_dl, hwf, i_split, near, far):

    # # load pretrained PoseNet model
    if args.DFNet_s:
        feat_model = DFNet_s()
    else:
        feat_model = DFNet()
    
    if args.pretrain_model_path != '':
        print("load posenet from ", args.pretrain_model_path)
        feat_model.load_state_dict(torch.load(args.pretrain_model_path))
    
    # # Freeze BN to not updating gamma and beta
    if args.freezeBN:
        feat_model = freeze_bn_layer(feat_model)

    feat_model.to(device)
    # summary(feat_model, (3, 240, 427))

    # set optimizer
    optimizer = optim.Adam(feat_model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, patience=args.patience[1], verbose=True)

    # set callbacks parameters
    early_stopping = EarlyStopping(args, patience=args.patience[0], verbose=False)

    # loss function
    loss_func = nn.MSELoss(reduction='mean')

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

    # load NeRF
    _, render_kwargs_test, start, _, _ = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    # render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    if args.reduce_embedding==2:
        render_kwargs_test['i_epoch'] = start

    N_epoch = args.epochs + 1 # epoch
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    world_setup_dict = {
        'pose_scale' : train_dl.dataset.pose_scale,
        'pose_scale2' : train_dl.dataset.pose_scale2,
        'move_all_cam_vec' : train_dl.dataset.move_all_cam_vec,
    }

    if args.eval:
        feat_model.eval()
        ### testing
        # get_error_in_q(args, train_dl, feat_model, len(train_dl.dataset), device, batch_size=1)
        get_error_in_q(args, test_dl, feat_model, len(val_dl.dataset), device, batch_size=1)
        sys.exit()

    if args.render_feature_only:
        targets, rgbs, poses, img_idxs = render_nerfw_imgs(args, test_dl, hwf, device, render_kwargs_test, world_setup_dict)
        dset_size = poses.shape[0]
        feat_model.eval()
        # extract features
        for i in range(dset_size):
            target_in = targets[i:i+1].permute(0,3,1,2).to(device)
            rgb_in = rgbs[i:i+1].permute(0,3,1,2).to(device)

            features, _ = feat_model(torch.cat([target_in, rgb_in]), True, upsampleH=H, upsampleW=W)
            if args.DFNet:
                features_target = features[0] # [3, B, C, H, W]
                features_rgb = features[1]

            # save features
            save_i = 2 # saving feature index, save_i out of 128
            ft = features_target[0, None, :, save_i] # [1,1,H,W]
            fr = features_rgb[0, None, :, save_i] # [1,1,H,W]

            scene = 'shop_gap/'
            save_path = './tmp/'+scene
            save_path_t = './tmp/'+scene+'target/'
            save_path_r = './tmp/'+scene+'rgb/'
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            if not os.path.isdir(save_path_t):
                os.mkdir(save_path_t)
            if not os.path.isdir(save_path_r):
                os.mkdir(save_path_r)
            save_image_saliancy_single(ft, save_path_t + '%04d.png'%i, True)
            save_image_saliancy_single(fr, save_path_r + '%04d.png'%i, True)

        print("render features done")
        sys.exit()


    targets, rgbs, poses, img_idxs = render_nerfw_imgs(args, train_dl, hwf, device, render_kwargs_test, world_setup_dict)

    dset_size = len(train_dl.dataset)
    # clean GPU memory before testing, try to avoid OOM
    torch.cuda.empty_cache()

    model_log = tqdm(total=0, position=1, bar_format='{desc}')
    for epoch in tqdm(range(N_epoch), desc='epochs'):

        if args.random_view_synthesis:
            ### this is the implementation of RVS ###
            isRVS = epoch % args.rvs_refresh_rate == 0 # decide if to resynthesis new views

            if isRVS:
                # random sample virtual camera locations, todo:
                rand_trans = args.rvs_trans
                rand_rot = args.rvs_rotation

                # determine bounding box
                b_min = [poses[:,0,3].min()-args.d_max, poses[:,1,3].min()-args.d_max, poses[:,2,3].min()-args.d_max]
                b_max = [poses[:,0,3].max()+args.d_max, poses[:,1,3].max()+args.d_max, poses[:,2,3].max()+args.d_max]
                
                poses_perturb = poses.clone().numpy()
                for i in range(dset_size):
                    poses_perturb[i] = perturb_single_render_pose(poses_perturb[i], rand_trans, rand_rot)
                    for j in range(3):
                        if poses_perturb[i,j,3] < b_min[j]:
                            poses_perturb[i,j,3] = b_min[j]
                        elif poses_perturb[i,j,3]> b_max[j]:
                            poses_perturb[i,j,3] = b_max[j]

                poses_perturb = torch.Tensor(poses_perturb).to(device) # [B, 3, 4]
                tqdm.write("renders RVS...")
                virtue_view = render_virtual_imgs(args, poses_perturb, img_idxs, hwf, device, render_kwargs_test, world_setup_dict)
            
            train_loss = train_on_batch_with_random_view_synthesis(args, targets, rgbs, poses, virtue_view, poses_perturb, feat_model, dset_size, loss_func, optimizer, hwf, img_idxs, render_kwargs_test)
            
        else:
            train_loss = train_on_batch(args, targets, rgbs, poses, feat_model, dset_size, loss_func, optimizer, hwf)

        feat_model.eval()
        val_loss_epoch = []
        for data, pose, _ in val_dl:
            inputs = data.to(device)
            labels = pose.to(device)
            
            # pose loss
            _, predict = feat_model(inputs)
            loss = loss_func(predict, labels)
            val_loss_epoch.append(loss.item())
        val_loss = np.mean(val_loss_epoch)

        # reduce LR on plateau
        scheduler.step(val_loss)

        # logging
        tqdm.write('At epoch {0:6d} : train loss: {1:.4f}, val loss: {2:.4f}'.format(epoch, train_loss, val_loss))

        # check wether to early stop
        early_stopping(val_loss, feat_model, epoch=epoch, save_multiple=(not args.no_save_multiple), save_all=args.save_all_ckpt)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if args.featurelossonly:
            global_step += 1
            continue

        model_log.set_description_str(f'Best val loss: {early_stopping.val_loss_min:.4f}')
        if epoch % args.i_eval == 0:
            get_error_in_q(args, test_dl, feat_model, len(test_dl.dataset), device, batch_size=1)
        global_step += 1

    return

def train():

    print(parser.format_values())

    # Load data
    if args.dataset_type == '7Scenes':

        train_dl, val_dl, test_dl, hwf, i_split, near, far = load_7Scenes_dataloader(args)
        near = near
        far = far
        print('NEAR FAR', near, far)
        train_feature(args, train_dl, val_dl, test_dl, hwf, i_split, near, far)
        return

    elif args.dataset_type == 'Cambridge':

        train_dl, val_dl, test_dl, hwf, i_split, near, far = load_Cambridge_dataloader(args)
        near = near
        far = far

        print('NEAR FAR', near, far)
        train_feature(args, train_dl, val_dl, test_dl, hwf, i_split, near, far)
        return

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

if __name__ == "__main__":

    train()