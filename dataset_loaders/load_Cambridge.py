import os
import os.path as osp
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch.cuda
from torch.utils.data import DataLoader
from torchvision import transforms, models

from dataset_loaders.cambridge_scenes import Cambridge2, normalize_recenter_pose, load_image
import pdb

#from dataset_loaders.frustum.frustum_util import initK, generate_sampling_frustum, compute_frustums_overlap

#focal_length = 555 # This is an approximate https://github.com/NVlabs/geomapnet/issues/8
                   # Official says (585,585)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# translation z axis
trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).astype(float)

# x rotation
rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).astype(float)

# y rotation
rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).astype(float)

# z rotation
rot_psi = lambda psi : np.array([
    [np.cos(psi),-np.sin(psi),0,0],
    [np.sin(psi),np.cos(psi),0,0],
    [0,0,1,0],
    [0,0,0,1]]).astype(float)

def is_inside_frustum(p, x_res, y_res):
    return (0 < p[0]) & (p[0] < x_res) & (0 < p[1]) & (p[1] < y_res)

def initK(f, cx, cy):
    K = np.eye(3, 3)
    K[0, 0] = K[1, 1] = f
    K[0, 2] = cx
    K[1, 2] = cy
    return K

def generate_sampling_frustum(step, depth, K, f, cx, cy, x_res, y_res):
    #pdb.set_trace()
    x_max = depth * (x_res - cx) / f
    x_min = -depth * cx / f
    y_max = depth * (y_res - cy) / f
    y_min = -depth * cy / f

    zs = np.arange(0, depth, step)
    xs = np.arange(x_min, x_max, step)
    ys = np.arange(y_min, y_max, step)

    X0 = []
    for z in zs:
        for x in xs:
            for y in ys:
                P = np.array([x, y, z])
                p = np.dot(K, P)
                if p[2] < 0.00001:
                    continue
                p = p / p[2]
                if is_inside_frustum(p, x_res, y_res):
                    X0.append(P)
    X0 = np.array(X0)
    return X0

def compute_frustums_overlap(pose0, pose1, sampling_frustum, K, x_res, y_res):
    R0 = pose0[0:3, 0:3]
    t0 = pose0[0:3, 3]
    R1 = pose1[0:3, 0:3]
    t1 = pose1[0:3, 3]

    R10 = np.dot(R1.T, R0)
    t10 = np.dot(R1.T, t0 - t1)

    _P = np.dot(R10, sampling_frustum.T).T + t10
    p = np.dot(K, _P.T).T
    pn = p[:, 2]
    p = np.divide(p, pn[:, None])
    res = np.apply_along_axis(is_inside_frustum, 1, p, x_res, y_res)
    return np.sum(res) / float(res.shape[0])

def perturb_rotation(c2w, theta, phi, psi=0):
    last_row = np.tile(np.array([0, 0, 0, 1]), (1, 1))  # (N_images, 1, 4)
    c2w = np.concatenate([c2w, last_row], 0)  # (N_images, 4, 4) homogeneous coordinate
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = rot_psi(psi/180.*np.pi) @ c2w
    c2w = c2w[:3,:4]
    return c2w

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)
    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)
    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)
    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)
    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    return pose_avg

def center_poses(poses, pose_avg_from_file=None):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)
        pose_avg_from_file: if not None, pose_avg is loaded from pose_avg_stats.txt

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """


    if pose_avg_from_file is None:
        pose_avg = average_poses(poses)  # (3, 4) # this need to be fixed throughout dataset
    else:
        pose_avg = pose_avg_from_file

    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation (4,4)
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg #np.linalg.inv(pose_avg_homo)

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5] # it's empty here...
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses

def average_poses(poses):
    """
    Same as in SingleCamVideoStatic.py
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    center = poses[..., 3].mean(0)  # (3)
    z = normalize(poses[..., 2].mean(0))  # (3)
    y_ = poses[..., 1].mean(0)  # (3)
    x = normalize(np.cross(y_, z))  # (3)
    y = np.cross(z, x)  # (3)
    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    return pose_avg

def generate_render_pose(poses, bds):
    idx = np.random.choice(poses.shape[0])
    c2w=poses[idx]
    print(c2w[:3,:4])
    
    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = bds.min()*.9, bds.max()*5.
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 20, 0) # views of 20 degrees
    c2w_path = c2w
    N_views = 120 # number of views in video
    N_rots = 2

    # Generate poses for spiral path
    render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
    return render_poses

def perturb_render_pose(poses, bds, x, angle):
    """
    Inputs:
        poses: (3, 4)
        bds: bounds
        x: translational perturb range
        angle: rotation angle perturb range in degrees
    Outputs:
        new_c2w: (N_views, 3, 4) new poses
    """
    idx = np.random.choice(poses.shape[0])
    c2w=poses[idx]
    
    N_views = 10 # number of views in video    
    new_c2w = np.zeros((N_views, 3, 4))

    # perturb translational pose
    for i in range(N_views):
        new_c2w[i] = c2w
        new_c2w[i,:,3] = new_c2w[i,:,3] + np.random.uniform(-x,x,3) # perturb pos between -1 to 1
        theta=np.random.uniform(-angle,angle,1) # in degrees
        phi=np.random.uniform(-angle,angle,1) # in degrees
        psi=np.random.uniform(-angle,angle,1) # in degrees
        new_c2w[i] = perturb_rotation(new_c2w[i], theta, phi, psi)
    return new_c2w, idx

def remove_overlap_data(train_set, val_set):
    ''' Remove some overlap data in val set so that train set and val set do not have overlap '''
    train = train_set.gt_idx
    val = val_set.gt_idx

    # find redundant data index in val_set
    index = np.where(np.in1d(val, train) == True) # this is a tuple
    # delete redundant data
    val_set.gt_idx = np.delete(val_set.gt_idx, index)
    val_set.poses = np.delete(val_set.poses, index, axis=0)
    for i in sorted(index[0], reverse=True):
        val_set.c_imgs.pop(i) 
        val_set.d_imgs.pop(i)
    return train_set, val_set

def fix_coord(args, train_set, val_set, pose_avg_stats_file='', rescale_coord=True):
    ''' fix coord for 7 Scenes to align with llff style dataset '''

    # This is only to store a pre-calculated pose average stats of the dataset
    if args.save_pose_avg_stats:
        pdb.set_trace()
        if pose_avg_stats_file == '':
            print('pose_avg_stats_file location unspecified, please double check...')
            sys.exit()

        all_poses = train_set.poses
        all_poses = all_poses.reshape(all_poses.shape[0], 3, 4)
        all_poses, pose_avg = center_poses(all_poses)

        # save pose_avg to pose_avg_stats.txt
        np.savetxt(pose_avg_stats_file, pose_avg)
        print('pose_avg_stats.txt successfully saved')
        sys.exit()

    # get all poses (train+val)
    train_poses = train_set.poses

    val_poses = val_set.poses
    all_poses = np.concatenate([train_poses, val_poses])

    # Center the poses for ndc
    all_poses = all_poses.reshape(all_poses.shape[0], 3, 4)

    # Here we use either pre-stored pose average stats or calculate pose average stats on the flight to center the poses
    if args.load_pose_avg_stats:
        pose_avg_from_file = np.loadtxt(pose_avg_stats_file)
        all_poses, pose_avg = center_poses(all_poses, pose_avg_from_file)
    else:
        all_poses, pose_avg = center_poses(all_poses)

    # Correct axis to LLFF Style y,z -> -y,-z
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(all_poses), 1, 1))  # (N_images, 1, 4)
    all_poses = np.concatenate([all_poses, last_row], 1)

    # rotate tpose 90 degrees at x axis # only corrected translation position
    all_poses = rot_phi(180/180.*np.pi) @ all_poses

    # correct view direction except mirror with gt view
    all_poses[:,:3,:3] = -all_poses[:,:3,:3]

    # camera direction mirror at x axis mod1 R' = R @ mirror matrix 
    # ref: https://gamedev.stackexchange.com/questions/149062/how-to-mirror-reflect-flip-a-4d-transformation-matrix
    all_poses[:,:3,:3] = all_poses[:,:3,:3] @ np.array([[-1,0,0],[0,1,0],[0,0,1]])

    all_poses = all_poses[:,:3,:4]

    bounds = np.array([train_set.near, train_set.far]) # manual tuned

    if rescale_coord:
        sc=train_set.pose_scale # manual tuned factor, align with colmap scale
        all_poses[:,:3,3] *= sc

        ### quite ugly ### 
        # move center of camera pose
        if train_set.move_all_cam_vec != [0.,0.,0.]:
            all_poses[:, :3, 3] += train_set.move_all_cam_vec

        if train_set.pose_scale2 != 1.0:
            all_poses[:,:3,3] *= train_set.pose_scale2
        # end of new mod1

    # Return all poses to dataset loaders
    all_poses = all_poses.reshape(all_poses.shape[0], 12)
    train_set.poses = all_poses[:train_poses.shape[0]]
    val_set.poses = all_poses[train_poses.shape[0]:]
    return train_set, val_set, bounds

def load_Cambridge_dataloader(args):
    ''' Data loader for Pose Regression Network '''
    if args.pose_only: # if train posenet is true
        pass
    else:
        raise Exception('load_Cambridge_dataloader() currently only support PoseNet Training, not NeRF training')
    data_dir, scene = osp.split(args.datadir) # ../data/7Scenes, chess
    dataset_folder, dataset = osp.split(data_dir) # ../data, 7Scenes

    # transformer
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    target_transform = transforms.Lambda(lambda x: torch.Tensor(x))

    ret_idx = False # return frame index
    fix_idx = False # return frame index=0 in training
    ret_hist = False

    if 'NeRFH' in args:
        if args.NeRFH == True:
            ret_idx = True
            if args.fix_index:
                fix_idx = True

    # encode hist experiment
    if args.encode_hist:
        ret_idx = False
        fix_idx = False
        ret_hist = True

    kwargs = dict(scene=scene, data_path=data_dir,
        transform=data_transform, target_transform=target_transform, 
        df=args.df, ret_idx=ret_idx, fix_idx=fix_idx,
        ret_hist=ret_hist, hist_bin=args.hist_bin)

    if args.finetune_unlabel: # direct-pn + unlabel
        train_set = Cambridge2(train=False, testskip=args.trainskip, **kwargs)
        val_set = Cambridge2(train=False, testskip=args.testskip, **kwargs)

        # if not args.eval:
        #     # remove overlap data in val_set that was already in train_set,
        #     train_set, val_set = remove_overlap_data(train_set, val_set)
    else:
        train_set = Cambridge2(train=True, trainskip=args.trainskip, **kwargs)
        val_set = Cambridge2(train=False, testskip=args.testskip, **kwargs)
    L = len(train_set)

    i_train = train_set.gt_idx
    i_val = val_set.gt_idx
    i_test = val_set.gt_idx
    # use a pose average stats computed earlier to unify posenet and nerf training
    if args.save_pose_avg_stats or args.load_pose_avg_stats:
        pose_avg_stats_file = osp.join(args.datadir, 'pose_avg_stats.txt')
        train_set, val_set, bounds = fix_coord(args, train_set, val_set, pose_avg_stats_file, rescale_coord=False) # only adjust coord. systems, rescale are done at training
    else:
        train_set, val_set, bounds = fix_coord(args, train_set, val_set, rescale_coord=False)

    train_shuffle=True
    if args.eval:
        train_shuffle=False

    train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=train_shuffle, num_workers=8) #num_workers=4 pin_memory=True
    val_dl = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=2)
    test_dl = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    hwf = [train_set.H, train_set.W, train_set.focal]
    i_split = [i_train, i_val, i_test]

    return train_dl, val_dl, test_dl, hwf, i_split, bounds.min(), bounds.max()

def load_Cambridge_dataloader_NeRF(args):
    ''' Data loader for NeRF '''

    data_dir, scene = osp.split(args.datadir) # ../data/7Scenes, chess
    dataset_folder, dataset = osp.split(data_dir) # ../data, 7Scenes

    data_transform = transforms.Compose([
        transforms.ToTensor()])
    target_transform = transforms.Lambda(lambda x: torch.Tensor(x))

    ret_idx = False # return frame index
    fix_idx = False # return frame index=0 in training
    ret_hist = False

    if 'NeRFH' in args:
        ret_idx = True
        if args.fix_index:
            fix_idx = True

    # encode hist experiment
    if args.encode_hist:
        ret_idx = False
        fix_idx = False
        ret_hist = True

    kwargs = dict(scene=scene, data_path=data_dir,
        transform=data_transform, target_transform=target_transform, 
        df=args.df, ret_idx=ret_idx, fix_idx=fix_idx, ret_hist=ret_hist, hist_bin=args.hist_bin)

    train_set = Cambridge2(train=True, trainskip=args.trainskip, **kwargs)
    val_set = Cambridge2(train=False, testskip=args.testskip, **kwargs)
 
    i_train = train_set.gt_idx
    i_val = val_set.gt_idx
    i_test = val_set.gt_idx

    # use a pose average stats computed earlier to unify posenet and nerf training
    if args.save_pose_avg_stats or args.load_pose_avg_stats:
        pose_avg_stats_file = osp.join(args.datadir, 'pose_avg_stats.txt')
        train_set, val_set, bounds = fix_coord(args, train_set, val_set, pose_avg_stats_file)
    else:
        train_set, val_set, bounds = fix_coord(args, train_set, val_set)

    render_poses = None
    render_img = None

    train_shuffle=True
    if args.render_video_train or args.render_test or args.dataset_type == 'Cambridge2':
        train_shuffle=False
    train_dl = DataLoader(train_set, batch_size=1, shuffle=train_shuffle) # default
    # train_dl = DataLoader(train_set, batch_size=1, shuffle=False) # for debug only
    val_dl = DataLoader(val_set, batch_size=1, shuffle=False)

    hwf = [train_set.H, train_set.W, train_set.focal]

    i_split = [i_train, i_val, i_test]
    
    return train_dl, val_dl, hwf, i_split, bounds, render_poses, render_img