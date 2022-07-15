'''
helper functions to train robust feature extractors
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pdb
from PIL import Image
from torchvision.utils import save_image
from math import pi
import cv2
# from pykalman import KalmanFilter

def freeze_bn_layer(model):
    ''' freeze bn layer by not require grad but still behave differently when model.train() vs. model.eval() '''
    print("Freezing BatchNorm Layers...")
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            # print("this is a BN layer:", module)
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
    return model

def freeze_bn_layer_train(model):
    ''' set batchnorm to eval() 
        it is useful to align train and testing result 
    '''
    # model.train()
    # print("Freezing BatchNorm Layers...")
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    return model

def save_image_saliancy(tensor, path, normalize: bool = False, scale_each: bool = False,):
    """
    Modification based on TORCHVISION.UTILS
    ::param: tensor (batch, channel, H, W)
    """
    # grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=32)
    grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=6)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    fig = plt.figure()
    plt.imshow(ndarr[:,:,0], cmap='jet') # viridis, plasma
    plt.axis('off')
    fig.savefig(path, bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    plt.close()

def save_image_saliancy_single(tensor, path, normalize: bool = False, scale_each: bool = False,):
    """
    Modification based on TORCHVISION.UTILS, save single feature map
    ::param: tensor (batch, channel, H, W)
    """
    # grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=32)
    grid = make_grid(tensor.detach(), normalize=normalize, scale_each=scale_each, nrow=1)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    fig = plt.figure()
    # plt.imshow(ndarr[:,:,0], cmap='plasma') # viridis, jet
    plt.imshow(ndarr[:,:,0], cmap='jet') # viridis, jet
    plt.axis('off')
    fig.savefig(path, bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    plt.close()

def print_feature_examples(features, path):
    """
    print feature maps
    ::param: features
    """
    kwargs = {'normalize' : True, } # 'scale_each' : True
    
    for i in range(len(features)):
        fn = path + '{}.png'.format(i)
        # save_image(features[i].permute(1,0,2,3), fn, **kwargs)
        save_image_saliancy(features[i].permute(1,0,2,3), fn, normalize=True)
    # pdb.set_trace()
    ###

def plot_features(features, path='f', isList=True):
    """
    print feature maps
    :param features: (3, [batch, H, W]) or [3, batch, H, W]
    :param path: save image path
    :param isList: wether the features is an list
    :return:
    """
    kwargs = {'normalize' : True, } # 'scale_each' : True
    
    if isList:
        dim = features[0].dim()
    else:
        dim = features.dim()
    assert(dim==3 or dim==4)

    if dim==4 and isList:
        print_feature_examples(features, path)
    elif dim==4 and (isList==False):
        fn = path
        lvl, b, H, W = features.shape
        for i in range(features.shape[0]):
            fn = path + '{}.png'.format(i)
            save_image_saliancy(features[i][None,...].permute(1,0,2,3).cpu(), fn, normalize=True)

        # # concat everything
        # features = features.reshape([-1, H, W])
        # # save_image(features[None,...].permute(1,0,2,3).cpu(), fn, **kwargs)
        # save_image_saliancy(features[None,...].permute(1,0,2,3).cpu(), fn, normalize=True) 

    elif dim==3 and isList: # print all images in the list
        for i in range(len(features)):
            fn = path + '{}.png'.format(i)
            # save_image(features[i][None,...].permute(1,0,2,3).cpu(), fn, **kwargs)
            save_image_saliancy(features[i][None,...].permute(1,0,2,3).cpu(), fn, normalize=True)
    elif dim==3 and (isList==False):
            fn = path
            save_image_saliancy(features[None,...].permute(1,0,2,3).cpu(), fn, normalize=True)

def sample_homography_np(
        shape, shift=0, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
        perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=pi/2,
        allow_artifacts=False, translation_overflow=0.):
    """Sample a random valid homography.

    Computes the homography transformation between a random patch in the original image
    and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
    transformed input point (original patch).
    The original patch, which is initialized with a simple half-size centered crop, is
    iteratively projected, scaled, rotated and translated.

    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography. (like crop size)
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.

    Returns:
        A `Tensor` of shape `[1, 8]` corresponding to the flattened homography transform.
    """

    # print("debugging")


    # Corners of the output image
    pts1 = np.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + np.array([[0, 0], [0, patch_ratio],
                                 [patch_ratio, patch_ratio], [patch_ratio, 0]])

    from numpy.random import normal
    from numpy.random import uniform
    from scipy.stats import truncnorm

    # Random perspective and affine perturbations
    # lower, upper = 0, 2
    std_trunc = 2
    # pdb.set_trace()
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        perspective_displacement = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_y/2).rvs(1)
        h_displacement_left = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        h_displacement_right = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = truncnorm(-1*std_trunc, std_trunc, loc=1, scale=scaling_amplitude/2).rvs(n_scales)
        scales = np.concatenate((np.array([1]), scales), axis=0)

        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if allow_artifacts:
            valid = np.arange(n_scales)  # all scales are valid except scale=1
        else:
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx,:,:]

    # Random translation
    if translation:
        # pdb.set_trace()
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += np.array([uniform(-t_min[0], t_max[0],1), uniform(-t_min[1], t_max[1], 1)]).T

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, num=n_angles)
        angles = np.concatenate((angles, np.array([0.])), axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul( (pts2 - center)[np.newaxis,:,:], rot_mat) + center
        if allow_artifacts:
            valid = np.arange(n_angles)  # all scales are valid except scale=1
        else: # find multiple valid option and choose the valid one
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx,:,:]

    # Rescale to actual size
    shape = shape[::-1]  # different convention [y, x]
    pts1 *= shape[np.newaxis,:]
    pts2 *= shape[np.newaxis,:]

    homography = cv2.getPerspectiveTransform(np.float32(pts1+shift), np.float32(pts2+shift))
    return homography

def warp_points(points, homographies, device='cpu'):
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2(x, y))).
        homography: batched or not (shapes (B, 3, 3) and (...) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2(x, y)) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.

    """
    # expand points len to (x, y, 1)
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies

    batch_size = homographies.shape[0]
    points = torch.cat((points.float(), torch.ones((points.shape[0], 1)).to(device)), dim=1)
    points = points.to(device)
    homographies = homographies.view(batch_size*3,3)

    warped_points = homographies@points.transpose(0,1)

    # normalize the points
    warped_points = warped_points.view([batch_size, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points[0,:,:] if no_batches else warped_points

def inv_warp_image_batch(img, mat_homo_inv, device='cpu', mode='bilinear'):
    '''
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [batch_size, 1, H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [batch_size, 3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [batch_size, 1, H, W]
    '''
    # compute inverse warped points
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1,1,img.shape[0], img.shape[1])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1,3,3)

    Batch, channel, H, W = img.shape
    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H), indexing='ij'), dim=2)
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous()

    src_pixel_coords = warp_points(coor_cells.view([-1, 2]), mat_homo_inv, device)
    src_pixel_coords = src_pixel_coords.view([Batch, H, W, 2])
    src_pixel_coords = src_pixel_coords.float()

    warped_img = F.grid_sample(img, src_pixel_coords, mode=mode, align_corners=True)
    return warped_img

def compute_valid_mask(image_shape, inv_homography, device='cpu', erosion_radius=0):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.

    Arguments:
        input_shape: Tensor of rank 2 representing the image shape, i.e. `[H, W]`.
        homography: Tensor of shape (B, 8) or (8,), where B is the batch size.
        `erosion_radius: radius of the margin to be discarded.

    Returns: a Tensor of type `tf.int32` and shape (H, W).
    """

    if inv_homography.dim() == 2:
        inv_homography = inv_homography.view(-1, 3, 3)
    batch_size = inv_homography.shape[0]
    mask = torch.ones(batch_size, 1, image_shape[0], image_shape[1]).to(device)
    mask = inv_warp_image_batch(mask, inv_homography, device=device, mode='nearest')
    mask = mask.view(batch_size, image_shape[0], image_shape[1])
    mask = mask.cpu().numpy()
    if erosion_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius*2,)*2)
        for i in range(batch_size):
            mask[i, :, :] = cv2.erode(mask[i, :, :], kernel, iterations=1)

    return torch.tensor(mask).to(device)

def Kalman1D(observations,damping=1):
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1
    initial_value_guess
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state

def Kalman3D(observations,damping=1):
    '''
    In:
    observation: Nx3
    Out:
    pred_state: Nx3
    '''
    # To return the smoothed time series data
    observation_covariance = damping
    transition_matrix = 1
    transition_covariance = 0.1
    initial_value_guess_x = observations[0,0]
    initial_value_guess_y = observations[0,1] # ?
    initial_value_guess_z = observations[0,2] # ?
    
    # perform 1D smooth for each axis
    kfx = KalmanFilter(
            initial_state_mean=initial_value_guess_x,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state_x, state_cov_x = kfx.smooth(observations[:, 0])
    
    kfy = KalmanFilter(
            initial_state_mean=initial_value_guess_y,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state_y, state_cov_y = kfy.smooth(observations[:, 1])
    
    kfz = KalmanFilter(
            initial_state_mean=initial_value_guess_z,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state_z, state_cov_z = kfy.smooth(observations[:, 2])
    
    pred_state = np.concatenate((pred_state_x, pred_state_y, pred_state_z), axis=1)
    return pred_state