import utils.set_sys_path
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import numpy as np

from dataset_loaders.load_llff import load_llff_data
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def prepare_data(args, images, poses_train, i_split):
    ''' prepare data for ready to train posenet, return dataloaders '''
    #TODO: Convert GPU friendly data generator later: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    #TODO: Probably a better implementation style here: https://github.com/PyTorchLightning/pytorch-lightning
    
    i_train, i_val, i_test = i_split

    img_train = torch.Tensor(images[i_train]).permute(0, 3, 1, 2) # now shape is [N, CH, H, W]
    pose_train = torch.Tensor(poses_train[i_train])
    
    trainset = TensorDataset(img_train, pose_train)
    train_dl = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    
    img_val = torch.Tensor(images[i_val]).permute(0, 3, 1, 2) # now shape is [N, CH, H, W]
    pose_val = torch.Tensor(poses_train[i_val])
    
    valset = TensorDataset(img_val, pose_val)
    val_dl = DataLoader(valset)

    img_test = torch.Tensor(images[i_test]).permute(0, 3, 1, 2) # now shape is [N, CH, H, W]
    pose_test = torch.Tensor(poses_train[i_test])

    testset = TensorDataset(img_test, pose_test)
    test_dl = DataLoader(testset)

    return train_dl, val_dl, test_dl

def load_dataset(args):
    ''' load posenet training data '''
    if args.dataset_type == 'llff':
        if args.no_bd_factor:
            bd_factor = None
        else:
            bd_factor = 0.75
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=bd_factor,
                                                                  spherify=args.spherify)

        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.

        if args.finetune_unlabel:
            i_train = i_test
        i_split = [i_train, i_val, i_test]
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    poses_train = poses[:,:3,:].reshape((poses.shape[0],12)) # get rid of last row [0,0,0,1]
    print("images.shape {}, poses_train.shape {}".format(images.shape, poses_train.shape))

    INPUT_SHAPE = images[0].shape
    print("=====================================================================")
    print("INPUT_SHAPE:", INPUT_SHAPE)
    return images, poses_train, render_poses, hwf, i_split, near, far