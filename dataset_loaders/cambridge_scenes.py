"""
pytorch data loader for the Cambridge Landmark dataset
"""
import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import sys
import pdb
import cv2
from dataset_loaders.utils.color import rgb_to_yuv
from torchvision.utils import save_image
import json

sys.path.insert(0, '../')
#from common.pose_utils import process_poses

## NUMPY
def qlog(q):
  """
  Applies logarithm map to q
  :param q: (4,)
  :return: (3,)
  """
  if all(q[1:] == 0):
    q = np.zeros(3)
  else:
    q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
  return q

def process_poses_rotmat(poses_in, rot_mat):
  """
  processes the position + quaternion raw pose from dataset to position + rotation matrix
  :param poses_in: N x 7
  :param rot_mat: N x 3 x 3
  :return: processed poses N x 12
  """

  poses = np.zeros((poses_in.shape[0], 3, 4))
  poses[:,:3,:3] = rot_mat
  poses[...,:3,3] = poses_in[:, :3]
  poses = poses.reshape(poses_in.shape[0], 12)
  return poses

from torchvision.datasets.folder import default_loader
def load_image(filename, loader=default_loader):
  try:
    img = loader(filename)
  except IOError as e:
    print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
    return None
  except:
    print('Could not load image {:s}, unexpected error'.format(filename))
    return None
  return img

def load_depth_image(filename):
  try:
    img_depth = Image.fromarray(np.array(Image.open(filename)).astype("uint16"))
  except IOError as e:
    print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
    return None
  return img_depth

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def normalize_recenter_pose(poses, sc, hwf):
  ''' normalize xyz into [-1, 1], and recenter pose '''
  target_pose = poses.reshape(poses.shape[0],3,4)
  target_pose[:,:3,3] = target_pose[:,:3,3] * sc


  x_norm = target_pose[:,0,3]
  y_norm = target_pose[:,1,3]
  z_norm = target_pose[:,2,3]

  tpose_ = target_pose+0

  # find the center of pose
  center = np.array([x_norm.mean(), y_norm.mean(), z_norm.mean()])
  bottom = np.reshape([0,0,0,1.], [1,4])

  # pose avg
  vec2 = normalize(tpose_[:, :3, 2].sum(0)) 
  up = tpose_[:, :3, 1].sum(0)
  hwf=np.array(hwf).transpose()
  c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
  c2w = np.concatenate([c2w[:3,:4], bottom], -2)

  bottom = np.tile(np.reshape(bottom, [1,1,4]), [tpose_.shape[0],1,1])
  poses = np.concatenate([tpose_[:,:3,:4], bottom], -2)
  poses = np.linalg.inv(c2w) @ poses
  return poses[:,:3,:].reshape(poses.shape[0],12)

def downscale_pose(poses, sc):
  ''' downscale translation pose to [-1:1] only '''
  target_pose = poses.reshape(poses.shape[0],3,4)
  target_pose[:,:3,3] = target_pose[:,:3,3] * sc
  return target_pose.reshape(poses.shape[0],12)

class Cambridge2(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None,
                 target_transform=None, mode=0, seed=7,
                 skip_images=False, df=2., trainskip=1, testskip=1, hwf=[480,854,744.],
                 ret_idx=False, fix_idx=False, ret_hist=False, hist_bin=10):
      """
      :param scene: scene name ['chess', 'pumpkin', ...]
      :param data_path: root 7scenes data directory.
      Usually '../data/deepslam_data/7Scenes'
      :param train: if True, return the training images. If False, returns the
      testing images
      :param transform: transform to apply to the images
      :param target_transform: transform to apply to the poses
      :param mode: (Obsolete) 0: just color image, 1: color image in NeRF 0-1 and resized
      :param skip_images: If True, skip loading images and return None instead
      :param df: downscale factor
      :param trainskip: due to 7scenes are so big, now can use less training sets # of trainset = 1/trainskip
      :param testskip: skip part of testset, # of testset = 1/testskip
      :param hwf: H,W,Focal from COLMAP
      """

      self.transform = transform
      self.target_transform = target_transform
      self.df = df

      self.H, self.W, self.focal = hwf
      self.H = int(self.H)
      self.W = int(self.W)
      np.random.seed(seed)

      self.train = train
      self.ret_idx = ret_idx
      self.fix_idx = fix_idx
      self.ret_hist = ret_hist
      self.hist_bin = hist_bin # histogram bin size

      if self.train:
        root_dir = osp.join(data_path, scene) + '/train'
      else:
        root_dir = osp.join(data_path, scene) + '/test'
      
      rgb_dir = root_dir + '/rgb/'
      
      pose_dir =  root_dir + '/poses/'

      world_setup_fn = osp.join(data_path, scene) + '/world_setup.json'

      # collect poses and image names
      self.rgb_files = os.listdir(rgb_dir)
      self.rgb_files = [rgb_dir + f for f in self.rgb_files]
      self.rgb_files.sort()

      self.pose_files = os.listdir(pose_dir)
      self.pose_files = [pose_dir + f for f in self.pose_files]
      self.pose_files.sort()

      # remove some abnormal data, need to fix later
      if scene == 'ShopFacade' and self.train:
        del self.rgb_files[42]
        del self.rgb_files[35]
        del self.pose_files[42]
        del self.pose_files[35]

      if len(self.rgb_files) != len(self.pose_files):
        raise Exception('RGB file count does not match pose file count!')

      # read json file
      with open(world_setup_fn, 'r') as myfile:
        data=myfile.read()

      # parse json file
      obj = json.loads(data)
      self.near = obj['near']
      self.far = obj['far']
      self.pose_scale = obj['pose_scale']
      self.pose_scale2 = obj['pose_scale2']
      self.move_all_cam_vec = obj['move_all_cam_vec']

      # trainskip and testskip
      frame_idx = np.arange(len(self.rgb_files))
      if train and trainskip > 1:
        frame_idx_tmp = frame_idx[::trainskip]
        frame_idx = frame_idx_tmp
      elif not train and testskip > 1:
        frame_idx_tmp = frame_idx[::testskip]
        frame_idx = frame_idx_tmp
      self.gt_idx = frame_idx

      self.rgb_files = [self.rgb_files[i] for i in frame_idx]
      self.pose_files = [self.pose_files[i] for i in frame_idx]
      
      if len(self.rgb_files) != len(self.pose_files):
        raise Exception('RGB file count does not match pose file count!')

      # read poses
      poses = []
      for i in range(len(self.pose_files)):
        pose = np.loadtxt(self.pose_files[i])
        poses.append(pose)
      poses = np.array(poses) # [N, 4, 4]
      self.poses = poses[:, :3, :4].reshape(poses.shape[0], 12)

      # debug read one img and get the shape of the img
      img = load_image(self.rgb_files[0])
      img_np = (np.array(img) / 255.).astype(np.float32) # (480,854,3)

      self.H, self.W = img_np.shape[:2]
      if self.df != 1.:
        self.H = int(self.H//self.df)
        self.W = int(self.W//self.df)
        self.focal = self.focal/self.df

    def __len__(self):
      return len(self.rgb_files)

    def __getitem__(self, index):
      img = load_image(self.rgb_files[index])
      pose = self.poses[index]
      if self.df != 1.:
        img_np = (np.array(img) / 255.).astype(np.float32)
        dims = (self.W, self.H)
        img_half_res = cv2.resize(img_np, dims, interpolation=cv2.INTER_AREA) # (H, W, 3)
        img = img_half_res

      if self.transform is not None:
        img = self.transform(img)

      if self.target_transform is not None:
        pose = self.target_transform(pose)

      if self.ret_idx:
        if self.train and self.fix_idx==False:
          return img, pose, index
        else:
          return img, pose, 0
      if self.ret_hist:
        yuv = rgb_to_yuv(img)
        y_img = yuv[0] # extract y channel only
        hist = torch.histc(y_img, bins=self.hist_bin, min=0., max=1.) # compute intensity histogram
        hist = hist/(hist.sum())*100 # convert to histogram density, in terms of percentage per bin
        hist = torch.round(hist)
        return img, pose, hist

      return img, pose

def main():
  """
  visualizes the dataset
  """
  #from common.vis_utils import show_batch, show_stereo_batch
  from torchvision.utils import make_grid
  import torchvision.transforms as transforms
  seq = 'ShopFacade'
  mode = 1
  # num_workers = 1

  # transformer
  data_transform = transforms.Compose([
        transforms.ToTensor(),
        ])
  target_transform = transforms.Lambda(lambda x: torch.Tensor(x))
  kwargs = dict(ret_hist=True)
  dset = Cambridge2(seq, '../data/Cambridge/', True, data_transform, target_transform=target_transform, mode=mode, df=7.15, trainskip=2, **kwargs)
  print('Loaded Cambridge sequence {:s}, length = {:d}'.format(seq, len(dset)))

  data_loader = data.DataLoader(dset, batch_size=4, shuffle=False)

  batch_count = 0
  N = 2
  for batch in data_loader:
    print('Minibatch {:d}'.format(batch_count))
    pdb.set_trace()

    batch_count += 1
    if batch_count >= N:
      break

if __name__ == '__main__':
  main()
