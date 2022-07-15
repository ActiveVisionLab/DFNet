import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from typing import List

# VGG-16 Layer Names and Channels
vgg16_layers = {
    "conv1_1": 64,
    "relu1_1": 64,
    "conv1_2": 64,
    "relu1_2": 64,
    "pool1": 64,
    "conv2_1": 128,
    "relu2_1": 128,
    "conv2_2": 128,
    "relu2_2": 128,
    "pool2": 128,
    "conv3_1": 256,
    "relu3_1": 256,
    "conv3_2": 256,
    "relu3_2": 256,
    "conv3_3": 256,
    "relu3_3": 256,
    "pool3": 256,
    "conv4_1": 512,
    "relu4_1": 512,
    "conv4_2": 512,
    "relu4_2": 512,
    "conv4_3": 512,
    "relu4_3": 512,
    "pool4": 512,
    "conv5_1": 512,
    "relu5_1": 512,
    "conv5_2": 512,
    "relu5_2": 512,
    "conv5_3": 512,
    "relu5_3": 512,
    "pool5": 512,
}

class AdaptLayers(nn.Module):
    """Small adaptation layers.
    """

    def __init__(self, hypercolumn_layers: List[str], output_dim: int = 128):
        """Initialize one adaptation layer for every extraction point.

        Args:
            hypercolumn_layers: The list of the hypercolumn layer names.
            output_dim: The output channel dimension.
        """
        super(AdaptLayers, self).__init__()
        self.layers = []
        channel_sizes = [vgg16_layers[name] for name in hypercolumn_layers]
        for i, l in enumerate(channel_sizes):
            layer = nn.Sequential(
                nn.Conv2d(l, 64, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, output_dim, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(output_dim),
            )
            self.layers.append(layer)
            self.add_module("adapt_layer_{}".format(i), layer) # ex: adapt_layer_0

    def forward(self, features: List[torch.tensor]):
        """Apply adaptation layers. # here is list of three levels of features
        """

        for i, _ in enumerate(features):
            features[i] = getattr(self, "adapt_layer_{}".format(i))(features[i])
        return features

class DFNet(nn.Module):
    ''' DFNet implementation '''
    default_conf = {
        'hypercolumn_layers': ["conv1_2", "conv3_3", "conv5_3"],
        'output_dim': 128,
    }
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, feat_dim=12, places365_model_path=''):
        super(DFNet, self).__init__()

        self.layer_to_index = {k: v for v, k in enumerate(vgg16_layers.keys())}
        self.hypercolumn_indices = [self.layer_to_index[n] for n in self.default_conf['hypercolumn_layers']] # [2, 14, 28]

        # Initialize architecture
        vgg16 = models.vgg16(pretrained=True)

        self.encoder = nn.Sequential(*list(vgg16.features.children()))

        self.scales = []
        current_scale = 0
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, torch.nn.MaxPool2d):
                current_scale += 1
            if i in self.hypercolumn_indices:
                self.scales.append(2**current_scale)

        ## adaptation layers, see off branches from fig.3 in S2DNet paper
        self.adaptation_layers = AdaptLayers(self.default_conf['hypercolumn_layers'], self.default_conf['output_dim'])

        # pose regression layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_pose = nn.Linear(512, feat_dim)

    def forward(self, x, return_feature=False, isSingleStream=False, return_pose=True, upsampleH=240, upsampleW=427):
        '''
        inference DFNet. It can regress camera pose as well as extract intermediate layer features.
            :param x: image blob (2B x C x H x W) two stream or (B x C x H x W) single stream
            :param return_feature: whether to return features as output
            :param isSingleStream: whether it's an single stream inference or siamese network inference
            :param upsampleH: feature upsample size H
            :param upsampleW: feature upsample size W
            :return feature_maps: (2, [B, C, H, W]) or (1, [B, C, H, W]) or None
            :return predict: [2B, 12] or [B, 12]
        '''
        # normalize input data
        mean, std = x.new_tensor(self.mean), x.new_tensor(self.std)
        x = (x - mean[:, None, None]) / std[:, None, None]

        ### encoder ###
        feature_maps = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)

            if i in self.hypercolumn_indices:
                feature = x.clone()
                feature_maps.append(feature)

                if i==self.hypercolumn_indices[-1]:
                    if return_pose==False:
                        predict = None
                        break

        ### extract and process intermediate features ###
        if return_feature:
            feature_maps = self.adaptation_layers(feature_maps) # (3, [B, C, H', W']), H', W' are different in each layer

            if isSingleStream: # not siamese network style inference
                feature_stacks = []
                for f in feature_maps:
                    feature_stacks.append(torch.nn.UpsamplingBilinear2d(size=(upsampleH, upsampleW))(f))
                feature_maps = [torch.stack(feature_stacks)] # (1, [3, B, C, H, W])
            else: # siamese network style inference
                feature_stacks_t = []
                feature_stacks_r = []
                for f in feature_maps:
                    # split real and nerf batches
                    batch = f.shape[0] # should be target batch_size + rgb batch_size
                    feature_t = f[:batch//2]
                    feature_r = f[batch//2:]

                    feature_stacks_t.append(torch.nn.UpsamplingBilinear2d(size=(upsampleH, upsampleW))(feature_t)) # GT img
                    feature_stacks_r.append(torch.nn.UpsamplingBilinear2d(size=(upsampleH, upsampleW))(feature_r)) # render img
                feature_stacks_t = torch.stack(feature_stacks_t) # [3, B, C, H, W]
                feature_stacks_r = torch.stack(feature_stacks_r) # [3, B, C, H, W]
                feature_maps = [feature_stacks_t, feature_stacks_r] # (2, [3, B, C, H, W])
        else:
            feature_maps = None

        if return_pose==False:
            return feature_maps, predict

        ### pose regression head ###
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        predict = self.fc_pose(x)
                
        return feature_maps, predict

class DFNet_s(nn.Module):
    ''' A slight accelerated version of DFNet, we experimentally found this version's performance is similar to original DFNet but inferences faster '''
    default_conf = {
        'hypercolumn_layers': ["conv1_2"],
        'output_dim': 128,
    }
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, feat_dim=12, places365_model_path=''):
        super(DFNet_s, self).__init__()

        self.layer_to_index = {k: v for v, k in enumerate(vgg16_layers.keys())}
        self.hypercolumn_indices = [self.layer_to_index[n] for n in self.default_conf['hypercolumn_layers']] # [2, 14, 28]

        # Initialize architecture
        vgg16 = models.vgg16(pretrained=True)

        self.encoder = nn.Sequential(*list(vgg16.features.children()))

        self.scales = []
        current_scale = 0
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, torch.nn.MaxPool2d):
                current_scale += 1
            if i in self.hypercolumn_indices:
                self.scales.append(2**current_scale)

        ## adaptation layers, see off branches from fig.3 in S2DNet paper
        self.adaptation_layers = AdaptLayers(self.default_conf['hypercolumn_layers'], self.default_conf['output_dim'])

        # pose regression layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_pose = nn.Linear(512, feat_dim)

    def forward(self, x, return_feature=False, isSingleStream=False, return_pose=True, upsampleH=240, upsampleW=427):
        '''
        inference DFNet_s. It can regress camera pose as well as extract intermediate layer features.
            :param x: image blob (2B x C x H x W) two stream or (B x C x H x W) single stream
            :param return_feature: whether to return features as output
            :param isSingleStream: whether it's an single stream inference or siamese network inference
            :param upsampleH: feature upsample size H
            :param upsampleW: feature upsample size W
            :return feature_maps: (2, [B, C, H, W]) or (1, [B, C, H, W]) or None
            :return predict: [2B, 12] or [B, 12]
        '''

        # normalize input data
        mean, std = x.new_tensor(self.mean), x.new_tensor(self.std)
        x = (x - mean[:, None, None]) / std[:, None, None]

        ### encoder ###
        feature_maps = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)

            if i in self.hypercolumn_indices:
                feature = x.clone()
                feature_maps.append(feature)

                if i==self.hypercolumn_indices[-1]:
                    if return_pose==False:
                        predict = None
                        break

        ### extract and process intermediate features ###
        if return_feature:
            feature_maps = self.adaptation_layers(feature_maps) # (3, [B, C, H', W']), H', W' are different in each layer

            if isSingleStream: # not siamese network style inference
                feature_stacks = []
                for f in feature_maps:
                    feature_stacks.append(torch.nn.UpsamplingBilinear2d(size=(upsampleH, upsampleW))(f))
                feature_maps = [torch.stack(feature_stacks)] # (1, [3, B, C, H, W])
            else: # siamese network style inference
                feature_stacks_t = []
                feature_stacks_r = []
                for f in feature_maps:
                    # split real and nerf batches
                    batch = f.shape[0] # should be target batch_size + rgb batch_size
                    feature_t = f[:batch//2]
                    feature_r = f[batch//2:]

                    feature_stacks_t.append(torch.nn.UpsamplingBilinear2d(size=(upsampleH, upsampleW))(feature_t)) # GT img
                    feature_stacks_r.append(torch.nn.UpsamplingBilinear2d(size=(upsampleH, upsampleW))(feature_r)) # render img
                feature_stacks_t = torch.stack(feature_stacks_t) # [3, B, C, H, W]
                feature_stacks_r = torch.stack(feature_stacks_r) # [3, B, C, H, W]
                feature_maps = [feature_stacks_t, feature_stacks_r] # (2, [3, B, C, H, W])
        else:
            feature_maps = None

        if return_pose==False:
            return feature_maps, predict

        ### pose regression head ###
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        predict = self.fc_pose(x)
                
        return feature_maps, predict