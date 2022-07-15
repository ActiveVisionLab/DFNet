import torch
from torch import nn
import torch.nn.functional as F
import copy, pdb
from typing import List
from efficientnet_pytorch import EfficientNet

# efficientnet-B3 Layer Names and Channels
EB3_layers = {
    "reduction_1": 24, # torch.Size([2, 24, 120, 213])
    "reduction_2": 32, # torch.Size([2, 32, 60, 106])
    "reduction_3": 48, # torch.Size([2, 48, 30, 53])
    "reduction_4": 136, # torch.Size([2, 136, 15, 26])
    "reduction_5": 384, # torch.Size([2, 384, 8, 13])
    "reduction_6": 1536, # torch.Size([2, 1536, 8, 13])
}

# efficientnet-B0 Layer Names and Channels
EB0_layers = {
    "reduction_1": 16, # torch.Size([2, 16, 120, 213])
    "reduction_2": 24, # torch.Size([2, 24, 60, 106])
    "reduction_3": 40, # torch.Size([2, 40, 30, 53])
    "reduction_4": 112, # torch.Size([2, 112, 15, 26])
    "reduction_5": 320, # torch.Size([2, 320, 8, 13])
    "reduction_6": 1280, # torch.Size([2, 1280, 8, 13])
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
        channel_sizes = [EB3_layers[name] for name in hypercolumn_layers]
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

class EfficientNetB3(nn.Module):
    ''' DFNet with EB3 backbone '''
    default_conf = {
        # 'hypercolumn_layers': ["reduction_1", "reduction_3", "reduction_6"],
        'hypercolumn_layers': ["reduction_1", "reduction_3", "reduction_5"],
        # 'hypercolumn_layers': ["reduction_2", "reduction_4", "reduction_6"],
        'output_dim': 128,
    }
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, feat_dim=12, places365_model_path=''):
        super(EfficientNetB3, self).__init__()
        # Initialize architecture
        self.backbone_net = EfficientNet.from_pretrained('efficientnet-b3')
        self.feature_extractor = self.backbone_net.extract_endpoints

        # self.feature_block_index = [1, 3, 6] # same as the 'hypercolumn_layers'
        self.feature_block_index = [1, 3, 5] # same as the 'hypercolumn_layers'
        # self.feature_block_index = [2, 4, 6] # same as the 'hypercolumn_layers'

        ## adaptation layers, see off branches from fig.3 in S2DNet paper
        self.adaptation_layers = AdaptLayers(self.default_conf['hypercolumn_layers'], self.default_conf['output_dim'])

        # pose regression layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_pose = nn.Linear(1536, feat_dim)

    def forward(self, x, return_feature=False, isSingleStream=False, upsampleH=120, upsampleW=213):
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
        list_x = self.feature_extractor(x)

        x = list_x['reduction_6'] # features to save
        for i in self.feature_block_index:
            fe = list_x['reduction_'+str(i)].clone()
            feature_maps.append(fe)

        ### extract and process intermediate features ###
        if return_feature:
            feature_maps = self.adaptation_layers(feature_maps) # (3, [B, C, H', W']), H', W' are different in each layer

            pdb.set_trace()
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

        ### pose regression head ###
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        predict = self.fc_pose(x)
                
        return feature_maps, predict

class AdaptLayers2(nn.Module):
    """Small adaptation layers.
    """

    def __init__(self, hypercolumn_layers: List[str], output_dim: int = 128):
        """Initialize one adaptation layer for every extraction point.

        Args:
            hypercolumn_layers: The list of the hypercolumn layer names.
            output_dim: The output channel dimension.
        """
        super(AdaptLayers2, self).__init__()
        self.layers = []
        channel_sizes = [EB0_layers[name] for name in hypercolumn_layers]
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

class EfficientNetB0(nn.Module):
    ''' DFNet with EB0 backbone, feature levels can be customized '''
    default_conf = {
        # 'hypercolumn_layers': ["reduction_1", "reduction_3", "reduction_6"],
        'hypercolumn_layers': ["reduction_1", "reduction_3", "reduction_5"],
        # 'hypercolumn_layers': ["reduction_2", "reduction_4", "reduction_6"],
        # 'hypercolumn_layers': ["reduction_1"],
        'output_dim': 128,
    }
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, feat_dim=12, places365_model_path=''):
        super(EfficientNetB0, self).__init__()
        # Initialize architecture
        self.backbone_net = EfficientNet.from_pretrained('efficientnet-b0')
        self.feature_extractor = self.backbone_net.extract_endpoints

        # self.feature_block_index = [1, 3, 6] # same as the 'hypercolumn_layers'
        self.feature_block_index = [1, 3, 5] # same as the 'hypercolumn_layers'
        # self.feature_block_index = [2, 4, 6] # same as the 'hypercolumn_layers'
        # self.feature_block_index = [1]

        ## adaptation layers, see off branches from fig.3 in S2DNet paper
        self.adaptation_layers = AdaptLayers2(self.default_conf['hypercolumn_layers'], self.default_conf['output_dim'])

        # pose regression layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_pose = nn.Linear(1280, feat_dim)

    def forward(self, x, return_feature=False, isSingleStream=False, return_pose=False, upsampleH=120, upsampleW=213):
        '''
        inference DFNet. It can regress camera pose as well as extract intermediate layer features.
            :param x: image blob (2B x C x H x W) two stream or (B x C x H x W) single stream
            :param return_feature: whether to return features as output
            :param isSingleStream: whether it's an single stream inference or siamese network inference
            :param return_pose: TODO: if only return_pose, we don't need to compute return_feature part
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
        list_x = self.feature_extractor(x)

        x = list_x['reduction_6'] # features to save
        for i in self.feature_block_index:
            fe = list_x['reduction_'+str(i)].clone()
            feature_maps.append(fe)

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

        ### pose regression head ###
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        predict = self.fc_pose(x)
                
        return feature_maps, predict

def main():
  """
  test model
  """
  from torchsummary import summary
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  feat_model = EfficientNetB3()
  # feat_model.load_state_dict(torch.load(''))
  feat_model.to(device)
  summary(feat_model, (3, 240, 427))

if __name__ == '__main__':
  main()
