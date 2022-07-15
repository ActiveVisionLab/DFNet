import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import copy, pdb
from efficientnet_pytorch import EfficientNet
from typing import List

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # # output layer (with tanh for scaling from -1 to 1)
        x = F.tanh(self.t_conv2(x))
        # output layer (with tanh for scaling from 0 to 1)
        # x = F.sigmoid(self.t_conv2(x))
                
        return x

class autoencoder_vgg1(nn.Module): # psnr 20.84
    def __init__(self):
        super(autoencoder_vgg1, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            # nn.Sigmoid() # if img is 0 to 1
            nn.Tanh() # if img is -1 to 1
        )
    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

class autoencoder_vgg2(nn.Module): # psnr 25.41
    ''' add skip connections '''
    def __init__(self):
        super(autoencoder_vgg2, self).__init__()
        conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 7)
        )
        self.encoder = nn.Sequential(
            conv1, conv2, conv3
        )

        deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(inplace=True),
        )
        deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        deconv3 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(deconv1, deconv2, deconv3)
    def forward(self, x):
        feat1 = self.encoder[0](x)
        feat2 = self.encoder[1](feat1)
        x = self.encoder[2](feat2)

        x = self.decoder[0](x)
        x = x + feat2
        x = self.decoder[1](x)
        x = x + feat1
        x = self.decoder[2](x)
        return None, x

class autoencoder_vgg3(nn.Module): # psnr: 37.77
    ''' vgg encoder '''
    def __init__(self):
        super(autoencoder_vgg3, self).__init__()
        self.encoder = models.vgg19(pretrained=True).features
        # receptive field not equal? so maybe it does not work very well
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 2, stride=2), # (b, 512, 14, 14)
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, stride=4), # (b, 256, 56, 56)
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4, stride=4), # (b, 64, 224, 224)
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh() # MSELoss
            # nn.Sigmoid() # BCELoss
        )

    def forward(self, x):
        feat = []
        feat_out = []
        for i in range(len(self.encoder)):
            # print("layer {} encoder layer: {}".format(i, self.encoder[i]))
            x = self.encoder[i](x)
            if i == 35: # ReLU-36
                feat.append(x)
            elif i == 17: # ReLU-17
                feat.append(x)
            elif i == 3: # ReLU-4
                feat.append(x)      
        for i in range(len(self.decoder)):
            # print("layer {} decoder layer: {}".format(i, self.decoder[i]))
            x = self.decoder[i](x)
            if i == 1:
                x = x + feat[2]
                feat_out.append(x)
            elif i == 3:
                x = x + feat[1]
                feat_out.append(x)
            elif i == 5:
                x = x + feat[0]
                feat_out.append(x)
        return feat_out, x

class autoencoder_vgg4(nn.Module): # 35.54 PSNR 36.05 BCELoss (120x120)
    ''' vgg encoder with bilinear upsampling'''
    def __init__(self):
        super(autoencoder_vgg4, self).__init__()
        self.encoder = models.vgg19(pretrained=True).features
        # receptive field not equal? so maybe it does not work very well
        self.decoder = nn.Sequential(
            # (b, 512, 14, 14)
            # nn.UpsamplingBilinear2d(scale_factor=2), # upsample to feature map's size
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(True),
            # (b, 256, 56, 56)
            # nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            # (b, 64, 224, 224)
            # nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Conv2d(256, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            # nn.Tanh() # MSELoss
            nn.Sigmoid() # BCELoss
        )
    def forward(self, x):
        # pdb.set_trace()
        feat = []
        feat_out = []
        for i in range(len(self.encoder)):
            # print("layer {} encoder layer: {}".format(i, self.encoder[i]))
            x = self.encoder[i](x)
            if i == 35: # ReLU-36
                feat.append(x)
            elif i == 17: # ReLU-17
                feat.append(x)
            elif i == 3: # ReLU-4
                feat.append(x)      

        for i in range(len(self.decoder)):
            # print("layer {} decoder layer: {}".format(i, self.decoder[i]))
            x = self.decoder[i](x)
            if i == 1:
                _, _, h, w = feat[2].shape
                x = nn.UpsamplingBilinear2d(size=(h,w))(x)
                x = x + feat[2]
                feat_out.append(x)
            elif i == 3:
                _, _, h, w = feat[1].shape
                x = nn.UpsamplingBilinear2d(size=(h,w))(x)
                x = x + feat[1]
                feat_out.append(x)
            elif i == 5:
                _, _, h, w = feat[0].shape
                x = nn.UpsamplingBilinear2d(size=(h,w))(x)
                x = x + feat[0]
                feat_out.append(x)
        return feat_out, x

class autoencoder_vgg5(nn.Module): # 36.78 PSNR
    ''' vgg encoder with bilinear upsampling'''
    def __init__(self):
        super(autoencoder_vgg5, self).__init__()
        self.encoder = models.vgg19(pretrained=True).features
        self.decoder = nn.Sequential(
            # (b, 512, 14, 14)
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(True),
            # (b, 512, 28, 28)
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(True),
            # (b, 256, 56, 56)
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            # (b, 128, 112, 112)
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            # (b, 64, 224, 224)
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            # nn.Tanh() # MSELoss
            nn.Sigmoid() # BCELoss
        )
    def forward(self, x):
        # pdb.set_trace()
        feat = []
        feat_out = [] # we only use high level features
        for i in range(len(self.encoder)):
            # print("layer {} encoder layer: {}".format(i, self.encoder[i]))
            x = self.encoder[i](x)
            if i == 3: # ReLU-4
                # pdb.set_trace()
                feat.append(x)
            elif i == 8: # ReLU-9
                # pdb.set_trace()
                feat.append(x)
            elif i == 17: # ReLU-18
                # pdb.set_trace()
                feat.append(x)
            elif i == 26: # ReLU-27
                # pdb.set_trace()
                feat.append(x)
            elif i == 35: # ReLU-36
                # pdb.set_trace()
                feat.append(x)    
        # pdb.set_trace()
        for i in range(len(self.decoder)):
            # print("layer {} decoder layer: {}".format(i, self.decoder[i]))
            x = self.decoder[i](x)
            if i == 1:
                # pdb.set_trace()
                _, _, h, w = feat[4].shape
                x = nn.UpsamplingBilinear2d(size=(h,w))(x)
                x = x + feat[4]
            elif i == 3:
                # pdb.set_trace()
                _, _, h, w = feat[3].shape
                x = nn.UpsamplingBilinear2d(size=(h,w))(x)
                x = x + feat[3]
            elif i == 5:
                # pdb.set_trace()
                _, _, h, w = feat[2].shape
                x = nn.UpsamplingBilinear2d(size=(h,w))(x)
                x = x + feat[2]
                feat_out.append(x)
            elif i == 7:
                # pdb.set_trace()
                _, _, h, w = feat[1].shape
                x = nn.UpsamplingBilinear2d(size=(h,w))(x)
                x = x + feat[1]
                feat_out.append(x)
            elif i == 9:
                # pdb.set_trace()
                _, _, h, w = feat[0].shape
                x = nn.UpsamplingBilinear2d(size=(h,w))(x)
                x = x + feat[0]
                feat_out.append(x)
        return feat_out, x

class autoencoder_vgg6(nn.Module): # robust feature extractors
    ''' vgg encoder with bilinear upsampling'''
    def __init__(self):
        super(autoencoder_vgg6, self).__init__()
        self.encoder = models.vgg19(pretrained=True).features
        self.decoder = nn.Sequential(
            # (b, 512, 14, 14)
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(True),
            # (b, 512, 28, 28)
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(True),
            # (b, 256, 56, 56)
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            # (b, 128, 112, 112)
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            # (b, 64, 224, 224)
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            # nn.Conv2d(64, 3, 3, stride=1, padding=1),
            # nn.Tanh() # MSELoss
            # nn.Sigmoid() # BCELoss
        )

    def forward(self, x, upsampleH, upsampleW): #
        feat = []
        feat_out = [] # we only use high level features
        for i in range(len(self.encoder)):
            # print("layer {} encoder layer: {}".format(i, self.encoder[i]))
            x = self.encoder[i](x)
            if i == 3: # ReLU-4
                feat.append(x)
            elif i == 8: # ReLU-9
                feat.append(x)
            elif i == 17: # ReLU-18
                feat.append(x)
            elif i == 26: # ReLU-27
                feat.append(x)
            elif i == 35: # ReLU-36
                feat.append(x)

        for i in range(len(self.decoder)):
            # print("layer {} decoder layer: {}".format(i, self.decoder[i]))
            x = self.decoder[i](x)
            if i == 1:
                _, _, h, w = feat[4].shape
                x = nn.UpsamplingBilinear2d(size=(h,w))(x)
                x = x + feat[4]
            elif i == 3:
                _, _, h, w = feat[3].shape
                x = nn.UpsamplingBilinear2d(size=(h,w))(x)
                x = x + feat[3]
            elif i == 5:
                _, _, h, w = feat[2].shape
                x = nn.UpsamplingBilinear2d(size=(h,w))(x)
                x = x + feat[2]
                feature = torch.mean(torch.nn.UpsamplingBilinear2d(size=(upsampleH, upsampleW))(x), dim=1)
                feat_out.append(feature)
            elif i == 7:
                _, _, h, w = feat[1].shape
                x = nn.UpsamplingBilinear2d(size=(h,w))(x)
                x = x + feat[1]
                feature = torch.mean(torch.nn.UpsamplingBilinear2d(size=(upsampleH, upsampleW))(x), dim=1)
                feat_out.append(feature)
            elif i == 9:
                _, _, h, w = feat[0].shape
                x = nn.UpsamplingBilinear2d(size=(h,w))(x)
                x = x + feat[0]
                feature = torch.mean(torch.nn.UpsamplingBilinear2d(size=(upsampleH, upsampleW))(x), dim=1)
                feat_out.append(feature)
        return feat_out, x

class autoencoder_vgg7(nn.Module): # no decoder
    ''' vgg encoder with bilinear upsampling'''
    def __init__(self):
        super(autoencoder_vgg7, self).__init__()
        self.encoder = models.vgg19(pretrained=True).features

    def forward(self, x, upsampleH=224, upsampleW=224):
        feat_out = [] # we only use high level features
        for i in range(len(self.encoder)):
            # print("layer {} encoder layer: {}".format(i, self.encoder[i]))
            x = self.encoder[i](x)
            if i == 3: # ReLU-4
                feature = torch.mean(torch.nn.UpsamplingBilinear2d(size=(upsampleH, upsampleW))(x), dim=1)
                feat_out.append(feature)
            elif i == 8: # ReLU-9
                feature = torch.mean(torch.nn.UpsamplingBilinear2d(size=(upsampleH, upsampleW))(x), dim=1)
                feat_out.append(feature)
            elif i == 17: # ReLU-18
                feature = torch.mean(torch.nn.UpsamplingBilinear2d(size=(upsampleH, upsampleW))(x), dim=1)
                feat_out.append(feature)
        return feat_out, x

# PoseNet (SE(3)) w/ mobilev2 backbone
class PoseNetV2(nn.Module):
    def __init__(self, feat_dim=12):
        super(PoseNetV2, self).__init__()
        self.backbone_net = models.mobilenet_v2(pretrained=True)
        self.feature_extractor = self.backbone_net.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_pose = nn.Linear(1280, feat_dim)
    
    def _aggregate_feature(self, x, upsampleH, upsampleW):
        '''
        assume target and nerf rgb are inferenced at the same time,
        slice target batch and nerf batch and aggregate features
        :param x: image blob (2B x C x H x W)
        :param upsampleH: New H
        :param upsampleW: New W
        :return feature: (2 x B x H x W)
        '''
        batch = x.shape[0] # should be target batch_size + rgb batch_size
        feature_t = torch.mean(torch.nn.UpsamplingBilinear2d(size=(upsampleH, upsampleW))(x[:batch//2]), dim=1)
        feature_r = torch.mean(torch.nn.UpsamplingBilinear2d(size=(upsampleH, upsampleW))(x[batch//2:]), dim=1)
        feature = torch.stack([feature_t, feature_r])
        return feature

    def _aggregate_feature2(self, x):
        '''
        assume target and nerf rgb are inferenced at the same time,
        slice target batch and nerf batch and output stacked features
        :param x: image blob (2B x C x H x W)
        :return feature: (2 x B x C x H x W)
        '''
        batch = x.shape[0] # should be target batch_size + rgb batch_size
        feature_t = x[:batch//2]
        feature_r = x[batch//2:]
        feature = torch.stack([feature_t, feature_r])
        return feature

    def forward(self, x, upsampleH=224, upsampleW=224, isTrain=False, isSingleStream=False):
        '''
        Currently under dev.
        :param x: image blob ()
        :param upsampleH: New H obsolete
        :param upsampleW: New W obsolete
        :param isTrain: True to extract features, False only return pose prediction. Really should be isExtractFeature
        :param isSingleStrea: True to inference single img, False to inference two imgs in siemese network fashion
        '''
        feat_out = [] # we only use high level features
        for i in range(len(self.feature_extractor)):
            # print("layer {} encoder layer: {}".format(i, self.feature_extractor[i]))
            x = self.feature_extractor[i](x)

            if isTrain: # collect aggregate features
                if i >= 17 and i <= 17: # 17th block
                    if isSingleStream:
                        feature = torch.stack([x])
                    else:
                        feature = self._aggregate_feature2(x)
                    feat_out.append(feature)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        predict = self.fc_pose(x)
        return feat_out, predict

class EfficientNetB3(nn.Module):
    ''' EfficientNet-B3 backbone,
    model ref: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py 
    '''
    def __init__(self, feat_dim=12, feature_block=6):
        super(EfficientNetB3, self).__init__()
        self.backbone_net = EfficientNet.from_pretrained('efficientnet-b3')
        self.feature_block = feature_block #  determine which block's feature to use, max=6
        if self.feature_block == 6:
            self.feature_extractor = self.backbone_net.extract_features
        else:
            self.feature_extractor = self.backbone_net.extract_endpoints
        
        # self.feature_extractor = self.backbone_net.extract_endpoints # it can restore middle layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_pose = nn.Linear(1536, feat_dim) # 1280 for efficientnet-b0, 1536 for efficientnet-b3

    def _aggregate_feature2(self, x):
        '''
        assume target and nerf rgb are inferenced at the same time,
        slice target batch and nerf batch and output stacked features
        :param x: image blob (2B x C x H x W)
        :return feature: (2 x B x C x H x W)
        '''
        batch = x.shape[0] # should be target batch_size + rgb batch_size
        feature_t = x[:batch//2]
        feature_r = x[batch//2:]
        feature = torch.stack([feature_t, feature_r])
        return feature

    def forward(self, x, return_feature=False, isSingleStream=False):
        '''
        Currently under dev.
        :param x: image blob ()
        :param return_feature: True to extract features, False only return pose prediction. Really should be isExtractFeature
        :param isSingleStream: True to inference single img, False to inference two imgs in siemese network fashion
        '''
        # pdb.set_trace()
        feat_out = [] # we only use high level features
        if self.feature_block == 6:
            x = self.feature_extractor(x)
            fe = x.clone() # features to save
        else:
            list_x = self.feature_extractor(x)
            fe = list_x['reduction_'+str(self.feature_block)]
            x = list_x['reduction_6'] # features to save
        if return_feature:
            if isSingleStream:
                feature = torch.stack([fe])
            else:
                feature = self._aggregate_feature2(fe)
            feat_out.append(feature)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        predict = self.fc_pose(x)
        return feat_out, predict