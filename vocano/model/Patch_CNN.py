import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return
    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.conv2d_1 = self.__conv(2, name='conv2d_1', in_channels=1, out_channels=8, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.conv2d_2 = self.__conv(2, name='conv2d_2', in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.dense_1 = self.__dense(name = 'dense_1', in_features = 1296, out_features = 128, bias = True)
        self.dense_2 = self.__dense(name = 'dense_2', in_features = 128, out_features = 64, bias = True)
        self.dense_3 = self.__dense(name = 'dense_3', in_features = 64, out_features = 2, bias = True)

    def forward(self, x):
        conv2d_1        = self.conv2d_1(x)
        conv2d_1_activation = F.relu(conv2d_1)
        dropout_1       = F.dropout(input = conv2d_1_activation, p = 0.25, training = self.training, inplace = True)
        conv2d_2        = self.conv2d_2(dropout_1)
        conv2d_2_activation = F.relu(conv2d_2)
        dropout_2       = F.dropout(input = conv2d_2_activation, p = 0.25, training = self.training, inplace = True)
        max_pooling2d_1, max_pooling2d_1_idx = F.max_pool2d(dropout_2, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        flatten_1       = max_pooling2d_1.view(max_pooling2d_1.size(0), -1)
        dense_1         = self.dense_1(flatten_1)
        dense_1_activation = F.relu(dense_1)
        dropout_3       = F.dropout(input = dense_1_activation, p = 0.25, training = self.training, inplace = True)
        dense_2         = self.dense_2(dropout_3)
        dense_2_activation = F.relu(dense_2)
        dropout_4       = F.dropout(input = dense_2_activation, p = 0.25, training = self.training, inplace = True)
        dense_3         = self.dense_3(dropout_4)
        dense_3_activation = F.softmax(dense_3, dim=1)
        return dense_3_activation


    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

