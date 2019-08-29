import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    """Flatten module
    """
    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)

class CNN_Model(nn.Module):
    """CNN Model
    """
    def __init__(self, num_input, num_classes, cnn_config):
        """ Initialize CNN Model

        params:
            num_input: number of input, tuple
        """
        cnn_filter_size = [c[0] for c in cnn_config]    # listing filter_size
        cnn_num_filters = [c[1] for c in cnn_config]    # listing cnn num of filters
        max_pool_ksize = [c[2] for c in cnn_config]     # listing max pool ksize
        dropout_rate = [c[3] for c in cnn_config]       # dropout rate

        assert len(cnn) == len(cnn_num_filters)

        # build module
        modules = []
        c_in = num_input
        for idx in range(len(cnn)):
            # TODO naming
            # conv2d
            conv = nn.Conv2d(
                c_in, 
                cnn_num_filters[idx], 
                cnn_filter_size[idx], 
                stride=1,
                padding=cnn_filter_size[idx]//2,
            )
            # maxpool2d
            maxpool = nn.MaxPool2d(
                max_pool_ksize[idx], 
                stride=1,
                padding=max_pool_ksize[idx]//2
            )
            # dropout
            dropout = nn.Dropout(dropout_rate[idx])

            modules.extend([conv, maxpool, dropout])
        flatten = Flatten()
        # TODO add linear
        self.logits = nn.Linear()
        
