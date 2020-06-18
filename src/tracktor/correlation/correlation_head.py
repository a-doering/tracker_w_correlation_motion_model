import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_correlation_sampler import SpatialCorrelationSampler

class CorrelationHead(nn.Module):
    def __init__(self):
        super(CorrelationHead, self).__init__()

        representation_size = 1024
        feature_map_res = 7
        correlation_patch_size = 16 # flownet uses 21, if we want to use faster-rcnn weight needs to be 16

        self.correlation_layer = SpatialCorrelationSampler(patch_size=correlation_patch_size, dilation_patch=2)
        self.fc1 = nn.Linear(correlation_patch_size ** 2 * feature_map_res ** 2, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)
        self.fc3 = nn.Linear(representation_size, 4)

    def forward(self, patch1, patch2):
        x = self.correlation_layer(patch1, patch2)
        
        x = x.flatten(start_dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pred_boxes = self.fc3(x)

        #pred_boxes = resize_boxes(pred_boxes, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])
        return pred_boxes

    def load_from_rcnn(self, rcnn_model):
        rcnn_model = torch.load(rcnn_model)
        with torch.no_grad():
            self.fc1.weight.copy_(rcnn_model["roi_heads.box_head.fc6.weight"])
            self.fc1.bias.copy_(rcnn_model["roi_heads.box_head.fc6.bias"])
            self.fc2.weight.copy_(rcnn_model["roi_heads.box_head.fc7.weight"])
            self.fc2.bias.copy_(rcnn_model["roi_heads.box_head.fc7.bias"])
            self.fc3.weight.copy_(rcnn_model["roi_heads.box_predictor.bbox_pred.weight"][-4:])
            self.fc3.bias.copy_(rcnn_model["roi_heads.box_predictor.bbox_pred.bias"][-4:])
