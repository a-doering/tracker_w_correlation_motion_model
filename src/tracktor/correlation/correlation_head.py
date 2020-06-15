
import torch
import torch.nn as nn

from spatial_correlation_sampler import SpatialCorrelationSampler

class CorrelationHead(nn.Module):
    def __init__(self):
        super(CorrelationHead, self).__init__()

        self.correlation_layer = SpatialCorrelationSampler(patch_size=21, dilation_patch=2)
        self.bbox_pred = nn.Linear(21 * 21 * 7 * 7, 4)

    def forward(self, patch1, patch2):
        #print(patch1.shape)
        #print(patch2.shape)
        x = self.correlation_layer(patch1, patch2)
        #print(x.shape)

        x = x.flatten(start_dim=1)
        #print(x.shape)
        bbox_deltas = self.bbox_pred(x)

        #pred_boxes = resize_boxes(pred_boxes, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])
        return bbox_deltas


