import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from spatial_correlation_sampler import SpatialCorrelationSampler

class CorrelationHead(nn.Module):
    def __init__(self):
        super(CorrelationHead, self).__init__()

        representation_size = 1024
        feature_map_res = 7
        correlation_patch_size = 16 # flownet uses 21, if we want to use faster-rcnn weight needs to be 16

        self.name = "CorrelationHead"

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

    def losses(self, batch, loss):

        patch1, patch2, gt_boxes = batch

        patch1 = Variable(patch1).cuda()
        patch2 = Variable(patch2).cuda()

        gt_boxes = gt_boxes.cuda()

        # print("fmap:")
        # print(patch1*100)
        # print("fmap_enlarged:")
        # print(patch2*100)
        # print("labels:")
        # print(gt_boxes)

        pred_boxes = self.forward(patch1, patch2)

        if loss == "GIoU":
            total_loss = self.giou_loss(pred_boxes, gt_boxes)
        elif loss == "MSE":
            total_loss = F.mse_loss(pred_boxes, gt_boxes)
        else:
            raise NotImplementedError("Loss: {}".format(loss))

        return total_loss

    def giou_loss(self, pred_boxes_in, gt_boxes):

        pred_boxes = pred_boxes_in.clone().cuda()
        pred_boxes[:,0] = torch.min(pred_boxes_in[:,0], pred_boxes_in[:,2])
        pred_boxes[:,1] = torch.min(pred_boxes_in[:,1], pred_boxes_in[:,3])
        pred_boxes[:,2] = torch.max(pred_boxes_in[:,0], pred_boxes_in[:,2])
        pred_boxes[:,3] = torch.max(pred_boxes_in[:,1], pred_boxes_in[:,3])

        enclosing_boxes = torch.zeros_like(pred_boxes).cuda()
        enclosing_boxes[:,0] = torch.min(pred_boxes[:,0], gt_boxes[:,0])
        enclosing_boxes[:,1] = torch.min(pred_boxes[:,1], gt_boxes[:,1])
        enclosing_boxes[:,2] = torch.max(pred_boxes[:,2], gt_boxes[:,2])
        enclosing_boxes[:,3] = torch.max(pred_boxes[:,3], gt_boxes[:,3])

        pred_boxes_areas = (pred_boxes[:, 2] - pred_boxes[:, 0] + 1) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1)
        gt_boxes_areas = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
        enclosing_areas = (enclosing_boxes[:, 2] - enclosing_boxes[:, 0] + 1) * (enclosing_boxes[:, 3] - enclosing_boxes[:, 1] + 1)

        iw = (torch.min(pred_boxes[:, 2], gt_boxes[:, 2]) - torch.max(pred_boxes[:, 0], gt_boxes[:, 0]) + 1).clamp(min=0)
        ih = (torch.min(pred_boxes[:, 3], gt_boxes[:, 3]) - torch.max(pred_boxes[:, 1], gt_boxes[:, 1]) + 1).clamp(min=0)
        ua = pred_boxes_areas + gt_boxes_areas - iw * ih
        iou = iw * ih / ua
        giou = iou - (enclosing_areas - ua) / enclosing_areas

        return torch.mean(1 - giou)

