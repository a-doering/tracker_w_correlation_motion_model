import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from spatial_correlation_sampler import SpatialCorrelationSampler
from torchvision.ops.boxes import box_iou
from torchvision.models.detection.transform import resize_boxes

class CorrelationHead(nn.Module):
    def __init__(self):
        super(CorrelationHead, self).__init__()

        representation_size = 1024
        feature_map_res = 7
        correlation_patch_size = 16 # flownet uses 21, if we want to use faster-rcnn weight needs to be 16

        self.name = "CorrelationHead"
        self.roi_heads = None

        self.correlation_layer = SpatialCorrelationSampler(patch_size=correlation_patch_size, dilation_patch=2)
        self.fc1 = nn.Linear(correlation_patch_size ** 2 * feature_map_res ** 2, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)
        self.fc3 = nn.Linear(representation_size, 4)

    def forward(self, patch1, patch2):
        x = self.correlation_layer(patch1, patch2)
        
        x = x.flatten(start_dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        boxes_deltas = self.fc3(x)

        return boxes_deltas

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

        patch1, patch2, gt_boxes, prev_boxes, _, _, _ , preprocessed_image_sizes, original_image_sizes = batch

        patch1 = Variable(patch1).cuda()
        patch2 = Variable(patch2).cuda()

        gt_boxes = gt_boxes.cuda()
        prev_boxes = prev_boxes.cuda()

        # print("fmap:")
        # print(patch1*100)
        # print("fmap_enlarged:")
        # print(patch2*100)
        # print("labels:")
        # print(gt_boxes)

        boxes_deltas = self.forward(patch1, patch2)

        pred_boxes = self.roi_heads.box_coder.decode(boxes_deltas, [prev_boxes]).squeeze(dim=1)
        #pred_boxes = resize_boxes(pred_boxes, preprocessed_image_sizes[0], original_image_sizes[0])

        if loss == "GIoU":
            total_loss = self.giou_loss(pred_boxes, gt_boxes)
        elif loss == "IoU":
            total_loss = box_iou(pred_boxes, gt_boxes).diag()
            total_loss = torch.mean(total_loss)
        elif loss == "MSE":
            total_loss = F.mse_loss(pred_boxes, gt_boxes)
        elif loss == "fasterRCNN":
            total_loss = self.smooth_l1_loss(pred_boxes, gt_boxes)
            total_loss /= len(gt_boxes)
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

    def smooth_l1_loss(self, input, target, beta: float = 1. / 9, size_average: bool = True):
        """
        very similar to the smooth_l1_loss from pytorch, but with
        the extra beta parameter
        """
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        if size_average:
            return loss.mean()
        return loss.sum()