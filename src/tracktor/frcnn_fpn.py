from collections import OrderedDict

import torch
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes


class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes, correlation_head=None):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes)
        # these values are cached to allow for feature reuse
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None

        self.prev_original_image_sizes = None
        self.prev_preprocessed_images = None
        self.prev_features = None

        self.correlation_head = correlation_head
        # if not self.correlation_head:
        #     self.correlation_head = CorrelationHead()

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach(), detections['scores'].detach()

    def predict_boxes(self, boxes):
        device = list(self.parameters())[0].device
        boxes = boxes.to(device)

        boxes = resize_boxes(boxes, self.original_image_sizes[0], self.preprocessed_images.image_sizes[0])
        proposals = [boxes]

        box_features = self.roi_heads.box_roi_pool(self.features, proposals, self.preprocessed_images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)

        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        # score_thresh = self.roi_heads.score_thresh
        # nms_thresh = self.roi_heads.nms_thresh

        # self.roi_heads.score_thresh = self.roi_heads.nms_thresh = 1.0
        # self.roi_heads.score_thresh = 0.0
        # self.roi_heads.nms_thresh = 1.0
        # detections, detector_losses = self.roi_heads(
        #     features, [boxes.squeeze(dim=0)], images.image_sizes, targets)

        # self.roi_heads.score_thresh = score_thresh
        # self.roi_heads.nms_thresh = nms_thresh

        # detections = self.transform.postprocess(
        #     detections, images.image_sizes, original_image_sizes)

        # detections = detections[0]
        # return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()

        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()
        pred_boxes = resize_boxes(pred_boxes, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()
        return pred_boxes, pred_scores

    def predict_with_correlation(self, prev_boxes, current_boxes):

        prev_boxes_features, current_boxes_features = self.get_feature_patches(prev_boxes, current_boxes)

        boxes_deltas = self.correlation_head(prev_boxes_features, current_boxes_features)
        
        pred_boxes = self.roi_heads.box_coder.decode(boxes_deltas, [prev_boxes]).squeeze(dim=1)
        #pred_boxes = resize_boxes(pred_boxes, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])

        return pred_boxes

    def get_feature_patches(self, prev_boxes, current_boxes):
        device = list(self.parameters())[0].device
        prev_boxes = prev_boxes.to(device)
        current_boxes = current_boxes.to(device)

        prev_boxes = resize_boxes(prev_boxes, self.prev_original_image_sizes[0], self.prev_preprocessed_images.image_sizes[0])
        current_boxes = resize_boxes(current_boxes, self.original_image_sizes[0], self.preprocessed_images.image_sizes[0])

        prev_boxes_features = self.roi_heads.box_roi_pool(self.prev_features, [prev_boxes], self.prev_preprocessed_images.image_sizes)
        current_boxes_features = self.roi_heads.box_roi_pool(self.features, [current_boxes], self.preprocessed_images.image_sizes)

        return prev_boxes_features, current_boxes_features

    def load_image(self, images):
        device = list(self.parameters())[0].device
        images = images.to(device)

        # saving last image
        self.prev_original_image_sizes = self.original_image_sizes
        self.prev_preprocessed_images = self.preprocessed_images
        self.prev_features = self.features

        # getting new feature map
        self.original_image_sizes = [img.shape[-2:] for img in images]

        preprocessed_images, _ = self.transform(images, None)
        self.preprocessed_images = preprocessed_images
        
        self.features = self.backbone(preprocessed_images.tensors)
        if isinstance(self.features, torch.Tensor):
            self.features = OrderedDict([(0, self.features)])
