from .mot_sequence import MOT17Sequence
from ..config import get_output_dir

import cv2
from PIL import Image
import numpy as np
import os.path as osp

from torchvision.ops.boxes import clip_boxes_to_image

class MOTcorrelation(MOT17Sequence):
    """Multiple object tracking dataset.
    
    This class builds samples for training a siamese net called correlation head. 
    It returns a pair of two patches that are cropped around the bounding box (bb) of frame t and 
    the enlarged bb of t in frame t+1 and also returns the bb ground truth in t+1. The crops can be precalculated.
    """

    def __init__(self, seq_name, split, vis_threshold, boxes_enlargement_factor, frames_apart, image_shape):
        super().__init__(seq_name, vis_threshold=vis_threshold)

        self.boxes_enlargement_factor = boxes_enlargement_factor
        self.frames_apart = frames_apart
        self.image_shape = image_shape

        self.build_samples()

        if split == 'train':
            pass
        elif split == 'small_train':
            self.data = self.data[0::5] + self.data[1::5] + self.data[2::5] + self.data[3::5]
        elif split == 'small_val':
            self.data = self.data[4::5]
        else:
            raise NotImplementedError("Split: {}".format(split))  

    def __getitem__(self, idx):
        """Returns the ith pair"""
        pair = self.data[idx]

        # concatenate the patches
        r = []
        r.append(pair[0])
        r.append(pair[1])
        patches = torch.stack(r,0)

        labels = np.array(pair[2])

        batch = [patches, labels]
        return batch

    def build_samples(self):
        """Builds the samples for the correlation layer out of the sequence"""
        
        tracks = {}

        for sample in self.data:
            im_path = sample['im_path']
            gt = sample['gt']

            for k,v in tracks.items():
                if k in gt.keys():
                    v.append({'id':k, 'im_path':im_path, 'gt':gt[k]})
                    del gt[k]

            # For all remaining BB in gt new tracks are created
            for k,v in gt.items():
                tracks[k] = [{'id':k, 'im_path':im_path, 'gt':v}]
        
        res = []
        # Loop through each track
        for frame, track in tracks.items():

            # Loop through each frame
            for idx in range(len(track)-1):
                
                # Check if frames are not too far apart, e.g. not lost for several frames
                if abs(int(osp.splitext(osp.basename(track[idx]['im_path']))[0]) - 
                    int(osp.splitext(osp.basename(track[idx+1]['im_path']))[0])) <= self.frames_apart:

                    pair = {}
                    # Cropped to the bounding box size and to the enlarged bb size
                    pair[0] = self.build_crop(track[idx]['im_path'], track[idx]['gt'])
                    pair[1] = self.build_crop(track[idx+1]['im_path'], self.clip_boxes_to_image(self.enlarge_boxes(track[idx]['gt']), self.image_shape))
                    # Ground truth for idx+1
                    pair[2] = track[idx+1]['gt']
                    res.append(np.array(pair))
                 
        if self._seq_name:
            print("[*] Loaded {} pairs from sequence {}.".format(len(res), self._seq_name))
        # crop t-1, crop enlarged t, put bb of t as output
        self.data = res
        
    def build_crop(self, im_path, bb):
        """Crops out a bounding box"""
        im = cv2.imread(im_path)

        # Assuming bb format  <bb_left>, <bb_top>, <bb_right>, <bb_bottom>
        # See mot_sequence.py  _sequence function for formatting
        crop = im[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]

        return crop

    def enlarge_boxes(self, bb):
        """Enlarges bounding box widht and height by some factor."""
        if self.boxes_enlargement_factor > 1.0:
            delta = (self.boxes_enlargement_factor - 1) / 2

            width_delta = (bb[2] - bb[0]) * delta
            height_delta = (bb[3] - bb[1]) * delta

            bb[0] -= width_delta
            bb[1] -= height_delta
            bb[2] += width_delta
            bb[3] += height_delta

        return bb

    def clip_boxes_to_image(self, bb, size):
        """Clips boxes to size"""
        height, width = size
        bb[0] = np.clip(bb[0], 0, width)
        bb[1] = np.clip(bb[1], 0, height)
        bb[2] = np.clip(bb[2], 0, width)
        bb[3] = np.clip(bb[3], 0, height)
        return bb