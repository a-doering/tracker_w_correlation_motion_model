from tracktor.config import cfg
import torch
from tracktor.frcnn_fpn import FRCNN_FPN
from torchvision.transforms import ToTensor

import configparser
import h5py 
import os.path as osp
from PIL import Image
import numpy as np
import csv


def enlarge_boxes(bb, boxes_enlargement_factor):
    """Enlarges bounding box widht and height by some factor."""
    if boxes_enlargement_factor > 1.0:
        delta = (boxes_enlargement_factor - 1) / 2

        width_delta = (bb[2] - bb[0]) * delta
        height_delta = (bb[3] - bb[1]) * delta

        bb[0] -= width_delta
        bb[1] -= height_delta
        bb[2] += width_delta
        bb[3] += height_delta

    return bb

def clip_boxes_to_image(bb, size):
    """Clips boxes to size"""
    height, width = size
    bb[0] = np.clip(bb[0], 0, width)
    bb[1] = np.clip(bb[1], 0, height)
    bb[2] = np.clip(bb[2], 0, width)
    bb[3] = np.clip(bb[3], 0, height)
    return bb

# Load model
print('Loading model...')
obj_detect = FRCNN_FPN(num_classes=2)
obj_detect.load_state_dict(torch.load("output/faster_rcnn_fpn_training_mot_17/model_epoch_27.model",
                            map_location=lambda storage, loc: storage))
obj_detect.eval()
obj_detect.cuda()
print('Model loaded!')

# Hardcoded loader for MOT17
mot_dir = osp.join(cfg.DATA_DIR, 'MOT17Det', 'train')
sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']

# Open hdf5 file and create arrays
print(100*'#')
h5_file = osp.join(cfg.DATA_DIR, 'correlation_dataset', 'correlation_dataset.hdf5')
print(h5_file)

h5 = h5py.File(h5_file, mode='w')
for seq in sequences:
    seq_im_dir = osp.join(mot_dir, seq)
    gt_file = osp.join(mot_dir, seq, 'gt', 'gt.txt')
    config_file = osp.join(mot_dir, seq, 'seqinfo.ini')

    assert osp.exists(config_file), \
        'Config file does not exist: {}'.format(config_file)

    config = configparser.ConfigParser()
    config.read(config_file)
    seqLength = int(config['Sequence']['seqLength'])
    imDir = config['Sequence']['imDir']
    imWidth = int(config['Sequence']['imWidth'])
    imHeight = int(config['Sequence']['imHeight'])

    # Constructing tracks, one sample per frame. Access boxes like: boxes[frame][id]
    total = {}
    boxes = {}
    visibility = {}

    for i in range(1, seqLength+1):
        boxes[i] = {}
        visibility[i] = {}

    #TODO set a parameter
    vis_threshold = 0.3 #
    boxes_enlargement_factor = 1.2

    no_gt = False
    if osp.exists(gt_file):
        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=',')
            for row in reader:
                # class person, certainity 1, visibility >= 0.25
                if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= vis_threshold:
                    # Make pixel indexes 0-based, should already be 0-based (or not)
                    x1 = int(row[2]) - 1
                    y1 = int(row[3]) - 1
                    # This -1 accounts for the width (width of 1 x1=x2)
                    x2 = x1 + int(row[4]) - 1
                    y2 = y1 + int(row[5]) - 1
                    bb = np.array([x1,y1,x2,y2], dtype=np.float32)
                    boxes[int(row[0])][int(row[1])] = bb
                    visibility[int(row[0])][int(row[1])] = float(row[8])
    else:
        no_gt = True

    id_in_frame = {}

    for i in range(1,seqLength+1):
        im_path = osp.join(imDir,"{:06d}.jpg".format(i))
        id_in_frame[i] = boxes[i].keys()
        sample = {'gt':boxes[i],
                    'im_path':im_path,
                    'vis':visibility[i]}

        total[i] = sample

    # Find all ids that are in this frame and next
    id_in_frame_and_next = {}
    for i in range(1, seqLength):
        id_in_frame_and_next[i] = set(id_in_frame[i]).intersection(id_in_frame[i+1])
        #print(id_in_frame_and_next[i])

    num_pairs = sum([len(x) for x in id_in_frame_and_next.values()])
    num_detections = sum([len(x) for x in id_in_frame.values()])
    print(100*'#')
    print("Sequence: {}".format(seq))
    print("Number of detections: {}".format(num_detections))
    print("Number of pairs: {}".format(num_pairs))
    # Create a group per sequence. 
    # This allows fixed size shape for the datasets of size sum()
    group = h5.create_group(seq)
    h5_dataset_1 = group.create_dataset("fmap", (num_pairs, 256, 7,7), dtype=np.float32)
    h5_dataset_2 = group.create_dataset("fmap_enlarged", (num_pairs, 256, 7,7), dtype=np.float32)
    h5_dataset_3 = group.create_dataset("labels",(num_pairs, 4), dtype=np.float32)
    h5_dataset_4 = group.create_dataset("names",(num_pairs,),dtype=h5py.special_dtype(vlen=str))

    pairs_stored = 0
    # Create feature maps
    for i in range(1, seqLength):

        print('####### Frame {:04} from sequence {} #######'.format(i, seq))
        print(id_in_frame_and_next[i])
        pairs_in_frame= len(id_in_frame_and_next[i])
        
        # Nothing to do here, prevent error
        if pairs_in_frame == 0:
            continue
        #print(pairs_in_frame)
        boxes = [total[i]['gt'][id] for id in id_in_frame_and_next[i]] # bounding boxes of the previous frame
        boxes_next = [total[i+1]['gt'][id] for id in id_in_frame_and_next[i]]
        enlarged_boxes = [clip_boxes_to_image(enlarge_boxes(total[i]['gt'][id], boxes_enlargement_factor), (imHeight, imWidth))  for id in id_in_frame_and_next[i]]# enlarged bounding boxes of the previous frame
        # Sequence Name _ frame (of first image) _ id
        # MOT17-02_000001_000001
        names = [seq + '_{:06}_'.format(i) + '{:06}'.format(id) for id in id_in_frame_and_next[i]]
        
        boxes = torch.tensor(boxes)
        enlarged_boxes = torch.tensor(enlarged_boxes)

        img = Image.open(osp.join(seq_im_dir, total[i]['im_path'])).convert("RGB")
        img = np.array(img,dtype=np.float32)
        img = img.transpose((2,0,1))
        img = torch.from_numpy(img)#1080,1920,3
        img.unsqueeze_(0)
        print(img.shape, img.type, img.type())#1,3,1080,1920 ---> yes, torch object
        obj_detect.load_image(img) # load previous frame

        img = Image.open(osp.join(seq_im_dir, total[i+1]['im_path'])).convert("RGB")
        img = np.array(img,dtype=np.float32)
        img = img.transpose((2,0,1))
        img = torch.from_numpy(img)
        img.unsqueeze_(0)
        obj_detect.load_image(img) # load current frame

        prev_7x7_features, current_7x7_features = obj_detect.get_feature_patches(boxes, enlarged_boxes)
        print(prev_7x7_features.shape) # _7x7_features.shape -> [#boxes, 256, 7, 7]

        h5_dataset_1[pairs_stored:pairs_stored+pairs_in_frame] = prev_7x7_features.data.cpu().numpy()
        h5_dataset_2[pairs_stored:pairs_stored+pairs_in_frame] = current_7x7_features.data.cpu().numpy()
        h5_dataset_3[pairs_stored:pairs_stored+pairs_in_frame] = boxes_next
        h5_dataset_4[pairs_stored:pairs_stored+pairs_in_frame] = names

        pairs_stored += pairs_in_frame
        print(pairs_stored)

    # For developing, only do the first couple.
    #     if i == 20:
    #         break
    # if seq == 'MOT17-02':
    #     break
h5.close()